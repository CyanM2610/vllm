# SPDX-License-Identifier: Apache-2.0
import io, json, time, unittest
import test_native_compute_selection as t

s = t.s
ROOT = t.ROOT
AREA = t.AREA
RUNTIME = t.RUNTIME


def model():
    mod = s.module(
        "_cpu_planner_tests", RUNTIME / "source/vllm/tests/v1/core/test_compute_slo.py"
    )
    return mod.controlled_model()


def enqueue(f, rid, length, age_ms=0):
    r = s.native.make_request(f.rt, rid, length)
    r.sampling_params.num_logprobs = None
    r.sampling_params.extra_args = dict(
        prefill_eligible_ns=time.monotonic_ns() - round(age_ms * 1e6),
        prefill_slo_ns=3_000_000_000,
        prefill_clock_id="controlled-clock",
    )
    f.scheduler.add_request(r)
    return r


W = []


class SLOIntegration(unittest.TestCase):
    def tearDown(self):
        s.close_fixtures()

    def test_common_preparation_same_step_information_for_all_modes(self):
        outputs = {}
        for mode in ("native", "sjf", "slo"):
            f = t.fixture(mode, completion_model=model(), prepare=True)
            long = enqueue(f, "long", 16384)
            s.t.known_result(f, long, 0)
            f.scheduler.schedule()
            short = enqueue(f, "short", 512)
            out = f.scheduler.schedule()
            self.assertEqual(out.num_scheduled_tokens, {"long": 2048})
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("short"), ([],))
            s.ops.ack(f.rpc, "short")
            f.scheduler.schedule()
            s.ops.status(f.rpc, 0, "short")
            out = f.scheduler.schedule()
            outputs[mode] = out.num_scheduled_tokens
            self.assertEqual(f.selection.keys["short"].step, f.scheduler.current_step)
            self.assertEqual(
                sum(c[0] == "LOOKUP" and c[2] == "short" for c in f.rpc.calls), 1
            )
            summaries = [
                e
                for e in f.rt.events
                if e["event"] == "prefill_first_match_prepare_step"
            ]
            self.assertEqual(sum(e["lookup_ack_observed"] for e in summaries), 1)
            self.assertEqual(sum(e["lookup_status_submitted"] for e in summaries), 1)
            self.assertEqual(sum(e["lookup_status_observed"] for e in summaries), 1)
            self.assertEqual(short.num_output_tokens, 0)
            self.assertFalse(f.scheduler._decoupled_control)
        self.assertEqual(outputs["native"], {"long": 2048})
        self.assertEqual(outputs["sjf"], {"short": 512, "long": 1536})
        self.assertEqual(outputs["slo"], outputs["sjf"])
        W.append(
            dict(kind="common_prepare_actual_same_step_three_modes", actual=outputs)
        )

    def test_explicit_small_batch_intent_still_ends_computation(self):
        f = t.fixture("slo", completion_model=model())
        s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
        long = enqueue(f, "long", 8192)
        s.t.known_result(f, long, 0)
        first = f.scheduler.schedule()
        w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
        cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
        f.scheduler.update_from_output(
            first,
            cls(req_ids=["long"], req_id_to_index={"long": 0}, sampled_token_ids=[[]]),
        )
        enqueue(f, "pending", 512)
        actual = f.selection.planner.plan

        def choose_real_small_candidate(*args, **kwargs):
            result = actual(*args, **kwargs)
            result["selected"] = next(
                c for c in result["candidates"] if c.get("end_compute_reason")
            )
            return result

        f.selection.planner.plan = choose_real_small_candidate
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"long": 1024})
        self.assertFalse(f.adapter.lookup_operation_state("pending")["submitted"])
        final = [e for e in f.rt.events if e["event"] == "native_compute_step"][-1]
        self.assertEqual(final["traversal_end_reason"], "policy_first_batch_cap")
        self.assertTrue(final["forecast_executed"])

    def test_preparation_cap_unknown_after_scope_remains_waiting_without_stop(self):
        f = t.fixture("slo", completion_model=model(), prepare=True, prepare_limit=1)
        s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
        r = enqueue(f, "modeled", 2304)
        s.t.known_result(f, r, 0)
        first = f.scheduler.schedule()
        w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
        cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
        f.scheduler.update_from_output(
            first,
            cls(
                req_ids=["modeled"],
                req_id_to_index={"modeled": 0},
                sampled_token_ids=[[]],
            ),
        )
        enqueue(f, "query_pending", 512)
        enqueue(f, "cap_unknown", 512)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"modeled": 256})
        self.assertIsNone(f.selection.plan_invalid)
        self.assertTrue(f.adapter.lookup_operation_state("query_pending")["submitted"])
        self.assertFalse(f.adapter.lookup_operation_state("cap_unknown")["submitted"])
        self.assertTrue(
            any(
                e.get("request_id") == "cap_unknown"
                and e.get("reason") == "first_match_not_prepared"
                for e in f.rt.events
            )
        )

    def test_common_slo_restore_pending_and_native_adoption(self):
        f = t.fixture("slo", completion_model=model(), prepare=True)
        restored = enqueue(f, "restore", 1024)
        cold = enqueue(f, "cold", 8192)
        s.t.known_result(f, restored, 2)
        s.t.known_result(f, cold, 0)
        first = f.scheduler.schedule()
        self.assertEqual(first.num_scheduled_tokens, {"cold": 2048})
        loads = [
            m for m in first.kv_connector_metadata.requests if m.direction == "RETRIEVE"
        ]
        self.assertEqual([m.request_id for m in loads], ["restore"])
        key, arrival = f.selection.keys["restore"], f.selection.arrivals["restore"]
        self.assertEqual(restored.status, f.rt.status.WAITING_FOR_REMOTE_KVS)
        pending = f.scheduler.schedule()
        self.assertEqual(pending.num_scheduled_tokens, {"cold": 2048})
        self.assertFalse(
            any(
                m.direction == "RETRIEVE"
                for m in pending.kv_connector_metadata.requests
            )
        )
        s.received(f, "restore")
        ready = f.scheduler.schedule()
        self.assertEqual(ready.num_scheduled_tokens["restore"], 512)
        self.assertGreater(ready.num_scheduled_tokens["cold"], 0)
        self.assertLessEqual(sum(ready.num_scheduled_tokens.values()), 2048)
        self.assertEqual(key, f.selection.keys["restore"])
        self.assertEqual(arrival, f.selection.arrivals["restore"])
        self.assertFalse(f.scheduler._decoupled_control)
        W.append(
            dict(
                kind="common_slo_native_restore_adoption",
                pending=pending.num_scheduled_tokens,
                after_native_adoption=ready.num_scheduled_tokens,
            )
        )

    def test_modeled_scope_exhaustion_continues_and_invalidates_on_compute(self):
        f = t.fixture("slo", completion_model=model())
        s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
        long = enqueue(f, "modeled", 2304)
        s.t.known_result(f, long, 0)
        first = f.scheduler.schedule()
        w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
        cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
        f.scheduler.update_from_output(
            first,
            cls(
                req_ids=["modeled"],
                req_id_to_index={"modeled": 0},
                sampled_token_ids=[[]],
            ),
        )
        new = enqueue(f, "new", 512)
        s.t.known_result(f, new, 0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"modeled": 256, "new": 512})
        self.assertEqual(
            f.selection.plan_invalid, "unmodeled_or_unselected_compute_proposal"
        )
        final = [e for e in f.rt.events if e["event"] == "native_compute_step"][-1]
        self.assertFalse(final["forecast_executed"])
        W.append(
            dict(
                kind="slo_scope_exhausted_compute_fallback",
                selected=out.num_scheduled_tokens,
                final=final,
            )
        )

    def test_modeled_scope_exhaustion_keeps_dependency_only_visit(self):
        f = t.fixture("slo", completion_model=model())
        s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
        r = enqueue(f, "modeled", 2304)
        s.t.known_result(f, r, 0)
        first = f.scheduler.schedule()
        w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
        cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
        f.scheduler.update_from_output(
            first,
            cls(
                req_ids=["modeled"],
                req_id_to_index={"modeled": 0},
                sampled_token_ids=[[]],
            ),
        )
        enqueue(f, "pending", 512)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"modeled": 256})
        self.assertEqual(
            sum(c[0] == "LOOKUP" and c[2] == "pending" for c in f.rpc.calls), 1
        )
        self.assertIsNone(f.selection.plan_invalid)
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("pending"), ([],))

    def test_actual_slo_allocation_protects_known_long_without_new_lookup(self):
        quotas = []
        for mode in ("sjf", "slo"):
            f = t.fixture(mode, completion_model=model())
            s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
            for rid in ("short1", "short2"):
                r = enqueue(f, rid, 1024)
                s.t.known_result(f, r, 2)
            long = enqueue(f, "long", 3584, age_ms=800)
            s.t.known_result(f, long, 0)
            first = f.scheduler.schedule()
            self.assertEqual(first.num_scheduled_tokens, {"long": 2048})
            w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
            cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
            f.scheduler.update_from_output(
                first,
                cls(
                    req_ids=["long"],
                    req_id_to_index={"long": 0},
                    sampled_token_ids=[[]],
                ),
            )
            self.assertFalse(f.selection.pending)
            s.received(f, "short1")
            s.received(f, "short2")
            calls = len(f.rpc.calls)
            second = f.scheduler.schedule()
            quotas.append(second.num_scheduled_tokens)
            self.assertFalse(
                any(
                    m.direction == "RETRIEVE"
                    for m in second.kv_connector_metadata.requests
                )
            )
            self.assertFalse(
                any(
                    c[0] in ("LOOKUP", "QUERY_PREFETCH_STATUS")
                    for c in f.rpc.calls[calls:]
                )
            )
            self.assertEqual(f.scheduler.running[0].request_id, "long")
            W.append(
                dict(
                    mode=mode,
                    first=first.num_scheduled_tokens,
                    second=second.num_scheduled_tokens,
                    deadline=f.selection.arrivals["long"].deadline_ns,
                    plan=f.selection.current_plan,
                )
            )
        self.assertEqual(quotas[0], {"short1": 512, "short2": 512, "long": 1024})
        self.assertEqual(quotas[1], {"long": 1536, "short1": 512})

    def test_deadline_uses_eligible_and_never_resets_after_steps(self):
        f = t.fixture("sjf")
        r = enqueue(f, "late", 8192, age_ms=4000)
        s.t.known_result(f, r, 0)
        arrival = f.selection.arrivals["late"]
        self.assertLess(arrival.deadline_ns, time.monotonic_ns())
        f.scheduler.schedule()
        f.scheduler.schedule()
        self.assertEqual(arrival, f.selection.arrivals["late"])
        self.assertEqual(arrival.deadline_ns - arrival.eligible_ns, 3_000_000_000)

    def test_unknown_metadata_does_not_restart_clock(self):
        f = t.fixture("slo", completion_model=model())
        r = s.t.enqueue(f, "q", 512)
        s.t.known_result(f, r, 0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"q": 512})
        self.assertIsNone(f.selection.arrivals["q"].deadline_ns)

    def test_missing_model_is_rejected_without_native_state_mutation(self):
        with self.assertRaisesRegex(ValueError, "completion model"):
            t.fixture("slo")

    def test_declining_model_keeps_pending_lookup_on_native_path(self):
        f = t.fixture("slo", completion_model=model())
        enqueue(f, "pending", 512)
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
        self.assertNotIn("pending", f.selection.keys)
        self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("pending"), ([],))


if __name__ == "__main__":
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SLOIntegration)
    f = t.fixture("sjf")
    pure = s.module(
        "_cpu_slo_tests", RUNTIME / "source/vllm/tests/v1/core/test_compute_slo.py"
    )
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(pure))
    s.close_fixtures()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (AREA / "stage-b.log").write_text(stream.getvalue(), encoding="utf-8")
    (AREA / "stage-b.json").write_text(
        json.dumps(
            dict(passed=result.wasSuccessful(), tests=result.testsRun, witnesses=W),
            indent=2,
        )
        + "\n"
    )
    print(stream.getvalue())
    raise SystemExit(not result.wasSuccessful())
