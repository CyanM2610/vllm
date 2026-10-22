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
