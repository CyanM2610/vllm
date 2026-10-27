# SPDX-License-Identifier: Apache-2.0
"""Joint plan behavior through native scheduler/cache/connector/adapter methods.

Artificial service coefficients test decisions, not real latency rankings.
"""

import time
import unittest
from dataclasses import replace

import test_recovery_path as rt

s = rt.s
WITNESSES = []


def costs():
    return dict(
        gpu=dict(
            coefficients=dict(
                theta0=2.0, thetaQ=0.1, thetaA=0.0, thetaN=0.0, floor_ms=0.0
            ),
            train_axis_ranges=dict(Q=[1, 2048], A=[0, 100000000], N=[1, 32]),
        ),
        services=dict(
            scheduler_step_ms=0.01,
            worker_host_ms=0.01,
            lookup_service_ms=0.1,
            status_rpc_ms=0.1,
            sample_copy_ms=0.01,
            result_return_ms=0.01,
            output_delivery_ms=0.01,
            rpc_reply_ms=0.01,
        ),
        kv_bytes_per_token=57344,
        transfer={
            kind: {
                medium: dict(fixed_ms=0.01, per_object_ms=0.0, bytes_per_ms=1e12)
                for medium in ("dram", "cxl")
            }
            for kind in ("LOAD", "STORE")
        },
    )


def config(enabled=False):
    return dict(
        coordinate=enabled,
        max_scores=12,
        max_wall_ms=30.0,
        costs=costs(),
        scenarios=[
            dict(
                name="controlled",
                gpu_scale=1.0,
                host_scale=1.0,
                transfer_scale=1.0,
                medium="dram",
                initial_remaining_fraction=1.0,
                unknown_match="cold",
                published_cache_survives=True,
                overlap_gpu_scale=1.0,
            )
        ],
    )


def install(f, enabled=False, cfg=None):
    for name in ("compute_budget", "joint_prefill_score", "joint_prefill"):
        mod = s.module(
            "vllm.v1.core.sched." + name, rt.RUNTIME / s.t.SCHED / (name + ".py")
        )
    f.scheduler.vllm_config.model_config.model = "Qwen2.5-7B-Instruct"
    f.scheduler.vllm_config.model_config.dtype = "bfloat16"
    f.scheduler.vllm_config.cache_config.block_size = 16
    f.scheduler.kv_cache_config.kv_cache_layout = "LBNHC"
    policy = mod.SRPTCoordinationPlan(
        f.scheduler.vllm_config.model_config.hf_config, cfg or config(enabled)
    )
    f.scheduler.set_prefill_plan_policy(policy)
    return policy


def fixture():
    return rt.fixture(new_lookup=True, prepare=True, num_blocks=6913)


def shared_pair(f, first_length=1349):
    requests = []
    for rid, length in (("a", first_length), ("b", 1602)):
        r = s.native.make_request(f.rt, rid, length, token_ids=list(range(length)))
        s.native.add(f.scheduler, r)
        if rid == "a":
            s.ops.seed_prefix(f, r, 16)
        s.t.known_result(f, r, 5)
        f.scheduler._match_waiting_prefix(
            r,
            phase="first_prepare",
            allow_lookup_submit=False,
            allow_lookup_progress=False,
            commit_match=False,
        )
        r.sampling_params.num_logprobs = None
        requests.append(r)
    return requests


class JointPrefillTests(unittest.TestCase):
    def tearDown(self):
        s.close_fixtures()

    def test_migrated_srpt_real_grants_agree_with_compute_only_for_known_work(self):
        grants = []
        for global_mode in (False, True):
            f = (
                fixture()
                if global_mode
                else rt.n.fixture("srpt", prepare=True, num_blocks=6913)
            )
            for rid, L in [
                ("long", 6000),
                ("middle", 700),
                ("short", 500),
                ("tiny", 256),
            ]:
                r = s.t.enqueue(f, rid, L)
                s.t.known_result(f, r, 0)
            if global_mode:
                install(f)
            out = f.scheduler.schedule()
            grants.append(dict(out.num_scheduled_tokens))
        self.assertEqual(grants[0], grants[1])
        self.assertEqual(
            grants[1], {"tiny": 256, "short": 500, "middle": 700, "long": 592}
        )
        WITNESSES.append(dict(name="migrated_srpt", grants=grants))

    def test_full_budget_still_submits_unknown_query_and_reply_changes_current_order(
        self,
    ):
        f = fixture()
        policy = install(f)
        long, _ = rt.matched(f, rid="long", length=12000, local=0, lower=0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"long": 2048})
        s.t.enqueue(f, "new", 512, 1)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"long": 2048})
        self.assertTrue(f.adapter.lookup_operation_state("new")["submitted"])
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("new"), ([],))
        s.ops.ack(f.rpc, "new")
        f.scheduler.schedule()
        s.ops.status(f.rpc, 0, "new")
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"new": 512, "long": 1536})
        self.assertEqual(
            sum(c[0] == "LOOKUP" and c[2] == "new" for c in f.rpc.calls), 1
        )
        self.assertEqual(policy.snapshot.requests[-1].last_match.matched_tokens, 0)
        WITNESSES.append(
            dict(name="unknown_to_selection", grants=dict(out.num_scheduled_tokens))
        )

    def test_five_requests_use_four_finishers_not_a_three_request_window(self):
        f = fixture()
        policy = install(f, True)
        for i in range(4):
            rt.matched(f, rid=f"s{i}", length=128, local=0, lower=0)
        rt.matched(f, rid="large", length=1536, local=0, lower=0)
        out = f.scheduler.schedule()
        self.assertEqual(
            out.num_scheduled_tokens,
            {f"s{i}": 128 for i in range(4)},
            policy.last_report,
        )
        self.assertTrue(policy.last_report["effective_change"])
        self.assertNotIn("large", out.num_scheduled_tokens)
        self.assertEqual(policy.current.end_after, "s3")
        WITNESSES.append(
            dict(
                name="five_request_coordinated_boundary",
                report=policy.last_report,
                grants=dict(out.num_scheduled_tokens),
            )
        )

    def test_generated_reservation_for_fourth_request_changes_actual_quota(self):
        f = fixture()
        policy = install(f)
        # Higher p can make a smaller token remainder costlier than a longer cold
        # request. This state is data from actual native prefix registration.
        for rid, L, h in [
            ("s0", 128, 0),
            ("s1", 512, 0),
            ("next", 1000, 0),
            ("fourth", 32700, 32000),
        ]:
            rt.matched(f, rid=rid, length=L, local=h, lower=0)
        view = f.scheduler.get_prefill_candidate_snapshot()
        Spec = policy.scorer.score.__globals__["CoordinationSpec"]
        candidates = list(
            policy.batch_candidates(view, Spec(), time.perf_counter_ns() + int(1e9))
        )
        candidate = next(
            c
            for k, t, c in candidates
            if k == "reserve_other" and t == ("next", "fourth")
        )

        class Once:
            def plan(self, v):
                return policy.build(v, candidate)

        f.scheduler.set_prefill_plan_policy(Once())
        out = f.scheduler.schedule()
        self.assertEqual(
            out.num_scheduled_tokens, {"s0": 128, "s1": 512, "next": 708, "fourth": 700}
        )
        WITNESSES.append(
            dict(
                name="fourth_request_budget",
                spec=repr(candidate),
                grants=dict(out.num_scheduled_tokens),
            )
        )

    def test_pending_restore_never_computes_or_blocks_other_requests(self):
        f = fixture()
        install(f)
        a, _ = rt.matched(f, rid="load", length=1600, local=16, lower=1280)
        rt.matched(f, rid="compute", length=6000, local=0, lower=0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"compute": 2048})
        self.assertEqual(rt.ranges(out, "RETRIEVE"), [(0, 1280)])
        self.assertEqual(a.status, f.rt.status.WAITING_FOR_REMOTE_KVS)
        self.assertNotIn("load", f.scheduler.schedule().num_scheduled_tokens)

    def test_recompute_rebuilds_all_queue_order_and_real_path_without_load(self):
        f = fixture()
        policy = install(f)
        rt.matched(f, rid="hot", length=4096, local=16, lower=3840)
        rt.matched(f, rid="cold", length=700, local=0, lower=0)
        view = f.scheduler.get_prefill_candidate_snapshot()
        Spec = policy.scorer.score.__globals__["CoordinationSpec"]
        spec = Spec(recompute=("hot",))
        self.assertEqual(policy.order(view, Spec())[0], "hot")
        self.assertEqual(policy.order(view, spec)[0], "cold")

        class Once:
            def plan(self, v):
                return policy.build(v, spec)

        f.scheduler.set_prefill_plan_policy(Once())
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"cold": 700, "hot": 1348})
        self.assertFalse(rt.ranges(out, "RETRIEVE"))
        self.assertEqual(f.scheduler.requests["hot"].num_computed_tokens, 1364)

    def test_snapshot_signature_read_and_plan_do_not_probe_or_advance_watermark(self):
        f = fixture()
        policy = install(f)
        a, b = shared_pair(f)
        calls = list(f.rpc.calls)
        free = f.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        view = f.scheduler.get_prefill_candidate_snapshot()
        again = f.scheduler.get_prefill_candidate_snapshot()
        self.assertEqual(view.requests, again.requests)
        self.assertIsNotNone(view.requests[0].logical_prefix)
        self.assertEqual(
            view.requests[0].logical_prefix.boundaries[:5],
            view.requests[1].logical_prefix.boundaries[:5],
        )
        groups, _ = policy.groups(view, time.perf_counter_ns() + int(1e9))
        self.assertEqual(groups, [(("a", "b"), 1280)])
        policy.plan(view)
        self.assertEqual(calls, f.rpc.calls)
        self.assertEqual(
            free, f.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        )

    def test_shared_candidate_waits_only_after_real_reservation_then_rematches(self):
        import verify_joint_slices_cpu as life

        f = fixture()
        cfg = config(True)
        cfg["costs"]["transfer"]["LOAD"]["dram"]["fixed_ms"] = 200.0
        policy = install(f, cfg=cfg)
        s.ops.prepare_full_output_fixture(f, runtime=rt.RUNTIME)
        a, b = shared_pair(f)
        w = rt.worker(f)
        w.adapter.kv_load_failure_policy = "recompute"
        rows = {"a": a, "b": b}
        loads = []
        quotas = []
        stores = []
        out = f.scheduler.schedule()
        self.assertTrue(policy.last_report["effective_change"], policy.last_report)
        self.assertTrue(policy.current.deferrals, policy.current)
        loads.extend(
            m.request_id
            for m in out.kv_connector_metadata.requests
            if m.direction == "RETRIEVE"
        )
        self.assertEqual(loads, ["a"], policy.last_report)
        self.assertEqual(out.num_scheduled_tokens, {})
        life.apply_worker_output(rt, f, w, out, rows)
        self.assertGreater(w.adapter.page_ownership.pending_count, 0)
        again = f.scheduler.schedule()
        self.assertFalse(rt.ranges(again, "RETRIEVE"))
        self.assertFalse(again.num_scheduled_tokens)
        life.apply_worker_output(rt, f, w, again, rows)
        self.assertTrue(f.adapter.lookup_operation_state("b")["source_ownership"])
        for _ in range(20):
            life.finish_worker_futures(w)
            out = f.scheduler.schedule()
            loads.extend(
                m.request_id
                for m in out.kv_connector_metadata.requests
                if m.direction == "RETRIEVE"
            )
            stores.extend(rt.ranges(out, "STORE"))
            if out.num_scheduled_tokens:
                quotas.append(dict(out.num_scheduled_tokens))
            life.apply_worker_output(rt, f, w, out, rows)
            if not f.scheduler.requests and not w.adapter.page_ownership.pending_count:
                break
        self.assertEqual(loads, ["a"])
        self.assertEqual((a.num_output_tokens, b.num_output_tokens), (1, 1))
        self.assertFalse(f.scheduler.requests)
        self.assertEqual(w.adapter.page_ownership.pending_count, 0)
        s.ops.wait_until(lambda: f.adapter.cleanup_status()["pending"] == 0)
        self.assertEqual(
            f.rpc.clients["server"].server._ctx.session_manager.active_count(), 0
        )
        WITNESSES.append(
            dict(
                name="shared_restore_actual_lifetime",
                loads=loads,
                quotas=quotas,
                stores=stores,
                outputs=[a.num_output_tokens, b.num_output_tokens],
                final_requests=0,
                final_page_protections=0,
                final_sources=0,
            )
        )

    def test_unreserved_leader_capacity_failure_does_not_hold_follower(self):
        f = rt.fixture(new_lookup=True, prepare=True, num_blocks=128)
        policy = install(f)
        a, b = shared_pair(f, 3000)
        Spec = policy.scorer.score.__globals__["CoordinationSpec"]
        spec = Spec(deferrals=(("b", "a", 1280),))

        class Once:
            def plan(self, v):
                policy.snapshot = v
                policy.current = spec
                return policy.build(v, spec)

            def on_feedback(self, result):
                policy.on_feedback(result)

        f.scheduler.set_prefill_plan_policy(Once())
        out = f.scheduler.schedule()
        self.assertEqual(
            [
                m.request_id
                for m in out.kv_connector_metadata.requests
                if m.direction == "RETRIEVE"
            ],
            ["b"],
        )
        self.assertNotIn("b", policy.waits)
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("a"), ([],))

    def test_failed_or_cancelled_leading_load_releases_wait_without_early_dma_release(
        self,
    ):
        import verify_joint_slices_cpu as life

        for cancel in (False, True):
            f = fixture()
            policy = install(f)
            s.ops.prepare_full_output_fixture(f, runtime=rt.RUNTIME)
            a, b = shared_pair(f)
            Spec = policy.scorer.score.__globals__["CoordinationSpec"]
            spec = Spec(deferrals=(("b", "a", 1280),))
            once = [True]

            class Explicit:
                def plan(self, v):
                    if once[0]:
                        once[0] = False
                        policy.snapshot = v
                        policy.current = spec
                        return policy.build(v, spec)
                    return policy.plan(v)

                def on_feedback(self, x):
                    policy.on_feedback(x)

            f.scheduler.set_prefill_plan_policy(Explicit())
            w = rt.worker(f)
            w.adapter.kv_load_failure_policy = "recompute"
            rows = {"a": a, "b": b}
            out = f.scheduler.schedule()
            life.apply_worker_output(rt, f, w, out, rows)
            self.assertGreater(w.adapter.page_ownership.pending_count, 0)
            if cancel:
                f.scheduler.finish_requests("a", f.rt.status.FINISHED_ABORTED)
                self.assertGreater(w.adapter.page_ownership.pending_count, 0)
            for future in w.adapter._fixture_futures:
                if not future.query():
                    future.set_result(cancel)
            for _ in range(24):
                out = f.scheduler.schedule()
                life.apply_worker_output(rt, f, w, out, rows)
                life.finish_worker_futures(w)
                if (
                    not f.scheduler.requests
                    and not w.adapter.page_ownership.pending_count
                ):
                    break
            self.assertEqual(b.num_output_tokens, 1)
            self.assertEqual(a.num_output_tokens, 0 if cancel else 1)
            self.assertFalse(f.scheduler.requests)
            self.assertEqual(w.adapter.page_ownership.pending_count, 0)
            self.assertFalse(policy.waits)

    def test_partial_cap_does_not_implicitly_end_remaining_computation(self):
        f = fixture()
        policy = install(f)
        for rid, L in [("a", 814), ("b", 184), ("c", 2443), ("fourth", 132)]:
            rt.matched(f, rid=rid, length=L, local=0, lower=0)
        Spec = policy.scorer.score.__globals__["CoordinationSpec"]

        class Once:
            def plan(self, v):
                return policy.build(v, Spec(caps=(("c", 395),)))

        f.scheduler.set_prefill_plan_policy(Once())
        out = f.scheduler.schedule()
        self.assertEqual(sum(out.num_scheduled_tokens.values()), 1525)
        self.assertEqual(out.num_scheduled_tokens["fourth"], 132)
        self.assertEqual(out.num_scheduled_tokens["c"], 395)

    def test_timeout_and_unknown_capacity_return_complete_srpt_baseline(self):
        f = fixture()
        cfg = config(True)
        cfg["max_wall_ms"] = 0.000001
        policy = install(f, cfg=cfg)
        rt.matched(f, rid="a", length=4096, local=0, lower=0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"a": 2048})
        self.assertTrue(policy.last_report["timed_out"])
        self.assertGreater(policy.last_report["measured_extra_wall_ns"], 0)

    def test_more_than_32_unknown_requests_remain_visible_and_progress_fairly(self):
        f = fixture()
        policy = install(f)
        for i in range(65):
            s.t.enqueue(f, f"q{i}", 256, i)
        for _ in range(3):
            out = f.scheduler.schedule()
            self.assertEqual(len(policy.snapshot.requests), 65)
            self.assertFalse(out.num_scheduled_tokens)
            self.assertFalse(rt.ranges(out, "RETRIEVE"))
        submitted = [c[2] for c in f.rpc.calls if c[0] == "LOOKUP"]
        self.assertEqual(len(submitted), 65)
        self.assertEqual(len(set(submitted)), 65)
        self.assertTrue(
            all(
                f.scheduler.kv_cache_manager.get_block_ids(f"q{i}") == ([],)
                for i in range(65)
            )
        )

    def test_nearly_finished_running_gets_tail_before_new_cold_request(self):
        f = fixture()
        policy = install(f)
        s.ops.prepare_full_output_fixture(f, runtime=rt.RUNTIME)
        long, _ = rt.matched(f, rid="near", length=6708, local=0, lower=0)
        long.sampling_params.num_logprobs = None
        s.t.enqueue(f, "cold", 2048, 1)
        w = rt.worker(f)
        cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
        for i in range(3):
            out = f.scheduler.schedule()
            f.scheduler.update_from_output(
                out,
                cls(
                    req_ids=["near"],
                    req_id_to_index={"near": 0},
                    sampled_token_ids=[[]],
                ),
            )
            if i == 0:
                s.ops.ack(f.rpc, "cold")
        self.assertEqual(
            (long.num_computed_tokens, long.num_in_flight_tokens), (6144, 0)
        )
        s.ops.status(f.rpc, 0, "cold")
        before = tuple(f.scheduler.running)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"near": 564, "cold": 1484})
        self.assertEqual(tuple(f.scheduler.running[: len(before)]), before)
        self.assertEqual(policy.predicted["near"], (6144, 564))

    def test_same_step_revoke_removes_prediction_ledger_and_pending_shape(self):
        f = fixture()
        policy = install(f)
        long, _ = rt.matched(f, rid="long", length=8192, local=0, lower=0)
        short = s.t.enqueue(f, "short", 512, 1)
        f.scheduler.schedule()
        s.ops.ack(f.rpc, "short")
        f.scheduler.schedule()
        s.ops.status(f.rpc, 0, "short")
        allocate = f.scheduler.kv_cache_manager.allocate_slots
        failed = []

        def fail_once(request, *args, **kwargs):
            if request is long and not failed:
                failed.append(True)
                return None
            return allocate(request, *args, **kwargs)

        f.scheduler.kv_cache_manager.allocate_slots = fail_once
        out = f.scheduler.schedule()
        self.assertNotIn("short", out.num_scheduled_tokens)
        self.assertNotIn("short", policy.actual)
        self.assertNotIn("short", f.scheduler._global_plan.current_grants)
        self.assertEqual(short.status, f.rt.status.PREEMPTED)
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("short"), ([],))
        batch = f.scheduler._global_plan.pending_batches[f.scheduler.current_step]
        self.assertEqual(
            {rid: q for rid, p, q in batch.shape}, out.num_scheduled_tokens
        )

    def test_actual_rematch_invalidates_cap_and_uses_safe_native_remainder(self):
        f = fixture()
        policy = install(f, True)
        request, _ = rt.matched(f, rid="cached", length=1024, local=512, lower=0)
        view = f.scheduler.get_prefill_candidate_snapshot()
        Spec = policy.scorer.score.__globals__["CoordinationSpec"]
        spec = Spec(caps=(("cached", 128),), end_after="cached")
        policy.snapshot = view
        policy.current = spec
        policy.predicted = policy.expected_grants(view, spec)
        blocks, _, _ = f.scheduler.kv_cache_manager.get_computed_blocks(request)
        f.scheduler.kv_cache_manager.evict_blocks(
            {b.block_id for group in blocks.blocks for b in group}
        )

        class Preset:
            def plan(self, v):
                return policy.build(v, spec)

            def check_compute_proposal(self, candidate, action):
                return policy.check_compute_proposal(candidate, action)

            def revise_attempt(self, item):
                return policy.revise_attempt(item)

            def on_feedback(self, result):
                policy.on_feedback(result)

        f.scheduler.set_prefill_plan_policy(Preset())
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"cached": 1024})
        self.assertEqual(policy.invalid, "native_match_or_budget_changed")

    def test_model_uncommitted_recompute_returns_to_restore_after_budget_exhaustion(
        self,
    ):
        f = fixture()
        policy = install(f)
        rt.matched(f, rid="small", length=2048, local=0, lower=0)
        rt.matched(f, rid="hot", length=4096, local=16, lower=3840)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        Spec = policy.scorer.score.__globals__["CoordinationSpec"]
        spec = Spec(recompute=("hot",))
        scene = config()["scenarios"][0]
        score = policy.scorer.score(
            snap,
            spec,
            policy.estimates(snap, spec),
            scene,
            0.0,
            time.perf_counter_ns() + int(1e9),
        )
        self.assertEqual(score.load_bytes, 3840 * 57344)
        self.assertEqual(set(dict(score.completion)), {"small", "hot"})
        self.assertEqual(
            score,
            policy.scorer.score(
                snap,
                spec,
                policy.estimates(snap, spec),
                scene,
                0.0,
                time.perf_counter_ns() + int(1e9),
            ),
        )

    def test_ranking_flip_keeps_baseline_and_second_round_uses_accepted_plan(self):
        f = fixture()
        cfg = config(True)
        cfg["max_scores"] = 4
        cfg["scenarios"] = [dict(cfg["scenarios"][0], name=x) for x in ("one", "two")]
        policy = install(f, cfg=cfg)
        for rid, L in [("a", 256), ("b", 300)]:
            rt.matched(f, rid=rid, length=L, local=0, lower=0)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        module = policy.scorer.score.__globals__
        Spec = module["CoordinationSpec"]
        Score = module["Score"]
        visited = []

        def candidates(snapshot, current, deadline):
            visited.append(current)
            if current == Spec():
                yield ("end", ("a",), Spec(end_after="a"))
            else:
                yield ("cap", ("b",), replace(current, caps=(("b", 100),)))

        policy.batch_candidates = candidates
        policy.recovery_candidates = lambda *args: iter(())

        def artificial(snapshot, spec, features, scene, delay, deadline):
            value = 100 if spec == Spec() else (90 if not spec.caps else 80)
            return Score((("a", value),), 0, 0, 0, 0, 0, 0, scene["name"])

        policy.scorer.score = artificial
        policy.plan(snap)
        self.assertEqual(visited, [Spec(), Spec(end_after="a")])
        self.assertEqual(len(policy.last_report["accepted"]), 2)
        self.assertEqual(policy.current, Spec(caps=(("b", 100),), end_after="a"))

        def flip(snapshot, spec, features, scene, delay, deadline):
            value = 100 if spec == Spec() else (90 if scene["name"] == "one" else 110)
            return Score((("a", value),), 0, 0, 0, 0, 0, 0, scene["name"])

        policy.scorer.score = flip
        policy.plan(snap)
        self.assertFalse(policy.last_report["effective_change"])


if __name__ == "__main__":
    unittest.main()
