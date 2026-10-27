# SPDX-License-Identifier: Apache-2.0
"""Rolling plans through the real scheduler; external futures are CPU inputs."""

import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_joint_prefill as base

WITNESSES = []
original_install = base.install


def install(f, enabled=True, mode="fixed"):
    original_install(f)
    s = base.s
    for name in (
        "joint_prefill_cache",
        "prefill_search_window",
        "joint_prefill_rolling",
    ):
        module = s.module(
            "vllm.v1.core.sched." + name,
            base.rt.RUNTIME / s.t.SCHED / (name + ".py"),
        )
    cfg = base.config(enabled)
    cfg.update(max_scores=8, max_wall_ms=8.0)
    cfg["rolling"] = dict(
        budget_mode=mode,
        absolute_wall_ms=8.0,
        absolute_cpu_ms=8.0,
        conservative_ms=2.0,
        control_delay_ms=2.0,
        submit_reserve_ms=0.5,
        cache_entries=128,
        cache_bytes=8 * 1024 * 1024,
    )
    p = module.SRPTRollingPlan(f.scheduler.vllm_config.model_config.hf_config, cfg)
    f.scheduler.set_prefill_plan_policy(p)
    return p


class RollingTests(unittest.TestCase):
    def tearDown(self):
        base.s.close_fixtures()

    def test_zero_control_step_does_not_consume_tail_and_new_query_progresses(self):
        f = base.fixture()
        p = install(f)
        base.s.t.enqueue(f, "unknown", 512)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertTrue(f.adapter.lookup_operation_state("unknown")["submitted"])
        self.assertEqual(p.rolling_report["nonempty_commits"], 0)

    def test_incremental_and_full_reference_match_all_metrics(self):
        f = base.fixture()
        p = install(f)
        for rid, length in (("small", 512), ("big", 3000)):
            base.rt.matched(f, rid=rid, length=length, local=0, lower=0)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        Spec = base.s.module(
            "test_rolling_score_reference",
            base.rt.RUNTIME / base.s.t.SCHED / "joint_prefill_score.py",
        ).CoordinationSpec
        deadline = time.perf_counter_ns() + int(3e9)
        for spec in (Spec(), Spec(caps=(("big", 395),)), Spec(end_after="small")):
            features = p.estimates(snap, spec)
            for scene in p.scenes:
                a = p.reference.score(snap, spec, features, scene, 2.0, deadline)
                b = p.scorer.score(snap, spec, features, scene, 2.0, deadline)
                self.assertEqual(a, b)
                self.assertEqual(
                    b, p.scorer.score(snap, spec, features, scene, 2.0, deadline)
                )
        self.assertGreater(p.cache.stats["full_hits"], 0)
        self.assertGreater(p.cache.stats["prefix_hits"], 0)

    def test_live_ready_flag_shrinks_budget_without_wait(self):
        f = base.fixture()
        p = install(f, mode="dynamic")
        W = p.window_type
        state = {"ready": False}
        p.set_execution_window_reader(lambda: W(result_ready=state["ready"]))
        budget = p.new_budget()
        self.assertLessEqual(budget.granted_ms, 2.0)
        state["ready"] = True
        budget.next_read = 0  # Advance the bounded readiness observation point.
        with self.assertRaises(p.timeout_type):
            budget.check()
        self.assertEqual(budget.reason, "result_ready")

    def test_tail_rolls_once_then_expired_generation_cannot_renew(self):
        f = base.fixture()
        p = install(f)
        budget_module = __import__(
            p.new_budget.__globals__["SearchBudget"].__module__,
            fromlist=["SearchBudget"],
        )
        # Controlled CPU time isolates lifetime logic from host scheduling noise;
        # all allocations and cache/connector operations remain native methods.
        p.new_budget = lambda: budget_module.SearchBudget(
            p.rolling_config, p.window_reader, clock=lambda: 0, cpu=lambda: 0
        )
        for rid, length in (("a", 512), ("b", 512), ("long", 10000)):
            base.rt.matched(f, rid=rid, length=length, local=0, lower=0)
        steps = []
        for _ in range(3):
            out = f.scheduler.schedule()
            steps.append(
                dict(
                    grants=dict(out.num_scheduled_tokens), report=dict(p.rolling_report)
                )
            )
        self.assertGreater(steps[0]["report"]["generation"], 0)
        self.assertTrue(steps[1]["report"]["reused"], steps)
        self.assertFalse(steps[2]["report"]["reused"], steps)
        self.assertEqual(sum(steps[1]["grants"].values()), 2048)
        self.assertEqual(
            [r.request_id for r in f.scheduler.running], ["a", "b", "long"]
        )
        WITNESSES.append(dict(name="t_t1_t2_actual_grants", steps=steps))

    def test_new_short_request_invalidates_tail_and_changes_real_allocation(self):
        f = base.fixture()
        p = install(f)
        base.rt.matched(f, rid="long", length=10000, local=0, lower=0)
        f.scheduler.schedule()
        base.s.t.enqueue(f, "short", 400)
        f.scheduler.schedule()
        base.s.ops.ack(f.rpc, "short")
        f.scheduler.schedule()
        base.s.ops.status(f.rpc, 0, "short")
        out = f.scheduler.schedule()
        self.assertEqual(next(iter(out.num_scheduled_tokens)), "short")
        self.assertEqual(out.num_scheduled_tokens["short"], 400)
        self.assertFalse(p.rolling_report["reused"])
        self.assertIn(
            "tail_differs_from_current_common_srpt",
            p.rolling_report["invalidation_reasons"],
        )
        WITNESSES.append(
            dict(
                name="arrival_rebuild",
                grants=dict(out.num_scheduled_tokens),
                report=p.rolling_report,
            )
        )

    def test_disabled_v2_matches_common_actual_quotas_and_rpc(self):
        outputs = []
        for version in (1, 2):
            f = base.fixture()
            base.install(f) if version == 1 else install(f, False)
            base.rt.matched(f, rid="long", length=6000, local=0, lower=0)
            base.rt.matched(f, rid="restore", length=1600, local=16, lower=1280)
            base.s.t.enqueue(f, "unknown", 512)
            out = f.scheduler.schedule()
            outputs.append(
                (
                    dict(out.num_scheduled_tokens),
                    base.rt.ranges(out, "RETRIEVE"),
                    list(f.rpc.calls),
                    [
                        (r.request_id, r.status.name, r.num_computed_tokens)
                        for r in f.scheduler.requests.values()
                    ],
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_restore_pending_and_observation_progress_without_compute(self):
        f = base.fixture()
        install(f, False)
        request, _ = base.rt.matched(
            f, rid="restore", length=1600, local=16, lower=1280
        )
        base.rt.matched(f, rid="long", length=10000, local=0, lower=0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"long": 2048})
        self.assertEqual(base.rt.ranges(out, "RETRIEVE"), [(0, 1280)])
        self.assertEqual(request.status, f.rt.status.WAITING_FOR_REMOTE_KVS)
        self.assertNotIn("restore", f.scheduler.schedule().num_scheduled_tokens)

    def test_score_state_changes_miss_and_do_not_poll_or_allocate(self):
        f = base.fixture()
        p = install(f)
        base.rt.matched(f, rid="long", length=3000, local=0, lower=0)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        Spec = type(p.current)
        scene = p.scenes[0]
        calls = list(f.rpc.calls)
        free = f.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        variants = (
            snap,
            replace(snap, requests=(replace(snap.requests[0], input_tokens=4000),)),
        )
        scores = []
        for v in variants:
            features = p.estimates(v, Spec())
            args = (v, Spec(), features, scene, 0, time.perf_counter_ns() + int(2e9))
            self.assertEqual(p.scorer.score(*args), p.reference.score(*args))
            scores.append(p.scorer.score(*args).objective)
        self.assertNotEqual(*scores)
        self.assertEqual(f.rpc.calls, calls)
        self.assertEqual(
            free, f.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        tight = replace(snap, free_hbm_blocks=0)
        for scorer in (p.scorer, p.reference):
            with self.assertRaisesRegex(Exception, "capacity_unconfirmed"):
                scorer.score(
                    tight,
                    Spec(),
                    p.estimates(tight, Spec()),
                    scene,
                    0,
                    time.perf_counter_ns() + int(1e9),
                )

    def test_engine_flags_keep_queue_order_and_never_consume_future(self):
        from collections import deque
        from concurrent.futures import Future
        from types import SimpleNamespace

        f = base.fixture()
        p = install(f)
        module = __import__(p.window_type.__module__, fromlist=["read_engine_window"])
        a, b = Future(), Future()
        a.set_result(None)
        queue = deque(
            [
                (b, SimpleNamespace(_action_step_id=2), b),
                (a, SimpleNamespace(_action_step_id=1, _coord_host_submit_ns=99), a),
            ]
        )
        original = tuple(queue)
        view = module.read_engine_window(queue)
        self.assertEqual([r.step_id for r in view.queued], [1, 2])
        self.assertTrue(view.result_ready)
        self.assertIsNone(view.device_remaining_lower_ms)
        self.assertEqual(tuple(queue), original)

    def test_actual_async_future_peeks_existing_event_without_consuming_output(self):
        import ast
        from concurrent.futures import Future
        from types import SimpleNamespace

        f = base.fixture()
        install(f)
        source = base.rt.RUNTIME / "source/vllm/vllm/v1/executor/uniproc_executor.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AsyncOutputFuture"
        )
        ns = {"Future": Future, "AsyncModelRunnerOutput": object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)

        class Output:
            def get_output(self):
                raise AssertionError("readiness must not consume output")

        runner = base.rt.RUNTIME / "source/vllm/vllm/v1/worker/gpu_model_runner.py"
        base.s.ops.add_methods(
            runner, "AsyncGPUModelRunnerOutput", Output, {"peek_ready"}, {}
        )
        state = {"ready": False, "queries": 0}

        def query():
            state["queries"] += 1
            return state["ready"]

        output = Output()
        output.async_copy_ready_event = SimpleNamespace(query=query)
        future = ns["AsyncOutputFuture"](output, True)
        self.assertFalse(future.done())
        self.assertFalse(future.peek_ready())
        state["ready"] = True
        self.assertTrue(future.peek_ready())
        self.assertFalse(future.done())
        self.assertEqual(state["queries"], 2)

    def test_lookup_ready_read_does_not_advance_ack_or_status(self):
        f = base.fixture()
        install(f)
        base.s.ops.add_methods(
            base.rt.RUNTIME
            / "source/LMCache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
            "LMCacheMPSchedulerAdapter",
            type(f.adapter),
            {"lookup_reply_readiness"},
            f.adapter.lookup_operation_state.__globals__,
        )
        base.s.t.enqueue(f, "unknown", 512)
        f.scheduler.schedule()
        before = list(f.rpc.calls)
        self.assertFalse(f.adapter.lookup_reply_readiness(("unknown",))["ready"])
        base.s.ops.ack(f.rpc, "unknown")
        self.assertTrue(f.adapter.lookup_reply_readiness(("unknown",))["ready"])
        self.assertEqual(f.rpc.calls, before)
        self.assertIn("unknown", f.adapter._unacked_lookups)

    def test_fixed_budget_keeps_completed_incumbent_and_discards_half_score(self):
        f = base.fixture()
        p = install(f)
        for rid, length in (("a", 512), ("b", 512), ("long", 4000)):
            base.rt.matched(f, rid=rid, length=length, local=0, lower=0)
        original = p.scorer.score
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise p.timeout_type("controlled_half_candidate")
            return original(*args, **kwargs)

        with patch.object(p.scorer, "score", interrupted):
            out = f.scheduler.schedule()
        self.assertTrue(p.last_report["search_budget_exhausted"])
        self.assertFalse(p.last_report["effective_change"])
        self.assertEqual(sum(out.num_scheduled_tokens.values()), 2048)
        self.assertIsNotNone(out.kv_connector_metadata)

    def test_native_lifecycle_cases_with_disabled_rolling(self):
        # These fixtures drive real worker/connector/allocator methods. Only
        # external replies/device events are controlled, not ownership logic.
        names = (
            "test_full_budget_still_submits_unknown_query_and_reply_changes_current_order",
            "test_pending_restore_never_computes_or_blocks_other_requests",
            "test_unreserved_leader_capacity_failure_does_not_hold_follower",
            "test_failed_or_cancelled_leading_load_releases_wait_without_early_dma_release",
            "test_more_than_32_unknown_requests_remain_visible_and_progress_fairly",
            "test_nearly_finished_running_gets_tail_before_new_cold_request",
            "test_same_step_revoke_removes_prediction_ledger_and_pending_shape",
        )
        for name in names:
            with (
                self.subTest(name=name),
                patch.object(base, "install", lambda f, *a, **k: install(f, False)),
            ):
                getattr(base.JointPrefillTests(), name)()

    def test_cache_bound_and_model_invalidation(self):
        f = base.fixture()
        p = install(f)
        cache = type(p.cache)(entries=3, max_bytes=4096)
        cache.bind_model({"revision": 1})
        for i in range(12):
            cache.put("full", i, (float(i),))
        self.assertLessEqual(len(cache.data), 3)
        self.assertLessEqual(cache.bytes, 4096)
        self.assertGreater(cache.stats["evictions"], 0)
        cache.bind_model({"revision": 2})
        self.assertEqual(len(cache.data), 0)
        self.assertEqual(cache.stats["model_invalidations"], 1)
        for i in range(10):
            cache.put("suffix", i % 2, ("x" * (100 + i),))
        self.assertLessEqual(len(cache.data), 3)
        self.assertEqual(set(cache.cheap) - set(cache.data), set())

    def test_suffix_reuse_requires_exact_reconvergence_not_equal_token_total(self):
        f = base.fixture()
        p = install(f)
        base.rt.matched(f, rid="a", length=512, local=0, lower=0)
        base.rt.matched(f, rid="b", length=5000, local=0, lower=0)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        Spec = type(p.current)
        deadline = time.perf_counter_ns() + int(3e9)
        # Nonbinding cap: different logical plan, identical actual first batch.
        for spec in (Spec(), Spec(caps=(("b", 2048),))):
            args = (snap, spec, p.estimates(snap, spec), p.scenes[0], 0, deadline)
            self.assertEqual(p.scorer.score(*args), p.reference.score(*args))
        self.assertGreater(p.cache.stats["suffix_hits"], 0)

    def test_expired_soft_reserve_does_not_discard_completed_plan_early(self):
        f = base.fixture()
        p = install(f)
        cls = p.new_budget.__globals__["SearchBudget"]
        clock = [0]
        budget = cls(
            p.rolling_config, p.window_reader, clock=lambda: clock[0], cpu=lambda: 0
        )
        clock[0] = 7_900_000
        with self.assertRaises(p.timeout_type):
            budget.check(200_000)
        self.assertFalse(budget.expired())
        clock[0] = 8_000_000
        self.assertTrue(budget.expired())
