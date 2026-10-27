# SPDX-License-Identifier: Apache-2.0
"""V3 decisions through real scheduler/cache/connector methods; CPU only."""

import unittest
from dataclasses import replace
from time import perf_counter_ns
from unittest.mock import patch

import test_joint_prefill as base

WITNESSES = []
ORIGINAL_INSTALL = base.install


def config(enabled=True):
    cfg = base.config(enabled)
    cfg.update(max_scores=8, max_wall_ms=8.0)
    cfg["scenarios"] = [dict(cfg["scenarios"][0], name=f"same-{i}") for i in range(8)]
    cfg["tiered"] = dict(
        max_candidates=16,
        wall_limit_ms=8.0,
        cpu_limit_ms=8.0,
        control_limit_ms=8.0,
        unknown_window_ms=8.0,
        submit_reserve_ms=0.5,
        coarse_scenario_index=0,
        gpu_window=dict(admitted=False, lower_scale=0.5),
        cpu_cost=dict(
            admitted=True,
            safety_factor=1.0,
            coefficients={k: [0.001, 0, 0, 0] for k in ("fine", "finalization")},
            floor_ms={k: 0.001 for k in ("fine", "finalization")},
        ),
    )
    return cfg


def install(f, enabled=True, cfg=None):
    ORIGINAL_INSTALL(f)
    for name in (
        "prefill_search_window",
        "prefill_tiered_budget",
        "joint_prefill_tiered_score",
        "joint_prefill_tiered",
    ):
        module = base.s.module(
            "vllm.v1.core.sched." + name,
            base.rt.RUNTIME / base.s.t.SCHED / (name + ".py"),
        )
    p = module.SRPTTieredPlan(
        f.scheduler.vllm_config.model_config.hf_config, cfg or config(enabled)
    )
    f.scheduler.set_prefill_plan_policy(p)
    return p


def controlled_budget(p, ready=lambda: False):
    p.set_execution_window_reader(lambda: p.window_type(result_ready=ready()))
    Budget = p.new_budget.__globals__["TieredBudget"]
    # Increasing observation clock lets readiness be re-read; still far below
    # the wall bound. This controls CPU test inputs, never a speed claim.
    tick = [0]

    def clock():
        tick[0] += 1000
        return tick[0]

    p.new_budget = lambda snap: Budget(
        p.tiered_config,
        snap,
        p.window_reader,
        p.window_model,
        clock=clock,
        cpu=lambda: 0,
        host_clock=lambda: 0,
    )
    return p


class TieredTests(unittest.TestCase):
    def tearDown(self):
        base.s.close_fixtures()

    def test_shaped_window_uses_host_forward_as_lower_start_not_device_time(self):
        from vllm.v1.core.sched.prefill_search_window import QueuedExecution

        f = base.fixture()
        p = install(f, False)
        base.rt.matched(f, rid="long", length=6000, local=0, lower=0)
        f.scheduler.schedule()
        snap = f.scheduler.get_prefill_candidate_snapshot()
        item = snap.submitted_batches[0]
        p.window_model.config = dict(
            schema="A100-shaped-window-r2",
            admitted=True,
            bins=[dict(q_min=1, q_max=2048, covered=True, lower_scale=0.8)],
        )
        queue = QueuedExecution(
            item.step_id,
            1_000_000_000,
            False,
            True,
            forward_host_start_ns=1_040_000_000,
        )
        window = p.window_type(queued=(queue,))
        end, basis = p.window_model.bound(snap, window, 1_050_000_000)
        service = p.scorer.gpu(item.shape, {"gpu_scale": 1}) * 0.8
        self.assertEqual(end, 1_040_000_000 + int(service * 1e6))
        self.assertEqual(basis, "host_forward_lower_start_shaped_service")
        self.assertEqual(queue.device_progress, "unknown")
        missing = replace(window, queued=(replace(queue, forward_host_start_ns=None),))
        earlier, _ = p.window_model.bound(snap, missing, 1_050_000_000)
        self.assertEqual(end - earlier, 40_000_000)
        future = replace(
            window, queued=(replace(queue, forward_host_start_ns=2_000_000_000),)
        )
        self.assertEqual(p.window_model.bound(snap, future, 1_050_000_000)[0], earlier)
        done = replace(window, queued=(replace(queue, result_ready=True),))
        self.assertEqual(
            p.window_model.bound(snap, done, 1_050_000_000)[0], 1_050_000_000
        )

    def test_native_window_reads_host_boundary_without_consuming_future(self):
        from collections import deque
        from types import SimpleNamespace as NS

        from vllm.v1.core.sched.prefill_search_window import read_engine_window

        f = base.fixture()
        install(f, False)
        base.rt.matched(f, rid="a", length=3000, local=0, lower=0)
        out = f.scheduler.schedule()
        out._coord_host_submit_ns = 100
        out._coord_forward_host_start_ns = 130
        calls = []
        future = NS(
            peek_ready=lambda: calls.append("peek") or False,
            result=lambda: (_ for _ in ()).throw(AssertionError("consume")),
        )
        view = read_engine_window(deque([(future, out, NS(done=lambda: True))]))
        self.assertEqual(calls, ["peek"])
        self.assertEqual(view.queued[0].host_submitted_ns, 100)
        self.assertEqual(view.queued[0].forward_host_start_ns, 130)
        self.assertEqual(view.queued[0].device_progress, "unknown")

    def test_sixteen_ms_budget_keeps_reserves_and_never_extends(self):
        f = base.fixture()
        p = install(f, False)
        base.rt.matched(f, rid="long", length=6000, local=0, lower=0)
        f.scheduler.schedule()
        snap = f.scheduler.get_prefill_candidate_snapshot()
        from vllm.v1.core.sched.prefill_search_window import QueuedExecution

        cfg = dict(
            p.tiered_config,
            wall_limit_ms=16,
            cpu_limit_ms=16,
            control_limit_ms=16,
            submit_reserve_ms=3,
        )
        p.window_model.config = dict(
            schema="A100-shaped-window-r2",
            admitted=True,
            bins=[dict(q_min=1, q_max=2048, covered=True, lower_scale=0.8)],
        )
        state = dict(now=0, control=False)
        read = lambda: p.window_type(
            queued=(
                QueuedExecution(
                    snap.submitted_batches[0].step_id,
                    0,
                    False,
                    True,
                    forward_host_start_ns=0,
                ),
            ),
            control_ready=state["control"],
        )
        Budget = p.new_budget.__globals__["TieredBudget"]
        b = Budget(
            cfg,
            snap,
            read,
            p.window_model,
            clock=lambda: state["now"],
            cpu=lambda: state["now"],
            host_clock=lambda: state["now"],
        )
        self.assertEqual(b.summary()["granted_ms"], 16)
        original = b.deadline
        state["now"] = 1_000_000
        b.check()
        self.assertLessEqual(b.deadline, original)
        state.update(now=2_000_000, control=True)
        with self.assertRaisesRegex(Exception, "control_ready"):
            b.slice(1).check()
        self.assertIsNone(b.completed_plan_discard_reason())
        state["now"] = 16_000_001
        self.assertEqual(b.completed_plan_discard_reason(), "cpu_limit")

    def test_disabled_is_common_srpt_and_no_new_control_operation(self):
        outputs = []
        for version in (1, 3):
            f = base.fixture()
            base.install(f, False) if version == 1 else install(f, False)
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
                        (r.request_id, r.status, r.num_computed_tokens)
                        for r in f.scheduler.requests.values()
                    ],
                )
            )
        self.assertEqual(*outputs)

    def test_no_budget_returns_common_without_coarse_or_fine(self):
        f = base.fixture()
        p = install(f)
        p.set_execution_window_reader(lambda: p.window_type(result_ready=True))
        for rid, length in (("a", 512), ("b", 512), ("long", 4000)):
            base.rt.matched(f, rid=rid, length=length, local=0, lower=0)
        with patch.object(p, "coarse", side_effect=AssertionError("must not screen")):
            out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"a": 512, "b": 512, "long": 1024})
        self.assertEqual(p.last_report["scenario_scores"], 0)
        self.assertEqual(p.tiered_report["configuration_serializations"], 0)

    def test_fine_adoption_changes_native_quota_and_keeps_queue_owners(self):
        f = base.fixture()
        p = install(f)
        for rid, length in (("a", 512), ("b", 512), ("long", 4000)):
            base.rt.matched(f, rid=rid, length=length, local=0, lower=0)
        # Controlled clocks isolate selection, not a CPU speed claim.
        Budget = p.new_budget.__globals__["TieredBudget"]
        p.new_budget = lambda snap: Budget(
            p.tiered_config,
            snap,
            p.window_reader,
            p.window_model,
            clock=lambda: 0,
            cpu=lambda: 0,
            host_clock=lambda: 0,
        )
        out = f.scheduler.schedule()
        self.assertTrue(p.last_report["effective_change"], p.last_report)
        self.assertGreater(p.tiered_report["fine_completed"], 0)
        self.assertLess(sum(out.num_scheduled_tokens.values()), 2048)
        self.assertEqual(
            [r.request_id for r in f.scheduler.running], list(out.num_scheduled_tokens)
        )
        self.assertIsNotNone(out.kv_connector_metadata)
        WITNESSES.append(
            dict(
                name="fine_actual_quota",
                grants=dict(out.num_scheduled_tokens),
                decision=p.last_report,
                tiered=p.tiered_report,
            )
        )

    def test_all_background_requests_and_retired_obligations_are_scored(self):
        f = base.fixture()
        p = install(f)
        for i in range(12):
            base.rt.matched(f, rid=f"r{i}", length=800 + i * 32, local=0, lower=0)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        p.snapshot, p.features, p.orders = snap, {}, {}
        spec = type(p.current)()
        result = p.coarse(snap, spec, p.scenes[0], 0, perf_counter_ns() + int(2e9))
        self.assertEqual(len(result.score.completion), 12)
        self.assertEqual(len(result.head_shapes), 2)
        changed = replace(snap, omitted_requests=1)
        with self.assertRaisesRegex(Exception, "incomplete_queue_view"):
            p.coarse(changed, spec, p.scenes[0], 0, perf_counter_ns() + int(2e9))

    def test_state_cache_does_not_reuse_prior_progress_and_no_query_for_scoring(self):
        f = base.fixture()
        p = install(f, False)
        base.rt.matched(f, rid="long", length=5000, local=0, lower=0)
        first = f.scheduler.schedule()
        p0 = p.estimates(p.snapshot, p.current)["long"].prefix_tokens
        calls = list(f.rpc.calls)
        second = f.scheduler.schedule()
        p1 = p.estimates(p.snapshot, p.current)["long"].prefix_tokens
        self.assertEqual(p1 - p0, first.num_scheduled_tokens["long"])
        self.assertEqual(second.num_scheduled_tokens, {"long": 2048})
        self.assertEqual(calls, f.rpc.calls)

    def test_scenario_merging_does_not_drop_unknown_match_or_pending_work(self):
        f = base.fixture()
        p = install(f, False)
        base.s.t.enqueue(f, "unknown", 512)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        merge = p.plan.__globals__["necessary_scenarios"]
        scenes = [
            dict(p.scenes[0], name=kind, unknown_match=kind)
            for kind in ("cold", "hbm_hot", "l1_hot")
        ]
        distinct, _ = merge(snap, scenes, False)
        self.assertEqual(len(distinct), 3)
        f.scheduler.schedule()
        base.s.ops.ack(f.rpc, "unknown")
        f.scheduler.schedule()
        base.s.ops.status(f.rpc, 0, "unknown")
        f.scheduler.schedule()
        distinct, groups = merge(
            f.scheduler.get_prefill_candidate_snapshot(), scenes, False
        )
        self.assertEqual(len(distinct), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_shared_screen_preserves_publish_dependency_and_fine_reference(self):
        f = base.fixture()
        p = install(f)
        base.shared_pair(f)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        p.snapshot, p.features, p.orders = snap, {}, {}
        spec = type(p.current)()
        groups, _ = p.groups(snap, perf_counter_ns() + int(2e9))
        choices = list(
            p.recovery_candidates(
                snap, spec, groups, perf_counter_ns() + int(2e9), "shared"
            )
        )
        self.assertTrue(choices)
        candidate = choices[0][2]
        original = p.coarse(snap, spec, p.scenes[0], 0, perf_counter_ns() + int(2e9))
        shared = p.coarse(snap, candidate, p.scenes[0], 0, perf_counter_ns() + int(2e9))
        self.assertLess(shared.score.load_bytes, original.score.load_bytes)
        fine = p.scorer.score(
            snap,
            candidate,
            p.estimates(snap, candidate),
            p.scenes[0],
            0,
            perf_counter_ns() + int(2e9),
        )
        reference = p.reference.score(
            snap,
            candidate,
            p.estimates(snap, candidate),
            p.scenes[0],
            0,
            perf_counter_ns() + int(2e9),
        )
        self.assertEqual(fine, reference)

    def test_budget_is_absolute_and_ready_stops_only_expansion(self):
        f = base.fixture()
        p = install(f)
        state = dict(now=0, ready=False)
        p.set_execution_window_reader(
            lambda: p.window_type(result_ready=state["ready"])
        )
        Budget = p.new_budget.__globals__["TieredBudget"]
        budget = Budget(
            p.tiered_config,
            f.scheduler.get_prefill_candidate_snapshot(),
            p.window_reader,
            p.window_model,
            clock=lambda: state["now"],
            cpu=lambda: 0,
            host_clock=lambda: state["now"],
        )
        original = budget.initial_deadline
        state.update(now=200_000, ready=True)
        with self.assertRaisesRegex(Exception, "result_ready"):
            budget.check()
        self.assertIsNone(budget.completed_plan_discard_reason())
        self.assertLess(budget.deadline, original)
        state["now"] = original + 1
        self.assertEqual(budget.completed_plan_discard_reason(), "wall_limit")

    def test_native_lifecycle_with_v3_disabled(self):
        for name in (
            "test_full_budget_still_submits_unknown_query_and_reply_changes_current_order",
            "test_pending_restore_never_computes_or_blocks_other_requests",
            "test_unreserved_leader_capacity_failure_does_not_hold_follower",
            "test_failed_or_cancelled_leading_load_releases_wait_without_early_dma_release",
            "test_same_step_revoke_removes_prediction_ledger_and_pending_shape",
        ):
            with (
                self.subTest(name=name),
                patch.object(base, "install", lambda f, *a, **k: install(f, False)),
            ):
                getattr(base.JointPrefillTests(), name)()

    def test_shared_fine_choice_runs_real_load_publication_store_and_release(self):
        def create(f, *args, **kwargs):
            cfg = config(True)
            original = kwargs.get("cfg")
            if original:
                cfg["costs"] = original["costs"]
            return controlled_budget(install(f, cfg=cfg))

        with patch.object(base, "install", create):
            base.JointPrefillTests().test_shared_candidate_waits_only_after_real_reservation_then_rematches()
        WITNESSES.extend(
            x
            for x in base.WITNESSES
            if x.get("name") == "shared_restore_actual_lifetime"
        )

    def test_completed_incumbent_survives_ready_but_partial_candidate_cannot_replace_it(
        self,
    ):
        f = base.fixture()
        ready = [False]
        p = controlled_budget(install(f), lambda: ready[0])
        for rid, length in (("a", 512), ("b", 512), ("long", 4000)):
            base.rt.matched(f, rid=rid, length=length, local=0, lower=0)
        score = p.scorer.score
        calls = [0]

        def observe(*a, **kw):
            value = score(*a, **kw)
            calls[0] += 1
            if calls[0] == 2:
                ready[0] = True
            return value

        with patch.object(p.scorer, "score", observe):
            out = f.scheduler.schedule()
        self.assertGreater(p.tiered_report["fine_completed"], 0)
        self.assertTrue(p.last_report["effective_change"])
        self.assertLess(sum(out.num_scheduled_tokens.values()), 2048)
        self.assertIsNone(p.last_report["final_plan_fallback"])

    def test_joint_families_are_generated_without_standalone_improvement(self):
        f = base.fixture()
        p = controlled_budget(install(f))
        base.shared_pair(f)
        for i, length in enumerate((512, 700, 1800, 2800, 3400, 4000)):
            base.rt.matched(f, rid=f"c{i}", length=length, local=0, lower=0)
        snap = f.scheduler.get_prefill_candidate_snapshot()
        p.snapshot, p.features, p.orders = snap, {}, {}
        spec = type(p.current)()
        until = perf_counter_ns() + int(2e9)
        groups, _ = p.groups(snap, until)
        kinds = []
        for category in ("path", "shared"):
            kinds.extend(
                k
                for k, _, _ in p.recovery_candidates(
                    snap, spec, groups, until, category
                )
            )
        self.assertIn("path", kinds)
        self.assertIn("shared", kinds)
        self.assertTrue(any(k.startswith("path_") for k in kinds))
        self.assertTrue(any(k.startswith("shared_") for k in kinds))
        self.assertEqual(len(snap.requests), 8)
        self.assertFalse(f.scheduler._global_plan.current_grants)

    def test_enabled_adaptation_rechecks_match_and_handles_rollback(self):
        for name in (
            "test_recompute_rebuilds_all_queue_order_and_real_path_without_load",
            "test_actual_rematch_invalidates_cap_and_uses_safe_native_remainder",
            "test_same_step_revoke_removes_prediction_ledger_and_pending_shape",
            "test_more_than_32_unknown_requests_remain_visible_and_progress_fairly",
        ):

            def create(f, *a, case=name, **k):
                policy = controlled_budget(install(f))
                if case == (
                    "test_same_step_revoke_removes_prediction_ledger_and_pending_shape"
                ):
                    # Exercise the allocation failure after the short request.
                    # An intentional earlier batch end would never reach it.
                    policy.batch_candidates = lambda *a, **k: iter(())
                return policy

            with self.subTest(name=name), patch.object(base, "install", create):
                getattr(base.JointPrefillTests(), name)()
