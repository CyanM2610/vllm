# SPDX-License-Identifier: Apache-2.0
"""Real scheduler/cache/connector methods via the project CPU harness."""

import io
import json
import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

ROOT = Path(os.environ["PREFILL_PROJECT_ROOT"])
AREA = Path(os.environ["PREFILL_CPU_OUTPUT"])
RUNTIME = Path(os.environ["PREFILL_NATIVE_RUNTIME"])
sys.path.insert(0, str(ROOT / "scripts/kv_wait_probe/global_plan"))
import cpu_support as s  # noqa: E402


def fixture(
    mode=None,
    runtime=RUNTIME,
    completion_model=None,
    prepare=False,
    prepare_limit=32,
    **kwargs,
):
    f = s.fixture(
        runtime=runtime,
        global_plan=False,
        control=False,
        store=True,
        prepare_load=False,
        **kwargs,
    )
    # The shared fixture's enabled flag selects action feedback AND legacy
    # progress. Compute-only production config enables only the former.
    f.adapter._lookup_progress_enabled = False
    f.adapter._lookup_progress_queue.clear()
    if mode is not None:
        trace = ModuleType("vllm.v1.kv_wait_trace")
        trace.emit = lambda event, request_id=None, **fields: f.rt.events.append(
            dict(event=event, request_id=request_id, **fields)
        )
        sys.modules[trace.__name__] = trace
        s.module(
            "vllm.v1.core.sched.compute_budget",
            runtime / s.t.SCHED / "compute_budget.py",
        )
        s.module(
            "vllm.v1.core.sched.compute_slo", runtime / s.t.SCHED / "compute_slo.py"
        )
        mod = s.module(
            "vllm.v1.core.sched.native_compute",
            runtime / s.t.SCHED / "native_compute.py",
        )
        mod.clock_identity = lambda: "controlled-clock"
        config = {"mode": mode}
        if completion_model is not None:
            config["completion_model"] = completion_model
        f.scheduler._init_native_compute_selection(config)
        f.selection = f.scheduler._native_compute_selection
    if prepare:
        extra = f.scheduler.vllm_config.kv_transfer_config.kv_connector_extra_config
        extra["lmcache.mp.first_match_preparation"] = True
        extra["lmcache.mp.first_match_prepare_limit"] = prepare_limit
        f.scheduler._init_first_match_preparation()
    return f


W = []


class FirstMatchPreparationTests(unittest.TestCase):
    def tearDown(self):
        s.close_fixtures()

    def test_pending_lookup_then_same_step_first_allocation_crosses_queues(self):
        results = {}
        chains = {}
        for mode in ("native", "sjf"):
            f = fixture(mode, prepare=True)
            long = s.t.enqueue(f, "long", 16384)
            s.t.known_result(f, long, 0)
            f.scheduler.schedule()
            short = s.t.enqueue(f, "short", 512, 1)
            first = f.scheduler.schedule()
            self.assertEqual(first.num_scheduled_tokens, {"long": 2048})
            self.assertTrue(f.adapter.lookup_operation_state("short")["submitted"])
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("short"), ([],))
            self.assertNotIn("short", f.selection.keys)
            s.ops.ack(f.rpc, "short")
            f.scheduler.schedule()  # ACK -> nonblocking STATUS
            s.ops.status(f.rpc, 0, "short")
            before = list(f.scheduler.running)
            out = f.scheduler.schedule()
            results[mode] = out.num_scheduled_tokens
            self.assertEqual(
                f.selection.keys["short"].step, f.selection.snapshot.step_id
            )
            self.assertEqual(f.scheduler.running[: len(before)], before)
            self.assertEqual(
                sum(c[0] == "LOOKUP" and c[2] == "short" for c in f.rpc.calls), 1
            )
            self.assertEqual(short.num_output_tokens, 0)
            chains[mode] = dict(
                first_match=asdict(f.scheduler._first_match_preparation.first["short"]),
                key=asdict(f.selection.keys["short"]),
                selected=dict(out.num_scheduled_tokens),
                selection_step=f.selection.snapshot.step_id,
                query_calls=[c for c in f.rpc.calls if c[2] == "short"],
                events=[
                    e
                    for e in f.rt.events
                    if e.get("request_id") == "short"
                    and e["event"]
                    in (
                        "native_compute_key_frozen",
                        "native_compute_key_first_selection",
                        "native_compute_first_allocation",
                    )
                ],
            )
        self.assertEqual(results["native"], {"long": 2048})
        self.assertEqual(results["sjf"], {"short": 512, "long": 1536})
        W.append(
            dict(
                kind="prepared_unknown_to_first_allocation",
                actual=results,
                chains=chains,
            )
        )

    def test_prepared_new_selection_rollback_and_retry_keep_frozen_information(self):
        f = fixture("sjf", prepare=True)
        long = s.t.enqueue(f, "long", 16384)
        s.t.known_result(f, long, 0)
        f.scheduler.schedule()
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
        key = f.selection.keys["short"]
        # Native retry keeps its already-computed 1536 proposal. Reclaimed
        # 512 tokens are free, not silently charged to the revoked short.
        self.assertEqual(out.num_scheduled_tokens, {"long": 1536})
        self.assertEqual(f.selection.selected, {"long": (6144, 1536)})
        self.assertEqual(short.status, f.rt.status.PREEMPTED)
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("short"), ([],))
        self.assertFalse(
            any(m.request_id == "short" for m in out.kv_connector_metadata.requests)
        )
        self.assertEqual(
            sum(e["event"] == "native_compute_revoked" for e in f.rt.events), 1
        )
        retry = f.scheduler.schedule()
        self.assertEqual(retry.num_scheduled_tokens, {"short": 512, "long": 1536})
        self.assertEqual(f.selection.keys["short"], key)
        self.assertEqual(
            sum(c[0] == "LOOKUP" and c[2] == "short" for c in f.rpc.calls), 1
        )

    def test_preparation_does_not_commit_match_watermarks_or_resources(self):
        for chunks, local, prefix in ((0, 992, 992), (2, 0, 512), (2, 992, 992)):
            with self.subTest(chunks=chunks, local=local):
                f = fixture("sjf", prepare=True)
                r = s.t.enqueue(f, "q", 1314)
                if local:
                    s.ops.seed_prefix(f, r, local)
                s.t.known_result(f, r, chunks)
                before = s.t.signature(f, [])
                p = f.scheduler._first_match_preparation
                p.begin_step(f.scheduler)
                p.prepare(f.scheduler)
                self.assertEqual(before, s.t.signature(f, []))
                self.assertEqual(p.first["q"].prefix_tokens, prefix)
                self.assertEqual(f.selection.keys["q"].prefix_tokens, prefix)
                out = f.scheduler.schedule()
                loads = [
                    m
                    for m in out.kv_connector_metadata.requests
                    if m.direction == "RETRIEVE"
                ]
                if chunks and not local:
                    self.assertEqual(out.num_scheduled_tokens, {})
                    self.assertEqual(len(loads), 1)
                    key = f.selection.keys["q"]
                    self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
                    s.received(f, "q")
                    self.assertEqual(
                        f.scheduler.schedule().num_scheduled_tokens, {"q": 802}
                    )
                    self.assertEqual(key, f.selection.keys["q"])
                else:
                    self.assertEqual(out.num_scheduled_tokens, {"q": 322})
                    self.assertEqual(loads, [])
                    ranges = [
                        (m.op.start, m.op.end)
                        for m in out.kv_connector_metadata.requests
                        if m.direction == "STORE"
                    ]
                    self.assertEqual(ranges, [(512 if chunks else 0, 1280)])

    def test_takeover_rechecks_evicted_prefix_without_changing_frozen_key(self):
        f = fixture("sjf", prepare=True)
        r = s.t.enqueue(f, "cached", 1024)
        s.ops.seed_prefix(f, r, 512)
        s.t.known_result(f, r, 0)
        p = f.scheduler._first_match_preparation
        p.begin_step(f.scheduler)
        p.prepare(f.scheduler)
        first, key = p.first["cached"], f.selection.keys["cached"]
        blocks, _, _ = f.scheduler.kv_cache_manager.get_computed_blocks(r)
        ids = [b.block_id for group in blocks.blocks for b in group]
        f.scheduler.kv_cache_manager.evict_blocks(set(ids))
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"cached": 1024})
        self.assertEqual(p.first["cached"], first)
        self.assertEqual(f.selection.keys["cached"], key)
        self.assertEqual(f.selection.reader.matches["cached"].local_tokens, 0)

    def test_prepare_existing_only_never_submits_or_commits(self):
        f = fixture("native")
        r = s.t.enqueue(f, "unselected", 512)
        before = s.t.signature(f, [])
        result = f.scheduler._match_waiting_prefix(
            r, phase="prepare", existing_lookup_only=True
        )
        self.assertIsNone(result)
        self.assertEqual(s.t.signature(f, []), before)
        self.assertFalse(f.adapter.lookup_operation_state("unselected")["submitted"])

    def test_round_robin_32_limit_and_empty_compute_progress(self):
        f = fixture("sjf", prepare=True)
        for i in range(70):
            s.t.enqueue(f, f"q{i}", 512, i)
        for expected in (32, 64, 70):
            self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
            self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), expected)
        # All remain pending: ACK query() visits rotate rather than rescanning
        # the oldest 32; metadata must not sneak in independent progress.
        summaries = [
            e for e in f.rt.events if e["event"] == "prefill_first_match_prepare_step"
        ]
        self.assertEqual(summaries[0]["not_visited"], 38)
        self.assertEqual(
            summaries[2]["request_ids"][:6], [f"q{i}" for i in range(64, 70)]
        )
        self.assertTrue(all(e["visited"] <= 32 for e in summaries))
        self.assertFalse(any(c[0] == "STATUS" for c in f.rpc.calls))
        self.assertEqual(summaries[0]["cap_not_visited"], 38)
        self.assertEqual(summaries[0]["query_pending_visited"], 32)
        self.assertEqual(summaries[1]["max_consecutive_cap_steps"], 2)
        for i in range(70, 75):
            s.t.enqueue(f, f"q{i}", 512, i)
        for _ in range(3):
            f.scheduler.schedule()
        self.assertTrue(
            all(
                f.adapter.lookup_operation_state(f"q{i}")["submitted"]
                for i in range(75)
            )
        )
        ended = [
            e
            for e in f.rt.events
            if e["event"] == "prefill_first_match_cap_wait" and e["phase"] == "end"
        ]
        self.assertTrue(
            any(e["args"] == ("q64",) and e["consecutive_steps"] == 2 for e in ended)
        )
        self.assertTrue(all(e["duration_ns"] >= 0 for e in ended))

    def test_one_observation_per_step_and_pending_does_not_block_ready(self):
        f = fixture("sjf", prepare=True)
        pending = s.t.enqueue(f, "pending", 512)
        ready = s.t.enqueue(f, "ready", 256, 1)
        s.t.known_result(f, ready, 0)
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"ready": 256})
        self.assertNotIn("pending", f.selection.keys)
        s.ops.ack(f.rpc, "pending")
        # Already observed pending in the current observation epoch. Even a
        # foreground native check must not advance it twice in that epoch.
        self.assertIsNone(f.adapter.check_lookup_result("pending"))
        self.assertFalse(
            any(c[0] == "STATUS" and c[2] == "pending" for c in f.rpc.calls)
        )
        f.scheduler.schedule()
        future = f.adapter._lookup_status["pending"]["server"][0]
        f.adapter.check_lookup_result("pending")
        self.assertIs(f.adapter._lookup_status["pending"]["server"][0], future)
        self.assertEqual(
            sum(c[0] == "STATUS" and c[2] == "pending" for c in f.rpc.calls), 1
        )
        self.assertEqual(pending.num_computed_tokens, 0)

    def test_remaining_budget_visits_unknown_after_known_candidates(self):
        f = fixture("sjf", prepare=True, prepare_limit=1)
        r = s.t.enqueue(f, "ready", 256)
        s.t.known_result(f, r, 0)
        s.t.enqueue(f, "later", 512, 1)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"ready": 256})
        self.assertIn("later", f.selection.visits)
        self.assertFalse(f.adapter.lookup_operation_state("later")["submitted"])
        f.scheduler.schedule()
        self.assertTrue(f.adapter.lookup_operation_state("later")["submitted"])
        self.assertNotIn("later", f.selection.keys)

    def test_global_plan_cached_preparation_requires_query_authorization(self):
        f = s.fixture(global_plan=True, store=True)
        extra = f.scheduler.vllm_config.kv_transfer_config.kv_connector_extra_config
        extra["lmcache.mp.first_match_preparation"] = True
        f.scheduler._init_first_match_preparation()
        T = f.plan_types

        class Policy:
            def plan(self, snapshot):
                return T.StepPlan(snapshot.snapshot_id)

        f.scheduler.set_prefill_plan_policy(Policy())
        s.t.enqueue(f, "q", 512)
        f.scheduler.schedule()
        self.assertFalse(f.adapter.lookup_operation_state("q")["submitted"])
        before = s.t.signature(f, [])
        pending = list(f.scheduler._first_match_preparation.pending)
        for _ in range(3):
            snap = f.scheduler.get_prefill_candidate_snapshot()
            self.assertIsNone(snap.requests[0].first_match)
        self.assertEqual(before, s.t.signature(f, []))
        self.assertEqual(pending, list(f.scheduler._first_match_preparation.pending))
        f.scheduler.set_prefill_plan_policy()  # NativePlan authorizes LOOKUP.
        f.scheduler.schedule()
        self.assertTrue(f.adapter.lookup_operation_state("q")["submitted"])
        self.assertNotIn("q", f.scheduler._first_match_preparation.first)
        s.ops.ack(f.rpc, "q")
        f.scheduler.schedule()
        s.ops.status(f.rpc, 0, "q")
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"q": 512})
        self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)
        self.assertEqual(
            f.scheduler._global_plan.snapshot.requests[0].first_match.prefix_tokens, 0
        )

    def test_failure_falls_back_locally_and_late_reply_cannot_resurrect(self):
        for failure in ("ack_timeout", "status_timeout", "rejected_ack", "unhealthy"):
            with self.subTest(failure=failure):
                f = fixture("sjf", prepare=True)
                r = s.t.enqueue(f, "q", 512)
                s.ops.seed_prefix(f, r, 256)
                if failure == "unhealthy":
                    for event in f.adapter._health_events.values():
                        event.clear()
                else:
                    f.scheduler.schedule()
                    if failure == "ack_timeout":
                        f.adapter._unacked_lookups["q"].submitted_at -= 301
                    elif failure == "rejected_ack":
                        f.rpc.clients["server"].acks["q"].set_result(False)
                    else:
                        s.ops.ack(f.rpc, "q")
                        f.scheduler.schedule()
                        future, at = f.adapter._lookup_status["q"]["server"]
                        f.adapter._lookup_status["q"]["server"] = (future, at - 301)
                out = f.scheduler.schedule()
                self.assertEqual(out.num_scheduled_tokens, {"q": 256})
                self.assertEqual(f.selection.keys["q"].prefix_tokens, 256)
                self.assertFalse(
                    any(
                        m.direction == "RETRIEVE"
                        for m in out.kv_connector_metadata.requests
                    )
                )
                f.scheduler.finish_requests("q", f.rt.status.FINISHED_ABORTED)
                from types import SimpleNamespace

                f.scheduler._update_from_kv_xfer_finished(
                    SimpleNamespace(
                        finished_recving=None,
                        finished_sending={"q"},
                        kv_connector_worker_meta=None,
                    )
                )
                s.ops.wait_until(lambda f=f: f.adapter.cleanup_status()["pending"] == 0)
                if failure in ("ack_timeout", "rejected_ack"):
                    self.assertFalse(f.rpc.clients["server"].finish_lookup("q"))
                self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
                s.ops.wait_until(lambda f=f: f.adapter.cleanup_status()["pending"] == 0)
                self.assertEqual(
                    f.rpc.clients["server"].server._ctx.session_manager.active_count(),
                    0,
                )

    def test_failed_estimate_rechecks_recovered_query_before_takeover(self):
        f = fixture("sjf", prepare=True)
        s.t.enqueue(f, "q", 1024)
        f.scheduler.max_num_running_reqs = 0
        f.scheduler.schedule()
        f.adapter._unacked_lookups["q"].submitted_at -= 301
        f.scheduler.schedule()
        key = f.selection.keys["q"]
        self.assertEqual(key.prefix_tokens, 0)
        for event in f.adapter._health_events.values():
            event.set()
        s.ops.ack(f.rpc, "q")
        f.scheduler.max_num_running_reqs = 32
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("q"), ([],))
        s.ops.status(f.rpc, 2, "q")
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertEqual(
            len(
                [
                    m
                    for m in out.kv_connector_metadata.requests
                    if m.direction == "RETRIEVE"
                ]
            ),
            1,
        )
        self.assertEqual(f.selection.keys["q"], key)
        self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)

    def test_prepared_capacity_sequence_and_cancel_keep_native_ownership(self):
        for blocks, seqs in ((33, 32), (4096, 0)):
            f = fixture("sjf", prepare=True, num_blocks=blocks)
            r = s.t.enqueue(f, "restore", 1024)
            s.t.known_result(f, r, 2)
            f.scheduler.max_num_running_reqs = seqs
            for _ in range(2):
                out = f.scheduler.schedule()
                self.assertEqual(out.num_scheduled_tokens, {})
                self.assertFalse(
                    any(
                        m.direction == "RETRIEVE"
                        for m in out.kv_connector_metadata.requests
                    )
                )
            self.assertEqual(
                f.scheduler.kv_cache_manager.get_block_ids("restore"), ([],)
            )
            self.assertEqual(
                f.connector.request_trackers["restore"].num_stored_tokens,
                512 if blocks == 33 else 0,
            )
            f.scheduler.finish_requests("restore", f.rt.status.FINISHED_ABORTED)
            s.ops.wait_until(lambda f=f: f.adapter.cleanup_status()["pending"] == 0)
            self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})

    def test_common_preparation_rejects_unbounded_or_blocking_modes(self):
        f = fixture("sjf")
        extra = f.scheduler.vllm_config.kv_transfer_config.kv_connector_extra_config
        extra["lmcache.mp.first_match_preparation"] = True
        for key, value in (
            ("first_match_prepare_limit", 33),
            ("nonblocking_lookup_status", False),
            ("lookup_independent_progress", True),
            ("prepare_load_operations", True),
        ):
            extra["lmcache.mp." + key] = value
            with self.assertRaises(ValueError):
                f.scheduler._init_first_match_preparation()
            extra.pop("lmcache.mp." + key)

    def test_pending_cancel_then_late_ack_releases_native_source_owner(self):
        f = fixture("sjf", prepare=True)
        s.t.enqueue(f, "cancel", 512)
        f.scheduler.schedule()
        f.scheduler.finish_requests("cancel", f.rt.status.FINISHED_ABORTED)
        s.ops.wait_until(lambda: f.adapter.cleanup_status()["pending"] == 0)
        self.assertFalse(f.rpc.clients["server"].finish_lookup("cancel"))
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
        self.assertNotIn("cancel", f.scheduler._first_match_preparation.pending)
        self.assertNotIn("cancel", f.adapter._owned_lookup_ids)
        self.assertEqual(
            f.rpc.clients["server"].server._ctx.session_manager.active_count(), 0
        )

    def test_real_full_hit_last_token_and_cache_growth_preserve_initial_scalar(self):
        f = fixture("sjf", prepare=True)
        r = s.t.enqueue(f, "full", 1024)
        s.t.known_result(f, r, 4)
        p = f.scheduler._first_match_preparation
        p.begin_step(f.scheduler)
        p.prepare(f.scheduler)
        self.assertEqual(
            (p.first["full"].prefix_tokens, p.first["full"].remaining_tokens), (1023, 1)
        )
        key = f.selection.keys["full"]
        s.ops.seed_prefix(f, r, 1024)
        out = f.scheduler.schedule()
        # Native local matching keeps complete block16 prefixes (1008 here),
        # so the remaining 15 matched tokens still require the native LOAD.
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertEqual(f.selection.reader.matches["full"].local_tokens, 1008)
        self.assertEqual(
            len(
                [
                    m
                    for m in out.kv_connector_metadata.requests
                    if m.direction == "RETRIEVE"
                ]
            ),
            1,
        )
        s.received(f, "full")
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"full": 1})
        self.assertEqual(f.selection.keys["full"], key)

    def test_native_order_preparation_without_policy_stores_no_cost(self):
        f = fixture(prepare=True)
        r = s.t.enqueue(f, "q", 512)
        s.t.known_result(f, r, 0)
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"q": 512})
        first = f.scheduler._first_match_preparation.first["q"]
        self.assertEqual(first.remaining_tokens, 512)
        self.assertFalse(hasattr(first, "work"))

    def test_prepared_same_step_native_rollback_preserves_ledger(self):
        f = fixture("native", prepare=True, num_blocks=81)
        a = s.t.active(f, "a", prompt=512, computed=256)
        s.t.active(f, "b", prompt=1280, computed=512)
        # Fault fixture exercises the accepted rollback branch; production
        # initialization still rejects priority and retains native FCFS.
        f.scheduler.policy = type(f.scheduler.policy).PRIORITY
        a.priority = 1
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"b": 768})
        self.assertEqual(f.selection.selected, {"b": (512, 768)})
        self.assertEqual(
            sum(e["event"] == "native_compute_revoked" for e in f.rt.events), 1
        )

    def test_prepared_request_outputs_one_token_and_store_drains_after_finish(self):
        f = fixture("sjf", prepare=True)
        s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
        r = s.t.enqueue(f, "store", 512)
        r.sampling_params.num_logprobs = None
        s.t.known_result(f, r, 0)
        scheduled = f.scheduler.schedule()
        stores = [
            m
            for m in scheduled.kv_connector_metadata.requests
            if m.direction == "STORE"
        ]
        self.assertEqual([(m.op.start, m.op.end) for m in stores], [(0, 512)])
        before = s.ops.cache_snapshot(f.scheduler.kv_cache_manager)["owned"]
        w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
        ns = w.connector.build_connector_meta.__globals__
        ns["LMCacheMPConnectorMetadata"] = (
            f.connector._fixture_code.LMCacheMPConnectorMetadata
        )
        ns["LMCacheMPWorkerMetadata"] = (
            f.connector._fixture_code.LMCacheMPWorkerMetadata
        )
        w.active.pre_forward(scheduled)
        committed = w.active.post_forward(set(), wait_for_save=True)
        cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
        sampled = cls(
            req_ids=["store"],
            req_id_to_index={"store": 0},
            sampled_token_ids=[[17, 18]],
            kv_connector_output=committed,
        )
        result = f.scheduler.update_from_output(scheduled, sampled)
        self.assertEqual(result[0].outputs[0].new_token_ids, [17])
        self.assertEqual(r.num_output_tokens, 1)
        self.assertEqual(
            s.ops.cache_snapshot(f.scheduler.kv_cache_manager)["owned"], before
        )
        zero = f.scheduler.schedule()
        self.assertEqual(zero.num_scheduled_tokens, {})
        f.scheduler.update_from_output(zero, w.active.no_forward(zero))
        self.assertEqual(len(w.adapter._fixture_calls), 1)
        self.assertIs(w.adapter._fixture_calls[0][1][5], w.producer_events[0])
        self.assertEqual(w.adapter.page_ownership.pending_count, 1)
        w.adapter._fixture_futures[0].set_result(True)
        final = f.scheduler.schedule()
        f.scheduler.update_from_output(final, w.active.no_forward(final))
        self.assertNotIn("store", f.scheduler.requests)
        self.assertEqual(w.adapter.page_ownership.pending_count, 0)
        self.assertEqual(f.connector._store_operation_views, {})
        self.assertEqual(f.scheduler._action_feedback.active, {})
        W.append(
            dict(
                kind="prepared_single_token_store_drain",
                tokens=[17],
                store_ranges=[(m.op.start, m.op.end) for m in stores],
                original_event=True,
                final_pending_pages=0,
            )
        )


class NativeComputeTests(unittest.TestCase):
    def tearDown(self):
        s.close_fixtures()

    def test_native_and_disabled_keep_real_operations(self):
        signatures = []
        for runtime, mode in (
            (s.REFERENCE, None),
            (RUNTIME, None),
            (RUNTIME, "native"),
        ):
            f = fixture(mode, runtime)
            a = s.t.enqueue(f, "long", 4096)
            s.t.known_result(f, a, 0)
            outputs = [f.scheduler.schedule()]
            b = s.t.enqueue(f, "short", 512, 1)
            s.t.known_result(f, b, 0)
            outputs += [f.scheduler.schedule(), f.scheduler.schedule()]
            signatures.append(s.t.signature(f, outputs))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_unknown_stays_in_native_slot_and_full_budget_does_not_query(self):
        f = fixture("sjf")
        a = s.t.enqueue(f, "long", 8192)
        s.t.known_result(f, a, 0)
        f.scheduler.schedule()
        s.t.enqueue(f, "unknown", 512, 1)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"long": 2048})
        self.assertFalse(f.adapter.lookup_operation_state("unknown")["submitted"])
        self.assertNotIn("unknown", f.selection.keys)
        self.assertEqual(f.selection.ordered_ids, ("long", "unknown"))

    def test_pending_unknown_can_submit_and_never_executes(self):
        f = fixture("sjf")
        s.t.enqueue(f, "pending", 512)
        b = s.t.enqueue(f, "knownlater", 512, 1)
        s.t.known_result(f, b, 0)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"knownlater": 512})
        self.assertTrue(f.adapter.lookup_operation_state("pending")["submitted"])
        self.assertNotIn("pending", f.selection.keys)
        self.assertIn("knownlater", f.selection.keys)

    def test_cross_queue_known_restore_has_actual_priority(self):
        f = fixture("sjf")
        # Native matching obtains B's key and submits its real controlled LOAD.
        b = s.t.enqueue(f, "short_restore", 1024)
        s.t.known_result(f, b, 2)
        first = f.scheduler.schedule()
        self.assertEqual(first.num_scheduled_tokens, {})
        self.assertEqual(
            [
                m.request_id
                for m in first.kv_connector_metadata.requests
                if m.direction == "RETRIEVE"
            ],
            ["short_restore"],
        )
        a = s.t.enqueue(f, "long", 8192, 1)
        s.t.known_result(f, a, 0)
        f.scheduler.schedule()
        self.assertEqual(b.status, f.rt.status.WAITING_FOR_REMOTE_KVS)
        s.received(f, "short_restore")
        order = list(f.scheduler.running)
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"short_restore": 512, "long": 1536})
        self.assertEqual(f.scheduler.running[: len(order)], order)
        self.assertEqual(sum(out.num_scheduled_tokens.values()), 2048)
        W.append(
            {
                "kind": "cross_queue",
                "tokens": out.num_scheduled_tokens,
                "running_order": [r.request_id for r in f.scheduler.running],
                "native_control": not f.scheduler._decoupled_control,
            }
        )

    def test_keys_do_not_change_with_progress_or_repeat_matching(self):
        f = fixture("sjf")
        a = s.t.enqueue(f, "long", 8192)
        s.t.known_result(f, a, 0)
        f.scheduler.schedule()
        key = f.selection.keys["long"]
        f.scheduler.schedule()
        self.assertEqual(key, f.selection.keys["long"])
        f.selection.matched(a, 512, 0, 512, f.scheduler.current_step)
        self.assertEqual(key, f.selection.keys["long"])

    def test_same_r_larger_p_has_larger_structure_cost(self):
        f = fixture("sjf")
        self.assertGreater(
            f.selection.model.estimate(512, 4096), f.selection.model.estimate(512, 0)
        )

    def test_native_match_l1_zero_preserves_store_progress(self):
        for mode in ("native", "sjf"):
            f = fixture(mode)
            r = s.t.enqueue(f, "q", 1314)
            s.ops.seed_prefix(f, r, 992)
            s.t.known_result(f, r, 0)
            o = f.scheduler.schedule()
            self.assertEqual(o.num_scheduled_tokens, {"q": 322})
            self.assertEqual(
                [
                    (m.op.start, m.op.end)
                    for m in o.kv_connector_metadata.requests
                    if m.direction == "STORE"
                ],
                [(0, 1280)],
            )
            self.assertEqual(f.selection.keys["q"].prefix_tokens, 992)

    def test_reader_is_pure(self):
        f = fixture("sjf")
        s.t.enqueue(f, "unmatched", 512)
        before = s.t.signature(f, [])
        snap = f.selection.reader.read_snapshot(f.scheduler, 2048, 2048)
        self.assertEqual(len(snap.requests), 1)
        self.assertEqual(before, s.t.signature(f, []))
        self.assertFalse(f.adapter.lookup_operation_state("unmatched")["submitted"])

    def test_cancellation_of_pending_lookup_does_not_compute(self):
        f = fixture("sjf")
        s.t.enqueue(f, "q", 512)
        f.scheduler.schedule()
        f.scheduler.finish_requests(["q"], f.rt.status.FINISHED_ABORTED)
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})

    def test_unknown_eventually_gets_native_lookup_after_running_chunks(self):
        f = fixture("sjf")
        a = s.t.enqueue(f, "long", 8192)
        s.t.known_result(f, a, 0)
        f.scheduler.schedule()
        s.t.enqueue(f, "unknown", 512, 1)
        for _ in range(4):
            f.scheduler.schedule()
        self.assertTrue(f.adapter.lookup_operation_state("unknown")["submitted"])
        self.assertEqual(
            sum(c[0] == "LOOKUP" and c[2] == "unknown" for c in f.rpc.calls), 1
        )

    def test_capacity_failure_keeps_no_resource_or_compute_commitment(self):
        f = fixture("sjf", num_blocks=33)
        r = s.t.enqueue(f, "large", 1024)
        s.t.known_result(f, r, 2)
        for _ in range(2):
            o = f.scheduler.schedule()
            self.assertEqual(o.num_scheduled_tokens, {})
            self.assertFalse(
                any(m.direction == "RETRIEVE" for m in o.kv_connector_metadata.requests)
            )
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("large"), ([],))
        self.assertIn("large", f.selection.keys)

    def test_sequence_limit_still_allows_running_progress(self):
        f = fixture("sjf")
        a = s.t.enqueue(f, "long", 8192)
        s.t.known_result(f, a, 0)
        f.scheduler.schedule()
        f.scheduler.max_num_running_reqs = 1
        b = s.t.enqueue(f, "short", 512, 1)
        s.t.known_result(f, b, 0)
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"long": 2048})
        self.assertEqual(b.status, f.rt.status.WAITING)

    def test_cancel_reserved_load_retains_pages_until_received(self):
        f = fixture("sjf")
        r = s.t.enqueue(f, "restore", 1024)
        s.t.known_result(f, r, 2)
        f.scheduler.schedule()
        before = s.ops.cache_snapshot(f.scheduler.kv_cache_manager)
        f.scheduler.finish_requests("restore", f.rt.status.FINISHED_ABORTED)
        self.assertEqual(s.ops.cache_snapshot(f.scheduler.kv_cache_manager), before)
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
        s.received(f, "restore")
        self.assertNotIn("restore", f.scheduler.requests)

    def test_native_selected_rollback_updates_compute_observer_once(self):
        f = fixture("native", num_blocks=81)
        a = s.t.active(f, "a", prompt=512, computed=256)
        s.t.active(f, "b", prompt=1280, computed=512)
        f.scheduler.policy = type(f.scheduler.policy).PRIORITY
        a.priority = 1
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"b": 768})
        self.assertEqual(f.selection.selected, {"b": (512, 768)})
        self.assertEqual(
            sum(e["event"] == "native_compute_revoked" for e in f.rt.events), 1
        )


if __name__ == "__main__":
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NativeComputeTests)
    )
    (AREA / "stage-a.log").write_text(stream.getvalue(), encoding="utf-8")
    (AREA / "stage-a.json").write_text(
        json.dumps(
            dict(passed=result.wasSuccessful(), tests=result.testsRun, witnesses=W),
            indent=2,
        )
        + "\n"
    )
    print(stream.getvalue())
    raise SystemExit(not result.wasSuccessful())
