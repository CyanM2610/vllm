# SPDX-License-Identifier: Apache-2.0
"""Explicit recovery choices through native scheduler, cache and connector methods."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType

import test_native_compute_selection as n

s = n.s
RUNTIME = Path(os.environ["PREFILL_NATIVE_RUNTIME"])
W = []


def fixture(*, enabled=True, new_lookup=False, prepare=False, **kwargs):
    f = s.fixture(
        global_plan=True,
        control=True,
        store=True,
        new_lookup=new_lookup,
        prepare_load=True,
        **kwargs,
    )
    trace = ModuleType("vllm.v1.kv_wait_trace")
    trace.emit = lambda event, request_id=None, **fields: f.rt.events.append(
        dict(event=event, request_id=request_id, **fields)
    )
    sys.modules[trace.__name__] = trace
    path = RUNTIME / "source/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py"
    if "def commit_recompute_path(" in path.read_text(encoding="utf-8"):
        s.ops.add_methods(
            path,
            "LMCacheMPConnector",
            type(f.connector),
            {"validate_recompute_path", "commit_recompute_path"},
            f.connector.build_connector_meta.__globals__,
        )
    cfg = f.scheduler.vllm_config.kv_transfer_config.kv_connector_extra_config
    cfg["lmcache.mp.recovery_path_choice"] = enabled
    f.scheduler._init_prefill_operations()
    if prepare:
        cfg["lmcache.mp.first_match_preparation"] = True
        f.scheduler._init_first_match_preparation()
    native_lease_release_transport(f)
    return f


def native_lease_release_transport(f):
    """Execute server lease release; control only RPC and object-key resolution."""
    client = f.rpc.clients["server"]
    server = client.server
    ns = server.lookup_guarded.__globals__

    def resolved(ctx, key, hit_chunks, locked_gids):
        lease = ctx.session_manager.get(key.request_id).hybrid_read_lease
        # Physical storage/hash lookup is a transport fixture. Request-owned
        # release, multiplicity and transferred-pin rules remain real methods.
        return [k for k in lease.remaining if key.start // 256 <= k[1] < key.end // 256]

    ns["resolve_prefetched_obj_keys"] = resolved
    source = RUNTIME / "source/LMCache/lmcache/v1/multiprocess/modules/lookup.py"
    s.ops.add_methods(
        source,
        "LookupModule",
        type(server),
        {"free_lookup_locks", "_free_lookup_locks"},
        ns,
    )

    def free(key, tp):
        f.rpc.calls.append(("FREE", "server", key.request_id, key.start, key.end))
        key.require_num_kv_readers = lambda: 1
        return server.free_lookup_locks(key, tp)

    client.free_lookup_locks = free
    finish_status = client.finish_status

    def finish(rid, hit_chunks):
        # The reusable endpoint initially supplies one controlled lease per
        # possible chunk. A real prefetch owns only returned hits; give the
        # native lifecycle that same external storage outcome before STATUS.
        session = server._ctx.session_manager.get(rid)
        if session is not None and isinstance(hit_chunks, int):
            lease = session.hybrid_read_lease
            if lease is not None:
                lease.release([k for k in lease.remaining if k[1] >= hit_chunks], 1)
        return finish_status(rid, hit_chunks)

    client.finish_status = finish


def choose(f, build):
    class Policy:
        def plan(self, view):
            return f.plan_types.StepPlan(view.snapshot_id, tuple(build(view)))

    f.scheduler.set_prefill_plan_policy(Policy())


def item(f, rid, path=None, *, tokens=None):
    T = f.plan_types
    kind = T.OperationKind.LOAD if path == "restore" else T.OperationKind.COMPUTE
    return T.PlanItem(
        kind,
        rid,
        max_tokens=tokens,
        recovery_path=T.RecoveryPath(path) if path else None,
    )


def matched(f, *, rid="q", length=4096, local=1024, lower=3072):
    r = s.t.enqueue(f, rid, length)
    if local:
        s.ops.seed_prefix(f, r, local)
    s.t.known_result(f, r, lower // 256)
    m = f.scheduler._match_waiting_prefix(
        r,
        phase="first_prepare",
        allow_lookup_submit=False,
        allow_lookup_progress=False,
        commit_match=False,
    )
    assert m is not None
    return r, m


def ranges(out, direction):
    return [
        (m.op.start, m.op.end)
        for m in out.kv_connector_metadata.requests
        if m.direction == direction
    ]


def worker(f):
    s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
    w = s.ops.worker_fixture(runtime=RUNTIME, decoupled_store=True)
    ns = w.connector.build_connector_meta.__globals__
    ns["LMCacheMPConnectorMetadata"] = (
        f.connector._fixture_code.LMCacheMPConnectorMetadata
    )
    ns["LMCacheMPWorkerMetadata"] = f.connector._fixture_code.LMCacheMPWorkerMetadata
    return w


def compute_output(f, w, r, out):
    w.active.pre_forward(out)
    kv = w.active.post_forward(set(), wait_for_save=True)
    cls = w.active.no_forward.__globals__["ModelRunnerOutput"]
    final = r.num_computed_tokens >= r.num_prompt_tokens
    value = cls(
        req_ids=[r.request_id],
        req_id_to_index={r.request_id: 0},
        sampled_token_ids=[[17, 18] if final else []],
        kv_connector_output=kv,
    )
    return f.scheduler.update_from_output(out, value)


class RecoveryPathTests(unittest.TestCase):
    def tearDown(self):
        s.close_fixtures()

    def test_failed_async_load_in_either_waiting_owner_is_not_published(self):
        """Worker failure must truncate the reservation before native adoption."""
        for owner in ("waiting", "skipped_waiting"):
            for failure_policy in ("recompute", "fail", "cancel"):
                with self.subTest(owner=owner, failure_policy=failure_policy):
                    f = fixture()
                    s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
                    r, _ = matched(f, length=1677, local=32, lower=1280)
                    r.sampling_params.num_logprobs = None
                    f.scheduler.recompute_kv_load_failures = failure_policy != "fail"
                    w = worker(f)
                    w.adapter.kv_load_failure_policy = (
                        "fail" if failure_policy == "fail" else "recompute"
                    )
                    choose(f, lambda v, f=f: (item(f, "q", "restore"),))
                    reserved = f.scheduler.schedule()
                    self.assertEqual(ranges(reserved, "RETRIEVE"), [(0, 1280)])
                    self.assertEqual(reserved.num_scheduled_tokens, {})
                    f.scheduler.update_from_output(
                        reserved, w.active.no_forward(reserved)
                    )
                    if owner == "skipped_waiting":
                        f.scheduler.waiting.remove_requests([r])
                        f.scheduler.skipped_waiting.prepend_request(r)
                    self.assertEqual(r.num_computed_tokens, 1280)
                    self.assertGreater(w.adapter.page_ownership.pending_count, 0)
                    choose(f, lambda v: ())
                    pending = f.scheduler.schedule()
                    f.scheduler.update_from_output(
                        pending, w.active.no_forward(pending)
                    )
                    self.assertEqual(r.num_computed_tokens, 1280)
                    if failure_policy == "cancel":
                        f.scheduler.finish_requests("q", f.rt.status.FINISHED_ABORTED)
                        self.assertTrue(
                            f.scheduler.kv_cache_manager.get_block_ids("q")[0]
                        )
                        self.assertGreater(w.adapter.page_ownership.pending_count, 0)
                    for future in w.adapter._fixture_futures:
                        future.set_result(False)
                    receipt = f.scheduler.schedule()
                    result = w.active.no_forward(receipt)
                    self.assertTrue(result.kv_connector_output.invalid_block_ids)
                    f.scheduler.update_from_output(receipt, result)
                    if failure_policy != "cancel":
                        self.assertLessEqual(r.num_computed_tokens, 32)
                    if failure_policy == "recompute":
                        self.assertIn("q", f.scheduler.failed_recving_kv_req_ids)
                        published = f.scheduler.schedule()
                        self.assertEqual(published.num_scheduled_tokens, {})
                        f.scheduler.update_from_output(
                            published, w.active.no_forward(published)
                        )
                        self.assertEqual(r.status, f.rt.status.WAITING)
                        self.assertLessEqual(r.num_computed_tokens, 32)
                        f.scheduler.set_prefill_plan_policy(f.plan_types.NativePlan())
                        out = f.scheduler.schedule()
                        self.assertEqual(
                            out.num_scheduled_tokens, {"q": 1645}, f.rt.events[-24:]
                        )
                        self.assertFalse(ranges(out, "RETRIEVE"))
                        compute_output(f, w, r, out)
                        self.assertEqual(r.num_output_tokens, 1)
                        self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)
                    elif failure_policy == "fail":
                        self.assertEqual(r.status, f.rt.status.FINISHED_ERROR)
                        self.assertEqual(r.num_output_tokens, 0)
                    else:
                        self.assertEqual(r.status, f.rt.status.FINISHED_ABORTED)
                        self.assertEqual(r.num_output_tokens, 0)
                    for _ in range(8):
                        for future in w.adapter._fixture_futures:
                            if not future.query():
                                future.set_result(True)
                        out = f.scheduler.schedule()
                        self.assertEqual(out.num_scheduled_tokens, {})
                        f.scheduler.update_from_output(out, w.active.no_forward(out))
                        if (
                            not f.scheduler.requests
                            and not w.adapter.page_ownership.pending_count
                        ):
                            break
                    self.assertFalse(f.scheduler.requests)
                    self.assertEqual(w.adapter.page_ownership.pending_count, 0)
                    self.assertEqual(
                        f.scheduler.kv_cache_manager.get_block_ids("q"), ([],)
                    )
                    s.ops.wait_until(
                        lambda f=f: f.adapter.cleanup_status()["pending"] == 0
                    )
                    self.assertEqual(
                        f.rpc.clients[
                            "server"
                        ].server._ctx.session_manager.active_count(),
                        0,
                    )

    def test_production_observation_symbol_is_imported_before_method_use(self):
        source = (
            RUNTIME / "source/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        # The CPU AST harness supplies observer endpoints. Check the real
        # module's import too, so that injection cannot hide a runtime NameError.
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.v1.kv_wait_trace"
                and any(
                    alias.name == "emit" and alias.asname == "wait_emit"
                    for alias in node.names
                )
                for node in tree.body
            )
        )

    def test_public_lookup_prepare_plan_flow_reaches_both_real_paths(self):
        outcomes = {}
        for path in ("restore", "recompute"):
            f = fixture(new_lookup=True, prepare=True)
            r = s.t.enqueue(f, "q", 4096)
            s.ops.seed_prefix(f, r, 1024)
            s.t.enqueue(f, "unselected", 512)
            T = f.plan_types
            seen = []

            def build(v, f=f, T=T, path=path, seen=seen):
                q = next(x for x in v.requests if x.request_id == "q")
                seen.append(q)
                if not q.lookup_submitted and not q.lookup_result_known:
                    return (T.PlanItem(T.OperationKind.LOOKUP, "q"),)
                if q.last_match is not None:
                    return (item(f, "q", path),)
                return ()

            choose(f, build)
            first = f.scheduler.schedule()
            self.assertEqual(first.num_scheduled_tokens, {})
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("q"), ([],))
            self.assertIsNone(seen[-1].last_match)
            self.assertTrue(f.adapter.lookup_operation_state("q")["submitted"])
            s.ops.ack(f.rpc, "q")
            pending = f.scheduler.schedule()
            self.assertEqual(pending.num_scheduled_tokens, {})
            self.assertIsNone(seen[-1].last_match)
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("q"), ([],))
            s.ops.status(f.rpc, 12, "q")
            out = f.scheduler.schedule()
            observed = seen[-1]
            self.assertEqual(
                (observed.accounted_tokens, observed.held_hbm_blocks), (0, 0)
            )
            self.assertEqual(
                (observed.last_match.local_tokens, observed.last_match.matched_tokens),
                (1024, 3072),
            )
            self.assertEqual(
                f.scheduler._first_match_preparation.first["q"].prefix_tokens, 3072
            )
            self.assertFalse(
                f.adapter.lookup_operation_state("unselected")["submitted"]
            )
            self.assertEqual(
                f.scheduler.kv_cache_manager.get_block_ids("unselected"), ([],)
            )
            self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)
            if path == "restore":
                self.assertEqual(out.num_scheduled_tokens, {})
                self.assertEqual(ranges(out, "RETRIEVE"), [(1024, 3072)])
            else:
                self.assertEqual(out.num_scheduled_tokens, {"q": 2048})
                self.assertFalse(ranges(out, "RETRIEVE"))
            outcomes[path] = dict(
                query_calls=list(f.rpc.calls),
                observed_before_ownership=dict(h=1024, p=3072, held_blocks=0),
                compute=dict(out.num_scheduled_tokens),
                loads=ranges(out, "RETRIEVE"),
            )
        W.append(dict(kind="public_lookup_prepare_plan_to_actual_path", paths=outcomes))

    def test_same_match_restore_and_recompute_take_distinct_real_paths(self):
        evidence = {}
        for path in ("restore", "recompute"):
            f = fixture()
            s.ops.prepare_full_output_fixture(f, runtime=RUNTIME)
            r, m = matched(f)
            r.sampling_params.num_logprobs = None
            self.assertEqual((m.local_tokens, m.matched_tokens), (1024, 3072))
            before = s.ops.cache_snapshot(f.scheduler.kv_cache_manager)["owned"]
            self.assertEqual(f.connector.request_trackers["q"].num_stored_tokens, 0)
            choose(f, lambda v, f=f, path=path: (item(f, "q", path),))
            out = f.scheduler.schedule()
            first = dict(out.num_scheduled_tokens)
            load_ranges = ranges(out, "RETRIEVE")
            self.assertNotEqual(
                s.ops.cache_snapshot(f.scheduler.kv_cache_manager)["owned"], before
            )
            if path == "restore":
                self.assertEqual(first, {})
                self.assertEqual(load_ranges, [(1024, 3072)])
                self.assertEqual(r.status, f.rt.status.WAITING_FOR_REMOTE_KVS)
                self.assertEqual(r.num_computed_tokens, 3072)
                choose(f, lambda v, f=f: (item(f, "q"),))
                self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
                s.received(f, "q")
                out = f.scheduler.schedule()
                self.assertEqual(out.num_scheduled_tokens, {"q": 1024})
            else:
                self.assertEqual(first, {"q": 2048})
                self.assertEqual(load_ranges, [])
                self.assertEqual(r.num_computed_tokens, 3072)
                self.assertEqual(ranges(out, "STORE"), [])
                choose(f, lambda v, f=f: (item(f, x.request_id) for x in v.requests))
                s.ops.wait_until(lambda f=f: f.adapter.cleanup_status()["pending"] == 0)
                lease = (
                    f.rpc.clients["server"]
                    .server._ctx.session_manager.get("q")
                    .hybrid_read_lease
                )
                self.assertEqual(lease.reservation_count(), 0)
            choose(f, lambda v, f=f: (item(f, x.request_id) for x in v.requests))
            w = worker(f)
            compute_quotas, stores = [], []
            while out.num_scheduled_tokens:
                compute_quotas.append(out.num_scheduled_tokens["q"])
                stores.extend(ranges(out, "STORE"))
                compute_output(f, w, r, out)
                if r.num_output_tokens:
                    break
                out = f.scheduler.schedule()
            self.assertEqual(r.num_output_tokens, 1)
            self.assertEqual(r.num_computed_tokens, 4096)
            self.assertEqual(stores, [(3072, 4096)])
            zero = f.scheduler.schedule()
            self.assertEqual(zero.num_scheduled_tokens, {})
            f.scheduler.update_from_output(zero, w.active.no_forward(zero))
            self.assertEqual(len(w.adapter._fixture_calls), 1)
            self.assertIs(w.adapter._fixture_calls[0][1][5], w.producer_events[-1])
            self.assertGreater(w.adapter.page_ownership.pending_count, 0)
            for pending in w.adapter._fixture_futures:
                pending.set_result(True)
            final = f.scheduler.schedule()
            f.scheduler.update_from_output(final, w.active.no_forward(final))
            self.assertNotIn("q", f.scheduler.requests)
            self.assertEqual(w.adapter.page_ownership.pending_count, 0)
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("q"), ([],))
            s.ops.wait_until(lambda f=f: f.adapter.cleanup_status()["pending"] == 0)
            evidence[path] = dict(
                match=dict(L=4096, h=1024, p=3072),
                first_quota=first,
                load_ranges=load_ranges,
                compute_quotas=compute_quotas,
                store_ranges=stores,
                output_tokens=r.num_output_tokens,
                final_page_protection=w.adapter.page_ownership.pending_count,
                final_sessions=f.rpc.clients[
                    "server"
                ].server._ctx.session_manager.active_count(),
            )
        self.assertEqual(evidence["restore"]["compute_quotas"], [1024])
        self.assertEqual(evidence["recompute"]["compute_quotas"], [2048, 1024])
        W.append(
            dict(
                kind="same_match_restore_or_recompute_complete_native_lifecycle",
                paths=evidence,
            )
        )

    def test_recompute_local_and_lower_boundaries_preserve_saved_ranges(self):
        for length, local, lower, expected, store in (
            (1314, 992, 0, 322, (0, 1280)),
            (1024, 0, 512, 1024, (512, 1024)),
            (1314, 992, 512, 322, (512, 1280)),
            (1024, 1024, 1024, 16, None),
        ):
            with self.subTest(length=length, local=local, lower=lower):
                f = fixture()
                r, _ = matched(f, length=length, local=local, lower=lower)
                choose(f, lambda v, f=f: (item(f, "q", "recompute"),))
                out = f.scheduler.schedule()
                self.assertEqual(out.num_scheduled_tokens, {"q": expected})
                self.assertFalse(ranges(out, "RETRIEVE"))
                self.assertEqual(ranges(out, "STORE"), [store] if store else [])
                self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)

    def test_restore_hbm_covered_does_not_implicitly_compute(self):
        f = fixture()
        r, _ = matched(f, length=1024, local=512, lower=512)
        before = s.ops.cache_snapshot(f.scheduler.kv_cache_manager)
        choose(f, lambda v: (item(f, "q", "restore"),))
        out = f.scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertFalse(out.kv_connector_metadata.requests)
        self.assertEqual(s.ops.cache_snapshot(f.scheduler.kv_cache_manager), before)
        self.assertTrue(any(e.get("result") == "not_required" for e in s.rows(f)))

    def test_full_lower_hit_reserves_effective_prefix_and_recomputes_last_token(self):
        for hbm in (1024, 1008):
            with self.subTest(hbm=hbm):
                f = fixture()
                r, m = matched(f, local=hbm, lower=4096)
                self.assertEqual(
                    (m.local_tokens, m.matched_tokens, m.external_tokens),
                    (hbm, 4095, 4095 - hbm),
                )
                choose(f, lambda v, f=f: (item(f, "q", "restore"),))
                out = f.scheduler.schedule()
                self.assertEqual(out.num_scheduled_tokens, {})
                self.assertEqual(ranges(out, "RETRIEVE"), [(hbm // 256 * 256, 4096)])
                load = next(
                    x
                    for x in out.kv_connector_metadata.requests
                    if x.direction == "RETRIEVE"
                )
                self.assertEqual(load.op.skip_first_n_tokens, hbm % 256)
                self.assertEqual(r.num_computed_tokens, 4095)
                committed = [
                    e
                    for e in f.scheduler._global_plan.path_feedback
                    if e.reason == "path_committed"
                ]
                self.assertEqual(len(committed), 1)
                self.assertEqual(committed[0].reserved_external_tokens, 4095 - hbm)
                choose(f, lambda v, f=f: (item(f, "q"),))
                self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
                s.received(f, "q")
                compute = f.scheduler.schedule()
                self.assertEqual(compute.num_scheduled_tokens, {"q": 1})
                self.assertEqual(ranges(compute, "STORE"), [])
                tokens, stopped = f.scheduler._update_request_with_output(r, [17, 18])
                self.assertEqual(tokens, [17])
                self.assertTrue(stopped)

    def test_takeover_reconfirms_changed_hbm_without_stale_pointers(self):
        f = fixture()
        r, first = matched(f)
        blocks, _, _ = f.scheduler.kv_cache_manager.get_computed_blocks(r)
        old = set(blocks.get_block_ids()[0])
        f.scheduler.kv_cache_manager.evict_blocks(old)
        choose(f, lambda v: (item(f, "q", "recompute"),))
        out = f.scheduler.schedule()
        self.assertEqual(first.local_tokens, 1024)
        self.assertEqual(out.num_scheduled_tokens, {"q": 2048})
        self.assertEqual(r.num_computed_tokens, 2048)
        self.assertEqual(
            f.connector.request_trackers["q"].local_compute_start_tokens, 0
        )
        self.assertFalse(ranges(out, "RETRIEVE"))
        self.assertFalse(ranges(out, "STORE"))

    def test_unknown_lookup_cannot_be_recompute_or_trigger_new_query(self):
        for submitted in (False, True):
            f = fixture()
            r = s.t.enqueue(f, "q", 1024)
            if submitted:
                self.assertEqual(
                    f.connector.get_num_new_matched_tokens(r, 0), (None, True)
                )
            before = list(f.rpc.calls)
            choose(f, lambda v, f=f: (item(f, "q", "recompute"),))
            out = f.scheduler.schedule()
            self.assertEqual(out.num_scheduled_tokens, {})
            self.assertFalse(out.kv_connector_metadata.requests)
            self.assertEqual(f.rpc.calls, before)
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("q"), ([],))

    def test_failed_match_switch_is_blocked_and_cancel_cleans_late_reply(
        self,
    ):
        for failure in ("pending", "ack_timeout", "negative_ack"):
            f = fixture()
            r = s.t.enqueue(f, "q", 1024)
            self.assertEqual(f.connector.get_num_new_matched_tokens(r, 0), (None, True))
            if failure == "ack_timeout":
                f.adapter._unacked_lookups["q"].submitted_at -= 301
                f.adapter.check_lookup_result("q")
            elif failure == "negative_ack":
                f.rpc.clients["server"].acks["q"].set_result(False)
                f.adapter.check_lookup_result("q")
            choose(f, lambda v, f=f: (item(f, "q", "recompute"),))
            out = f.scheduler.schedule()
            self.assertEqual(out.num_scheduled_tokens, {})
            self.assertFalse(out.kv_connector_metadata.requests)
            self.assertIsNone(
                f.connector.request_trackers["q"].local_compute_start_tokens
            )
            f.scheduler.finish_requests("q", f.rt.status.FINISHED_ABORTED)
            s.ops.wait_until(lambda f=f: f.adapter.cleanup_status()["pending"] == 0)
            self.assertFalse(f.rpc.clients["server"].finish_lookup("q"))
            self.assertEqual(
                f.rpc.clients["server"].server._ctx.session_manager.active_count(), 0
            )
            self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)

    def test_one_recompute_releases_only_its_shared_lower_read_obligation(self):
        f = fixture()
        a, _ = matched(f, rid="a", length=1024, local=256, lower=512)
        matched(f, rid="b", length=1024, local=256, lower=512)
        server = f.rpc.clients["server"].server
        sessions = server._ctx.session_manager
        lease_type = type(sessions.get("a").hybrid_read_lease)
        for rid in ("a", "b"):
            sessions.get(rid).hybrid_read_lease.close()
        key = ("shared-object", 0)
        server.directory.logical[key] = 2
        for rid in ("a", "b"):
            sessions.get(rid).hybrid_read_lease = lease_type(server.directory, [key], 1)
        b_lease = sessions.get("b").hybrid_read_lease
        choose(f, lambda v: (item(f, "a", "recompute"),))
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"a": 768})
        s.ops.wait_until(lambda: f.adapter.cleanup_status()["pending"] == 0)
        self.assertEqual(server.directory.logical[key], 1)
        self.assertEqual(b_lease.reservation_count(), 1)
        self.assertFalse(b_lease.closed)
        result = b_lease.acquire_transfer([key])
        self.assertEqual(result[key][0], "success")
        self.assertEqual(server.directory.logical[key], 0)
        self.assertEqual(server.directory.physical[key], 1)
        f.adapter.end_session("b")
        s.ops.wait_until(lambda: f.adapter.cleanup_status()["pending"] == 0)
        self.assertEqual(server.directory.physical[key], 1)
        server.directory.release_transfer([key])
        self.assertEqual(server.directory.physical[key], 0)

    def test_same_step_revoke_then_native_reentry_keeps_lower_coverage(self):
        f = fixture()
        long = s.t.enqueue(f, "long", 16384)
        s.t.known_result(f, long, 0)
        choose(f, lambda v: (item(f, "long"),))
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"long": 2048})
        r, _ = matched(f, rid="short", length=512, local=0, lower=256)
        allocate = f.scheduler.kv_cache_manager.allocate_slots
        failed = []

        def fail_once(request, *args, **kwargs):
            if request is long and not failed:
                failed.append(True)
                return None
            return allocate(request, *args, **kwargs)

        f.scheduler.kv_cache_manager.allocate_slots = fail_once
        choose(f, lambda v: (item(f, "short", "recompute"), item(f, "long")))
        revoked = f.scheduler.schedule()
        self.assertNotIn("short", revoked.num_scheduled_tokens)
        self.assertEqual(r.status, f.rt.status.PREEMPTED)
        self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("short"), ([],))
        self.assertFalse(
            any(m.request_id == "short" for m in revoked.kv_connector_metadata.requests)
        )
        choose(f, lambda v: (item(f, "short"), item(f, "long")))
        retried = f.scheduler.schedule()
        self.assertEqual(retried.num_scheduled_tokens["short"], 512)
        self.assertEqual(
            [
                (m.op.start, m.op.end)
                for m in retried.kv_connector_metadata.requests
                if m.request_id == "short" and m.direction == "STORE"
            ],
            [(256, 512)],
        )
        self.assertFalse(ranges(retried, "RETRIEVE"))
        self.assertEqual(
            sum(c[0] == "LOOKUP" and c[2] == "short" for c in f.rpc.calls), 1
        )

    def test_capacity_failure_keeps_path_uncommitted_and_retry_rechecks(self):
        f = fixture(num_blocks=129)
        r, _ = matched(f, local=0)
        choose(f, lambda v: (item(f, "q", "recompute"),))
        for _ in range(2):
            out = f.scheduler.schedule()
            self.assertEqual(out.num_scheduled_tokens, {})
            self.assertFalse(out.kv_connector_metadata.requests)
            self.assertIsNone(
                f.connector.request_trackers["q"].local_compute_start_tokens
            )
            self.assertEqual(f.scheduler.kv_cache_manager.get_block_ids("q"), ([],))
        self.assertTrue(f.adapter.lookup_operation_state("q")["source_ownership"])
        self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 1)
        f.scheduler.finish_requests("q", f.rt.status.FINISHED_ABORTED)
        s.ops.wait_until(lambda: f.adapter.cleanup_status()["pending"] == 0)
        self.assertEqual(
            f.rpc.clients["server"].server._ctx.session_manager.active_count(), 0
        )

    def test_sequence_limit_does_not_commit_path_or_consume_source(self):
        f = fixture()
        matched(f)
        f.scheduler.max_num_running_reqs = 0
        choose(f, lambda v: (item(f, "q", "recompute"),))
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
        self.assertIsNone(f.connector.request_trackers["q"].local_compute_start_tokens)
        self.assertTrue(f.adapter.lookup_operation_state("q")["source_ownership"])
        f.scheduler.max_num_running_reqs = 32
        self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {"q": 2048})

    def test_reserved_load_rejects_switch_and_cancel_keeps_protected_pages(self):
        f = fixture()
        r, _ = matched(f)
        choose(f, lambda v: (item(f, "q", "restore"),))
        out = f.scheduler.schedule()
        self.assertTrue(ranges(out, "RETRIEVE"))
        before = s.ops.cache_snapshot(f.scheduler.kv_cache_manager)
        choose(f, lambda v: (item(f, "q", "recompute"),))
        attempted = f.scheduler.schedule()
        self.assertEqual(attempted.num_scheduled_tokens, {})
        self.assertFalse(ranges(attempted, "RETRIEVE"))
        self.assertEqual(r.status, f.rt.status.WAITING_FOR_REMOTE_KVS)
        self.assertEqual(s.ops.cache_snapshot(f.scheduler.kv_cache_manager), before)
        f.scheduler.finish_requests("q", f.rt.status.FINISHED_ABORTED)
        self.assertEqual(s.ops.cache_snapshot(f.scheduler.kv_cache_manager), before)
        s.received(f, "q")
        self.assertNotIn("q", f.scheduler.requests)

    def test_default_disabled_plan_has_same_native_query_allocation_and_metadata(self):
        outcomes = []
        for explicit_flag in (False, True):
            f = fixture(enabled=explicit_flag)
            r = s.t.enqueue(f, "q", 1024)
            s.t.known_result(f, r, 2)
            T = f.plan_types
            choose(
                f,
                lambda v, T=T: (
                    T.PlanItem(T.OperationKind.LOAD, "q"),
                    T.PlanItem(T.OperationKind.COMPUTE, "q"),
                ),
            )
            out = f.scheduler.schedule()
            outcomes.append(s.t.signature(f, [out]))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_invalid_operation_path_pair_is_rejected_before_resource_work(self):
        for kind, path in (
            ("compute", "restore"),
            ("load", "recompute"),
            ("lookup", "recompute"),
        ):
            f = fixture()
            matched(f)
            T = f.plan_types
            choose(
                f,
                lambda v, T=T, kind=kind, path=path: (
                    T.PlanItem(
                        T.OperationKind(kind), "q", recovery_path=T.RecoveryPath(path)
                    ),
                ),
            )
            before = s.ops.cache_snapshot(f.scheduler.kv_cache_manager)
            with self.assertRaises(ValueError):
                f.scheduler.schedule()
            self.assertEqual(s.ops.cache_snapshot(f.scheduler.kv_cache_manager), before)


if __name__ == "__main__":
    unittest.main()
