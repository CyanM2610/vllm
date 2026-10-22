# SPDX-License-Identifier: Apache-2.0
"""Real scheduler, KVCacheManager and connector integration via the project CPU harness."""

from pathlib import Path
import io
import json
import sys
import unittest
from types import ModuleType

import os

ROOT = Path(os.environ["PREFILL_PROJECT_ROOT"])
AREA = Path(os.environ["PREFILL_CPU_OUTPUT"])
RUNTIME = Path(os.environ["PREFILL_NATIVE_RUNTIME"])
sys.path.insert(0, str(ROOT / "scripts/kv_wait_probe/global_plan"))
import cpu_support as s


def fixture(mode=None, runtime=RUNTIME, completion_model=None, **kwargs):
    f = s.fixture(
        runtime=runtime,
        global_plan=False,
        control=False,
        store=True,
        prepare_load=False,
        **kwargs,
    )
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
    return f


W = []


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
