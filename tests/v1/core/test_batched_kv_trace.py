# SPDX-License-Identifier: Apache-2.0
"""Actual observer and native block-pool calls, with local CPU-only files."""

import atexit
import importlib.util
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


class BatchedTraceTests(unittest.TestCase):
    def setUp(self):
        self.area = Path(os.environ["PREFILL_CPU_OUTPUT"]) / self.id().split(".")[-1]
        self.area.mkdir(parents=True, exist_ok=True)
        self.stack = []
        self.root = Path(os.environ["PREFILL_NATIVE_RUNTIME"]) / "source/vllm/vllm/v1"
        self.hd = self.load("_batch_test_host", self.root / "host_diagnostic.py")
        with patch.object(
            sys.modules["vllm.v1"], "host_diagnostic", self.hd, create=True
        ):
            self.trace = self.load("_batch_test_trace", self.root / "kv_wait_trace.py")
        self.trace._ROOT = str(self.area)
        self.trace._BATCH_ENABLED = True
        self.path = self.area / f"probe-{os.getpid()}.jsonl"
        if self.path.exists():
            self.path.unlink()

    def load(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        atexit.unregister(getattr(mod, "close", getattr(mod, "dump", None)))
        return mod

    def tearDown(self):
        self.trace.close()

    def rows(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_occurrence_time_order_and_fewer_writes(self):
        times = []
        with self.trace.batch_events():
            for i in range(32):
                self.trace.emit("evict", "r", block=i)
                times.append(self.trace.time.monotonic_ns())
            self.assertEqual(self.trace._WRITES, 0)
        self.assertEqual(self.trace._WRITES, 1)
        rows = self.rows()
        self.assertEqual([r["block"] for r in rows], list(range(32)))
        self.assertTrue(all(r["ts_ns"] <= t for r, t in zip(rows, times)))
        self.trace.close()
        end = self.rows()[-1]
        self.assertEqual(end["events"], 32)
        self.assertEqual(end["buffer_events"], 0)
        self.assertEqual(end["dropped_events"], 0)

    def test_count_and_byte_limits_and_large_event(self):
        with self.trace.batch_events():
            for i in range(300):
                self.trace.emit("evict", block=i)
            self.trace.emit("large", payload="x" * 300000)
            for i in range(5):
                self.trace.emit("medium", payload="x" * 100000)
            self.assertLessEqual(self.trace._BUFFER_BYTES, 262144)
        self.assertEqual(len(self.rows()), 306)
        self.assertLessEqual(self.trace._PEAK_EVENTS, 128)
        self.assertLessEqual(self.trace._PEAK_BYTES, 262144)

    def test_exception_and_nested_calls_flush_in_original_order(self):
        with self.assertRaisesRegex(ValueError, "native failure"):  # noqa: SIM117
            with self.trace.batch_events():
                self.trace.emit("outer")
                with self.trace.batch_events():
                    self.trace.emit("inner")
                self.trace.emit("after")
                raise ValueError("native failure")
        self.assertEqual([r["event"] for r in self.rows()], ["outer", "inner", "after"])
        self.assertEqual(self.trace._DEPTH, 0)

    def test_other_thread_and_unbatched_event_do_not_overtake(self):
        with self.trace.batch_events():
            self.trace.emit("before")
            thread = threading.Thread(target=self.trace.emit, args=("other",))
            thread.start()
            thread.join()
            self.trace.emit("after")
        self.trace.emit("outside")
        self.assertEqual(
            [r["event"] for r in self.rows()], ["before", "other", "after", "outside"]
        )

    def test_disabled_path_writes_immediately(self):
        self.trace._BATCH_ENABLED = False
        with self.trace.batch_events():
            self.trace.emit("first")
            self.trace.emit("second")
            self.assertEqual(self.trace._WRITES, 2)
            self.assertFalse(self.trace._BUFFER)

    def test_partial_os_write_is_completed_without_duplicate_lines(self):
        original = os.write
        with patch.object(  # noqa: SIM117
            self.trace.os, "write", side_effect=lambda fd, b: original(fd, b[:31])
        ):
            with self.trace.batch_events():
                self.trace.emit("one", block=1)
                self.trace.emit("two", block=2)
        self.assertEqual([r["event"] for r in self.rows()], ["one", "two"])
        self.assertGreater(self.trace._WRITES, 1)

    def test_io_error_invalidates_observer(self):
        with patch.object(self.trace.os, "write", side_effect=OSError("disk full")):  # noqa: SIM117
            with self.assertRaises(OSError):
                with self.trace.batch_events():
                    self.trace.emit("critical")
        self.assertEqual(self.trace._WRITE_ERRORS, 1)
        self.trace.close()
        self.assertEqual(self.rows()[-1]["write_errors"], 1)

    def test_real_block_pool_eviction_keeps_native_state_and_ids(self):
        import test_native_compute_selection as n

        results = []
        for enabled in (False, True):
            f = n.fixture("native", prepare=True)
            req = n.s.t.enqueue(f, "seed", 1024)
            n.s.ops.seed_prefix(f, req, 512)
            pool = f.scheduler.kv_cache_manager.block_pool
            # CPU loader executes exact native method bodies; install the
            # production observer decorator and real passive probe as well.
            with patch.dict(sys.modules, {"vllm.v1.kv_wait_trace": self.trace}):
                retirement = self.load(
                    "_batch_test_retirement",
                    self.root / "core/retirement_observation.py",
                )
                probe = self.load(
                    "_batch_test_eviction", self.root / "eviction_probe.py"
                )
                fn = type(pool).get_new_blocks
                wrapped = retirement.retirement_call("pool_allocate")(fn)
                self.trace._BATCH_ENABLED = enabled
                before = self.trace._WRITES
                with patch.object(sys.modules["vllm.v1"], "eviction_probe", probe):  # noqa: SIM117
                    # The loader's module global is the original injected probe.
                    with patch.dict(fn.__globals__, eviction_probe=probe):
                        blocks = wrapped(pool, pool.get_num_free_blocks())
                results.append(
                    (
                        [b.block_id for b in blocks],
                        [b.ref_cnt for b in blocks],
                        self.trace._WRITES - before,
                    )
                )
            n.s.close_fixtures()
        self.assertEqual(results[0][:2], results[1][:2])
        self.assertGreater(results[0][2], results[1][2])
        evictions = [r for r in self.rows() if r["event"] == "hbm_hash_evicted"]
        count = len(evictions) // 2
        self.assertGreater(count, 1)
        self.assertEqual(
            [r["block_id"] for r in evictions[:count]],
            [r["block_id"] for r in evictions[count:]],
        )
