# SPDX-License-Identifier: Apache-2.0
"""Estimated remaining-work ordering through actual native allocation paths."""

import unittest
from dataclasses import asdict
from unittest.mock import patch

import test_native_compute_selection as n

W = []


class RemainingPreparationBoundaries(n.FirstMatchPreparationTests):
    """Run the accepted preparation/lifecycle suite under remaining-work order."""

    def setUp(self):
        original = n.fixture

        def fixture(mode=None, *args, **kwargs):
            return original("srpt" if mode == "sjf" else mode, *args, **kwargs)

        replacement = patch.object(n, "fixture", fixture)
        replacement.start()
        self.addCleanup(replacement.stop)

    test_sequence_limit = (
        n.NativeComputeTests.test_sequence_limit_still_allows_running_progress
    )
    test_cancel_load = (
        n.NativeComputeTests.test_cancel_reserved_load_retains_pages_until_received
    )
    test_reader_pure = n.NativeComputeTests.test_reader_is_pure


class RemainingComputeTests(unittest.TestCase):
    def tearDown(self):
        n.s.close_fixtures()

    def test_nearly_finished_running_precedes_larger_waiting_with_actual_quotas(self):
        results = {}
        chains = {}
        for mode in ("native", "sjf", "srpt"):
            f = n.fixture(mode, prepare=True)
            n.s.ops.prepare_full_output_fixture(f, runtime=n.RUNTIME)
            long = n.s.t.enqueue(f, "near", 6708)
            long.sampling_params.num_logprobs = None
            n.s.t.known_result(f, long, 0)
            worker = n.s.ops.worker_fixture(runtime=n.RUNTIME, decoupled_store=True)
            cls = worker.active.no_forward.__globals__["ModelRunnerOutput"]

            def consume_step(f=f, cls=cls):
                out = f.scheduler.schedule()
                f.scheduler.update_from_output(
                    out,
                    cls(
                        req_ids=["near"],
                        req_id_to_index={"near": 0},
                        sampled_token_ids=[[]],
                    ),
                )

            consume_step()
            n.s.t.enqueue(f, "cold", 2048, 1)
            consume_step()  # New LOOKUP; the existing request still gets 2048.
            n.s.ops.ack(f.rpc, "cold")
            consume_step()  # Nonblocking STATUS, existing compute continues.
            n.s.ops.status(f.rpc, 0, "cold")
            self.assertEqual(long.num_in_flight_tokens, 0)
            self.assertEqual(long.num_computed_tokens, 6144)
            frozen = f.selection.keys["near"]
            before = list(f.scheduler.running)
            out = f.scheduler.schedule()
            results[mode] = out.num_scheduled_tokens
            self.assertEqual(sum(out.num_scheduled_tokens.values()), 2048)
            self.assertEqual(f.scheduler.running[: len(before)], before)
            self.assertEqual(f.selection.keys["near"], frozen)
            self.assertFalse(f.scheduler._decoupled_control)
            self.assertEqual(sum(c[0] == "LOOKUP" for c in f.rpc.calls), 2)
            chains[mode] = dict(
                frozen={k: asdict(v) for k, v in f.selection.keys.items()},
                remaining={k: asdict(v) for k, v in f.selection.remaining.items()},
                order=f.selection.ordered_ids,
                tokens=out.num_scheduled_tokens,
                states=[asdict(r) for r in f.selection.snapshot.requests],
            )
        self.assertEqual(results["sjf"], {"cold": 2048})
        self.assertEqual(results["srpt"], {"near": 564, "cold": 1484})
        self.assertEqual(results["native"], results["srpt"])
        W.append(
            dict(
                kind="remaining_tail_cross_queue_actual", results=results, chains=chains
            )
        )

    def test_inflight_submissions_are_counted_once_and_no_extra_output(self):
        f = n.fixture("srpt", prepare=True)
        req = n.s.t.enqueue(f, "r", 2560)
        n.s.t.known_result(f, req, 0)
        first = f.scheduler.schedule()
        self.assertEqual(first.num_scheduled_tokens, {"r": 2048})
        self.assertEqual(req.num_in_flight_tokens, 2048)
        second = f.scheduler.schedule()
        self.assertEqual(second.num_scheduled_tokens, {"r": 512})
        estimate = f.selection.remaining["r"]
        self.assertEqual(
            (estimate.prefix_tokens, estimate.remaining_tokens), (2048, 512)
        )
        self.assertEqual(estimate.inflight_tokens, 2048)
        for _ in range(2):
            self.assertEqual(f.scheduler.schedule().num_scheduled_tokens, {})
        self.assertEqual(f.selection.remaining["r"].remaining_tokens, 0)
        self.assertEqual(req.num_output_tokens, 0)

    def test_pending_restore_cost_does_not_grant_compute(self):
        f = n.fixture("srpt", prepare=True)
        restore = n.s.t.enqueue(f, "restore", 768)
        n.s.t.known_result(f, restore, 2)
        cold = n.s.t.enqueue(f, "cold", 8192, 1)
        n.s.t.known_result(f, cold, 0)
        first = f.scheduler.schedule()
        self.assertEqual(first.num_scheduled_tokens, {"cold": 2048})
        self.assertEqual(
            [
                m.request_id
                for m in first.kv_connector_metadata.requests
                if m.direction == "RETRIEVE"
            ],
            ["restore"],
        )
        second = f.scheduler.schedule()
        self.assertEqual(second.num_scheduled_tokens, {"cold": 2048})
        self.assertTrue(f.selection.remaining["restore"].restore_reserved)
        self.assertEqual(f.selection.remaining["restore"].remaining_tokens, 256)
        n.s.received(f, "restore")
        self.assertEqual(
            f.scheduler.schedule().num_scheduled_tokens, {"restore": 256, "cold": 1792}
        )

    def test_stale_match_is_only_an_estimate_and_takeover_can_compute_more(self):
        f = n.fixture("srpt", prepare=True)
        req = n.s.t.enqueue(f, "cached", 1024)
        n.s.ops.seed_prefix(f, req, 512)
        n.s.t.known_result(f, req, 0)
        prep = f.scheduler._first_match_preparation
        prep.begin_step(f.scheduler)
        prep.prepare(f.scheduler)
        first = f.selection.keys["cached"]
        blocks, _, _ = f.scheduler.kv_cache_manager.get_computed_blocks(req)
        f.scheduler.kv_cache_manager.evict_blocks(
            {b.block_id for group in blocks.blocks for b in group}
        )
        out = f.scheduler.schedule()
        self.assertEqual(f.selection.remaining["cached"].remaining_tokens, 512)
        self.assertEqual(out.num_scheduled_tokens, {"cached": 1024})
        self.assertEqual(f.selection.keys["cached"], first)
        self.assertEqual(f.selection.reader.matches["cached"].matched_tokens, 0)

    def test_equal_remaining_tokens_still_include_context_cost(self):
        f = n.fixture("srpt", prepare=True)
        long = n.s.t.enqueue(f, "long", 9216)
        n.s.t.known_result(f, long, 0)
        f.scheduler.schedule()
        n.s.t.enqueue(f, "cold", 1024, 1)
        f.scheduler.schedule()
        n.s.ops.ack(f.rpc, "cold")
        f.scheduler.schedule()
        f.scheduler.schedule()
        n.s.ops.status(f.rpc, 0, "cold")
        out = f.scheduler.schedule()
        self.assertEqual(list(out.num_scheduled_tokens), ["cold", "long"])
        self.assertGreater(
            f.selection.remaining["long"].work, f.selection.remaining["cold"].work
        )
