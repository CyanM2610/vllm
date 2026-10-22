# SPDX-License-Identifier: Apache-2.0
"""Completion planning contract; CPU-only model values are deliberately artificial."""

import unittest
from unittest.mock import Mock, patch

from vllm.v1.core.sched.compute_slo import CompletionPlanner, PlanningRequest


def controlled_model():
    return dict(
        coefficients=dict(theta0=0, thetaQ=1, thetaA=0, thetaN=0, floor_ms=0),
        train_axis_ranges=dict(Q=[1, 4096], A=[1, 100_000_000], N=[1, 8]),
        control_model=dict(
            launch_gap_ms=0,
            output_delay_ms=0,
            per_batch_margin_ms=0,
            output_margin_ms=0,
        ),
        cpu_limit_ns=100_000_000,
        wall_limit_ns=500_000_000,
        max_candidates=5,
        max_requests=32,
        max_continuation_batches=128,
    )


class CompletionPlannerTests(unittest.TestCase):
    def requests(self, long_deadline_ms=2300):
        now = 10_000_000_000
        return now, [
            PlanningRequest("short1", 0, 512, now, now + 3_000_000_000, (1,)),
            PlanningRequest("short2", 0, 512, now, now + 3_000_000_000, (2,)),
            PlanningRequest(
                "long",
                0,
                1536,
                now - 700_000_000,
                now + long_deadline_ms * 1_000_000,
                (3,),
            ),
        ]

    def test_deadline_candidate_completes_all_modeled_requests(self):
        now, requests = self.requests()
        result = CompletionPlanner(controlled_model()).plan(requests, now, now)
        chosen = result["selected"]
        self.assertTrue(chosen["feasible"])
        self.assertEqual(chosen["allocations"], (("long", 1536), ("short1", 512)))
        self.assertEqual(set(chosen["completion_ns"]), {r.request_id for r in requests})
        self.assertTrue(result["changed_from_sjf"])

    def test_infeasible_candidates_minimize_max_lateness_then_total(self):
        now, requests = self.requests(100)
        result = CompletionPlanner(controlled_model()).plan(requests, now, now)
        self.assertTrue(result["no_feasible_candidate"])
        valid = [c for c in result["candidates"] if "unknown" not in c]
        self.assertEqual(
            result["selected"],
            min(valid, key=lambda c: (c["max_lateness_ns"], c["total_latency_ns"])),
        )

    def test_expired_deadline_remains_expired(self):
        now, requests = self.requests(-100)
        deadline = requests[-1].deadline_ns
        result = CompletionPlanner(controlled_model()).plan(requests, now, now)
        self.assertGreater(result["selected"]["max_lateness_ns"], 0)
        self.assertEqual(requests[-1].deadline_ns, deadline)

    def test_fixed_submitted_work_shifts_all_candidate_completions(self):
        now, requests = self.requests()
        p = CompletionPlanner(controlled_model())
        normal = p.plan(requests, now, now)
        pending = p.plan(requests, now, now + 500_000_000)
        left = {c["name"]: c for c in normal["candidates"]}
        for candidate in pending["candidates"]:
            for rid, t in candidate["completion_ns"].items():
                self.assertEqual(
                    t - left[candidate["name"]]["completion_ns"][rid], 500_000_000
                )

    def test_unknown_baseline_shape_returns_safe_fallback(self):
        now, requests = self.requests()
        cfg = controlled_model()
        cfg["train_axis_ranges"]["N"] = [1, 1]
        result = CompletionPlanner(cfg).plan(requests, now, now)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["fallback"], "sjf_continuation_unmodeled")

    def test_timeout_does_not_return_partial_search_winner(self):
        now, requests = self.requests()
        cfg = controlled_model()
        cfg["wall_limit_ns"] = 1
        with patch.dict(
            CompletionPlanner.plan.__globals__,
            perf_counter_ns=Mock(side_effect=[0, 2, 3]),
        ):
            result = CompletionPlanner(cfg).plan(requests, now, now)
        self.assertTrue(result["timeout"])
        self.assertIsNone(result["selected"])

    def test_pending_output_is_part_of_deadline_feasibility(self):
        now, requests = self.requests()
        requests.append(
            PlanningRequest(
                "output", 128, 0, now - 3_000_000_000, now, (0,), now + 100_000_000
            )
        )
        result = CompletionPlanner(controlled_model()).plan(requests, now, now)
        self.assertTrue(result["no_feasible_candidate"])
        self.assertIn("output", result["selected"]["completion_ns"])

    def test_forecast_continues_beyond_first_chunk(self):
        now = 10_000_000_000
        r = PlanningRequest("long", 0, 5000, now, now + 6_000_000_000, (1,))
        result = CompletionPlanner(controlled_model()).plan([r], now, now)
        self.assertEqual(
            result["selected"]["completion_ns"]["long"], now + 5_000_000_000
        )
        self.assertEqual(result["selected"]["predicted_batches"], 3)
