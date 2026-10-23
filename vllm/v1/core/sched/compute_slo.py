# SPDX-License-Identifier: Apache-2.0
"""Bounded rolling completion forecasts; no runtime or resource operations."""

from dataclasses import dataclass
from time import perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.compute_budget import BatchTimeModel


@dataclass(frozen=True)
class PlanningRequest:
    request_id: str
    p: int
    remaining: int
    eligible_ns: int
    deadline_ns: int
    sort_key: tuple
    pending_completion_ns: int | None = None


@dataclass(frozen=True)
class FirstBatch:
    name: str
    allocations: tuple[tuple[str, int], ...]
    end_compute_reason: str | None = None


class ModelUnknown(ValueError):
    pass


class PlanningTimeout(RuntimeError):
    pass


class CompletionPlanner:
    """Evaluate fixed candidates, using frozen-SJF greedy continuation for all."""

    def __init__(self, config):
        self.config = config
        self.time_model = BatchTimeModel(tau_ms=1, **config["coefficients"])
        self.ranges = config["train_axis_ranges"]
        self.control = config["control_model"]
        self.cpu_limit_ns = config["cpu_limit_ns"]
        self.wall_limit_ns = config["wall_limit_ns"]
        if (
            config["max_candidates"] != 5
            or config["max_requests"] != 32
            or config["max_continuation_batches"] != 128
            or self.cpu_limit_ns <= 0
            or self.wall_limit_ns <= 0
        ):
            raise ValueError("unsupported frozen planner bounds")

    def duration(self, shapes):
        Q = sum(q for p, q in shapes)
        A = sum(q * p + q * (q + 1) // 2 for p, q in shapes)
        N = len(shapes)
        if not N:
            return 0
        if any(
            not self.ranges[k][0] <= v <= self.ranges[k][1]
            for k, v in [("Q", Q), ("A", A), ("N", N)]
        ):
            raise ModelUnknown("batch_shape_outside_calibrated_axes")
        return round(self.time_model.predict(Q, A, N) * 1e6)

    @staticmethod
    def batch(order, remaining, cap):
        result = []
        for rid in order:
            q = min(remaining[rid], cap)
            if q:
                result.append((rid, q))
                cap -= q
            if cap == 0:
                break
        return tuple(result)

    def evaluate(
        self, requests, first, now_ns, gpu_available_ns, fixed_margin_ns, check
    ):
        """Roll every modeled request to completion, including pending outputs."""
        by_id = {r.request_id: r for r in requests}
        remaining = {r.request_id: r.remaining for r in requests}
        positions = {r.request_id: r.p for r in requests}
        continuation = [
            r.request_id for r in sorted(requests, key=lambda r: r.sort_key)
        ]
        completion = {}
        margins = {}
        for r in requests:
            if r.remaining == 0:
                if r.pending_completion_ns is None:
                    raise ModelUnknown("unobserved_output_predecessor")
                completion[r.request_id] = max(now_ns, r.pending_completion_ns)
                margins[r.request_id] = fixed_margin_ns + round(
                    self.control["output_margin_ms"] * 1e6
                )
        gpu_tail = max(now_ns, gpu_available_ns)
        batch = first.allocations
        batch_count = 0
        cumulative_margin = fixed_margin_ns
        while any(remaining.values()):
            check()
            if not batch or batch_count >= self.config["max_continuation_batches"]:
                raise ModelUnknown("continuation_limit_or_no_progress")
            shapes = []
            for rid, q in batch:
                if rid not in by_id or not 0 < q <= remaining[rid]:
                    raise ModelUnknown("invalid_proposed_range")
                shapes.append((positions[rid], q))
            gpu_tail += round(self.control["launch_gap_ms"] * 1e6) + self.duration(
                shapes
            )
            cumulative_margin += round(self.control["per_batch_margin_ms"] * 1e6)
            batch_count += 1
            for rid, q in batch:
                remaining[rid] -= q
                positions[rid] += q
                if remaining[rid] == 0:
                    completion[rid] = gpu_tail + round(
                        self.control["output_delay_ms"] * 1e6
                    )
                    margins[rid] = cumulative_margin + round(
                        self.control["output_margin_ms"] * 1e6
                    )
            batch = self.batch(continuation, remaining, 2048)
        if set(completion) != set(by_id):
            raise ModelUnknown("incomplete_completion_forecast")
        lateness = {
            rid: max(0, completion[rid] + margins[rid] - r.deadline_ns)
            for rid, r in by_id.items()
        }
        return dict(
            name=first.name,
            allocations=first.allocations,
            end_compute_reason=first.end_compute_reason,
            completion_ns=completion,
            margin_ns=margins,
            max_lateness_ns=max(lateness.values(), default=0),
            total_latency_ns=sum(
                completion[rid] - r.eligible_ns for rid, r in by_id.items()
            ),
            predicted_batches=batch_count,
            feasible=not any(lateness.values()),
        )

    def plan(
        self, requests, now_ns, gpu_available_ns, fixed_margin_ns=0, *, started=None
    ):
        wall, cpu = started or (perf_counter_ns(), thread_time_ns())

        def check():
            if (
                thread_time_ns() - cpu > self.cpu_limit_ns
                or perf_counter_ns() - wall > self.wall_limit_ns
            ):
                raise PlanningTimeout("planner_budget")

        result = dict(selected=None, candidates=[], fallback=None)
        try:
            if not requests or len(requests) > self.config["max_requests"]:
                raise ModelUnknown("empty_or_oversized_planning_scope")
            sjf = [r.request_id for r in sorted(requests, key=lambda r: r.sort_key)]
            edf = [
                r.request_id
                for r in sorted(requests, key=lambda r: (r.deadline_ns, r.sort_key))
            ]
            urgent = [edf[0]] + [rid for rid in sjf if rid != edf[0]]
            remaining = {r.request_id: r.remaining for r in requests}
            recipes = [
                ("sjf_greedy", sjf, 2048),
                ("deadline_greedy", edf, 2048),
                ("deadline_1024", edf, 1024),
                ("urgent_greedy", urgent, 2048),
                ("urgent_1024", urgent, 1024),
            ]
            seen = set()
            for name, order, cap in recipes:
                check()
                allocations = self.batch(order, remaining, cap)
                # A deliberate smaller quota is a batch boundary. Exhausting
                # the modeled set alone is not: native candidates may remain.
                batch = FirstBatch(
                    name,
                    allocations,
                    "policy_first_batch_cap"
                    if cap < 2048 and sum(q for _, q in allocations) == cap
                    else None,
                )
                if batch.allocations in seen:
                    continue
                seen.add(batch.allocations)
                try:
                    candidate = self.evaluate(
                        requests,
                        batch,
                        now_ns,
                        gpu_available_ns,
                        fixed_margin_ns,
                        check,
                    )
                except ModelUnknown as error:
                    candidate = dict(name=name, unknown=str(error))
                result["candidates"].append(candidate)
                # If even the baseline cannot be forecast, do not pick a
                # modeled alternative by silently discarding that baseline.
                if name == "sjf_greedy" and "unknown" in candidate:
                    raise ModelUnknown("sjf_continuation_unmodeled")
            valid = [c for c in result["candidates"] if "unknown" not in c]
            if not valid:
                raise ModelUnknown("no_modeled_candidate")
            feasible = [c for c in valid if c["feasible"]]
            result["selected"] = min(
                feasible or valid,
                key=lambda c: (
                    (0 if feasible else c["max_lateness_ns"]),
                    c["total_latency_ns"],
                ),
            )
            result["no_feasible_candidate"] = not feasible
            result["changed_from_sjf"] = (
                result["selected"]["allocations"] != valid[0]["allocations"]
            )
            check()
        except (ModelUnknown, PlanningTimeout) as error:
            result.update(
                selected=None,
                fallback=str(error),
                timeout=isinstance(error, PlanningTimeout),
            )
        result.update(
            wall_ns=perf_counter_ns() - wall, thread_cpu_ns=thread_time_ns() - cpu
        )
        return result
