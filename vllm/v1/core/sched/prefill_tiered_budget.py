# SPDX-License-Identifier: Apache-2.0
"""One absolute search allowance, from validated service bounds and host signals."""

from dataclasses import dataclass
from time import monotonic_ns, perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.joint_prefill_score import ScoreTimeout, ScoreUnknown


class GPUWindow:
    def __init__(self, config, scorer):
        self.config, self.scorer = config, scorer

    def bound(self, snapshot, window, now_ns):
        cfg = self.config
        if not cfg.get("admitted") or not window.queued:
            return None, "uncalibrated_or_empty_queue"
        shapes = {b.step_id: b.shape for b in snapshot.submitted_batches}
        oldest = window.queued[0]
        if oldest.result_ready is True:
            return now_ns, "result_ready"
        shape = shapes.get(oldest.step_id)
        submitted = oldest.host_submitted_ns
        if not shape or submitted is None or submitted > now_ns:
            return None, "shape_or_submission_unknown"
        try:
            duration = self.scorer.gpu(shape, {"gpu_scale": 1.0})
        except ScoreUnknown:
            return None, "shape_outside_model"
        start = submitted
        basis = "validated_host_plus_lower_service"
        if cfg.get("schema") == "A100-shaped-window-r2":
            q = sum(tokens for _, _, tokens in shape)
            group = next(
                (b for b in cfg["bins"] if b["q_min"] <= q <= b["q_max"]), None
            )
            if group is None or not group["covered"]:
                return None, "shape_bin_uncovered"
            duration *= group["lower_scale"]
            forward = oldest.forward_host_start_ns
            if forward is not None and submitted <= forward <= now_ns:
                start = forward
                basis = "host_forward_lower_start_shaped_service"
            else:
                basis = "host_submit_fallback_shaped_service"
        else:
            duration *= cfg["lower_scale"]
        # Device start cannot precede native execute_model submission. A lower
        # service duration therefore gives a conditional lower completion bound.
        return start + int(duration * 1e6), basis


class TieredBudget:
    def __init__(
        self,
        config,
        snapshot,
        reader,
        model,
        *,
        clock=perf_counter_ns,
        cpu=thread_time_ns,
        host_clock=monotonic_ns,
    ):
        self.cfg, self.snapshot, self.reader, self.model = (
            config,
            snapshot,
            reader,
            model,
        )
        self.clock, self.cpu, self.host_clock = clock, cpu, host_clock
        self.started, self.cpu_started = clock(), cpu()
        self.reason = self.discard_reason = None
        self.checks = self.reads = self.shrinks = 0
        self.window = reader()
        self.reads += 1
        now, host_now = clock(), host_clock()
        self.observation_ns = host_now
        bound, self.basis = model.bound(snapshot, self.window, host_now)
        if bound is None:
            available = config["unknown_window_ms"]
            window_deadline = self.started + int(available * 1e6)
        else:
            available = max(0.0, (bound - host_now) / 1e6 - config["submit_reserve_ms"])
            window_deadline = now + int(available * 1e6)
        available = min(available, config["control_limit_ms"], config["wall_limit_ms"])
        if self.ready():
            available = 0.0
            window_deadline = self.started
        self.deadline = min(
            self.started + int(config["wall_limit_ms"] * 1e6),
            self.started + int(config["control_limit_ms"] * 1e6),
            window_deadline,
        )
        self.initial_deadline = self.deadline
        self.cpu_deadline = self.cpu_started + int(config["cpu_limit_ms"] * 1e6)
        self.next_read = now + 100_000
        self.initial_window_ms = max(0.0, (bound - host_now) / 1e6) if bound else None
        self.initial_queue = tuple(self.window.queued)

    def ready(self):
        return (
            self.window.result_ready
            or self.window.control_ready
            or self.window.cancel_ready
        )

    def refresh(self):
        now = self.clock()
        if now < self.next_read:
            return
        self.window = self.reader()
        self.reads += 1
        self.next_read = now + 100_000
        bound, _ = self.model.bound(self.snapshot, self.window, self.host_clock())
        reduced = self.deadline
        if self.ready():
            reduced = min(reduced, now)
            self.reason = (
                "cancel_ready"
                if self.window.cancel_ready
                else ("result_ready" if self.window.result_ready else "control_ready")
            )
        elif bound is not None:
            remaining = (
                bound - self.host_clock() - int(self.cfg["submit_reserve_ms"] * 1e6)
            )
            reduced = min(reduced, now + max(0, remaining))
        self.shrinks += reduced < self.deadline
        self.deadline = reduced

    def check(self, reserve_ns=0):
        self.checks += 1
        self.refresh()
        if self.ready():
            self.reason = (
                "cancel_ready"
                if self.window.cancel_ready
                else ("result_ready" if self.window.result_ready else "control_ready")
            )
            raise ScoreTimeout(self.reason)
        if self.cpu() >= self.cpu_deadline - reserve_ns:
            self.reason = "cpu_limit_or_reserve"
            raise ScoreTimeout(self.reason)
        if self.clock() >= self.deadline - reserve_ns:
            self.reason = "wall_limit_or_reserve"
            raise ScoreTimeout(self.reason)

    def remaining_ns(self):
        return max(0, min(self.deadline - self.clock(), self.cpu_deadline - self.cpu()))

    def slice(self, reserve_ms):
        return TieredSlice(self, int(reserve_ms * 1e6))

    def completed_plan_discard_reason(self):
        self.refresh()
        if self.window.cancel_ready:
            reason = "cancel_ready"
        elif self.cpu() >= self.cpu_deadline:
            reason = "cpu_limit"
        elif self.clock() >= self.initial_deadline:
            reason = "wall_limit"
        else:
            reason = None
        self.discard_reason = reason
        return reason

    def summary(self):
        return dict(
            granted_ms=max(0, self.initial_deadline - self.started) / 1e6,
            observation_ns=self.observation_ns,
            remaining_ms=self.remaining_ns() / 1e6,
            consumed_wall_ns=self.clock() - self.started,
            consumed_cpu_ns=self.cpu() - self.cpu_started,
            reason=self.reason,
            discard_reason=self.discard_reason,
            checks=self.checks,
            reads=self.reads,
            shrinks=self.shrinks,
            basis=self.basis,
            initial_gpu_window_ms=self.initial_window_ms,
            result_ready=self.window.result_ready,
            control_ready=self.window.control_ready,
            queued_executions=[
                dict(
                    step_id=q.step_id,
                    host_submitted_ns=q.host_submitted_ns,
                    forward_host_start_ns=q.forward_host_start_ns,
                    result_ready=q.result_ready,
                    device_progress=q.device_progress,
                )
                for q in self.initial_queue
            ],
        )


@dataclass(frozen=True)
class TieredSlice:
    parent: TieredBudget
    reserve_ns: int

    def check(self):
        self.parent.check(self.reserve_ns)
