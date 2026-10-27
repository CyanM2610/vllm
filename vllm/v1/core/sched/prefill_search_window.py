# SPDX-License-Identifier: Apache-2.0
"""Read-only host progress, explicitly distinct from device progress."""

from dataclasses import asdict, dataclass
from time import perf_counter_ns, thread_time_ns
from weakref import ref

from vllm.v1.core.sched.joint_prefill_score import ScoreTimeout


@dataclass(frozen=True)
class QueuedExecution:
    step_id: int | None
    host_submitted_ns: int | None
    result_ready: bool | None
    execution_reply_ready: bool
    device_progress: str = "unknown"
    # Host entry to the actual eager model call; a lower bound on device start.
    # This is not a CUDA timestamp and carries no device progress claim.
    forward_host_start_ns: int | None = None


@dataclass(frozen=True)
class ExecutionWindow:
    queued: tuple[QueuedExecution, ...] = ()  # oldest first
    result_ready: bool = False
    control_ready: bool = False
    cancel_ready: bool = False
    control_checked: int = 0
    control_omitted: int = 0
    # Current EngineCore never populates a device-derived window. A future
    # calibrated producer must explicitly provide a conservative bound.
    device_remaining_lower_ms: float | None = None
    device_basis: str = "unknown"


def read_engine_window(batch_queue, cancel_ready=False):
    """Non-consuming host/producer-event readiness; never result() or wait()."""
    queued = tuple(
        QueuedExecution(
            getattr(output, "_action_step_id", None),
            getattr(output, "_coord_host_submit_ns", None),
            future.peek_ready() if hasattr(future, "peek_ready") else future.done(),
            execution.done(),
            forward_host_start_ns=getattr(output, "_coord_forward_host_start_ns", None),
        )
        for future, output, execution in reversed(batch_queue or ())
    )
    return ExecutionWindow(
        queued, any(q.result_ready is True for q in queued), cancel_ready=cancel_ready
    )


def engine_window_reader(engine):
    """The policy reader must not prolong EngineCore/executor/IPC ownership."""
    owner = ref(engine)

    def read():
        core = owner()
        if core is None:
            return ExecutionWindow()
        return read_engine_window(core.batch_queue, not core.aborts_queue.empty())

    return read


class SearchBudget:
    """One non-renewable, cooperatively checked wall/CPU allowance per step."""

    def __init__(self, config, reader, clock=perf_counter_ns, cpu=thread_time_ns):
        self.config, self.reader, self.clock, self.cpu = config, reader, clock, cpu
        self.started, self.cpu_started = clock(), cpu()
        self.reason = None
        self.checks = self.reads = self.shrinks = 0
        self.window = reader()
        self.reads += 1
        self.next_read = self.started + 100_000
        self.granted_ms = self.allowance(self.window)
        self.deadline = self.started + int(self.granted_ms * 1e6)
        self.initial_deadline = self.deadline
        self.cpu_deadline = self.cpu_started + int(config["absolute_cpu_ms"] * 1e6)
        self.discard_reason = None

    def allowance(self, window):
        cfg = self.config
        if cfg["budget_mode"] == "fixed":
            return min(8.0, cfg["absolute_wall_ms"])
        if window.result_ready or window.control_ready or window.cancel_ready:
            return 0.0
        remaining = window.device_remaining_lower_ms
        if remaining is None or window.device_basis != "calibrated_lower_bound":
            gpu_limit = cfg["conservative_ms"]
        else:
            gpu_limit = max(0.0, remaining - cfg["submit_reserve_ms"])
        return min(cfg["absolute_wall_ms"], cfg["control_delay_ms"], gpu_limit)

    def check(self, reserve_ns=0):
        self.checks += 1
        now = self.clock()
        if self.config["budget_mode"] == "dynamic" and now >= self.next_read:
            self.window = self.reader()
            self.reads += 1
            self.next_read = now + 100_000
            remaining = self.allowance(self.window)
            reduced = min(self.deadline, self.started + int(remaining * 1e6))
            self.shrinks += reduced < self.deadline
            self.deadline = reduced
        violation = None
        if (
            self.window.result_ready
            or self.window.control_ready
            or self.window.cancel_ready
        ) and self.config["budget_mode"] == "dynamic":
            violation = (
                "result_ready"
                if self.window.result_ready
                else "control_or_cancel_ready"
            )
        if self.cpu() >= self.cpu_deadline:
            violation = violation or "cpu_limit"
        if now >= self.deadline - reserve_ns:
            violation = violation or "wall_or_finalization_reserve"
        if violation:
            self.reason = violation
            raise ScoreTimeout(violation)

    def with_reserve(self, reserve_ns):
        return BudgetSlice(self, reserve_ns)

    def completed_plan_discard_reason(self):
        """Readiness stops search, not a complete plan within its original limit."""
        try:
            self.check()
        except ScoreTimeout:
            pass
        if self.config["budget_mode"] == "dynamic" and self.window.cancel_ready:
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
            mode=self.config["budget_mode"],
            granted_ms=self.granted_ms,
            allowed_ns=max(0, self.deadline - self.started),
            consumed_wall_ns=self.clock() - self.started,
            consumed_cpu_ns=self.cpu() - self.cpu_started,
            reason=self.reason,
            completed_plan_discard_reason=self.discard_reason,
            initial_allowed_ns=max(0, self.initial_deadline - self.started),
            checks=self.checks,
            readiness_reads=self.reads,
            shrinks=self.shrinks,
            device_basis=self.window.device_basis,
            queued=len(self.window.queued),
            queued_executions=[asdict(q) for q in self.window.queued],
            control_checked=self.window.control_checked,
            control_omitted=self.window.control_omitted,
        )


@dataclass(frozen=True)
class BudgetSlice:
    parent: SearchBudget
    reserve_ns: int

    def check(self):
        self.parent.check(self.reserve_ns)
