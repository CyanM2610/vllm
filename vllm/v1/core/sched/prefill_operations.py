# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-local observations and decisions; native owners retain every resource."""

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter_ns, thread_time_ns


@dataclass(frozen=True)
class OperationView:
    operation_id: str
    request_id: str
    kind: str
    start_token: int | None
    end_token: int | None
    stage: str
    dependencies: tuple[tuple[str, str], ...] = ()
    blocked_reason: str | None = None
    proposed_tokens: int | None = None
    granted_tokens: int | None = None
    owner: str = "native_runtime"
    native_phase: str | None = None
    source_ownership: bool | None = None
    physical_ready: str = "unknown"
    resources: tuple[tuple[str, int], ...] = ()
    request_status: str | None = None
    input_tokens: int | None = None
    accounted_computed_tokens: int | None = None
    in_flight_tokens: int | None = None


@dataclass(frozen=True)
class OperationDecision:
    # "attempt" requests native checks and acquisition, not a readiness promise.
    action: str = "attempt"
    compute_tokens: int | None = None


@dataclass(frozen=True)
class StepOperationView:
    step_id: int
    remaining_token_budget: int
    remaining_input_budget: int
    free_hbm_blocks: int
    inflight_prefill_reserved_blocks: int
    operations: tuple[OperationView, ...]
    connector_operations: tuple[OperationView, ...] = ()


class NativeOperationDecider:
    """Retain native traversal and quota while allowing operation advancement."""

    def decide(
        self, step: StepOperationView, candidate: OperationView
    ) -> OperationDecision:
        return OperationDecision()


class ActionKind(str, Enum):
    ATTEMPT = "attempt"
    SKIP_CANDIDATE = "skip_candidate"
    END_COMPUTE_BATCH = "end_compute_batch"
    RETAIN_STORE = "retain_store"
    SUBMIT_RETAINED_STORE = "submit_retained_store"
    KEEP_RETAINED_STORE = "keep_retained_store"


@dataclass(frozen=True)
class OperationAction:
    """A proposal, never a reservation. Zero is deliberately not a quota."""

    kind: ActionKind = ActionKind.ATTEMPT
    compute_tokens: int | None = None
    end_after_commit: bool = False

    def validate(self, candidate: OperationView) -> "OperationAction":
        if not isinstance(self.kind, ActionKind):
            raise TypeError("typed actions require ActionKind")
        if candidate.kind == "store":
            allowed = (
                {ActionKind.ATTEMPT, ActionKind.RETAIN_STORE}
                if candidate.native_phase == "candidate"
                else {ActionKind.SUBMIT_RETAINED_STORE, ActionKind.KEEP_RETAINED_STORE}
            )
        else:
            allowed = {ActionKind.ATTEMPT, ActionKind.SKIP_CANDIDATE}
            if candidate.kind == "compute":
                allowed.add(ActionKind.END_COMPUTE_BATCH)
        if self.kind not in allowed:
            raise ValueError(f"{self.kind.value} is not valid for {candidate.kind}/{candidate.native_phase}")
        if type(self.end_after_commit) is not bool:
            raise TypeError("end_after_commit must be bool")
        if self.compute_tokens is not None and (
            candidate.kind != "compute" or self.kind != ActionKind.ATTEMPT
            or type(self.compute_tokens) is not int
            or not 0 < self.compute_tokens <= candidate.proposed_tokens
        ):
            raise ValueError("quota must be positive and within the native proposal")
        if self.end_after_commit and (
            candidate.kind != "compute" or self.kind != ActionKind.ATTEMPT
        ):
            raise ValueError("end_after_commit only applies to an attempted computation")
        return self


class ResultKind(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    BLOCKED = "blocked"
    RESOURCES_ACQUIRED = "resources_acquired"
    SELECTED = "compute_selected"
    REVOKED = "selection_revoked"
    METADATA = "metadata_formed"
    SUBMITTED = "submission_observed"
    COMPLETION = "completion_received"
    FAILURE = "failure_received"
    ADOPTED = "completion_adopted"
    CANCEL_PENDING = "cancel_or_finish_pending"
    ENDED = "obligation_ended"
    TRACKING_ENDED = "query_tracking_ended"


@dataclass(frozen=True)
class OperationFeedback:
    """Scalar evidence from a native boundary; no device time is inferred."""

    operation_id: str
    attempt_id: str
    commitment_id: str | None
    request_id: str
    kind: str
    step_id: int
    start_token: int | None
    end_token: int | None
    result: ResultKind
    reason: str | None = None
    granted_tokens: int | None = None
    native_phase: str | None = None
    physical_ready: str = "unknown"
    resource_obligation: str = "unknown"
    observed_step_id: int | None = None
    transfer_start_token: int | None = None
    transfer_end_token: int | None = None
    skip_first_n_tokens: int | None = None


class ActionFeedback:
    """Bounded observation/identity adapter, not a resource state machine.

    Active entries identify native obligations until their owner ends them.
    Only recent ended identities are retained for duplicate receipt detection.
    Native operations remain the sole authority for allocation and release.
    Callback exceptions poison further dispatch; they never turn a failed
    resource operation into a successful SchedulerOutput.
    """

    def __init__(self, limit, emit):
        self.limit = limit
        self.policy = None
        self.emit = emit
        self.in_callback = False
        self.failed = False
        self.step_id = -1
        self.sequence = 0
        self.attempts = {}
        self.visited = set()
        self.skipped_requests = set()
        self.active = {}
        self.by_request = {}
        self.recent = OrderedDict()
        self.history = deque(maxlen=limit)
        self.timings = {name: deque(maxlen=128) for name in
                        ("state", "decision", "feedback", "observation")}

    def check_entry(self):
        if self.in_callback:
            raise RuntimeError("operation callbacks cannot reenter native mutations")
        if self.failed:
            raise RuntimeError("operation callback failed; do not resume partial dispatch")

    def begin(self, step_id):
        self.check_entry()
        for evidence in self.attempts.values():
            self.active.pop(evidence.attempt_id, None)
        self.step_id = step_id
        self.attempts.clear()
        self.visited.clear()
        self.skipped_requests.clear()

    def measured(self, name, function, *args):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        try:
            return function(*args)
        finally:
            self.timings[name].append((perf_counter_ns() - wall, thread_time_ns() - cpu))

    def _callback(self, function, *args):
        self.check_entry()
        self.in_callback = True
        try:
            return function(*args)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.in_callback = False

    def decide(self, candidate, snapshot):
        key = (candidate.request_id, candidate.kind, candidate.operation_id)
        if key in self.visited:
            # An extra LOAD visit and native WAITING visit must not attempt the
            # same operation twice. No additional matching is requested here.
            return OperationAction(ActionKind.SKIP_CANDIDATE)
        if self.policy is None:
            kind = (ActionKind.SUBMIT_RETAINED_STORE
                    if candidate.kind == "store" and candidate.native_phase != "candidate"
                    else ActionKind.ATTEMPT)
            return OperationAction(kind)
        view = self.measured("state", snapshot)
        action = self.measured("decision", self._callback, self.policy.decide, view, candidate)
        try:
            if not isinstance(action, OperationAction):
                raise TypeError("action policy must return OperationAction")
            action.validate(candidate)
        except BaseException:
            self.failed = True
            raise
        if action.kind == ActionKind.SKIP_CANDIDATE:
            self.visited.add(key)
            self.skipped_requests.add(candidate.request_id)
        return action

    def attempt(self, request_id, kind, start=None, end=None, *, operation_id=None, renew=False):
        key = (request_id, kind, operation_id or (start, end))
        previous = self.attempts.get(key)
        if previous is None or renew:
            if previous is not None:
                self.active.pop(previous.attempt_id, None)
            self.sequence += 1
            identity = operation_id or f"{request_id}:{kind}:{self.step_id}:{self.sequence}"
            previous = OperationFeedback(
                identity, f"attempt:{self.step_id}:{self.sequence}", None,
                request_id, kind, self.step_id, start, end, ResultKind.NOT_ATTEMPTED,
            )
            self.attempts[key] = previous
        return previous

    def report(self, evidence, result, *, commitment_id=None, effect=None, **fields):
        evidence = replace(evidence, result=result, observed_step_id=self.step_id, **fields)
        if commitment_id is not None:
            evidence = replace(evidence, commitment_id=commitment_id)
        identity = evidence.commitment_id or evidence.attempt_id
        transition = (result, evidence.reason if result == ResultKind.BLOCKED else None,
                      evidence.native_phase)
        if identity in self.recent:
            if transition in self.recent[identity]:
                return None
            raise ValueError("late feedback contradicts an ended native obligation")
        state = self.active.setdefault(identity, [evidence, set()])
        if transition in state[1]:
            return None
        state[0] = evidence
        state[1].add(transition)
        if evidence.commitment_id:
            self.by_request.setdefault((evidence.request_id, evidence.kind), {})[identity] = None
        if effect is not None:
            # Only a runtime-owned accounting adapter, never a policy action.
            effect(evidence)
        self.history.append(evidence)
        self.measured("observation", self.emit, evidence)
        callback = getattr(self.policy, "on_feedback", None)
        if callback is not None:
            self.measured("feedback", self._callback, callback, evidence)
        if result in (ResultKind.ENDED, ResultKind.REVOKED, ResultKind.TRACKING_ENDED):
            self.recent[identity] = state[1]
            self.active.pop(identity, None)
            index = self.by_request.get((evidence.request_id, evidence.kind), {})
            index.pop(identity, None)
            if not index:
                self.by_request.pop((evidence.request_id, evidence.kind), None)
            while len(self.recent) > self.limit:
                self.recent.popitem(last=False)
        return evidence

    def find(self, request_id, kind, *, commitment_id=None):
        if commitment_id is not None:
            state = self.active.get(commitment_id)
            return None if state is None else state[0]
        index = self.by_request.get((request_id, kind))
        return self.active[next(reversed(index))][0] if index else None
