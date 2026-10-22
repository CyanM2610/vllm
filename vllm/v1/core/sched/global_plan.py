# SPDX-License-Identifier: Apache-2.0
"""Read-only pre-selection views and step-local plans for frozen P-only.

No allocator, query checker or worker method lives here. Request references are
used only by the internal traversal adapter; policies receive frozen scalars.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
from time import monotonic_ns, perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.prefill_operations import ActionKind, OperationAction


@dataclass(frozen=True)
class MatchObservation:
    local_tokens: int
    external_tokens: int
    matched_tokens: int
    observed_ns: int
    step_id: int


@dataclass(frozen=True)
class RequestCandidate:
    request_id: str
    enqueue_sequence: int
    queue: str
    queue_position: int
    arrival_time: float
    status: str
    input_tokens: int
    accounted_tokens: int
    inflight_tokens: int
    output_placeholders: int
    stale_output_tokens: int
    next_eligible_step: int
    lookup_submitted: bool
    lookup_result_known: bool
    lookup_result_tokens: int | None
    lookup_observed_ns: int | None
    last_match: MatchObservation | None
    restore_reserved: bool
    restore_completion_received: bool
    restore_completion_adopted: bool
    held_hbm_blocks: int
    dependencies: tuple[tuple[str, str], ...]
    native_lookup_fallback: str | None = None
    restore_failed: bool = False


@dataclass(frozen=True)
class ExistingOperation:
    operation_id: str
    request_id: str
    kind: str
    stage: str
    start_token: int | None
    end_token: int | None
    native_phase: str | None
    source_ownership: bool | None
    physical_ready: str
    dependencies: tuple[tuple[str, str], ...]
    resources: tuple[tuple[str, int], ...]
    retained_store: bool = False
    enqueue_sequence: int | None = None
    queue: str = "native_owner"
    queue_position: int | None = None
    observed_ns: int | None = None


@dataclass(frozen=True)
class CandidateSnapshot:
    snapshot_id: str
    step_id: int
    observed_ns: int
    requests: tuple[RequestCandidate, ...]
    operations: tuple[ExistingOperation, ...]
    token_budget: int
    input_budget: int
    max_num_seqs: int
    running_slots: int
    free_hbm_blocks: int
    inflight_prefill_reserved_blocks: int
    control_attempt_limit: int
    restore_preparation_enabled: bool
    store_control_enabled: bool
    new_lookup_enabled: bool
    lookup_submission_limit: int
    load_attempt_limit: int
    complete_request_view: bool = True
    omitted_requests: int = 0
    physical_completion_scope: str = "only native received/adopted evidence; no device polling"


class OperationKind(str, Enum):
    LOOKUP = "lookup"
    LOAD = "load"
    COMPUTE = "compute"


@dataclass(frozen=True)
class PlanItem:
    kind: OperationKind
    request_id: str
    max_tokens: int | None = None
    end_after_commit: bool = False


@dataclass(frozen=True)
class StoreChoice:
    operation_id: str
    submit: bool = True


@dataclass(frozen=True)
class NewStoreIntent:
    request_id: str
    retain: bool = False


@dataclass(frozen=True)
class StepPlan:
    snapshot_id: str
    # Cross-type ordered attempts; duplicate (kind, request) keeps the first.
    attempts: tuple[PlanItem, ...] = ()
    retained_stores: tuple[StoreChoice, ...] = ()
    new_stores: tuple[NewStoreIntent, ...] = ()
    end_compute: bool = False


class NativePlan:
    """Native request order and greedy quota, with explicit operation choices.

    New LOOKUP is independently opt-in. Unlike disabled compatibility mode,
    this plan does not implicitly chain a COMPUTE attempt into LOOKUP/LOAD.
    Newly submitted results are intentionally left to a later step.
    """
    def plan(self, snapshot):
        items = []
        for r in snapshot.requests:
            if (r.queue == "running" or r.accounted_tokens > 0
                    or r.status == "PREEMPTED" or r.native_lookup_fallback):
                items.append(PlanItem(OperationKind.COMPUTE, r.request_id))
            elif r.lookup_result_known:
                if r.lookup_result_tokens > 0 and snapshot.restore_preparation_enabled:
                    items.append(PlanItem(OperationKind.LOAD, r.request_id))
                items.append(PlanItem(OperationKind.COMPUTE, r.request_id))
            elif not r.lookup_submitted and snapshot.new_lookup_enabled:
                items.append(PlanItem(OperationKind.LOOKUP, r.request_id))
        return StepPlan(snapshot.snapshot_id, tuple(items))


def _operation(view, retained=False, sequence=None):
    dependencies = view.get("dependencies", ())
    if isinstance(dependencies, dict):
        dependencies = dependencies.items()
    return ExistingOperation(
        view["operation_id"], view["request_id"], view["kind"], view["stage"],
        view.get("start_token"), view.get("end_token"), view.get("native_phase"),
        view.get("source_ownership"), view.get("source_readable", "unknown"),
        tuple((str(k), {"pending": "blocked", "unknown": "unchecked",
                       "not_recorded": "blocked"}.get(str(v), str(v))) for k, v in dependencies),
        tuple((str(k), int(v)) for k, v in view.get("resources", ())), retained,
        sequence, view.get("queue", "native_owner"), view.get("queue_position"),
        view.get("observed_ns"),
    )


class CandidateReader:
    """Pure scalar snapshot shared by operation plans and compute-only policies."""
    def __init__(self):
        self.sequences = {}
        self.sequence = 0
        self.matches = {}
        self.worker_initialized = set()
        self.new_lookup = False
        self.lookup_limit = self.load_limit = 0

    def register(self, request_id):
        self.sequence += 1
        self.sequences[request_id] = self.sequence

    def read_snapshot(self, scheduler, token_budget, input_budget):
        """Pure read: no result query(), match, allocation, pin or RR rotation."""
        connector = scheduler.connector
        manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        requests = []
        seen = set()
        for origin in ("running", "skipped_waiting", "waiting"):
            for position, request in enumerate(getattr(scheduler, origin)):
                rid = request.request_id
                if rid in seen:
                    continue
                seen.add(rid)
                query = connector.lookup_operation_state(rid)
                load = scheduler._action_loads.get(rid)
                received = rid in scheduler.finished_recving_kv_req_ids
                adopted = load is not None and load.result.value == "completion_adopted"
                pending_restore = request.status.name == "WAITING_FOR_REMOTE_KVS"
                blockers = (
                    ("lookup_result", "confirmed" if query["result_known"] else
                     "blocked" if query["pending"] else "unchecked"),
                    ("restore_completion", "confirmed" if received or adopted else
                     "blocked" if pending_restore else "unchecked"),
                    ("output_inflight", "blocked" if request.num_output_placeholders or
                     request.num_stale_output_tokens else "unchecked"),
                    ("current_hbm_match", "unchecked"),
                    ("capacity_and_sequence_admission", "unchecked"),
                    ("physical_inputs_ready", "unchecked"),
                )
                requests.append(RequestCandidate(
                    rid, self.sequences[rid], origin, position, float(request.arrival_time),
                    request.status.name, request.num_prompt_tokens, request.num_computed_tokens,
                    request.num_in_flight_tokens, request.num_output_placeholders,
                    request.num_stale_output_tokens, request.next_decode_eligible_step,
                    query["submitted"], query["result_known"], query["result_tokens"],
                    query.get("observed_ns"), self.matches.get(rid), pending_restore,
                    received, adopted, len(manager.req_to_blocks.get(rid, ())), blockers,
                    query.get("native_fallback_reason"),
                    rid in scheduler.failed_recving_kv_req_ids,
                ))
        operations = [_operation(v, bool(v.get("retained_store")), self.sequences.get(v["request_id"]))
                      for v in connector.get_global_operation_snapshot()]
        # The action records contain only scalar evidence. Include LOAD and
        # unfinished lifetimes even after their request left the queues.
        for evidence, _ in scheduler._action_feedback.active.values():
            if evidence.kind not in ("load", "lifetime") or evidence.commitment_id is None:
                continue
            operations.append(ExistingOperation(
                evidence.commitment_id, evidence.request_id, evidence.kind,
                evidence.result.value, evidence.start_token, evidence.end_token,
                evidence.native_phase, None, evidence.physical_ready, (), (), False,
                self.sequences.get(evidence.request_id), "native_action_obligation"))
        return CandidateSnapshot(
            f"prefill-step:{scheduler.current_step}", scheduler.current_step, monotonic_ns(),
            tuple(requests), tuple(operations), token_budget, input_budget,
            scheduler.max_num_running_reqs,
            len(scheduler.running) + scheduler.num_waiting_for_streaming_input,
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            scheduler._inflight_prefill_reserved_blocks(), scheduler._operation_limit,
            bool(scheduler._decoupled_control and connector.prepare_load_operations),
            bool(connector.decoupled_store_enabled),
            self.new_lookup, self.lookup_limit, self.load_limit,
        )


class PlanRuntime(CandidateReader):
    """Plan validation/accounting only. Native owners perform all side effects."""
    def __init__(self, *, new_lookup=False, lookup_limit=32, load_limit=32):
        self.policy = NativePlan()
        self.sequences = {}
        self.sequence = 0
        self.matches = {}
        self.worker_initialized = set()
        self.snapshot = None
        self.current = None
        self.attempts = ()
        self.active_item = None
        self.new_lookup = new_lookup
        self.lookup_limit, self.load_limit = lookup_limit, load_limit
        self.request_views = {}
        self.stores = {}
        self.new_stores = {}
        self.checked = set()
        self.check_count = 0
        self.check_cpu_ns = 0
        self.check_wall_ns = 0
        self.timings = deque(maxlen=128)

    def begin(self, scheduler, token_budget, input_budget):
        # Retirement is outside the pure reader and does not hold Request refs.
        wall, cpu = perf_counter_ns(), thread_time_ns()
        snapshot = self.read_snapshot(scheduler, token_budget, input_budget)
        snapshot_time = (perf_counter_ns()-wall, thread_time_ns()-cpu)
        owned = set(scheduler.requests) | {v.request_id for v in snapshot.operations}
        for rid in self.sequences.keys() - owned:
            self.sequences.pop(rid)
            self.matches.pop(rid, None)
            self.worker_initialized.discard(rid)
        wall, cpu = perf_counter_ns(), thread_time_ns()
        plan = scheduler._action_feedback._callback(self.policy.plan, snapshot)
        plan_time = (perf_counter_ns()-wall, thread_time_ns()-cpu)
        wall, cpu = perf_counter_ns(), thread_time_ns()
        try:
            duplicates = self._install(snapshot, plan)
        except BaseException:
            scheduler._action_feedback.failed = True
            raise
        adaptation_time = (perf_counter_ns()-wall, thread_time_ns()-cpu)
        self.snapshot = snapshot
        self.checked = set()
        self.check_count = self.check_cpu_ns = self.check_wall_ns = 0
        self.lookup_attempted = self.lookup_submitted = self.load_attempted = 0
        self.native_wall_ns = self.native_cpu_ns = 0
        self.dispatch_wall_ns = self.dispatch_cpu_ns = 0
        self.timings.append(dict(step=snapshot.step_id, candidates=len(snapshot.requests),
            operations=len(snapshot.operations), snapshot_wall_ns=snapshot_time[0],
            snapshot_thread_cpu_ns=snapshot_time[1], plan_wall_ns=plan_time[0],
            plan_thread_cpu_ns=plan_time[1], adaptation_wall_ns=adaptation_time[0],
            adaptation_thread_cpu_ns=adaptation_time[1], duplicates_removed=duplicates))
        return self.timings[-1]

    def _install(self, snapshot, plan):
        if not isinstance(plan, StepPlan) or plan.snapshot_id != snapshot.snapshot_id:
            raise ValueError("plan must target this step's snapshot")
        if type(plan.end_compute) is not bool:
            raise TypeError("end_compute must be bool")
        request_ids = {r.request_id for r in snapshot.requests}
        retained = {o.operation_id for o in snapshot.operations if o.retained_store}
        duplicates = 0

        def unique(rows, cls, key, allowed):
            nonlocal duplicates
            if type(rows) is not tuple:
                raise TypeError("plan collections must be immutable tuples")
            result = {}
            for row in rows:
                if not isinstance(row, cls) or getattr(row, key) not in allowed:
                    raise ValueError("unknown plan identity or item type")
                identity = getattr(row, key)
                if identity in result:
                    duplicates += 1
                else:
                    result[identity] = row
            return result

        if type(plan.attempts) is not tuple:
            raise TypeError("plan attempts must be an immutable tuple")
        attempts = {}
        for item in plan.attempts:
            if (not isinstance(item, PlanItem) or not isinstance(item.kind, OperationKind)
                    or item.request_id not in request_ids):
                raise ValueError("unknown request or operation in plan")
            if item.kind == OperationKind.LOOKUP and not snapshot.new_lookup_enabled:
                raise ValueError("new LOOKUP selection requires plan_new_lookup=true")
            if item.kind == OperationKind.LOAD and not snapshot.restore_preparation_enabled:
                raise ValueError("LOAD selection requires the existing extra LOAD capability")
            if type(item.end_after_commit) is not bool:
                raise TypeError("end_after_commit must be bool")
            if item.max_tokens is not None and (item.kind != OperationKind.COMPUTE
                    or type(item.max_tokens) is not int or item.max_tokens <= 0):
                raise ValueError("positive token cap only applies to COMPUTE")
            if item.end_after_commit and item.kind != OperationKind.COMPUTE:
                raise ValueError("end_after_commit only applies to COMPUTE")
            key = (item.kind, item.request_id)
            if key in attempts:
                duplicates += 1
            else:
                attempts[key] = item
        stores = unique(plan.retained_stores, StoreChoice, "operation_id", retained)
        for choice in stores.values():
            if type(choice.submit) is not bool:
                raise TypeError("STORE submit must be bool")
        new_stores = unique(plan.new_stores, NewStoreIntent, "request_id", request_ids)
        for choice in new_stores.values():
            if type(choice.retain) is not bool or (choice.retain and not snapshot.store_control_enabled):
                raise ValueError("retaining new STORE requires the existing STORE capability")
        self.current, self.attempts = plan, tuple(attempts.values())
        self.request_views = {r.request_id: r for r in snapshot.requests}
        self.stores, self.new_stores = stores, new_stores
        return duplicates

    def action(self, candidate):
        started, cpu = perf_counter_ns(), thread_time_ns()
        try:
            if candidate.kind == "store":
                if candidate.native_phase == "candidate":
                    intent = self.new_stores.get(candidate.request_id)
                    return OperationAction(ActionKind.RETAIN_STORE if intent and intent.retain else ActionKind.ATTEMPT)
                intent = self.stores.get(candidate.operation_id)
                return OperationAction(ActionKind.KEEP_RETAINED_STORE if intent and not intent.submit
                                       else ActionKind.SUBMIT_RETAINED_STORE)
            item = self.active_item
            if item is None or item.request_id != candidate.request_id or item.kind.value != candidate.kind:
                raise RuntimeError("native operation lacks the current plan authorization")
            return OperationAction(compute_tokens=min(item.max_tokens, candidate.proposed_tokens)
                                   if item.max_tokens is not None else None,
                                   end_after_commit=item.end_after_commit)
        finally:
            self.check_wall_ns += perf_counter_ns()-started
            self.check_cpu_ns += thread_time_ns()-cpu

    def on_feedback(self, result):
        callback = getattr(self.policy, "on_feedback", None)
        if callback is not None:
            callback(result)

    def call_native(self, function, *args, **kwargs):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self.native_wall_ns += perf_counter_ns()-wall
            self.native_cpu_ns += thread_time_ns()-cpu

    def final_stats(self, output):
        self.worker_initialized.update(r.req_id for r in output.scheduled_new_reqs)
        self.worker_initialized.difference_update(output.finished_req_ids)
        self.timings[-1].update(checked_requests=len(self.checked), actual_checks=self.check_count,
            lookup_submitted=self.lookup_submitted, load_attempted=self.load_attempted,
            lookup_attempted=self.lookup_attempted,
            adaptation_wall_ns=self.timings[-1]["adaptation_wall_ns"]+self.check_wall_ns
                + max(0, self.dispatch_wall_ns-self.native_wall_ns),
            adaptation_thread_cpu_ns=self.timings[-1]["adaptation_thread_cpu_ns"]+self.check_cpu_ns
                + max(0, self.dispatch_cpu_ns-self.native_cpu_ns),
            dispatch_wall_ns=self.dispatch_wall_ns, dispatch_thread_cpu_ns=self.dispatch_cpu_ns,
            native_checks_wall_ns=self.native_wall_ns, native_checks_thread_cpu_ns=self.native_cpu_ns,
            selected_requests=len(output.num_scheduled_tokens),
            selected_tokens=sum(output.num_scheduled_tokens.values()),
            traversal="explicit_operations_native_checks")
        return self.timings[-1]
