# SPDX-License-Identifier: Apache-2.0
"""Compute selection sharing native matching, admission and runtime preparation.

Unknown keys keep their native traversal slots. Known keys are sorted only
inside the remaining slots. A match observed during traversal participates next
step; optional runtime preparation publishes it before this step's selection.
Neither observation acquires HBM or determines physical readiness.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic_ns, perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.compute_cost import ComputeCostModel
from vllm.v1.core.sched.compute_slo import (
    CompletionPlanner,
    ModelUnknown,
    PlanningRequest,
)
from vllm.v1.core.sched.global_plan import CandidateReader, MatchObservation
from vllm.v1.kv_wait_trace import emit


def clock_identity():
    """Identify the Linux monotonic clock shared by local replay and EngineCore."""
    return (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        + ":"
        + str(Path("/proc/self/ns/time").stat().st_ino)
    )


@dataclass(frozen=True)
class Arrival:
    arrival_time: float
    sequence: int
    eligible_ns: int | None
    deadline_ns: int | None
    clock_valid: bool


@dataclass(frozen=True)
class ComputeKey:
    input_tokens: int
    prefix_tokens: int
    remaining_tokens: int
    work: int
    frozen_ns: int
    step: int


@dataclass(frozen=True)
class ComputeGrant:
    tokens: int
    end_after_commit: bool = False


class NativeComputeSelection:
    """Scalar policy state; native scheduler retains all resource ownership."""

    def __init__(self, config, hf_config, *, local_clock=None):
        self.mode = config["mode"]
        if self.mode not in ("native", "sjf", "slo"):
            raise ValueError("supported native compute modes: native, sjf, slo")
        self.model = ComputeCostModel.from_config(hf_config)
        self.local_clock = clock_identity() if local_clock is None else local_clock
        self.reader = CandidateReader()
        self.arrivals = {}
        self.keys = {}
        self.ordered_ids = ()
        self.snapshot = None
        self.visits = []
        self.selected = {}
        self.first_allocated = set()
        self.witnesses = 0
        self.planner = (
            CompletionPlanner(config["completion_model"])
            if "completion_model" in config
            else None
        )
        if self.mode == "slo" and self.planner is None:
            raise ValueError(
                "SLO mode requires an independently frozen completion model"
            )
        self.pending = {}
        self.current_plan = None
        self.plan_invalid = None
        self.fallback_requested = False
        self.slo_witnesses = 0
        emit(
            "native_compute_enabled",
            mode=self.mode,
            model=self.model.to_dict(),
            clock_id=self.local_clock,
            slo_ns=3_000_000_000,
            unknown_merge=(
                "native slots retained; known slots sorted; "
                "runtime preparation before selection, traversal match next step"
            ),
            completion_model=config.get("completion_model"),
        )

    def register(self, request):
        """Copy arrival metadata once, without changing the Request."""
        rid = request.request_id
        if rid in self.arrivals:
            raise ValueError("compute-only P-only scope excludes reused request IDs")
        if request.max_tokens != 1 or request.resumable:
            raise ValueError("compute-only policy requires non-streaming P-only")
        extra = request.sampling_params.extra_args or {}
        eligible = extra.get("prefill_eligible_ns")
        valid = (
            type(eligible) is int
            and 0 < eligible <= monotonic_ns()
            and extra.get("prefill_slo_ns") == 3_000_000_000
            and extra.get("prefill_clock_id") == self.local_clock
        )
        self.reader.register(rid)
        self.arrivals[rid] = Arrival(
            request.arrival_time,
            self.reader.sequences[rid],
            eligible if valid else None,
            eligible + 3_000_000_000 if valid else None,
            valid,
        )
        emit(
            "native_compute_arrival",
            rid,
            **asdict(self.arrivals[rid]),
            metadata_present=eligible is not None,
        )

    def matched(
        self, request, local, external, matched, step, *, source="native", first=None
    ):
        """Observe the complete native merge; retain no physical block pointer."""
        rid = request.request_id
        now = monotonic_ns()
        self.reader.matches[rid] = MatchObservation(local, external, matched, now, step)
        if rid not in self.keys:
            length = request.num_prompt_tokens if first is None else first.input_tokens
            matched = matched if first is None else first.prefix_tokens
            if not 0 <= matched < length:
                emit(
                    "native_compute_key_unknown",
                    rid,
                    reason="match_not_prefill_boundary",
                    matched=matched,
                    input_tokens=length,
                )
                return
            remaining = length - matched
            self.keys[rid] = ComputeKey(
                length,
                matched,
                remaining,
                self.model.estimate(remaining, matched),
                now if first is None else first.observed_ns,
                step if first is None else first.step_id,
            )
            emit(
                "native_compute_key_frozen",
                rid,
                **asdict(self.keys[rid]),
                a=self.model.a,
                b=self.model.b,
                participation="current_selection"
                if source == "first_prepare"
                else "next_step",
                source=source,
            )
        else:
            emit(
                "native_compute_match_confirmed",
                rid,
                step=step,
                frozen_prefix=self.keys[rid].prefix_tokens,
                current_prefix=matched,
                local=local,
                external=external,
            )

    def begin(self, scheduler, token_budget, input_budget):
        """Read only, then order references indirectly by immutable request IDs."""
        wall, cpu = perf_counter_ns(), thread_time_ns()
        self.snapshot = self.reader.read_snapshot(scheduler, token_budget, input_budget)
        read_wall, read_cpu = perf_counter_ns() - wall, thread_time_ns() - cpu
        wall, cpu = perf_counter_ns(), thread_time_ns()
        native = [r.request_id for r in self.snapshot.requests]
        known = sorted((rid for rid in native if rid in self.keys), key=self.sort_key)
        stream = iter(known)
        ordered = (
            [next(stream) if rid in self.keys else rid for rid in native]
            if self.mode != "native"
            else native
        )
        self.ordered_ids = tuple(ordered)
        participated = getattr(self, "_participated_keys", None)
        if participated is None:
            participated = self._participated_keys = set()
        for rid in known:
            if rid not in participated:
                participated.add(rid)
                emit(
                    "native_compute_key_first_selection",
                    rid,
                    step=scheduler.current_step,
                    frozen_step=self.keys[rid].step,
                    prior_compute=scheduler.requests[rid].num_computed_tokens,
                )
        self.sjf_ids = self.ordered_ids
        self.visits, self.selected = [], {}
        self.scope_exhausted = False
        self.intentional_end_reason = None
        self.current_plan, self.plan_invalid, self.fallback_requested = (
            None,
            None,
            False,
        )
        if self.mode == "slo":
            self.plan_slo()
        self.timings = dict(
            snapshot_wall_ns=read_wall,
            snapshot_thread_cpu_ns=read_cpu,
            plan_wall_ns=perf_counter_ns() - wall,
            plan_thread_cpu_ns=thread_time_ns() - cpu,
        )
        changed = native != list(self.ordered_ids)
        fields = dict(
            step=scheduler.current_step,
            mode=self.mode,
            candidates=len(native),
            known=len(known),
            unknown=len(native) - len(known),
            order_changed=changed,
            **self.timings,
        )
        if changed and self.witnesses < 8:
            self.witnesses += 1
            fields.update(
                native_order=native,
                proposed_order=list(self.ordered_ids),
                candidates_detail=[asdict(r) for r in self.snapshot.requests],
                keys={rid: asdict(self.keys[rid]) for rid in known},
            )
        emit("native_compute_plan", **fields)

    def plan_slo(self):
        """Forecast only scalar states with known native predecessors."""
        started = perf_counter_ns(), thread_time_ns()
        now = monotonic_ns()
        gpu_available = now
        fixed_margin = 0
        pending_outputs = {}
        pending_requests = set()
        failure = None
        for batch in self.pending.values():
            pending_requests.update(batch["tokens"])
            if batch["gpu_end_ns"] is None:
                failure = "unmodeled_submitted_batch"
                continue
            if now > batch["gpu_end_ns"] + batch["margin_ns"] + round(
                (
                    self.planner.control["output_delay_ms"]
                    + self.planner.control["output_margin_ms"]
                )
                * 1e6
            ):
                failure = "submitted_completion_forecast_expired"
            gpu_available = max(gpu_available, batch["gpu_end_ns"])
            fixed_margin += batch["margin_ns"]
            for rid in batch["finished_prefill"]:
                pending_outputs[rid] = batch["gpu_end_ns"] + round(
                    self.planner.control["output_delay_ms"] * 1e6
                )
        modeled, unknown = [], []
        for r in self.snapshot.requests:
            arrival = self.arrivals[r.request_id]
            reason = None
            if r.request_id not in self.keys:
                reason = "unknown_first_match"
            elif not arrival.clock_valid:
                reason = "eligible_clock_unknown"
            elif r.stale_output_tokens:
                reason = "stale_output_dependency"
            elif r.restore_failed:
                reason = "restore_failed"
            elif r.restore_reserved and not r.restore_completion_received:
                reason = "restore_not_received"
            elif r.held_hbm_blocks == 0 or r.status not in (
                "RUNNING",
                "WAITING",
                "WAITING_FOR_REMOTE_KVS",
                "PREEMPTED",
            ):
                reason = "current_inputs_not_owned_or_native_state_unknown"
            elif r.inflight_tokens and r.request_id not in pending_requests:
                reason = "submitted_predecessor_unknown"
            p = min(r.input_tokens, r.accounted_tokens)
            remaining = r.input_tokens - p
            if (
                reason is None
                and remaining == 0
                and r.request_id not in pending_outputs
            ):
                reason = "output_completion_unknown"
            if reason:
                unknown.append(
                    dict(
                        request_id=r.request_id,
                        reason=reason,
                        eligible_ns=arrival.eligible_ns,
                        deadline_ns=arrival.deadline_ns,
                    )
                )
            else:
                modeled.append(
                    PlanningRequest(
                        r.request_id,
                        p,
                        remaining,
                        arrival.eligible_ns,
                        arrival.deadline_ns,
                        self.sort_key(r.request_id),
                        pending_outputs.get(r.request_id),
                    )
                )
        result = (
            dict(
                selected=None,
                candidates=[],
                fallback=failure,
                wall_ns=perf_counter_ns() - started[0],
                thread_cpu_ns=thread_time_ns() - started[1],
            )
            if failure
            else self.planner.plan(
                modeled, now, gpu_available, fixed_margin, started=started
            )
        )
        chosen = result["selected"]
        if chosen is not None:
            self.current_plan = chosen
            self.plan_starts = {r.request_id: r.p for r in modeled}
            first = [rid for rid, _ in chosen["allocations"]]
            modeled_ids = {r.request_id for r in modeled}
            ordered_modeled = iter(
                first + [r for r in self.sjf_ids if r in modeled_ids and r not in first]
            )
            # Unknown and unmodeled slots keep their native/SJF opportunity.
            # If one actually acquires compute, invalidate this projection.
            self.ordered_ids = tuple(
                next(ordered_modeled) if rid in modeled_ids else rid
                for rid in self.sjf_ids
            )
        fields = dict(
            step=self.snapshot.step_id,
            now_ns=now,
            modeled=len(modeled),
            total=len(self.snapshot.requests),
            unknown=unknown,
            fixed_submitted_batches=len(self.pending),
            gpu_available_ns=gpu_available,
            fixed_margin_ns=fixed_margin,
            selected=chosen,
            candidates_evaluated=len(result["candidates"]),
            fallback=result.get("fallback"),
            timeout=result.get("timeout", False),
            no_feasible_candidate=result.get("no_feasible_candidate", False),
            changed_from_sjf=result.get("changed_from_sjf", False),
            wall_ns=result["wall_ns"],
            thread_cpu_ns=result["thread_cpu_ns"],
            scope=(
                "conditional completion prediction; "
                "native admission remains authoritative"
            ),
        )
        if self.slo_witnesses < 8 and chosen and len(modeled) > 1:
            self.slo_witnesses += 1
            fields.update(
                requests=[asdict(r) for r in modeled],
                candidates=result["candidates"],
                sjf_order=self.sjf_ids,
                proposed_order=self.ordered_ids,
            )
        emit("native_compute_slo_plan", **fields)

    def invalidate(self, reason):
        """Keep committed allocations; return remaining references to SJF."""
        if self.current_plan is not None and self.plan_invalid is None:
            self.plan_invalid = reason
            self.fallback_requested = True
            emit(
                "native_compute_slo_invalidated",
                step=self.snapshot.step_id,
                reason=reason,
            )

    def fallback_order(self, remaining_ids):
        remaining = set(remaining_ids)
        self.fallback_requested = False
        return [rid for rid in self.sjf_ids if rid in remaining]

    def sort_key(self, rid):
        arrival = self.arrivals[rid]
        return self.keys[rid].work, arrival.arrival_time, arrival.sequence

    def observe(self, request, **fields):
        """Record real checks; presence in a candidate list is not readiness."""
        self.visits.append(request.request_id)
        if fields.get("reason") in (
            "waiting_allocation_failed",
            "running_allocation_failed",
            "operation_max_num_seqs",
            "stale_output_pending",
        ):
            self.invalidate(fields["reason"])
        emit(
            "native_compute_execution",
            request.request_id,
            step=self.snapshot.step_id,
            **fields,
        )

    def propose(self, request, start, native_tokens, branch):
        """Limit a native proposal; never allocate or refresh a prefix."""
        tokens, end = native_tokens, False
        if self.current_plan is not None and self.plan_invalid is None:
            quotas = dict(self.current_plan["allocations"])
            rid = request.request_id
            if rid not in quotas:
                self.invalidate("unmodeled_or_unselected_compute_proposal")
            elif start != self.plan_starts[rid] or native_tokens < quotas[rid]:
                self.invalidate("native_range_changed")
            else:
                tokens = quotas[rid]
                end = bool(self.current_plan.get("end_compute_reason")) and (
                    rid == self.current_plan["allocations"][-1][0]
                    and all(
                        other == rid or self.selected.get(other, (None, None))[1] == q
                        for other, q in quotas.items()
                    )
                )
        emit(
            "native_compute_proposal",
            request.request_id,
            step=self.snapshot.step_id,
            branch=branch,
            start=start,
            native_tokens=native_tokens,
            granted=tokens,
            end_after_commit=end,
            end_reason=self.current_plan.get("end_compute_reason") if end else None,
            plan_invalid=self.plan_invalid,
        )
        return ComputeGrant(tokens, end)

    def committed(self, request, start, tokens, branch):
        """Remember only successful native allocations."""
        self.selected[request.request_id] = (start, tokens)
        if self.current_plan is not None and self.plan_invalid is None:
            quotas = dict(self.current_plan["allocations"])
            if all(
                self.selected.get(rid, (None, None))[1] == q
                for rid, q in quotas.items()
            ):
                self.scope_exhausted = True
                self.intentional_end_reason = self.current_plan.get(
                    "end_compute_reason"
                )
                emit(
                    "native_compute_slo_scope_completed",
                    step=self.snapshot.step_id,
                    explicit_end_reason=self.intentional_end_reason,
                    selected_tokens=sum(q for _, q in self.selected.values()),
                )
        emit(
            "native_compute_selected",
            request.request_id,
            step=self.snapshot.step_id,
            start=start,
            tokens=tokens,
            branch=branch,
        )

    def revoked(self, rid):
        if rid not in self.selected:
            return
        self.selected.pop(rid, None)
        self.invalidate("same_step_native_rollback")
        emit("native_compute_revoked", rid, step=self.snapshot.step_id)

    def finish(self, output, *, remaining_tokens=None, compute_ended=False):
        """Check the one native output, without rewriting it."""
        if {
            rid: q for rid, (_, q) in self.selected.items()
        } != output.num_scheduled_tokens:
            raise AssertionError(
                "compute selection diverged from native allocation ledger"
            )
        self.reader.worker_initialized.update(
            r.req_id for r in output.scheduled_new_reqs
        )
        for rid, (start, tokens) in self.selected.items():
            if rid not in self.first_allocated:
                self.first_allocated.add(rid)
                emit(
                    "native_compute_first_allocation",
                    rid,
                    step=self.snapshot.step_id,
                    start=start,
                    tokens=tokens,
                    frozen_step=self.keys[rid].step if rid in self.keys else None,
                )
        output._native_compute_step_id = self.snapshot.step_id
        if self.planner is not None and self.selected:
            now = monotonic_ns()
            try:
                previous = [b["gpu_end_ns"] for b in self.pending.values()]
                if any(t is None for t in previous):
                    raise ModelUnknown("submitted_predecessor_unknown")
                end = (
                    max([now] + previous)
                    + round(self.planner.control["launch_gap_ms"] * 1e6)
                    + self.planner.duration(list(self.selected.values()))
                )
            except ModelUnknown:
                end = None
            views = {r.request_id: r for r in self.snapshot.requests}
            self.pending[self.snapshot.step_id] = dict(
                gpu_end_ns=end,
                margin_ns=round(self.planner.control["per_batch_margin_ms"] * 1e6),
                submitted_ns=now,
                tokens=dict(output.num_scheduled_tokens),
                finished_prefill=[
                    rid
                    for rid, (p, q) in self.selected.items()
                    if p + q >= views[rid].input_tokens
                ],
            )
        emit(
            "native_compute_step",
            step=self.snapshot.step_id,
            actual_checks=len(set(self.visits)),
            selected=dict(output.num_scheduled_tokens),
            plan_scope_exhausted=self.scope_exhausted,
            plan_invalid=self.plan_invalid,
            forecast_executed=(
                self.current_plan is not None
                and self.plan_invalid is None
                and dict(self.current_plan["allocations"])
                == output.num_scheduled_tokens
            ),
            explicit_end_reason=self.intentional_end_reason
            if self.plan_invalid is None
            else None,
            traversal_end_reason=(
                "token_budget_exhausted"
                if remaining_tokens == 0
                else self.intentional_end_reason
                if compute_ended
                else "native_traversal_complete"
            ),
            remaining_tokens=remaining_tokens,
            **self.timings,
        )

    def output_received(self, output):
        """Adopt an already returned worker result; never poll external work."""
        step = getattr(output, "_native_compute_step_id", None)
        batch = self.pending.pop(step, None)
        if batch is not None:
            emit(
                "native_compute_submitted_completed",
                step=step,
                observed_ns=monotonic_ns(),
                predicted_gpu_end_ns=batch["gpu_end_ns"],
                scope=(
                    "scheduler observation of worker output, "
                    "not GPU completion timestamp"
                ),
            )
