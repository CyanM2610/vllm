"""Frozen first-match cost policies on the common global request driver.

Preparation obtains no HBM ownership. Only scalar estimates survive it. The
shared WAITING handler matches HBM again immediately before native allocation.
The lower lookup future and reader reservation remain owned by LMCache.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from vllm.v1.core.sched.compute_cost import ComputeCostModel
from vllm.v1.core.sched.global_sjf_input import GlobalSJFInputScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.kv_wait_trace import emit
from vllm.v1.request import RequestStatus


@dataclass(frozen=True)
class FirstMatchKey:
    L: int
    p0: int
    r0: int
    estimated_compute_cost: int
    arrival_time: float
    arrival_sequence: int
    frozen_monotonic_ns: int


class GlobalSJFInputPreparedScheduler(GlobalSJFInputScheduler):
    """Common match preparation, then fixed full input length ordering."""

    _global_sjf_event_prefix = "global_sjf_prepared"
    _global_sjf_match_prepare = True
    _global_sjf_key_kind = "input"

    def __init__(self, *args, **kwargs):
        self._prepared_keys = {}
        self._prepared_attempts = {}
        super().__init__(*args, **kwargs)
        config = self.vllm_config.model_config
        self._compute_cost_model = ComputeCostModel.from_config(
            getattr(config, "hf_text_config", None) or config.hf_config
        )
        emit(
            "global_sjf_prepared_cost_model",
            key_kind=self._global_sjf_key_kind,
            cost_model=self._compute_cost_model.to_dict(),
            semantics="theoretical model work, not a fitted GPU latency predictor",
        )

    def _emit_policy(self, suffix, *args, **fields):
        # The inherited immutable input key still identifies enqueue order.
        # It is NOT a first-match compute estimate or a ready certificate.
        if suffix == "enqueue_key_frozen":
            suffix = "arrival_registered"
        if suffix == "enabled":
            fields["key_definition"] = [
                "original_num_prompt_tokens" if self._global_sjf_key_kind == "input"
                else "theoretical_compute_cost_from_first_complete_native_match",
                "original_arrival_time", "stable_add_request_sequence",
            ]
        emit(
            self._global_sjf_event_prefix + "_" + suffix,
            *args,
            key_kind=self._global_sjf_key_kind,
            **fields,
        )

    def _global_sjf_retire_observer_references(self):
        super()._global_sjf_retire_observer_references()
        for rid in self._prepared_keys.keys() - self._global_sjf_requests.keys():
            self._prepared_keys.pop(rid)
            self._prepared_attempts.pop(rid, None)

    def _prepare_first_matches(self, requests):
        started = time.monotonic_ns()
        pending = sorted(
            (r for r in requests if r.request_id not in self._prepared_keys),
            key=lambda r: (
                self._global_sjf_keys[r.request_id].arrival_time,
                self._global_sjf_keys[r.request_id].arrival_sequence,
            ),
        )
        emit(
            "global_sjf_prepared_match_prepare_start",
            batch_id=self._global_sjf_batch_id(),
            key_kind=self._global_sjf_key_kind,
            unfrozen_count=len(pending),
        )
        if self._pause_state != PauseState.UNPAUSED:
            emit(
                "global_sjf_prepared_match_prepare_end",
                batch_id=self._global_sjf_batch_id(),
                key_kind=self._global_sjf_key_kind,
                unfrozen_count=len(pending), attempted_count=0, frozen_count=0,
                elapsed_ns=time.monotonic_ns() - started,
                skipped_reason="native_admission_paused",
            )
            return
        attempted, frozen = 0, 0
        # This is a finite snapshot, once per unfrozen request per step. No
        # waiting, allocate_slots, queue mutation, block touch or progress edit.
        for request in pending:
            rid = request.request_id
            if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                self._global_sjf_reject(
                    "first_match_missing_before_native_ownership", request,
                    native_status=request.status.name,
                )
            if (
                request.num_computed_tokens != 0
                or request.num_in_flight_tokens != 0
                or request.num_output_placeholders != 0
                or request.num_stale_output_tokens != 0
            ):
                self._global_sjf_observe_result(
                    request, outcome="not_checked", branch="prepare",
                    reason="first_match_state_not_available", readiness="unknown",
                )
                continue
            attempted += 1
            self._prepared_attempts[rid] = self._prepared_attempts.get(rid, 0) + 1
            match = self._match_waiting_prefix(request, phase="prepare")
            if match is None:
                self._global_sjf_observe_result(
                    request, outcome="native_check", branch="prepare",
                    reason="lookup_pending", readiness="dependency_pending",
                )
                continue
            arrival = self._global_sjf_keys[rid]
            L, p0 = request.num_prompt_tokens, match.matched_tokens
            if not 0 <= p0 < L:
                self._global_sjf_reject("invalid_first_match", request, L=L, p0=p0)
            r0 = L - p0
            key = FirstMatchKey(
                L, p0, r0, self._compute_cost_model.estimate(r0, p0),
                arrival.arrival_time, arrival.arrival_sequence, time.monotonic_ns(),
            )
            self._prepared_keys[rid] = key
            frozen += 1
            self._emit_policy(
                "key_frozen", rid,
                **asdict(key), batch_id=self._global_sjf_batch_id(),
                a=self._compute_cost_model.a, b=self._compute_cost_model.b,
                cost_model=self._compute_cost_model.to_dict(),
                local_tokens=match.local_tokens, external_tokens=match.external_tokens,
                async_load=match.load_kv_async,
                preparation_attempts=self._prepared_attempts[rid],
                semantics="first complete native match; immutable estimate, not physical readiness or retained HBM pointers",
            )
            # Do not keep match.blocks or its shared boundary in request state.
        emit(
            "global_sjf_prepared_match_prepare_end",
            batch_id=self._global_sjf_batch_id(),
            key_kind=self._global_sjf_key_kind,
            unfrozen_count=len(pending), attempted_count=attempted,
            frozen_count=frozen, elapsed_ns=time.monotonic_ns() - started,
            elapsed_scope="preparation including nested observation emission",
        )

    def _global_sjf_candidates(self):
        self._global_sjf_retire_observer_references()
        requests, origins, seen = [], {}, set()
        for name in ("running", "skipped_waiting", "waiting"):
            for request in getattr(self, name):
                rid = request.request_id
                if self._global_sjf_requests.get(rid) is not request:
                    self._global_sjf_reject("unregistered_request_identity", request)
                if request.num_prompt_tokens != self._global_sjf_keys[rid].prompt_tokens:
                    self._global_sjf_reject("input_length_changed_after_freeze", request)
                origins.setdefault(rid, []).append(name)
                if id(request) not in seen:
                    seen.add(id(request))
                    requests.append(request)
        self._global_sjf_result_count = 0
        self._prepare_first_matches(requests)
        started = time.monotonic_ns()
        thread_started = time.thread_time_ns()
        known = [r for r in requests if r.request_id in self._prepared_keys]
        sort_started = time.monotonic_ns()
        sort_thread_started = time.thread_time_ns()
        known.sort(key=self._prepared_sort_key)
        sort_elapsed = time.monotonic_ns() - sort_started
        sort_thread_elapsed = time.thread_time_ns() - sort_thread_started
        self._global_sjf_ordered_ids = [r.request_id for r in known]
        rows = []
        for request in known:
            key = self._prepared_keys[request.request_id]
            rows.append(dict(
                request_id=request.request_id, queue_origins=origins[request.request_id],
                status=request.status.name, prompt_tokens=key.L, **asdict(key),
                a=self._compute_cost_model.a, b=self._compute_cost_model.b,
                key_kind=self._global_sjf_key_kind,
                computed_tokens=request.num_computed_tokens,
                in_flight_tokens=request.num_in_flight_tokens,
                output_placeholders=request.num_output_placeholders,
                worker_new_committed=request.request_id in self._global_sjf_worker_initialized,
                readiness="unchecked; native branch remains authoritative",
            ))
        self._emit_policy(
            "candidates", batch_id=self._global_sjf_batch_id(), step=self.current_step,
            native_bucket_view=[r.request_id for r in requests],
            unknown_cost_request_ids=[r.request_id for r in requests if r.request_id not in self._prepared_keys],
            ordered_candidates=self._global_sjf_ordered_ids, candidates=rows,
            elapsed_ns=time.monotonic_ns() - started,
            thread_cpu_ns=time.thread_time_ns() - thread_started,
            sort_elapsed_ns=sort_elapsed,
            sort_thread_cpu_ns=sort_thread_elapsed,
            elapsed_scope="known-candidate sort and event construction; preparation and emission excluded",
        )
        return known

    def _prepared_sort_key(self, request):
        key = self._prepared_keys[request.request_id]
        size = key.L if self._global_sjf_key_kind == "input" else key.estimated_compute_cost
        return size, key.arrival_time, key.arrival_sequence

    def _observe_takeover_match(self, request, match):
        key = self._prepared_keys[request.request_id]
        self._emit_policy(
            "takeover_match", request.request_id,
            batch_id=self._global_sjf_batch_id(), frozen_p0=key.p0,
            frozen_r0=key.r0, current_merged_tokens=match.matched_tokens,
            actual_compute_tokens=request.num_tokens - match.matched_tokens,
            local_tokens=match.local_tokens, external_tokens=match.external_tokens,
            async_load=match.load_kv_async, num_preemptions=request.num_preemptions,
            semantics="fresh pre-allocation match; allocation and dependency checks remain authoritative",
        )


class GlobalSJFComputeScheduler(GlobalSJFInputPreparedScheduler):
    """Identical preparation and execution, fixed theoretical compute-cost key."""

    _global_sjf_key_kind = "compute"
