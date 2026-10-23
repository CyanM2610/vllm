# SPDX-License-Identifier: Apache-2.0
"""Ephemeral native matches and runtime-owned immutable first-match scalars."""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import monotonic_ns, perf_counter_ns, thread_time_ns
from typing import Any


@dataclass(frozen=True)
class PrefixMatch:
    blocks: Any
    local_tokens: int
    external_tokens: int
    matched_tokens: int
    load_kv_async: bool
    shared_prefix_boundary: int
    connector_queries: int


@dataclass(frozen=True)
class FirstMatch:
    input_tokens: int
    prefix_tokens: int
    remaining_tokens: int
    observed_ns: int
    step_id: int
    source: str


class FirstMatchPreparation:
    """Bounded fair preparation; no HBM ownership or policy cost lives here."""

    def __init__(self, *, planned, limit, emit):
        self.planned, self.limit, self.emit = planned, limit, emit
        self.first = {}
        self.pending = OrderedDict()
        self.new = {}
        self.sequence = 0

    def register(self, request):
        self.sequence += 1
        self.new[request.request_id] = (request.arrival_time, self.sequence)

    def record(self, request, match, step, source):
        rid = request.request_id
        if rid not in self.first:
            length, prefix = request.num_prompt_tokens, match.matched_tokens
            if not 0 <= prefix < length:
                raise ValueError(
                    "first complete match must leave the native last token"
                )
            self.first[rid] = FirstMatch(
                length, prefix, length - prefix, monotonic_ns(), step, source
            )
            self.emit("prefill_first_match", rid, **asdict(self.first[rid]))
        self.pending.pop(rid, None)
        self.new.pop(rid, None)
        return self.first[rid]

    def begin_step(self, scheduler):
        """Set the shared observation epoch before any bounded control polling."""
        scheduler.connector.begin_first_match_step(scheduler.current_step)

    def prepare(self, scheduler):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        # Newly arrived requests join in arrival order. Unfinished requests
        # rotate, so later arrivals cannot starve earlier pending queries.
        for rid in sorted(self.new, key=self.new.get):
            self.pending[rid] = None
        self.new.clear()
        for rid in self.first.keys() - scheduler.requests.keys():
            self.first.pop(rid)
        for rid in self.pending.keys() - scheduler.requests.keys():
            self.pending.pop(rid)
        total = len(self.pending)
        attempted, frozen, submitted, observed = 0, 0, 0, 0
        visited = []
        paused = scheduler._pause_state.name != "UNPAUSED"
        for _ in range(0 if paused else min(total, self.limit)):
            rid = next(iter(self.pending))
            self.pending.move_to_end(rid)
            request = scheduler.requests[rid]
            visited.append(rid)
            if request.is_finished():
                self.pending.pop(rid)
                continue
            if (
                request.status.name not in ("WAITING", "PREEMPTED")
                or request.num_computed_tokens
                or request.num_in_flight_tokens
                or request.num_output_placeholders
                or request.num_stale_output_tokens
            ):
                continue
            before = scheduler.connector.lookup_operation_state(rid)
            if self.planned and not (
                before["result_known"] or before.get("native_fallback_reason")
            ):
                continue
            attempted += 1
            match = scheduler._match_waiting_prefix(
                request,
                phase="first_prepare",
                existing_lookup_only=True,
                allow_lookup_submit=not self.planned,
                allow_lookup_progress=not self.planned,
                commit_match=False,
            )
            after = scheduler.connector.lookup_operation_state(rid)
            submitted += int(not before["submitted"] and after["submitted"])
            observed += int(not before["result_known"] and after["result_known"])
            if match is not None:
                frozen += 1  # _match_waiting_prefix publishes scalars only.
        self.emit(
            "prefill_first_match_prepare_step",
            step=scheduler.current_step,
            mode="plan_cached_only" if self.planned else "common_query_prepare",
            limit=self.limit,
            total_unprepared=total,
            visited=len(visited),
            attempted=attempted,
            frozen=frozen,
            lookup_submitted=submitted,
            lookup_results_observed=observed,
            not_visited=total - len(visited),
            remaining=len(self.pending),
            paused=paused,
            request_ids=visited,
            wall_ns=perf_counter_ns() - wall,
            thread_cpu_ns=thread_time_ns() - cpu,
        )
