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
        self.cap_waits = {}

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
        self._end_cap_wait(rid, step, monotonic_ns(), "matched")
        return self.first[rid]

    def begin_step(self, scheduler):
        """Set the shared observation epoch before any bounded control polling."""
        scheduler.connector.begin_first_match_step(scheduler.current_step)

    def prepare(self, scheduler):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        now = monotonic_ns()
        counters = scheduler.connector.lookup_observation_counters()
        # Newly arrived requests join in arrival order. Unfinished requests
        # rotate, so later arrivals cannot starve earlier pending queries.
        for rid in sorted(self.new, key=self.new.get):
            self.pending[rid] = None
        self.new.clear()
        for rid in self.first.keys() - scheduler.requests.keys():
            self.first.pop(rid)
        for rid in self.pending.keys() - scheduler.requests.keys():
            self.pending.pop(rid)
            self._end_cap_wait(rid, scheduler.current_step, now, "retired")
        total = len(self.pending)
        attempted, frozen, submitted, observed = 0, 0, 0, 0
        pending_queries, state_skipped, unauthorized = 0, 0, 0
        initial_ids = tuple(self.pending)
        visited = []
        paused = scheduler._pause_state.name != "UNPAUSED"
        for _ in range(0 if paused else min(total, self.limit)):
            rid = next(iter(self.pending))
            self.pending.move_to_end(rid)
            request = scheduler.requests[rid]
            visited.append(rid)
            self._end_cap_wait(rid, scheduler.current_step, now, "visited")
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
                state_skipped += 1
                continue
            before = scheduler.connector.lookup_operation_state(rid)
            if self.planned and not (
                before["result_known"] or before.get("native_fallback_reason")
            ):
                unauthorized += 1
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
            pending_queries += int(match is None and after["pending"])
            if match is not None:
                frozen += 1  # _match_waiting_prefix publishes scalars only.
        omitted = set(initial_ids) - set(visited) if not paused else set()
        for rid in omitted:
            if rid not in self.cap_waits:
                self.cap_waits[rid] = [now, 0]
                self.emit(
                    "prefill_first_match_cap_wait",
                    rid,
                    step=scheduler.current_step,
                    phase="begin",
                    observed_ns=now,
                    limit=self.limit,
                )
            self.cap_waits[rid][1] += 1
        after_counters = scheduler.connector.lookup_observation_counters()
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
            cap_not_visited=len(omitted),
            max_consecutive_cap_steps=max(
                (v[1] for v in self.cap_waits.values()), default=0
            ),
            query_pending_visited=pending_queries,
            native_state_skipped=state_skipped,
            plan_result_unavailable=unauthorized,
            **{
                "lookup_" + key: after_counters[key] - value
                for key, value in counters.items()
            },
            remaining=len(self.pending),
            paused=paused,
            request_ids=visited,
            wall_ns=perf_counter_ns() - wall,
            thread_cpu_ns=thread_time_ns() - cpu,
        )

    def _end_cap_wait(self, rid, step, now, reason):
        interval = self.cap_waits.pop(rid, None)
        if interval is not None:
            self.emit(
                "prefill_first_match_cap_wait",
                rid,
                step=step,
                phase="end",
                reason=reason,
                observed_ns=now,
                consecutive_steps=interval[1],
                duration_ns=now - interval[0],
            )
