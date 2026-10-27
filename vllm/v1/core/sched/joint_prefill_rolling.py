# SPDX-License-Identifier: Apache-2.0
"""Two logical batches, exact incremental scoring, bounded search windows."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.joint_prefill import SRPTCoordinationPlan
from vllm.v1.core.sched.joint_prefill_cache import EvaluationCache, freeze
from vllm.v1.core.sched.joint_prefill_score import (
    BatchScorer,
    CoordinationSpec,
    ScoreTimeout,
)
from vllm.v1.core.sched.prefill_operations import ResultKind
from vllm.v1.core.sched.prefill_search_window import ExecutionWindow, SearchBudget
from vllm.v1.kv_wait_trace import emit


@dataclass(frozen=True)
class LogicalBatch:
    # No operation objects or old snapshot IDs: current controls are rebuilt.
    shape: tuple[tuple[str, int, int], ...]
    generation: int
    remaining_reuses: int
    information: tuple
    source: str = "common_srpt_tail"


def information(snapshot):
    """Evaluation assumptions, separate from requested next compute progress."""
    return freeze(
        (
            tuple(
                (
                    r.request_id,
                    r.input_tokens,
                    r.enqueue_sequence,
                    r.arrival_time,
                    r.lookup_submitted,
                    r.lookup_result_known,
                    r.lookup_result_tokens,
                    None
                    if r.last_match is None
                    else (
                        r.last_match.local_tokens,
                        r.last_match.external_tokens,
                        r.last_match.matched_tokens,
                        r.last_match.local_only_tokens,
                    ),
                    r.restore_reserved,
                    r.restore_completion_received,
                    r.restore_completion_adopted,
                    r.restore_failed,
                    r.dependencies,
                    r.held_hbm_blocks,
                    r.committed_recovery_path,
                    r.store_committed_or_saved_tokens,
                )
                for r in snapshot.requests
            ),
            snapshot.operations,
            snapshot.submitted_batches,
            snapshot.free_hbm_blocks,
            snapshot.inflight_prefill_reserved_blocks,
            snapshot.max_num_seqs,
            snapshot.token_budget,
            snapshot.input_budget,
        )
    )


class SRPTRollingPlan(SRPTCoordinationPlan):
    """V1 action/search space, with one successor reuse per search generation.

    The cheap tail is common SRPT, not a separately optimized second step.
    It can avoid a fresh search only when its full compute proposal agrees with
    CURRENT common SRPT. Controls always come from the current state. That is
    an identity certificate, not reuse of an old improvement score. A divergent
    or stale proposal is rebuilt/searched; insufficient budget uses common SRPT.
    """

    window_type = ExecutionWindow
    timeout_type = ScoreTimeout

    def __init__(self, hf_config, config):
        super().__init__(hf_config, config)
        if self.limit > 8 or self.time_ms != 8.0:
            raise ValueError("rolling v2 retains eight attempts and fixed8 reference")
        self.rolling_config = dict(config["rolling"])
        cfg = self.rolling_config
        if cfg["budget_mode"] not in ("fixed", "dynamic"):
            raise ValueError("budget mode must be fixed or dynamic")
        if any(
            not 0 < cfg[k] <= 8.0
            for k in (
                "absolute_wall_ms",
                "absolute_cpu_ms",
                "conservative_ms",
                "control_delay_ms",
                "submit_reserve_ms",
            )
        ):
            raise ValueError("rolling allowances must be in (0, 8] ms")
        self.cache = EvaluationCache(cfg["cache_entries"], cfg["cache_bytes"])
        self.reference = self.scorer
        self.scorer = BatchScorer(config["costs"], self.order, self.cache)
        self.window_reader = lambda: ExecutionWindow()
        self.tail = None
        self.generation = 0
        self.nonempty_commits = 0
        self.reused = False
        self.rolling_report = {}
        self.committed_step = None
        self.feedback_invalid = None

    def set_execution_window_reader(self, reader):
        self.window_reader = reader

    def new_budget(self):
        return SearchBudget(self.rolling_config, self.window_reader)

    def common_spec(self, snapshot):
        rows = {r.request_id: r for r in snapshot.requests}
        return CoordinationSpec(
            deferrals=tuple(
                (rid, leader, p)
                for rid, (leader, p) in self.waits.items()
                if rid in rows
                and leader in rows
                and rows[leader].restore_reserved
                and not rows[leader].restore_completion_received
                and not rows[leader].restore_failed
            )
        )

    def plan(self, snapshot):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        self.snapshot = snapshot
        budget = self.new_budget()
        self.reused = False
        self.committed_step = None
        previous_error, self.feedback_invalid = self.feedback_invalid, None
        cfg_key = freeze((self.config["costs"], self.scenes))
        previous_key = getattr(self, "model_scenarios", cfg_key)
        if cfg_key != previous_key:
            # Configuration is immutable during a run. Never silently refit.
            raise ValueError("cost/scenario identity changed; create a new policy")
        self.model_scenarios = cfg_key
        common = self.common_spec(snapshot)
        current = tuple(
            (rid, *pq) for rid, pq in self.expected_grants(snapshot, common).items()
        )
        info = information(snapshot)
        reasons = []
        execution_valid = False
        score_valid = False
        if self.tail is not None:
            rows = {r.request_id: r for r in snapshot.requests}
            estimates = self.estimates(snapshot, common)
            execution_valid = all(
                rid in rows
                and estimates[rid] is not None
                and estimates[rid].prefix_tokens == p
                and not rows[rid].restore_reserved
                and not rows[rid].output_placeholders
                for rid, p, q in self.tail.shape
            )
            if not execution_valid:
                reasons.append("progress_dependency_or_members_changed")
            if self.tail.information != info:
                reasons.append("request_match_or_obligation_changed")
            if previous_error:
                reasons.append(previous_error)
            if self.tail.remaining_reuses == 0:
                reasons.append("search_generation_expired")
            # Equality includes ALL selected current requests, not just overlap
            # with old members. New controls still appear in the rebuilt plan.
            identity = self.tail.shape == current
            score_valid = identity and execution_valid
            self.reused = bool(
                self.enabled
                and self.tail.remaining_reuses == 1
                and score_valid
                and previous_error is None
                and current
            )
            if not identity:
                reasons.append("tail_differs_from_current_common_srpt")
        validation_wall, validation_cpu = (
            perf_counter_ns() - wall,
            thread_time_ns() - cpu,
        )
        result = super().plan(
            snapshot, allow_search=not self.reused, search_budget=budget
        )
        searched = not self.reused and self.last_report["scores"] > 0
        if searched:
            self.generation += 1
        self.next_reuses = (
            0
            if self.reused
            else 1
            if searched and not self.last_report["timed_out"]
            else 0
        )
        self.rolling_report = dict(
            step=snapshot.step_id,
            generation=self.generation,
            execution_proposal_valid=execution_valid,
            old_evaluation_invalid=bool(reasons),
            invalidation_reasons=reasons,
            evaluation_certificate="current_common_srpt_identity"
            if self.reused
            else "fresh_search_or_common_fallback",
            reused=self.reused,
            reused_optimized_score=False,
            avoided_fresh_search=int(self.reused),
            next_reuses=self.next_reuses,
            nonempty_commits=self.nonempty_commits,
            validation_wall_ns=validation_wall,
            validation_cpu_ns=validation_cpu,
            budget=budget.summary(),
            cache=dict(self.cache.stats),
            cache_entries=len(self.cache.data),
            cache_bytes=self.cache.bytes,
            total_wall_ns=perf_counter_ns() - wall,
            total_thread_cpu_ns=thread_time_ns() - cpu,
        )
        emit("srpt_rolling_step", **self.rolling_report)
        return result

    def on_step_committed(self, step_id, shape, mapping_complete=True):
        """Final native allocation, after all same-step revokes; scalar only."""
        wall, cpu = perf_counter_ns(), thread_time_ns()
        if self.snapshot is None or step_id != self.snapshot.step_id:
            return
        if self.committed_step == step_id:
            raise RuntimeError("duplicate final allocation feedback")
        self.committed_step = step_id
        if not mapping_complete:
            self.tail = None
            self.feedback_invalid = "unmapped_final_allocation"
            self.rolling_report["commit_invalidated"] = self.feedback_invalid
            return
        if not shape:
            self.rolling_report["zero_step_preserved_tail"] = self.tail is not None
            return
        self.nonempty_commits += 1
        self.rolling_report["nonempty_commits"] = self.nonempty_commits
        expected = tuple((rid, *pq) for rid, pq in self.predicted.items())
        exact = shape == expected and self.invalid is None
        if not exact:
            self.tail = None
            self.rolling_report["commit_invalidated"] = (
                "actual_allocation_or_feedback_changed"
            )
            emit(
                "srpt_rolling_commit",
                step=step_id,
                shape=shape,
                reused=False,
                reason=self.rolling_report["commit_invalidated"],
            )
            return
        assigned = {rid: (p, q) for rid, p, q in shape}
        # A logical unsubmitted-work projection. These estimates confer no
        # capacity/readiness; next step always rechecks current native state.
        projected = replace(
            self.snapshot,
            requests=tuple(
                replace(
                    r,
                    accounted_tokens=assigned[r.request_id][0]
                    + assigned[r.request_id][1],
                    inflight_tokens=r.inflight_tokens + assigned[r.request_id][1],
                )
                if r.request_id in assigned
                else r
                for r in self.snapshot.requests
            ),
        )
        tail = tuple(
            (rid, *pq)
            for rid, pq in self.expected_grants(
                projected, self.common_spec(projected)
            ).items()
        )
        self.tail = LogicalBatch(
            tail, self.generation, self.next_reuses, information(self.snapshot)
        )
        self.rolling_report.update(
            finalization_wall_ns=perf_counter_ns() - wall,
            finalization_thread_cpu_ns=thread_time_ns() - cpu,
        )
        emit(
            "srpt_rolling_commit",
            step=step_id,
            shape=shape,
            reused=self.reused,
            tail=asdict(self.tail) | {"information": "scalar dependency summary"},
            nonempty_commits=self.nonempty_commits,
        )

    def on_feedback(self, result):
        super().on_feedback(result)
        if result.result in (
            ResultKind.FAILURE,
            ResultKind.REVOKED,
            ResultKind.CANCEL_PENDING,
        ):
            self.feedback_invalid = result.result.value
            if self.tail and any(
                result.request_id == rid for rid, p, q in self.tail.shape
            ):
                self.tail = None
