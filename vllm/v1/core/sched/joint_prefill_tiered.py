# SPDX-License-Identifier: Apache-2.0
"""Global coarse screening and selectively detailed, single-budget coordination."""

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.joint_prefill import SRPTCoordinationPlan
from vllm.v1.core.sched.joint_prefill_score import (
    CoordinationSpec,
    ScoreTimeout,
    ScoreUnknown,
)
from vllm.v1.core.sched.joint_prefill_tiered_score import (
    ShapeCachedScorer,
    coarse_score,
    cpu_estimate,
    necessary_scenarios,
    size_features,
)
from vllm.v1.core.sched.prefill_search_window import ExecutionWindow
from vllm.v1.core.sched.prefill_tiered_budget import GPUWindow, TieredBudget
from vllm.v1.kv_wait_trace import emit


def family(kind):
    if kind.startswith("shared_"):
        return "shared_joint"
    if kind.startswith("path_"):
        return "path_joint"
    return kind if kind in ("shared", "path") else "batch"


class SRPTTieredPlan(SRPTCoordinationPlan):
    window_type = ExecutionWindow

    def __init__(self, hf_config, config):
        super().__init__(hf_config, deepcopy(config))
        self.tiered_config = self.config["tiered"]
        cfg = self.tiered_config
        if not 1 <= cfg["max_candidates"] <= 32 or len(self.scenes) != 8:
            raise ValueError(
                "V3 requires <=32 bounded proposals and eight source scenarios"
            )
        ceiling = (
            16 if cfg["gpu_window"].get("schema") == "A100-shaped-window-r2" else 8
        )
        if any(
            not 0 < cfg[k] <= ceiling
            for k in (
                "wall_limit_ms",
                "cpu_limit_ms",
                "control_limit_ms",
                "unknown_window_ms",
                "submit_reserve_ms",
            )
        ):
            raise ValueError(f"V3 permits at most {ceiling}ms wall/CPU/control delay")
        self.reference = self.scorer
        self.scorer = ShapeCachedScorer(self.config["costs"], self.order)
        self.window_model = GPUWindow(cfg["gpu_window"], self.scorer)
        self.window_reader = lambda: ExecutionWindow()
        self.features = self.orders = None
        self.tiered_report = {}

    def set_execution_window_reader(self, reader):
        self.window_reader = reader

    def new_budget(self, snapshot):
        return TieredBudget(
            self.tiered_config, snapshot, self.window_reader, self.window_model
        )

    def estimates(self, snapshot, spec):
        if snapshot is not self.snapshot or self.features is None:
            return super().estimates(snapshot, spec)
        key = spec.recompute
        if key not in self.features:
            self.features[key] = super().estimates(snapshot, spec)
        return self.features[key]

    def order(self, snapshot, spec, states=None):
        if snapshot is not self.snapshot or self.orders is None or states is not None:
            return super().order(snapshot, spec, states)
        if spec.recompute not in self.orders:
            self.orders[spec.recompute] = super().order(snapshot, spec)
        return self.orders[spec.recompute]

    def coarse(self, snapshot, spec, scene, delay, deadline):
        return coarse_score(
            snapshot,
            spec,
            self.estimates(snapshot, spec),
            scene,
            self.scorer,
            self.order,
            delay,
            deadline,
        )

    def plan(self, snapshot):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        self.snapshot, self.invalid, self.actual = snapshot, None, {}
        self.features, self.orders = {}, {}
        budget = self.new_budget(snapshot)
        rows = {r.request_id: r for r in snapshot.requests}
        self.waits = {
            rid: (leader, p)
            for rid, (leader, p) in self.waits.items()
            if rid in rows
            and leader in rows
            and rows[leader].restore_reserved
            and not rows[leader].restore_failed
            and not rows[leader].restore_completion_adopted
        }
        baseline = CoordinationSpec(
            deferrals=tuple((rid, *pair) for rid, pair in self.waits.items())
        )
        chosen = baseline
        estimates = self.estimates(snapshot, baseline)
        sizes = size_features(snapshot, estimates)
        report = dict(
            step=snapshot.step_id,
            coordinate=self.enabled,
            candidates=len(rows),
            unknown=sum(v is None for v in estimates.values()),
            groups=0,
            groups_covered=0,
            requests_covered=0,
            scores=0,
            scenario_scores=0,
            fully_scored_candidates=0,
            candidate_kinds={},
            rounds=1,
            accepted=[],
            rejected={},
            timed_out=False,
            final_plan_fallback=None,
            prediction_invalid=None,
            effective_change=False,
        )
        detail = dict(
            step=snapshot.step_id,
            requests=len(rows),
            operations=len(snapshot.operations),
            submitted_batches=len(snapshot.submitted_batches),
            scalar_preparations=1,
            coarse_completed=0,
            coarse_unknown=0,
            coarse_families={},
            fine_families={},
            fine_completed=0,
            fine_early_rejected=0,
            fine_incomplete=0,
            state_score_cache_hits=0,
            coarse_shapes=[],
            fallback=None,
            configuration_serializations=0,
            scenario_groups=[],
        )
        phases = {k: [0, 0] for k in ("coarse", "fine", "generation", "finalization")}
        prepare_wall, prepare_cpu = perf_counter_ns() - wall, thread_time_ns() - cpu
        touched, groups_touched = set(), set()
        baseline_scores = {}
        fine_cache = {}  # Exact same immutable snapshot; discarded next step.
        fine_scenes, equivalences = necessary_scenarios(
            snapshot,
            self.scenes,
            bool(self.waits)
            or sum(
                self.initial_choice(r) or r.restore_reserved for r in snapshot.requests
            )
            >= 2,
            estimates,
        )
        detail["scenario_groups"] = equivalences
        cfg = self.tiered_config
        finalize_ms = 0.0
        try:
            if not self.enabled or budget.remaining_ns() <= 0 or budget.ready():
                detail["fallback"] = "disabled_or_no_budget"
            elif sum(
                e is not None and e.remaining_tokens > 0 for e in estimates.values()
            ) < 2 and not any(self.initial_choice(r) for r in snapshot.requests):
                detail["fallback"] = "no_available_action_family"
            else:
                finalize_ms = cpu_estimate(cfg["cpu_cost"], "finalization", sizes)
                fine_ms = cpu_estimate(cfg["cpu_cost"], "fine", sizes, len(fine_scenes))
                pair_ms = 2 * fine_ms
                detail.update(
                    reserved_finalize_ms=finalize_ms, reserved_fine_pair_ms=pair_ms
                )
                if budget.remaining_ns() < (finalize_ms + pair_ms) * 1e6:
                    detail["fallback"] = "insufficient_one_complete_pair"
                else:
                    pool = []
                    seen = {baseline}
                    # Coarse work may spend only what remains after a complete
                    # baseline/candidate pair and native-plan materialization.
                    coarse_deadline = budget.slice(finalize_ms + pair_ms)
                    center = self.scenes[cfg["coarse_scenario_index"]]
                    start, thread = perf_counter_ns(), thread_time_ns()
                    try:
                        groups, overlap = self.groups(snapshot, coarse_deadline)
                        report.update(
                            groups=len(groups), overlapping_group_nodes_omitted=overlap
                        )
                        generators = [
                            self.batch_candidates(snapshot, baseline, coarse_deadline),
                            self.recovery_candidates(
                                snapshot, baseline, groups, coarse_deadline, "path"
                            ),
                            self.recovery_candidates(
                                snapshot, baseline, groups, coarse_deadline, "shared"
                            ),
                        ]
                    finally:
                        phases["generation"][0] += perf_counter_ns() - start
                        phases["generation"][1] += thread_time_ns() - thread
                    live = (
                        generators[self.rotation % 3 :]
                        + generators[: self.rotation % 3]
                    )
                    try:
                        while live and len(seen) <= cfg["max_candidates"]:
                            for generator in tuple(live):
                                coarse_deadline.check()
                                start, thread = perf_counter_ns(), thread_time_ns()
                                try:
                                    item = next(generator)
                                except StopIteration:
                                    live.remove(generator)
                                    continue
                                finally:
                                    phases["generation"][0] += perf_counter_ns() - start
                                    phases["generation"][1] += thread_time_ns() - thread
                                kind, ids, spec = item
                                if spec in seen:
                                    continue
                                if len(seen) > cfg["max_candidates"]:
                                    break
                                seen.add(spec)
                                touched.update(ids)
                                report["candidate_kinds"][kind] = (
                                    report["candidate_kinds"].get(kind, 0) + 1
                                )
                                bucket = family(kind)
                                start, thread = perf_counter_ns(), thread_time_ns()
                                try:
                                    coarse = self.coarse(
                                        snapshot,
                                        spec,
                                        center,
                                        budget.summary()["granted_ms"],
                                        coarse_deadline,
                                    )
                                    score = coarse.score.objective
                                    detail["coarse_completed"] += 1
                                    if len(detail["coarse_shapes"]) < 4:
                                        detail["coarse_shapes"].append(
                                            dict(
                                                kind=kind,
                                                shape=coarse.head_shapes,
                                                finishers=coarse.head_finishers,
                                                score_ms=score,
                                            )
                                        )
                                except ScoreUnknown:
                                    score = None
                                    detail["coarse_unknown"] += 1
                                finally:
                                    phases["coarse"][0] += perf_counter_ns() - start
                                    phases["coarse"][1] += thread_time_ns() - thread
                                pool.append((score, kind, ids, spec))
                                detail["coarse_families"][bucket] = (
                                    detail["coarse_families"].get(bucket, 0) + 1
                                )
                    except ScoreTimeout:
                        detail["coarse_budget_stopped"] = True
                    buckets = {}
                    for item in pool:
                        buckets.setdefault(family(item[1]), []).append(item)
                    for values in buckets.values():
                        values.sort(key=lambda x: (x[0] is None, x[0] or 0))
                    names = sorted(buckets)
                    if names:
                        shift = self.rotation % len(names)
                        names = names[shift:] + names[:shift]
                    best_value = 0.0
                    while any(buckets.values()):
                        for name in names:
                            if not buckets[name]:
                                continue
                            baseline_fraction = 1 - len(baseline_scores) / len(
                                fine_scenes
                            )
                            required = fine_ms * (1 + baseline_fraction) + finalize_ms
                            if budget.remaining_ns() < required * 1e6:
                                raise ScoreTimeout(
                                    "insufficient_next_complete_candidate"
                                )
                            budget.check(int(finalize_ms * 1e6))
                            _, kind, ids, spec = buckets[name].pop(0)
                            report["scores"] += 1
                            detail["fine_families"][name] = (
                                detail["fine_families"].get(name, 0) + 1
                            )
                            diffs, complete = [], True
                            start, thread = perf_counter_ns(), thread_time_ns()
                            try:
                                for index, scene in enumerate(fine_scenes):
                                    until = budget.slice(finalize_ms)
                                    until.check()
                                    if index not in baseline_scores:
                                        baseline_scores[index] = self.scorer.score(
                                            snapshot,
                                            baseline,
                                            estimates,
                                            scene,
                                            0.0,
                                            until,
                                        )
                                        report["scenario_scores"] += 1
                                    key = (spec, index)
                                    if key not in fine_cache:
                                        fine_cache[key] = self.scorer.score(
                                            snapshot,
                                            spec,
                                            self.estimates(snapshot, spec),
                                            scene,
                                            budget.summary()["granted_ms"],
                                            until,
                                        )
                                        report["scenario_scores"] += 1
                                    else:
                                        detail["state_score_cache_hits"] += 1
                                    delta = (
                                        fine_cache[key].objective
                                        - baseline_scores[index].objective
                                    )
                                    diffs.append(delta)
                                    if delta >= -1e-9:
                                        detail["fine_early_rejected"] += 1
                                        complete = False
                                        break
                            except ScoreUnknown as error:
                                reason = str(error)
                                report["rejected"][reason] = (
                                    report["rejected"].get(reason, 0) + 1
                                )
                                complete = False
                            except ScoreTimeout:
                                detail["fine_incomplete"] += 1
                                raise
                            finally:
                                phases["fine"][0] += perf_counter_ns() - start
                                phases["fine"][1] += thread_time_ns() - thread
                            if complete:
                                detail["fine_completed"] += 1
                                report["fully_scored_candidates"] += 1
                                value = sum(
                                    delta * len(group)
                                    for delta, group in zip(
                                        diffs, equivalences, strict=True
                                    )
                                )
                                if value < best_value:
                                    chosen, best_value = spec, value
                                    report["accepted"].append(
                                        dict(
                                            kind=kind,
                                            requests=ids,
                                            deltas_ms=diffs,
                                            scenario_groups=equivalences,
                                            spec=asdict(spec),
                                        )
                                    )
                                    if name.startswith("shared"):
                                        groups_touched.add(tuple(ids))
        except ScoreTimeout as error:
            detail["search_stopped"] = str(error)
            report["search_budget_exhausted"] = True
        except ScoreUnknown as error:
            detail["fallback"] = str(error)
        start, thread = perf_counter_ns(), thread_time_ns()
        discard = budget.completed_plan_discard_reason() if chosen != baseline else None
        if discard:
            chosen = baseline
            report.update(
                final_plan_fallback=discard, timed_out=discard != "cancel_ready"
            )
        result = self.build(snapshot, chosen)
        self.current = chosen
        self.predicted = self.expected_grants(snapshot, chosen)
        discard = budget.completed_plan_discard_reason() if chosen != baseline else None
        if discard:
            chosen = baseline
            result = self.build(snapshot, chosen)
            self.current = chosen
            self.predicted = self.expected_grants(snapshot, chosen)
            report.update(
                final_plan_fallback=discard, timed_out=discard != "cancel_ready"
            )
        for rid, leader, p in chosen.deferrals:
            if rows[leader].restore_reserved and not rows[leader].restore_failed:
                self.waits[rid] = (leader, p)
        self.rotation += 1
        phases["finalization"] = [perf_counter_ns() - start, thread_time_ns() - thread]
        report.update(
            effective_change=chosen != baseline,
            requests_covered=len(touched),
            requests_not_in_candidate_edits=len(rows.keys() - touched),
            groups_covered=len(groups_touched),
            score_wall_ns=phases["fine"][0],
            score_thread_cpu_ns=phases["fine"][1],
            generation_wall_ns=phases["generation"][0],
            generation_thread_cpu_ns=phases["generation"][1],
            total_wall_ns=perf_counter_ns() - wall,
            total_thread_cpu_ns=thread_time_ns() - cpu,
        )
        detail.update(
            budget=budget.summary(),
            prepare_wall_ns=prepare_wall,
            prepare_thread_cpu_ns=prepare_cpu,
            phases=phases,
            shape_cache_hits=self.scorer.hits,
            shape_cache_misses=self.scorer.misses,
            shape_cache_entries=len(self.scorer.shapes),
            effective_change=report["effective_change"],
            total_wall_ns=perf_counter_ns() - wall,
            total_thread_cpu_ns=thread_time_ns() - cpu,
        )
        self.last_report, self.tiered_report = report, detail
        emit("srpt_joint_step", **report)
        emit("srpt_tiered_step", **detail)
        if report["effective_change"] and self.witnesses < 8:
            self.witnesses += 1
            emit(
                "srpt_tiered_witness",
                step=snapshot.step_id,
                chosen=asdict(chosen),
                predicted=self.predicted,
                accepted=report["accepted"],
                requests=[
                    dict(
                        id=r.request_id,
                        accounted=r.accounted_tokens,
                        inflight=r.inflight_tokens,
                        reserved=r.restore_reserved,
                        lookup_known=r.lookup_result_known,
                    )
                    for r in snapshot.requests
                ],
            )
        return result
