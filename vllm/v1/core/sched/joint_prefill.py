# SPDX-License-Identifier: Apache-2.0
"""Estimated SRPT with optional bounded full-queue batch/recovery coordination."""

from dataclasses import asdict, replace
from time import perf_counter_ns, thread_time_ns

from vllm.v1.core.sched.compute_cost import ComputeCostModel
from vllm.v1.core.sched.global_plan import (
    NativePlan,
    OperationKind,
    RecoveryPath,
    StepPlan,
)
from vllm.v1.core.sched.joint_prefill_score import (
    BatchScorer,
    CoordinationSpec,
    ScoreTimeout,
    ScoreUnknown,
    check_time,
)
from vllm.v1.core.sched.prefill_estimate import remaining_estimate
from vllm.v1.core.sched.prefill_operations import OperationAction, ResultKind
from vllm.v1.kv_wait_trace import emit


def replace_cap(spec, rid, cap):
    caps = dict(spec.caps)
    caps[rid] = cap
    return replace(spec, caps=tuple(sorted(caps.items())), end_after=None)


class SRPTCoordinationPlan:
    """No resource references. Only native feedback can establish persistent waits.

    Control attempts retain NativePlan's native relative order and precede the
    SRPT computation pass in BOTH modes. This is a new common plan baseline,
    not the legacy compute-only plugin's query-trigger timing.
    """

    def __init__(self, hf_config, config):
        self.model = ComputeCostModel.from_config(hf_config)
        self.a, self.b = self.model.a, self.model.b
        self.config = config
        self.enabled = config["coordinate"]
        if type(self.enabled) is not bool:
            raise ValueError("coordinate must be bool")
        self.scenes = tuple(config["scenarios"])
        self.limit = config["max_scores"]
        self.time_ms = config["max_wall_ms"]
        if not 1 <= self.limit <= 64 or not 0 < self.time_ms <= 100:
            raise ValueError("invalid frozen decision budget")
        if not self.scenes:
            raise ValueError("explicit uncertainty scenarios required")
        self.scorer = BatchScorer(config["costs"], self.order)
        self.rotation = 0
        self.waits = {}
        self.snapshot = None
        self.current = CoordinationSpec()
        self.predicted = {}
        self.actual = {}
        self.invalid = None
        self.last_report = {}
        self.witnesses = 0

    def estimates(self, snapshot, spec):
        return {
            r.request_id: remaining_estimate(
                r,
                self.model,
                snapshot.observed_ns,
                r.first_match,
                recompute=r.request_id in spec.recompute,
            )
            for r in snapshot.requests
        }

    def order(self, snapshot, spec, states=None):
        native = [r.request_id for r in snapshot.requests]
        rows = {r.request_id: r for r in snapshot.requests}
        if states is None:
            estimates = self.estimates(snapshot, spec)
            work = {rid: e.work for rid, e in estimates.items() if e is not None}
        else:
            # Predicted scalar positions were checked on model entry. Reuse the
            # same structural formula without re-reading constant geometry.
            work = {
                rid: self.a * (rows[rid].input_tokens - s["p"])
                + self.b
                * (
                    (rows[rid].input_tokens - s["p"]) * s["p"]
                    + (rows[rid].input_tokens - s["p"])
                    * (rows[rid].input_tokens - s["p"] + 1)
                    // 2
                )
                for rid, s in states.items()
                if s["known"]
            }
        known = iter(
            sorted(
                work,
                key=lambda rid: (
                    work[rid],
                    rows[rid].arrival_time,
                    rows[rid].enqueue_sequence,
                ),
            )
        )
        return tuple(next(known) if rid in work else rid for rid in native)

    def build(self, snapshot, spec):
        native = NativePlan().plan(snapshot)
        by_request = {r.request_id: r for r in snapshot.requests}
        controls = [x for x in native.attempts if x.kind is not OperationKind.COMPUTE]
        computes = {
            x.request_id: x for x in native.attempts if x.kind is OperationKind.COMPUTE
        }
        recompute = set(spec.recompute)
        controls = [
            x
            for x in controls
            if not (x.kind is OperationKind.LOAD and x.request_id in recompute)
        ]
        # New LOOKUP permission and its ordering never depend on the score.
        # A group leader may move only its existing LOAD before its followers.
        waits = {rid: (leader, p) for rid, leader, p in spec.deferrals}
        for leader in dict.fromkeys(leader for leader, _ in waits.values()):
            lead = next(
                (
                    x
                    for x in controls
                    if x.kind is OperationKind.LOAD and x.request_id == leader
                ),
                None,
            )
            targets = [
                i
                for i, x in enumerate(controls)
                if x.kind is OperationKind.LOAD
                and (
                    x.request_id == leader
                    or waits.get(x.request_id, (None,))[0] == leader
                )
            ]
            if lead is not None and targets:
                controls.remove(lead)
                controls.insert(min(targets), lead)
        controls = [
            replace(x, defer_while_loading=waits[x.request_id][0])
            if x.kind is OperationKind.LOAD and x.request_id in waits
            else x
            for x in controls
        ]
        items = []
        caps = dict(spec.caps)
        for rid in self.order(snapshot, spec):
            if rid not in computes:
                continue
            item = computes[rid]
            item = replace(
                item,
                max_tokens=caps.get(rid),
                end_after_commit=rid == spec.end_after,
                recovery_path=RecoveryPath.RECOMPUTE
                if rid in recompute and not by_request[rid].committed_recovery_path
                else None,
                defer_while_loading=waits[rid][0] if rid in waits else None,
            )
            items.append(item)
        return StepPlan(snapshot.snapshot_id, tuple(controls + items))

    @staticmethod
    def initial_choice(r):
        return (
            r.status == "WAITING"
            and not r.held_hbm_blocks
            and not r.accounted_tokens
            and not r.inflight_tokens
            and not r.output_placeholders
            and not r.stale_output_tokens
            and not r.committed_recovery_path
            and r.lookup_result_known
            and r.last_match is not None
            and r.lookup_result_tokens > 0
        )

    def groups(self, snapshot, deadline):
        index = {}
        rows = {r.request_id: r for r in snapshot.requests}
        for r in snapshot.requests:
            check_time(deadline)
            if r.logical_prefix is None or r.last_match is None:
                continue
            if not (self.initial_choice(r) or r.restore_reserved):
                continue
            for end, digest in r.logical_prefix.boundaries:
                if end > r.last_match.matched_tokens:
                    break
                key = (r.logical_prefix.namespace, end, digest)
                index.setdefault(key, []).append(r.request_id)
        used = set()
        groups = []
        overlap = 0
        for key, ids in sorted(index.items(), key=lambda kv: (-kv[0][1], kv[0])):
            check_time(deadline)
            members = sorted(
                (r for r in ids if r not in used),
                key=lambda r: (rows[r].arrival_time, rows[r].enqueue_sequence),
            )
            if len(ids) >= 2 and len(members) < 2:
                overlap += 1
            if len(members) < 2:
                continue
            leaders = [
                r
                for r in members
                if rows[r].restore_reserved or self.initial_choice(rows[r])
            ]
            if not leaders:
                continue
            used.update(members)
            groups.append((tuple(members), key[1]))
        return groups, overlap

    def batch_candidates(self, snapshot, spec, deadline):
        estimates = self.estimates(snapshot, spec)
        rows = {r.request_id: r for r in snapshot.requests}
        order = self.order(snapshot, spec)
        known = [
            rid
            for rid in order
            if estimates[rid] is not None
            and estimates[rid].remaining_tokens > 0
            and not rows[rid].restore_reserved
            and not rows[rid].output_placeholders
            and not rows[rid].stale_output_tokens
            and (
                rows[rid].held_hbm_blocks
                or not rows[rid].last_match
                or rows[rid].last_match.external_tokens == 0
                or rid in spec.recompute
            )
        ]
        if not known:
            return
        # Rotate proposal traversal, not execution order. Every known background
        # finisher can supply a budget-preserving point; no fixed request window.
        for offset in range(len(known)):
            check_time(deadline)
            index = (offset + self.rotation) % len(known)
            rid = known[index]
            prefix = known[:index]
            prior_caps = dict(spec.caps)
            consumed = sum(
                min(estimates[r].remaining_tokens, prior_caps.get(r, 2048))
                for r in prefix
            )
            if consumed >= snapshot.token_budget:
                continue
            budget = snapshot.token_budget - consumed
            r = estimates[rid].remaining_tokens
            upper = min(budget, r)
            if prefix and all(
                prior_caps.get(r, 2048) >= estimates[r].remaining_tokens for r in prefix
            ):
                yield ("end", tuple(prefix), replace(spec, end_after=prefix[-1]))
            align = r - snapshot.token_budget
            if 0 < align < upper:
                yield ("align", (rid,), replace_cap(spec, rid, align))
            for later in known[index + 1 :]:
                check_time(deadline)
                reserve = estimates[later].remaining_tokens
                cap = budget - reserve
                if 0 < cap < upper:
                    yield ("reserve_other", (rid, later), replace_cap(spec, rid, cap))
        if spec.caps or spec.end_after:
            yield ("fill", tuple(known), replace(spec, caps=(), end_after=None))

    def recovery_candidates(self, snapshot, spec, groups, deadline, category):
        rows = {r.request_id: r for r in snapshot.requests}
        # Every matching request can propose recompute, including singleton groups.
        choices = [r.request_id for r in snapshot.requests if self.initial_choice(r)]
        rotated = (
            choices[self.rotation % len(choices) :]
            + choices[: self.rotation % len(choices)]
            if choices
            else []
        )
        for rid in rotated if category == "path" else ():
            check_time(deadline)
            route = replace(
                spec,
                recompute=tuple(sorted(set(spec.recompute) ^ {rid})),
                end_after=None,
            )
            yield ("path", (rid,), route)
            # A route and a batch boundary are accepted as one complete edit.
            yield from (
                ("path_" + kind, (rid, *touched), candidate)
                for kind, touched, candidate in self.batch_candidates(
                    snapshot, route, deadline
                )
            )
        for members, prefix in (
            groups[self.rotation % len(groups) :]
            + groups[: self.rotation % len(groups)]
            if groups and category == "shared"
            else []
        ):
            check_time(deadline)
            earliest = members[0]
            cheapest = min(
                members,
                key=lambda r: (
                    self.model.estimate(
                        rows[r].input_tokens
                        - (
                            rows[r].last_match.local_only_tokens
                            or rows[r].last_match.local_tokens
                        ),
                        rows[r].last_match.local_only_tokens
                        or rows[r].last_match.local_tokens,
                    ),
                    rows[r].enqueue_sequence,
                ),
            )
            for leader in dict.fromkeys((earliest, cheapest)):
                followers = [
                    r for r in members if r != leader and self.initial_choice(rows[r])
                ]
                if not followers:
                    continue
                old = tuple(x for x in spec.deferrals if x[0] not in members)
                route = replace(
                    spec,
                    deferrals=old + tuple((r, leader, prefix) for r in followers),
                    recompute=tuple(r for r in spec.recompute if r not in members),
                    end_after=None,
                )
                yield ("shared", members, route)
                yield from (
                    ("shared_" + kind, members + tuple(touched), candidate)
                    for kind, touched, candidate in self.batch_candidates(
                        snapshot, route, deadline
                    )
                )

    def plan(self, snapshot, *, allow_search=True, search_budget=None):
        wall, cpu = perf_counter_ns(), thread_time_ns()
        self.snapshot = snapshot
        self.invalid = None
        self.actual = {}
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
        baseline_spec = CoordinationSpec(
            deferrals=tuple((rid, *pair) for rid, pair in self.waits.items())
        )
        baseline = self.build(snapshot, baseline_spec)
        baseline_ns = perf_counter_ns() - wall
        self.current = baseline_spec
        report = dict(
            step=snapshot.step_id,
            coordinate=self.enabled,
            candidates=len(rows),
            unknown=sum(
                v is None for v in self.estimates(snapshot, baseline_spec).values()
            ),
            groups=0,
            groups_covered=0,
            requests_covered=0,
            scores=0,
            scenario_scores=0,
            fully_scored_candidates=0,
            candidate_kinds={},
            rounds=0,
            accepted=[],
            rejected={},
            timed_out=False,
            prediction_invalid=None,
            baseline_wall_ns=baseline_ns,
            scope="conditional closed set; no GPU ranking guarantee",
        )
        started = perf_counter_ns()
        time_ms = self.time_ms if search_budget is None else search_budget.granted_ms
        searching = self.enabled and allow_search and time_ms > 0
        hard_deadline = started + int(time_ms * 1e6)
        # Bound the expensive evaluator while reserving measured full-queue
        # work for materializing the already validated fallback/final plan.
        reserve_ns = max(250_000, 4 * baseline_ns)
        deadline = (
            hard_deadline - reserve_ns
            if search_budget is None
            else search_budget.with_reserve(reserve_ns)
        )

        def expired():
            return (
                perf_counter_ns() >= hard_deadline
                if search_budget is None
                else search_budget.expired()
            )

        score_wall = score_cpu = generate_wall = generate_cpu = select_wall = (
            select_cpu
        ) = 0
        touched_all = set()
        groups_touched = set()
        chosen = baseline_spec
        baseline_cache = {}

        def evaluate(spec, incumbent):
            nonlocal score_wall, score_cpu
            w, c = perf_counter_ns(), thread_time_ns()
            try:
                features = self.estimates(snapshot, spec)
                values = []
                for i, scene in enumerate(self.scenes):
                    check_time(deadline)
                    if i not in baseline_cache:
                        baseline_cache[i] = self.scorer.score(
                            snapshot,
                            baseline_spec,
                            self.estimates(snapshot, baseline_spec),
                            scene,
                            0.0,
                            deadline,
                        )
                        report["scenario_scores"] += 1
                    value = self.scorer.score(
                        snapshot, spec, features, scene, time_ms, deadline
                    )
                    report["scenario_scores"] += 1
                    if i == len(self.scenes) - 1:
                        report["fully_scored_candidates"] += 1
                    limit = (
                        baseline_cache[i].objective
                        if incumbent is None
                        else incumbent[i].objective
                    )
                    if (
                        value.objective >= limit - 1e-9
                        or value.objective >= baseline_cache[i].objective - 1e-9
                    ):
                        report["early_scenario_rejections"] = (
                            report.get("early_scenario_rejections", 0) + 1
                        )
                        return None
                    values.append(value)
                return values
            finally:
                score_wall += perf_counter_ns() - w
                score_cpu += thread_time_ns() - c

        best = None
        best_scores = None
        try:
            if searching:
                w, c = perf_counter_ns(), thread_time_ns()
                groups, overlap = self.groups(snapshot, deadline)
                generate_wall += perf_counter_ns() - w
                generate_cpu += thread_time_ns() - c
                report.update(
                    groups=len(groups), overlapping_group_nodes_omitted=overlap
                )
                incumbent = None
                seen = {baseline_spec}
                for round_id in range(2):
                    report["rounds"] = round_id + 1
                    best = None
                    best_scores = None
                    generators = (
                        self.batch_candidates(snapshot, chosen, deadline),
                        self.recovery_candidates(
                            snapshot, chosen, groups, deadline, "shared"
                        ),
                        self.recovery_candidates(
                            snapshot, chosen, groups, deadline, "path"
                        ),
                    )
                    live = list(generators)
                    while live:
                        for generator in tuple(live):
                            gw, gc = perf_counter_ns(), thread_time_ns()
                            try:
                                proposal = next(generator)
                            except StopIteration:
                                live.remove(generator)
                                proposal = None
                            finally:
                                generate_wall += perf_counter_ns() - gw
                                generate_cpu += thread_time_ns() - gc
                            if proposal is None:
                                continue
                            check_time(deadline)
                            kind, touched, spec = proposal
                            if spec in seen:
                                continue
                            if report["scores"] >= self.limit:
                                break
                            seen.add(spec)
                            report["scores"] += 1
                            touched_all.update(touched)
                            report["candidate_kinds"][kind] = (
                                report["candidate_kinds"].get(kind, 0) + 1
                            )
                            if kind.startswith("shared"):
                                groups_touched.update(
                                    g for g, p in groups if set(g) <= set(touched)
                                )
                            try:
                                scored = evaluate(spec, incumbent)
                            except ScoreUnknown as e:
                                reason = str(e)
                                report["rejected"][reason] = (
                                    report["rejected"].get(reason, 0) + 1
                                )
                                continue
                            if scored is None:
                                continue
                            sw, sc = perf_counter_ns(), thread_time_ns()
                            diffs = [
                                s.objective - baseline_cache[i].objective
                                for i, s in enumerate(scored)
                            ]
                            if best_scores is None or sum(
                                s.objective for s in scored
                            ) < sum(s.objective for s in best_scores):
                                best = (spec, kind, touched, diffs)
                                best_scores = scored
                            select_wall += perf_counter_ns() - sw
                            select_cpu += thread_time_ns() - sc
                        if report["scores"] >= self.limit:
                            break
                    if best is None:
                        break
                    chosen, kind, touched, diffs = best
                    incumbent = best_scores
                    report["accepted"].append(
                        dict(
                            kind=kind,
                            requests=touched,
                            deltas_ms=diffs,
                            spec=asdict(chosen),
                        )
                    )
                    if report["scores"] >= self.limit:
                        break
        except ScoreTimeout:
            report["search_budget_exhausted"] = True
            # Only a fully evaluated complete plan may survive this cooperative
            # stop. A half-scored candidate never replaces the incumbent.
            if best is not None:
                chosen, kind, touched, diffs = best
                report["accepted"].append(
                    dict(
                        kind=kind,
                        requests=touched,
                        deltas_ms=diffs,
                        spec=asdict(chosen),
                        stop="bounded_partial_neighborhood",
                    )
                )
        except ScoreUnknown as e:
            report["baseline_uncovered"] = str(e)
            chosen = baseline_spec
        result = self.build(snapshot, chosen)
        if searching and expired():
            report["timed_out"] = True
            chosen = baseline_spec
            result = baseline
        self.rotation += 1
        self.current = chosen
        self.predicted = self.expected_grants(snapshot, chosen)
        if searching and expired():
            report["timed_out"] = True
            chosen = baseline_spec
            result = baseline
            self.current = chosen
            self.predicted = self.expected_grants(snapshot, chosen)
        for rid, leader, p in chosen.deferrals:
            if rows[leader].restore_reserved and not rows[leader].restore_failed:
                self.waits[rid] = (leader, p)
        report.update(
            effective_change=chosen != baseline_spec,
            requests_covered=len(touched_all),
            requests_not_in_candidate_edits=len(rows.keys() - touched_all),
            groups_covered=len(groups_touched),
            score_wall_ns=score_wall,
            score_thread_cpu_ns=score_cpu,
            generation_wall_ns=generate_wall,
            generation_thread_cpu_ns=generate_cpu,
            selection_wall_ns=select_wall,
            selection_thread_cpu_ns=select_cpu,
            total_wall_ns=perf_counter_ns() - wall,
            total_thread_cpu_ns=thread_time_ns() - cpu,
            charged_candidate_delay_ms=time_ms if searching else 0.0,
            finalization_reserve_ns=reserve_ns,
            measured_extra_wall_ns=perf_counter_ns() - started,
        )
        self.last_report = report
        emit("srpt_joint_step", **report)
        if report["effective_change"] and self.witnesses < 8:
            self.witnesses += 1
            estimates = self.estimates(snapshot, chosen)
            emit(
                "srpt_joint_witness",
                step=snapshot.step_id,
                spec=asdict(chosen),
                predicted_grants=self.predicted,
                candidates=[
                    dict(
                        id=r.request_id,
                        status=r.status,
                        estimate=asdict(estimates[r.request_id])
                        if estimates[r.request_id]
                        else None,
                        restore_reserved=r.restore_reserved,
                        lookup_known=r.lookup_result_known,
                        first_match=r.first_match is not None,
                    )
                    for r in snapshot.requests
                ],
            )
        return result

    def expected_grants(self, snapshot, spec):
        estimates = self.estimates(snapshot, spec)
        rows = {r.request_id: r for r in snapshot.requests}
        caps = dict(spec.caps)
        budget = snapshot.token_budget
        result = {}
        for rid in self.order(snapshot, spec):
            r = rows[rid]
            e = estimates[rid]
            if (
                e is None
                or r.restore_reserved
                or r.output_placeholders
                or e.remaining_tokens <= 0
            ):
                continue
            if (
                not r.held_hbm_blocks
                and r.last_match
                and r.last_match.external_tokens > 0
                and rid not in spec.recompute
            ):
                continue
            q = min(budget, e.remaining_tokens, caps.get(rid, budget))
            if q <= 0:
                break
            result[rid] = (e.prefix_tokens, q)
            budget -= q
            if rid == spec.end_after:
                break
        return result

    def invalidate(self, reason):
        if self.invalid is None:
            self.invalid = reason
            emit(
                "srpt_joint_prediction_invalid",
                step=self.snapshot.step_id,
                reason=reason,
            )

    def check_compute_proposal(self, candidate, action):
        if self.enabled and self.current != CoordinationSpec():
            expected = self.predicted.get(candidate.request_id)
            q = action.compute_tokens or candidate.proposed_tokens
            if expected != (candidate.start_token, q):
                self.invalidate("native_match_or_budget_changed")
        return OperationAction() if self.invalid else action

    def revise_attempt(self, item):
        if self.invalid:
            return replace(
                item, max_tokens=None, end_after_commit=False, recovery_path=None
            )
        return item

    def on_feedback(self, result):
        if self.snapshot is None:
            return
        if result.result in (
            ResultKind.FAILURE,
            ResultKind.CANCEL_PENDING,
            ResultKind.REVOKED,
        ):
            self.waits = {
                r: pair
                for r, pair in self.waits.items()
                if result.request_id not in (r, pair[0])
            }
            self.invalidate(result.result.value)
        if result.kind == "load" and result.result == ResultKind.RESOURCES_ACQUIRED:
            for rid, leader, p in self.current.deferrals:
                if leader == result.request_id:
                    self.waits[rid] = (leader, p)
        if result.kind == "compute" and result.step_id == self.snapshot.step_id:
            if result.result == ResultKind.SELECTED:
                self.actual[result.request_id] = result.granted_tokens
            elif result.result == ResultKind.REVOKED:
                self.actual.pop(result.request_id, None)
            elif (
                result.result == ResultKind.BLOCKED
                and result.request_id in self.predicted
                and self.current != CoordinationSpec()
            ):
                self.invalidate(result.reason or "native_blocked")
