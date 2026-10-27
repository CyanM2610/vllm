# SPDX-License-Identifier: Apache-2.0
"""Cheap closed-set screening and unchanged detailed event scoring."""

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil

from vllm.v1.core.sched.joint_prefill_score import (
    BatchScorer,
    CoordinationSpec,
    Score,
    ScoreUnknown,
    check_time,
)


class ShapeCachedScorer(BatchScorer):
    """Only immutable GPU shape service values survive a scheduler step."""

    def __init__(self, costs, order_function, limit=256):
        super().__init__(costs, order_function)
        self.shapes = OrderedDict()
        self.limit = limit
        self.hits = self.misses = 0

    def gpu(self, shape, scene):
        if not shape:
            return 0.0
        key = tuple(sorted((p, q) for _, p, q in shape))
        value = self.shapes.get(key)
        if value is None:
            self.misses += 1
            value = self._gpu(shape, {"gpu_scale": 1.0})
            self.shapes[key] = value
            if len(self.shapes) > self.limit:
                self.shapes.popitem(last=False)
        else:
            self.hits += 1
            self.shapes.move_to_end(key)
        return value * scene["gpu_scale"]


def necessary_scenarios(snapshot, scenes, shared_possible, estimates=None):
    """Merge only dimensions absent from every candidate in this step."""
    unknown = (
        any(e is None for e in estimates.values())
        if estimates is not None
        else any(r.last_match is None for r in snapshot.requests)
    )
    pending = (
        bool(snapshot.submitted_batches)
        or any(r.restore_reserved for r in snapshot.requests)
        or any(
            o.kind == "store"
            and o.stage not in ("obligation_ended", "completion_received")
            for o in snapshot.operations
        )
    )
    ids = {r.request_id for r in snapshot.requests}
    retired = any(o.request_id not in ids for o in snapshot.operations)
    unique, groups = {}, {}
    for scene in scenes:
        key = tuple(
            (k, v)
            for k, v in sorted(scene.items())
            if k != "name"
            and not (k == "unknown_match" and not unknown)
            and not (k == "initial_remaining_fraction" and not pending)
            and not (k == "output_remaining_fraction" and not retired)
            and not (k == "published_cache_survives" and not shared_possible)
        )
        unique.setdefault(key, scene)
        groups.setdefault(key, []).append(scene["name"])
    return tuple(unique.values()), tuple(tuple(v) for v in groups.values())


def size_features(snapshot, estimates):
    remaining = sum(
        e.remaining_tokens if e else r.input_tokens
        for r in snapshot.requests
        for e in (estimates[r.request_id],)
    )
    dependencies = (
        len(snapshot.operations)
        + len(snapshot.submitted_batches)
        + sum(
            e is None or r.restore_reserved
            for r in snapshot.requests
            for e in (estimates[r.request_id],)
        )
    )
    return (
        1.0,
        float(len(snapshot.requests)),
        float(ceil(remaining / 2048)),
        float(dependencies),
    )


def cpu_estimate(model, kind, features, scenarios=1):
    if not model.get("admitted"):
        raise ScoreUnknown("cpu_cost_model_unadmitted")
    ms = max(
        model["floor_ms"][kind],
        sum(x * y for x, y in zip(model["coefficients"][kind], features, strict=True)),
    )
    return ms * scenarios * model["safety_factor"]


@dataclass(frozen=True)
class CoarseResult:
    score: Score
    head_shapes: tuple
    head_finishers: tuple
    unknown_requests: int
    scope: str = "two-batch shapes plus cheap SRPT closed-set tail; screening only"


def coarse_score(snapshot, spec, features, scene, scorer, order, delay_ms, deadline):
    """Approximate service lanes; detailed scoring remains the adoption gate.

    All requests and STORE/retired obligations are kept. Unknown queries incur
    service and cold work; they are not executable at the initial decision.
    The tail uses common SRPT greedy batches, without fine event/IPC polling.
    """
    check_time(deadline)
    if not snapshot.complete_request_view or snapshot.omitted_requests:
        raise ScoreUnknown("incomplete_queue_view")
    if (
        sum(
            max(0, (r.input_tokens + 15) // 16 - r.held_hbm_blocks)
            for r in snapshot.requests
        )
        + snapshot.inflight_prefill_reserved_blocks
        > snapshot.free_hbm_blocks
    ):
        raise ScoreUnknown("simultaneous_full_input_capacity_unconfirmed")
    if any(r.restore_failed or r.stale_output_tokens for r in snapshot.requests):
        raise ScoreUnknown("failure_or_stale_state")
    rows = {r.request_id: r for r in snapshot.requests}
    cfg, host = scorer.costs["services"], scene["host_scale"]
    launch = cfg["worker_host_ms"] * host
    output = (
        cfg["sample_copy_ms"] + cfg["result_return_ms"] + cfg["output_delivery_ms"]
    ) * host
    publish = cfg["rpc_reply_ms"] * host
    gpu = sum(
        scorer.gpu(b.shape, scene) * scene["initial_remaining_fraction"]
        for b in snapshot.submitted_batches
    )
    gpu_work = gpu
    copy = 0.0
    loads = stores = batches = 0
    seen, retired = set(), set()
    for op in snapshot.operations:
        check_time(deadline)
        if op.request_id not in rows:
            retired.add(op.request_id)
        if (
            op.operation_id in seen
            or op.kind != "store"
            or op.stage in ("obligation_ended", "completion_received")
        ):
            continue
        seen.add(op.operation_id)
        if op.start_token is None or op.end_token is None:
            raise ScoreUnknown("store_range_unknown")
        ms, size = scorer.transfer("STORE", op.start_token, op.end_token, scene)
        copy = max(copy, gpu) + ms * scene["initial_remaining_fraction"]
        stores += size
    states, ready, saved, completion = {}, {}, {}, {}
    deferred = {f: (leader, common) for f, leader, common in spec.deferrals}
    pending_loads = {}
    unknown = 0
    for rid, r in rows.items():
        check_time(deadline)
        f = features[rid]
        p = f.prefix_tokens if f else 0
        local = (
            r.last_match.local_only_tokens
            if r.last_match and r.last_match.local_only_tokens is not None
            else r.last_match.local_tokens
            if r.last_match
            else 0
        )
        if r.store_committed_or_saved_tokens is None and (
            r.held_hbm_blocks or r.accounted_tokens
        ):
            raise ScoreUnknown("native_STORE_coverage_unknown")
        states[rid] = dict(p=p, known=True)
        ready[rid] = delay_ms
        saved[rid] = max(
            (r.lookup_result_tokens or 0) // 256 * 256,
            r.store_committed_or_saved_tokens or 0,
        )
        if f is None:
            unknown += 1
            ready[rid] += (cfg["lookup_service_ms"] + cfg["status_rpc_ms"]) * host
        if (r.restore_reserved and not r.restore_completion_received) or (
            f
            and not r.held_hbm_blocks
            and p > local
            and rid not in spec.recompute
            and rid not in deferred
        ):
            ms, size = scorer.transfer(
                "LOAD", local // 256 * 256, p // 256 * 256, scene
            )
            if r.restore_reserved:
                copy = max(copy, gpu) + ms
                loads += size
                ready[rid] = max(ready[rid], copy + publish)
            else:
                pending_loads[rid] = (ms, size)
                ready[rid] = float("inf")
        if rid in deferred:
            ready[rid] = float("inf")
        if p >= r.input_tokens:
            if not any(
                rid == member
                for b in snapshot.submitted_batches
                for member, _, _ in b.shape
            ):
                raise ScoreUnknown("unmapped_submitted_batch")
            completion[rid] = max(delay_ms, gpu) + output
    for follower, leader, _common in spec.deferrals:
        if follower not in states or leader not in states:
            raise ScoreUnknown("shared_dependency_missing")
    t = delay_ms
    caps = dict(spec.caps)
    heads, finishers = [], []
    first = True
    for _ in range(4096):
        check_time(deadline)
        for follower, (leader, common) in tuple(deferred.items()):
            if ready[leader] > t:
                continue
            ready[follower] = t + publish
            if scene["published_cache_survives"]:
                states[follower]["p"] = max(states[follower]["p"], common)
            else:
                r = rows[follower]
                local = r.last_match.local_only_tokens or r.last_match.local_tokens
                ms, size = scorer.transfer(
                    "LOAD",
                    local // 256 * 256,
                    states[follower]["p"] // 256 * 256,
                    scene,
                )
                copy = max(copy, gpu, t) + ms
                loads += size
                ready[follower] = copy + publish
            del deferred[follower]
        if len(completion) == len(rows):
            break
        available = [
            rid
            for rid in order(snapshot, spec if first else CoordinationSpec(), states)
            if rid not in completion and ready[rid] <= t
        ]
        if not available:
            if pending_loads:
                for rid, (ms, size) in pending_loads.items():
                    copy = max(copy, gpu, t + launch) + ms
                    loads += size
                    ready[rid] = copy + publish
                pending_loads.clear()
                if len(heads) < 2:
                    heads.append(())
                    finishers.append(())
                first = False
            future = min(ready[rid] for rid in rows if rid not in completion)
            if future == float("inf"):
                raise ScoreUnknown("coarse_dependency_unresolved")
            t = max(t + launch, future)
            continue
        quota, shape = snapshot.token_budget, []
        for rid in available:
            if quota <= 0 or len(shape) >= snapshot.max_num_seqs:
                break
            p = states[rid]["p"]
            q = min(
                quota,
                rows[rid].input_tokens - p,
                caps.get(rid, quota) if first else quota,
            )
            if q <= 0:
                continue
            shape.append((rid, p, q))
            states[rid]["p"] += q
            quota -= q
            if first and rid == spec.end_after:
                break
        if not shape:
            raise ScoreUnknown("coarse_no_progress")
        duration = scorer.gpu(shape, scene)
        start = max(gpu, t + launch)
        if copy > start:
            duration *= scene["overlap_gpu_scale"]
        gpu = start + duration
        gpu_work += duration
        producer = gpu + cfg["sample_copy_ms"] * host
        for rid, (ms, size) in pending_loads.items():
            copy = max(copy, producer) + ms
            loads += size
            ready[rid] = copy + publish
        pending_loads.clear()
        done = []
        for rid, p, q in shape:
            upto = (p + q) // 256 * 256
            if upto > saved[rid]:
                ms, size = scorer.transfer("STORE", saved[rid], upto, scene)
                stores += size
                copy = max(copy, gpu + cfg["sample_copy_ms"] * host) + ms
                saved[rid] = upto
            if states[rid]["p"] >= rows[rid].input_tokens:
                completion[rid] = gpu + output
                done.append(rid)
        if len(heads) < 2:
            heads.append(tuple(shape))
            finishers.append(tuple(done))
        batches += 1
        first = False
        # Cheap continuation is deliberately serialized, not a second IPC model.
        t = gpu + cfg["scheduler_step_ms"] * host
    else:
        raise ScoreUnknown("coarse_tail_incomplete")
    tails = {rid: cfg["output_delivery_ms"] * host for rid in retired}
    drain = max([t, copy + publish, *completion.values(), *tails.values()])
    upper_area = drain * sum((r.input_tokens + 15) // 16 for r in rows.values())
    return CoarseResult(
        Score(
            tuple(sorted((completion | tails).items())),
            gpu_work,
            loads,
            stores,
            drain,
            upper_area,
            batches,
            scene["name"],
        ),
        tuple(heads),
        tuple(finishers),
        unknown,
    )
