# SPDX-License-Identifier: Apache-2.0
"""Bounded, conditional batch continuation; never a runtime resource owner."""

from dataclasses import dataclass
from time import perf_counter_ns

from vllm.v1.core.sched.compute_budget import BatchTimeModel


class ScoreUnknown(Exception):
    pass


class ScoreTimeout(Exception):
    pass


def check_time(deadline):
    if hasattr(deadline, "check"):
        deadline.check()
        return
    if perf_counter_ns() >= deadline:
        raise ScoreTimeout("decision_time_limit")


@dataclass(frozen=True)
class CoordinationSpec:
    caps: tuple[tuple[str, int], ...] = ()
    end_after: str | None = None
    recompute: tuple[str, ...] = ()
    # follower, actual LOAD leader, known common logical prefix end
    deferrals: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True)
class Score:
    completion: tuple[tuple[str, float], ...]
    gpu_ms: float
    load_bytes: int
    store_bytes: int
    drain_ms: float
    owner_block_ms_upper: float
    batches: int
    scenarios: str

    @property
    def objective(self):
        return sum(t for _, t in self.completion)


class BatchScorer:
    """Same calibrated GPU formula; aggregate steps/obligations, not RPC replay.

    Cost uncertainty is a frozen scenario set, not a confidence interval.
    CPU decision latency delays new submissions and completion observation;
    already submitted GPU/copy work continues on its existing calendar.
    """

    def __init__(self, costs, order_function, cache=None):
        self.costs = costs
        self.cache = cache
        if cache is not None:
            cache.bind_model(costs)
        self.order = order_function
        c = costs["gpu"]["coefficients"]
        self.gpu_model = BatchTimeModel(**c, tau_ms=1.0)
        self.anchors = {
            tuple(sorted(map(tuple, a["shape_pq"]))): a
            for a in costs.get("batch_service_anchors", ())
        }

    def gpu(self, shape, scene):
        # Existing exact shape anchors are the service-cost cache. The affine
        # fallback and transfer arithmetic are cheaper than general LRU writes;
        # do not let cost entries evict reusable continuation checkpoints.
        return self._gpu(shape, scene)

    def _gpu(self, shape, scene):
        if not shape:
            return 0.0
        q = sum(q for _, p, q in shape)
        attn = sum(q * p + q * (q + 1) // 2 for _, p, q in shape)
        n = len(shape)
        for axis, value in (("Q", q), ("A", attn), ("N", n)):
            lo, hi = self.costs["gpu"]["train_axis_ranges"][axis]
            if not lo <= value <= hi:
                raise ScoreUnknown("gpu_shape_outside_" + axis)
        anchor = self.anchors.get(tuple(sorted((p, q) for _, p, q in shape)))
        ms = anchor["gpu_forward_ms"] if anchor else self.gpu_model.predict(q, attn, n)
        return ms * scene["gpu_scale"]

    def transfer(self, kind, start, end, scene):
        return self._transfer(kind, start, end, scene)

    def _transfer(self, kind, start, end, scene):
        if end <= start:
            return 0.0, 0
        if start % 256 or end % 256:
            raise ScoreUnknown("transfer_range_not_chunk_aligned")
        spec = self.costs["transfer"][kind][scene["medium"]]
        size = (end - start) * self.costs["kv_bytes_per_token"]
        return (
            spec["fixed_ms"]
            + spec["per_object_ms"] * ((end - start) // 256)
            + size / spec["bytes_per_ms"]
        ) * scene["transfer_scale"], size

    def score(self, snapshot, spec, features, scene, delay_ms, deadline):
        check_time(deadline)
        session = None
        prefix = None
        if self.cache is not None:
            from vllm.v1.core.sched.joint_prefill_cache import ScoreSession

            session = ScoreSession(
                self.cache, snapshot, spec, features, scene, delay_ms
            )
            cached = session.full()
            check_time(deadline)
            if cached is not None:
                return cached
            prefix = session.prefix()
        if not snapshot.complete_request_view or snapshot.omitted_requests:
            raise ScoreUnknown("incomplete_queue_view")
        # Conservative sufficient capacity region; free includes reclaimable
        # cache. This is NOT a second allocator or a proof for tight capacity.
        required = sum(
            max(0, (r.input_tokens + 15) // 16 - r.held_hbm_blocks)
            for r in snapshot.requests
        )
        if (
            required + snapshot.inflight_prefill_reserved_blocks
            > snapshot.free_hbm_blocks
        ):
            raise ScoreUnknown("simultaneous_full_input_capacity_unconfirmed")
        if any(
            r.restore_failed
            or r.status
            not in ("RUNNING", "WAITING", "PREEMPTED", "WAITING_FOR_REMOTE_KVS")
            or r.stale_output_tokens
            for r in snapshot.requests
        ):
            raise ScoreUnknown("failure_streaming_or_stale_state")
        cfg = self.costs["services"]
        host = scene["host_scale"]
        sched_ms = cfg["scheduler_step_ms"] * host
        launch = cfg["worker_host_ms"] * host
        sample = cfg["sample_copy_ms"] * host
        reply = cfg["result_return_ms"] * host
        delivery = cfg["output_delivery_ms"] * host
        publish = cfg["rpc_reply_ms"] * host
        rows = {r.request_id: r for r in snapshot.requests}
        caps = dict(spec.caps)
        if prefix is None:
            states = {}
            for rid, r in rows.items():
                check_time(deadline)
                f = features[rid]
                known = f is not None
                p = f.prefix_tokens if known else 0
                local = (
                    r.last_match.local_only_tokens
                    if r.last_match and r.last_match.local_only_tokens is not None
                    else r.last_match.local_tokens
                    if r.last_match
                    else 0
                )
                lower = r.lookup_result_tokens or 0
                if r.store_committed_or_saved_tokens is None and (
                    r.held_hbm_blocks or r.accounted_tokens
                ):
                    raise ScoreUnknown("native_STORE_coverage_unknown")
                states[rid] = dict(
                    p=p,
                    local=local,
                    lower=lower,
                    known=known,
                    query_known=r.lookup_result_known,
                    query_due=0.0
                    if r.lookup_result_known and not known
                    else (cfg["lookup_service_ms"] + cfg["status_rpc_ms"]) * host
                    if r.lookup_submitted and not known
                    else None,
                    reserved=r.restore_reserved,
                    load_due=None,
                    received=r.restore_completion_received,
                    held=bool(r.held_hbm_blocks or r.accounted_tokens),
                    saved=max(
                        lower // 256 * 256, r.store_committed_or_saved_tokens or 0
                    ),
                    release=None,
                    save_received=False,
                    done=None,
                    dep=0.0,
                    published=r.restore_completion_adopted,
                )
                if (
                    not known
                    and not r.lookup_submitted
                    and not snapshot.new_lookup_enabled
                ):
                    raise ScoreUnknown("unknown_query_has_no_submission_permission")
            gpu_end = copy_end = gpu_work = 0.0
            loads = stores = batches = 0
            in_flight = []
            mapped = set()
            for batch in snapshot.submitted_batches:
                check_time(deadline)
                duration = (
                    self.gpu(batch.shape, scene) * scene["initial_remaining_fraction"]
                )
                gpu_end += duration
                gpu_work += duration
                members = []
                for rid, p, q in batch.shape:
                    mapped.add(rid)
                    if rid in states:
                        states[rid]["dep"] = gpu_end
                        members.append((rid, p + q >= rows[rid].input_tokens))
                in_flight.append((gpu_end + sample + reply, members, (), (), False))
            if any(
                r.inflight_tokens and r.request_id not in mapped for r in rows.values()
            ):
                raise ScoreUnknown("unmapped_submitted_batch")
            # Existing STOREs and retired owners remain on the copy/resource tail.
            # Each obligation identity is counted once, never once per reference.
            seen = set()
            for op in snapshot.operations:
                check_time(deadline)
                if op.operation_id in seen or op.kind != "store":
                    continue
                seen.add(op.operation_id)
                if op.stage in ("obligation_ended", "completion_received"):
                    continue
                if op.start_token is None or op.end_token is None:
                    raise ScoreUnknown("store_range_unknown")
                ms, size = self.transfer("STORE", op.start_token, op.end_token, scene)
                copy_end = (
                    max(copy_end, gpu_end) + ms * scene["initial_remaining_fraction"]
                )
                stores += size
            retired = {
                o.request_id for o in snapshot.operations if o.request_id not in rows
            }
            retired_due = copy_end + publish if seen else 0.0
            retired_observed = not seen
            for rid, r in rows.items():
                s = states[rid]
                if r.restore_reserved:
                    p = (
                        r.last_match.matched_tokens
                        if r.last_match
                        else r.accounted_tokens
                    )
                    start = (s["local"] // 256) * 256
                    end = (p // 256) * 256
                    ms, size = self.transfer("LOAD", start, end, scene)
                    copy_end = (
                        max(copy_end, gpu_end)
                        + ms * scene["initial_remaining_fraction"]
                    )
                    loads += size
                    s["load_due"] = copy_end + publish
            for _, members, _, _, _ in in_flight:
                for rid, finish in members:
                    if finish and rid in states:
                        states[rid]["release"] = max(
                            copy_end + publish, gpu_end + sample + reply
                        )
            t = delay_ms
            first = True
            preparation_cursor = 0
            completed = {}
            # Prior core outputs have already been enqueued by _process_engine_step
            # before the next schedule. Their external delivery is a common uncertain
            # tail, not deleted background or newly rescheduled GPU work.
            common_tails = {
                rid: delivery * scene.get("output_remaining_fraction", 1.0)
                for rid in retired
            }
            defer = {r: (leader, p) for r, leader, p in spec.deferrals}
            # A path proposal that receives no first-step allocation is not committed.
            # Its next step must return to the ordinary restore path, not silently
            # retain a hypothetical recompute choice.
            uncommitted_recompute = {
                rid: rows[rid].last_match.matched_tokens
                for rid in spec.recompute
                if not states[rid]["held"]
            }
            area = 0.0
        else:
            states = prefix["states"]
            gpu_end = prefix["gpu_end"]
            copy_end = prefix["copy_end"]
            gpu_work = prefix["gpu_work"]
            loads = prefix["loads"]
            stores = prefix["stores"]
            batches = prefix["batches"]
            in_flight = prefix["in_flight"]
            retired_due = prefix["retired_due"]
            retired_observed = prefix["retired_observed"]
            t = prefix["t"]
            first = prefix["first"]
            preparation_cursor = prefix["preparation_cursor"]
            completed = prefix["completed"]
            common_tails = prefix["common_tails"]
            defer = prefix["defer"]
            uncommitted_recompute = prefix["uncommitted_recompute"]
            area = prefix["area"]
            before = prefix["before"]
            pending_loads = prefix["pending_loads"]
            held = prefix["held"]
        shape = ()
        for iteration in range(4096):
            check_time(deadline)
            if session is not None and not first and shape and iteration <= 2:
                cached = session.boundary(locals())
                check_time(deadline)
                if cached is not None:
                    return session.finish(cached)
            if not (first and prefix is not None):
                before = t
                preparing = [rid for rid, s in states.items() if not s["known"]]
                if preparing:
                    pos = preparation_cursor % len(preparing)
                    visit = set(
                        (preparing[pos:] + preparing[:pos])[
                            : snapshot.control_attempt_limit
                        ]
                    )
                    preparation_cursor += min(
                        len(preparing), snapshot.control_attempt_limit
                    )
                else:
                    visit = set()
                for rid, s in states.items():
                    check_time(deadline)
                    if (
                        not first
                        and rid in visit
                        and s["query_due"] is not None
                        and s["query_due"] <= t
                    ):
                        s["known"] = True
                        mode = scene["unknown_match"]
                        s["local"] = (
                            (rows[rid].input_tokens - 1) // 16 * 16
                            if mode == "hbm_hot"
                            else 0
                        )
                        if not s["query_known"]:
                            s["lower"] = (
                                (rows[rid].input_tokens - 1) // 256 * 256
                                if mode == "l1_hot"
                                else 0
                            )
                        s["p"] = min(
                            rows[rid].input_tokens - 1, max(s["local"], s["lower"])
                        )
                        s["saved"] = max(s["saved"], s["lower"] // 256 * 256)
                    if s["reserved"] and s["received"]:
                        s["reserved"] = False
                        s["published"] = True
                    if s["save_received"] and s["done"] is not None:
                        s["held"] = False
                if (
                    len(completed) == len(states)
                    and not any(s["held"] for s in states.values())
                    and retired_observed
                ):
                    drain = max(
                        [t, copy_end, *completed.values(), *common_tails.values()]
                    )
                    area += max(0, drain - t) * sum(
                        (s["p"] + 15) // 16 for s in states.values() if s["held"]
                    )
                    result = Score(
                        tuple(sorted((completed | common_tails).items())),
                        gpu_work,
                        loads,
                        stores,
                        drain,
                        area,
                        batches,
                        scene["name"],
                    )
                    return session.finish(result) if session else result
                t += sched_ms
                query_count = load_count = 0
                pending_loads = []
                held = sum(s["held"] for s in states.values())
                control_order = list(rows)
                for leader in dict.fromkeys(leader for leader, p in defer.values()):
                    positions = [
                        control_order.index(r)
                        for r, (owner, p) in defer.items()
                        if owner == leader
                    ]
                    at = min([control_order.index(leader), *positions])
                    control_order.remove(leader)
                    control_order.insert(at, leader)
                for rid in control_order:
                    check_time(deadline)
                    s = states[rid]
                    if s["done"] is not None:
                        continue
                    if not s["known"]:
                        if (
                            s["query_due"] is None
                            and query_count < snapshot.lookup_submission_limit
                        ):
                            s["query_due"] = (
                                t
                                + (cfg["lookup_service_ms"] + cfg["status_rpc_ms"])
                                * host
                            )
                            query_count += 1
                        continue
                    if rid in defer:
                        leader, common = defer[rid]
                        if states[leader]["published"]:
                            if scene["published_cache_survives"]:
                                s["local"] = max(s["local"], common)
                                s["p"] = max(s["p"], common)
                            del defer[rid]
                        elif states[leader]["reserved"] or leader in pending_loads:
                            continue
                    if first and rid in spec.recompute and not s["held"]:
                        s["p"] = s["local"]
                        continue
                    if (
                        not s["held"]
                        and s["p"] > s["local"]
                        and load_count < snapshot.load_attempt_limit
                        and held < snapshot.max_num_seqs
                    ):
                        pending_loads.append(rid)
                        load_count += 1
                        held += 1
                        s["held"] = True
                        s["reserved"] = True
                if session is not None and first:
                    session.save_prefix(locals())
            if self.cache is not None:
                self.cache.stats["rollout_nodes"] += 1
            order = self.order(snapshot, spec if first else CoordinationSpec(), states)
            budget = snapshot.token_budget
            shape = []
            stop = False
            for rid in order:
                check_time(deadline)
                s = states[rid]
                if stop or budget <= 0:
                    break
                if (
                    not s["known"]
                    or s["reserved"]
                    or s["done"] is not None
                    or s["p"] >= rows[rid].input_tokens
                ):
                    continue
                if rid in defer and states[defer[rid][0]]["reserved"]:
                    continue
                if not s["held"] and held >= snapshot.max_num_seqs:
                    continue
                if not s["held"]:
                    s["held"] = True
                    held += 1
                q = min(
                    budget,
                    rows[rid].input_tokens - s["p"],
                    caps.get(rid, budget) if first else budget,
                )
                shape.append((rid, s["p"], q))
                s["p"] += q
                budget -= q
                stop = first and rid == spec.end_after
            if first:
                for rid, original in uncommitted_recompute.items():
                    if not states[rid]["held"]:
                        states[rid]["p"] = original
            start = max(t + launch, gpu_end)
            duration = self.gpu(shape, scene)
            if copy_end > start:
                duration *= scene["overlap_gpu_scale"]
            end = start + duration
            if shape:
                gpu_end = end
                gpu_work += duration
                batches += 1
            # Native LOAD producer is the sample-tail event of this worker step;
            # empty steps still wait behind preceding work on its GPU stream.
            producer = (end + sample) if shape else max(t + launch, gpu_end)
            for rid in pending_loads:
                s = states[rid]
                ms, size = self.transfer(
                    "LOAD", s["local"] // 256 * 256, s["p"] // 256 * 256, scene
                )
                copy_end = (
                    max(copy_end, producer, scene.get("source_ready_delay_ms", 0.0))
                    + ms
                )
                loads += size
                s["load_due"] = copy_end + publish
            for rid, p, q in shape:
                s = states[rid]
                upto = (p + q) // 256 * 256
                if upto > s["saved"]:
                    ms, size = self.transfer("STORE", s["saved"], upto, scene)
                    copy_end = max(copy_end, producer) + ms
                    stores += size
                    s["saved"] = upto
                if p + q >= rows[rid].input_tokens:
                    s["release"] = max(copy_end + publish, end + sample + reply)
            poll = end + sample if shape else t + launch
            received = tuple(
                rid
                for rid, s in states.items()
                if s["reserved"] and s["load_due"] is not None and s["load_due"] <= poll
            )
            saved = tuple(
                rid
                for rid, s in states.items()
                if s["release"] is not None and s["release"] <= poll
            )
            in_flight.append(
                (
                    (end + sample + reply) if shape else t + launch,
                    [(rid, p + q >= rows[rid].input_tokens) for rid, p, q in shape],
                    received,
                    saved,
                    retired_due <= poll,
                )
            )
            first = False
            t += launch
            # Frozen EngineCore fills its depth-2 pipeline before consuming one
            # oldest result. Ready futures are not adopted at arbitrary CPU times.
            if len(in_flight) >= 2 or not any(
                s["done"] is None for s in states.values()
            ):
                due, members, received, saved, retired_done = in_flight.pop(0)
                t = max(t, due)
                for rid, finish in members:
                    if finish and rid in states and states[rid]["done"] is None:
                        states[rid]["done"] = t + delivery
                        completed[rid] = t + delivery
                for rid in received:
                    states[rid]["received"] = True
                for rid in saved:
                    states[rid]["save_received"] = True
                retired_observed = retired_observed or retired_done
            if not shape:
                futures = [row[0] for row in in_flight if row[0] > t]
                futures += [
                    s[k]
                    for s in states.values()
                    for k in ("query_due", "load_due", "release")
                    if s[k] is not None and s[k] > t
                ]
                if (
                    not futures
                    and not in_flight
                    and any(s["done"] is None for s in states.values())
                ):
                    raise ScoreUnknown("no_progress_or_unmodeled_output")
                t = max(t + 1.0, min(futures)) if futures else t + 1.0
            area += max(0, t - before) * sum(
                (s["p"] + 15) // 16 for s in states.values() if s["held"]
            )
        raise ScoreUnknown("bounded_continuation_not_complete")
