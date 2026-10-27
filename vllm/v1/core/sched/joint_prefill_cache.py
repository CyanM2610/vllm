# SPDX-License-Identifier: Apache-2.0
"""Bounded scalar checkpoints; never an owner of runtime pages or futures."""

import io
import pickle
from collections import Counter, OrderedDict
from dataclasses import fields, is_dataclass, replace


def freeze(value):
    if is_dataclass(value):
        return tuple((f.name, freeze(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(freeze(v) for v in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not a predictive scalar: {type(value).__name__}")


class ScalarSizer(pickle.Pickler):
    """Size frozen observations without importing runtime classes or copying leaves."""

    def reducer_override(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return tuple, (tuple(value.__dict__.values()),)
        return NotImplemented


class EvaluationCache:
    """One aggregate LRU bound, including keys and values (serialized payload).

    Prefix reuse ends immediately before the first changed compute action.
    Suffix reuse requires identical absolute calendars, dependencies and owners;
    it never translates completion timestamps. Costs and full scores have
    different namespaces. Partial/timeout results are never cached.
    """

    def __init__(self, entries=128, max_bytes=8 * 1024 * 1024):
        if entries <= 0 or max_bytes <= 0:
            raise ValueError("positive cache bounds required")
        self.limit, self.max_bytes = entries, max_bytes
        self.data = OrderedDict()
        self.cheap = OrderedDict()
        self.stats = Counter()
        self.bytes = 0
        self.model = None
        self.sequence = 0

    def bind_model(self, identity):
        key = freeze(identity)
        if self.model != key:
            self.stats["model_invalidations"] += bool(self.data)
            self.data.clear()
            self.cheap.clear()
            self.bytes = 0
            self.model = key

    def get(self, kind, key):
        pair = (kind, key)
        entry = self.data.get(pair)
        self.stats[kind + ("_hits" if entry is not None else "_misses")] += 1
        if entry is None:
            return None
        self.data.move_to_end(pair)
        if pair in self.cheap:
            self.cheap.move_to_end(pair)
        return entry[0]

    def put(self, kind, key, value):
        pair = (kind, key)
        # C-level serialization avoids a recursive Python heap walk on every
        # rollout boundary. This is a payload bound, not claimed process RSS;
        # offline tracemalloc separately measures actual Python allocation.
        stream = io.BytesIO()
        ScalarSizer(stream, protocol=5).dump((pair, value))
        size = stream.tell()
        if size > self.max_bytes:
            self.stats["oversized"] += 1
            return
        if pair in self.data:
            self.bytes -= self.data.pop(pair)[1]
            self.cheap.pop(pair, None)
        while self.data and (
            len(self.data) >= self.limit or self.bytes + size > self.max_bytes
        ):
            # Prefer dropping cheap costs/suffixes to discarding the contexts
            # and first-action checkpoints reused across the eight scenarios.
            victim = (
                self.cheap.popitem(last=False)[0]
                if self.cheap
                else next(iter(self.data))
            )
            self.bytes -= self.data.pop(victim)[1]
            self.stats["evictions"] += 1
        self.data[pair] = (value, size)
        if kind in ("gpu", "transfer", "suffix"):
            self.cheap[pair] = None
        self.bytes += size
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.bytes)


# Explicit allowlist. No rows/Request, model, callback, allocator or future.
CHECKPOINT_FIELDS = (
    "states",
    "gpu_end",
    "copy_end",
    "gpu_work",
    "loads",
    "stores",
    "batches",
    "in_flight",
    "retired_due",
    "retired_observed",
    "t",
    "first",
    "preparation_cursor",
    "completed",
    "common_tails",
    "defer",
    "uncommitted_recompute",
    "area",
    "before",
    "pending_loads",
    "held",
)
ACCUMULATORS = ("gpu_work", "loads", "stores", "batches", "area")


def pack(values):
    result = {k: values[k] for k in CHECKPOINT_FIELDS if k in values}
    # All producers are the scalar interpreter below. Copy the mutable leaves,
    # not the immutable snapshot/model on every batch boundary.
    for key, value in tuple(result.items()):
        if key == "states":
            result[key] = {rid: dict(state) for rid, state in value.items()}
        elif isinstance(value, dict):
            result[key] = dict(value)
        elif key == "in_flight":
            result[key] = [
                (due, tuple(members), received, saved, retired)
                for due, members, received, saved, retired in value
            ]
        elif isinstance(value, list):
            result[key] = list(value)
    return result


class ScoreSession:
    def __init__(self, cache, snapshot, spec, features, scene, delay_ms):
        self.cache = cache
        # The public snapshot/estimates are frozen scalar dataclasses. Reuse
        # them directly as keys; recursively converting every request for each
        # candidate would itself recreate the expensive full-state traversal.
        context = (snapshot, tuple(sorted(features.items())), freeze(scene), delay_ms)
        self.common = cache.get("context", context)
        if self.common is None:
            cache.sequence += 1
            self.common = cache.sequence
            cache.put("context", context, self.common)
        self.full_key = (self.common, freeze(spec))
        self.prefix_key = (self.common, spec.recompute, spec.deferrals)
        self.boundaries = []

    def full(self):
        return self.cache.get("full", self.full_key)

    def prefix(self):
        saved = self.cache.get("prefix", self.prefix_key)
        return pack(saved) if saved is not None else None

    def save_prefix(self, values):
        self.cache.put("prefix", self.prefix_key, pack(values))

    def boundary(self, values):
        state = pack(values)
        totals = tuple(state.pop(k) for k in ACCUMULATORS)
        # First-step-specific caps/end are gone here. Exact absolute time and
        # dependency state are required before reusing ANY subsequent event.
        # Interpreter field/row insertion order is fixed. Avoid recursively
        # converting the same field names on every continuation node.
        state["states"] = tuple(
            (rid, tuple(s.values())) for rid, s in state["states"].items()
        )
        for k in ("completed", "common_tails", "defer", "uncommitted_recompute"):
            state[k] = tuple(state[k].items())
        state["in_flight"] = tuple(state["in_flight"])
        state["pending_loads"] = tuple(state.get("pending_loads", ()))
        key = (self.common, tuple(state.items()))
        saved = self.cache.get("suffix", key)
        if saved is not None:
            score, original = saved
            d = [x - y for x, y in zip(totals, original, strict=True)]
            return replace(
                score,
                gpu_ms=score.gpu_ms + d[0],
                load_bytes=score.load_bytes + d[1],
                store_bytes=score.store_bytes + d[2],
                batches=score.batches + d[3],
                owner_block_ms_upper=score.owner_block_ms_upper + d[4],
            )
        if len(self.boundaries) < 32:
            self.boundaries.append((key, totals))
        return None

    def finish(self, score):
        self.cache.put("full", self.full_key, score)
        for key, totals in self.boundaries:
            self.cache.put("suffix", key, (score, totals))
        return score
