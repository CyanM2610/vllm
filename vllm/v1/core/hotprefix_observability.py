# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead HotPrefix observations collected as scheduler deltas."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Protocol


class HotPrefixEventKind(str, Enum):
    """Stable aggregate event families."""

    DECISION = "decision"
    CPU = "cpu"
    CONTROL_RPC = "control_rpc"
    PROMOTION = "promotion"
    HBM_BLOCKS = "hbm_blocks"
    DIVERGENCE = "divergence"


class HotPrefixStage(str, Enum):
    """Stable HotPrefix execution stages."""

    LOCAL_TREE = "local_tree"
    PROJECTION = "projection"
    EVICTION = "eviction"
    CONTROL = "control"
    STORE = "store"
    PROMOTION = "promotion"


class HotPrefixAction(str, Enum):
    """Stable actions used by decision and lifecycle observations."""

    NONE = "none"
    OBSERVE = "observe"
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    DEDUP = "dedup"
    RESERVE = "reserve"
    RELEASE = "release"
    PLAN = "plan"
    COPY = "copy"
    PUBLISH = "publish"
    FAIL = "fail"
    COALESCE = "coalesce"


class HotPrefixOutcome(str, Enum):
    """Stable terminal outcomes."""

    NONE = "none"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class HotPrefixReason(str, Enum):
    """Bounded reasons safe for Prometheus labels."""

    NONE = "none"
    HEADROOM = "headroom"
    UNALIGNED = "unaligned"
    NOT_HOTTER = "not_hotter"
    SOURCE_MISSING = "source_missing"
    RPC_UNHEALTHY = "rpc_unhealthy"
    CAPACITY = "capacity"
    CONFLICT = "conflict"
    DISABLED = "disabled"
    INFLIGHT = "inflight"
    INVALID = "invalid"
    FREQUENCY = "frequency"


@dataclass(frozen=True)
class HotPrefixObservation:
    """One immutable observation emitted by a HotPrefix producer."""

    kind: HotPrefixEventKind
    stage: HotPrefixStage
    action: HotPrefixAction = HotPrefixAction.NONE
    outcome: HotPrefixOutcome = HotPrefixOutcome.NONE
    reason: HotPrefixReason = HotPrefixReason.NONE
    duration_ns: int = 0
    tokens: int = 0
    blocks: int = 0
    bytes: int = 0
    request_id: str | None = None
    prefix_digest: str | None = None
    free_blocks_before: int | None = None
    free_blocks_after: int | None = None
    local_frequency: int | None = None
    local_clock: int | None = None

    def __post_init__(self) -> None:
        for name in ("duration_ns", "tokens", "blocks", "bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "free_blocks_before",
            "free_blocks_after",
            "local_frequency",
            "local_clock",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when set")


@dataclass(frozen=True)
class HotPrefixStatEntry:
    """One aggregate label tuple and its interval values."""

    kind: HotPrefixEventKind
    stage: HotPrefixStage
    action: HotPrefixAction
    outcome: HotPrefixOutcome
    reason: HotPrefixReason
    count: int
    durations_ns: tuple[int, ...]
    tokens: int
    blocks: int
    bytes: int
    free_blocks_before: int | None
    free_blocks_after: int | None


@dataclass(frozen=True)
class HotPrefixStats:
    """Immutable interval delta returned by a collector drain."""

    entries: tuple[HotPrefixStatEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether the interval contains no observations."""
        return not self.entries

    @property
    def work(self) -> dict[str, int]:
        """Return aggregate work totals across all entries."""
        return {
            "tokens": sum(entry.tokens for entry in self.entries),
            "blocks": sum(entry.blocks for entry in self.entries),
            "bytes": sum(entry.bytes for entry in self.entries),
        }

    def decision_count(
        self,
        stage: HotPrefixStage,
        action: HotPrefixAction,
        reason: HotPrefixReason,
    ) -> int:
        """Return a decision count for one fixed label tuple."""
        return sum(
            entry.count
            for entry in self.entries
            if entry.kind is HotPrefixEventKind.DECISION
            and entry.stage is stage
            and entry.action is action
            and entry.reason is reason
        )

    def cpu_duration_ns(self, stage: HotPrefixStage) -> tuple[int, ...]:
        """Return all CPU duration observations for one stage."""
        return tuple(
            duration
            for entry in self.entries
            if entry.stage is stage
            for duration in entry.durations_ns
        )

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        """Return a serialization-safe scheduler payload."""
        return {
            "entries": [
                {
                    "kind": entry.kind.value,
                    "stage": entry.stage.value,
                    "action": entry.action.value,
                    "outcome": entry.outcome.value,
                    "reason": entry.reason.value,
                    "count": entry.count,
                    "durations_ns": list(entry.durations_ns),
                    "tokens": entry.tokens,
                    "blocks": entry.blocks,
                    "bytes": entry.bytes,
                    "free_blocks_before": entry.free_blocks_before,
                    "free_blocks_after": entry.free_blocks_after,
                }
                for entry in self.entries
            ]
        }


class HotPrefixObservationCollector(Protocol):
    """Observation seam used by HotPrefix producers."""

    def record(self, event: HotPrefixObservation) -> None:
        """Record one observation."""

    def drain(self) -> HotPrefixStats:
        """Return and reset interval deltas."""


@dataclass
class _MutableStatEntry:
    count: int = 0
    durations_ns: list[int] | None = None
    tokens: int = 0
    blocks: int = 0
    bytes: int = 0
    free_blocks_before: int | None = None
    free_blocks_after: int | None = None


_StatKey = tuple[
    HotPrefixEventKind,
    HotPrefixStage,
    HotPrefixAction,
    HotPrefixOutcome,
    HotPrefixReason,
]


class InMemoryHotPrefixObservationCollector:
    """Thread-safe in-memory adapter used by scheduler metrics and tests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[_StatKey, _MutableStatEntry] = defaultdict(
            _MutableStatEntry
        )
        self._trace_events = os.environ.get("HOTPREFIX_TRACE_EVENTS") == "1"
        self._run_id = os.environ.get("HOTPREFIX_RUN_ID", "")
        if self._trace_events:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("vllm.hotprefix")
        else:
            self._tracer = None

    def record(self, event: HotPrefixObservation) -> None:
        """Aggregate an event without retaining high-cardinality fields."""
        key = (event.kind, event.stage, event.action, event.outcome, event.reason)
        with self._lock:
            entry = self._entries[key]
            entry.count += 1
            if event.duration_ns:
                if entry.durations_ns is None:
                    entry.durations_ns = []
                entry.durations_ns.append(event.duration_ns)
            entry.tokens += event.tokens
            entry.blocks += event.blocks
            entry.bytes += event.bytes
            if event.free_blocks_before is not None:
                entry.free_blocks_before = event.free_blocks_before
            if event.free_blocks_after is not None:
                entry.free_blocks_after = event.free_blocks_after
        if self._tracer is not None:
            attributes: dict[str, str | int] = {
                "hotprefix.kind": event.kind.value,
                "hotprefix.stage": event.stage.value,
                "hotprefix.action": event.action.value,
                "hotprefix.outcome": event.outcome.value,
                "hotprefix.reason": event.reason.value,
                "hotprefix.tokens": event.tokens,
                "hotprefix.blocks": event.blocks,
                "hotprefix.bytes": event.bytes,
            }
            if self._run_id:
                attributes["hotprefix.run_id"] = self._run_id
            if event.request_id:
                attributes["hotprefix.request_id"] = event.request_id
            if event.prefix_digest:
                attributes["hotprefix.prefix_digest"] = event.prefix_digest
            span = self._tracer.start_span(
                f"hotprefix.{event.stage.value}.{event.kind.value}",
                attributes=attributes,
            )
            span.end()

    def drain(self) -> HotPrefixStats:
        """Return immutable interval deltas and atomically reset state."""
        with self._lock:
            entries = self._entries
            self._entries = defaultdict(_MutableStatEntry)
        result = tuple(
            HotPrefixStatEntry(
                kind=key[0],
                stage=key[1],
                action=key[2],
                outcome=key[3],
                reason=key[4],
                count=value.count,
                durations_ns=tuple(value.durations_ns or ()),
                tokens=value.tokens,
                blocks=value.blocks,
                bytes=value.bytes,
                free_blocks_before=value.free_blocks_before,
                free_blocks_after=value.free_blocks_after,
            )
            for key, value in sorted(
                entries.items(), key=lambda item: tuple(part.value for part in item[0])
            )
        )
        return HotPrefixStats(result)


class NoOpHotPrefixObservationCollector:
    """Allocation-free adapter used when HotPrefix is disabled."""

    def record(self, event: HotPrefixObservation) -> None:
        """Discard an observation."""
        del event

    def drain(self) -> HotPrefixStats:
        """Return an empty interval."""
        return HotPrefixStats()
