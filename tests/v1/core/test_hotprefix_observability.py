# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import pytest

from vllm.v1.core.hotprefix_observability import (
    HotPrefixAction,
    HotPrefixEventKind,
    HotPrefixObservation,
    HotPrefixReason,
    HotPrefixStage,
    InMemoryHotPrefixObservationCollector,
    NoOpHotPrefixObservationCollector,
)

pytestmark = pytest.mark.cpu_test


def test_collector_drains_delta_and_resets() -> None:
    collector = InMemoryHotPrefixObservationCollector()
    event = HotPrefixObservation(
        kind=HotPrefixEventKind.DECISION,
        stage=HotPrefixStage.PROMOTION,
        action=HotPrefixAction.REJECT,
        reason=HotPrefixReason.HEADROOM,
        duration_ns=2_000_000,
        tokens=32,
        blocks=2,
        bytes=4096,
        tree_nodes=11,
        path_nodes=3,
        request_blocks=4,
        groups=2,
        component_blocks=3,
        discard_calls=1,
        signature_keys=2,
        invalidated_signatures=1,
        planning_steps=1,
        candidates=3,
        free_blocks_before=3,
        free_blocks_after=3,
    )

    collector.record(event)
    collector.record(event)
    first = collector.drain()
    second = collector.drain()

    assert (
        first.decision_count(
            HotPrefixStage.PROMOTION,
            HotPrefixAction.REJECT,
            HotPrefixReason.HEADROOM,
        )
        == 2
    )
    assert first.cpu_duration_ns(HotPrefixStage.PROMOTION) == (2_000_000, 2_000_000)
    assert first.work == {
        "tokens": 64,
        "blocks": 4,
        "bytes": 8192,
        "tree_nodes": 22,
        "path_nodes": 6,
        "request_blocks": 8,
        "groups": 4,
        "component_blocks": 6,
        "discard_calls": 2,
        "signature_keys": 4,
        "invalidated_signatures": 2,
        "planning_steps": 2,
        "candidates": 6,
    }
    assert second.is_empty


def test_event_is_immutable_and_rejects_negative_measurements() -> None:
    event = HotPrefixObservation(
        kind=HotPrefixEventKind.CPU,
        stage=HotPrefixStage.LOCAL_TREE,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.tokens = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="duration_ns"):
        HotPrefixObservation(
            kind=HotPrefixEventKind.CPU,
            stage=HotPrefixStage.LOCAL_TREE,
            duration_ns=-1,
        )


def test_noop_collector_never_accumulates() -> None:
    collector = NoOpHotPrefixObservationCollector()
    collector.record(
        HotPrefixObservation(
            kind=HotPrefixEventKind.DECISION,
            stage=HotPrefixStage.EVICTION,
            action=HotPrefixAction.ACCEPT,
        )
    )

    assert collector.drain().is_empty
