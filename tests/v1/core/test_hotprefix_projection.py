# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm.v1.core.hotprefix import (
    EvictionGroup,
    EvictionNode,
    ExactHotnessStore,
    HotnessRecord,
    HotPrefixBlockEvictionSelector,
    HotPrefixNodeSnapshot,
    LocalHotPrefixTree,
)
from vllm.v1.core.hotprefix_projection import HotPrefixBlockProjection
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def _slow_reconcile(
    selector: HotPrefixBlockEvictionSelector,
    tree: LocalHotPrefixTree,
    token_ids: tuple[int, ...],
    physical_block_ids: tuple[int, ...],
    block_size: int,
) -> None:
    path = [
        node
        for node in tree.snapshot()
        if node.valid
        and len(node.full_prefix) <= len(token_ids)
        and token_ids[: len(node.full_prefix)] == node.full_prefix
    ]
    groups: list[EvictionGroup] = []
    for node in path:
        node_end = len(node.full_prefix)
        node_start = node_end - len(node.segment)
        block_ids = tuple(
            block_id
            for index, block_id in enumerate(physical_block_ids)
            if index * block_size < node_end and (index + 1) * block_size > node_start
        )
        if not block_ids:
            continue
        groups.append(
            EvictionGroup(
                nodes=(
                    EvictionNode(
                        node.prefix_id,
                        node.record.frequency,
                        node.record.clock,
                        tree.namespace,
                        node.full_prefix,
                        physical_block_ids[: (node_end + block_size - 1) // block_size],
                    ),
                ),
                block_ids=block_ids,
                block_size=block_size,
            )
        )
    selector.update_groups(groups)


def _group_facts(
    selector: HotPrefixBlockEvictionSelector,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            group.block_ids,
            tuple(
                (
                    node.prefix_id,
                    node.frequency,
                    node.clock,
                    node.full_prefix,
                    node.full_block_ids,
                )
                for node in group.nodes
            ),
            group.reclaimable_length,
            group.priority,
        )
        for group in selector.eviction_groups()
    )


def test_identical_projection_signature_skips_group_rebuild() -> None:
    selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    projection = HotPrefixBlockProjection(selector)
    path = (
        HotPrefixNodeSnapshot(
            prefix_id=b"prefix",
            full_prefix=(1, 2, 3, 4),
            segment=(1, 2, 3, 4),
            parent=None,
            children=(),
            record=HotnessRecord(frequency=3, clock=8, depth=1),
        ),
    )

    first = projection.reconcile(
        namespace=b"model\0tenant",
        cached_tokens=4,
        path=path,
        physical_block_ids=(10,),
        block_size=4,
        aging_epoch=0,
    )
    second = projection.reconcile(
        namespace=b"model\0tenant",
        cached_tokens=4,
        path=path,
        physical_block_ids=(10,),
        block_size=4,
        aging_epoch=0,
    )

    assert first.skipped is False
    assert second.skipped is True
    assert [group.block_ids for group in selector.eviction_groups()] == [(10,)]


def test_phase_a_matches_slow_projection_across_state_changes() -> None:
    tree = LocalHotPrefixTree(
        hotness_store=ExactHotnessStore(),
        namespace=b"model\0tenant",
        aging_interval=2,
    )
    tree.publish((1, 2, 3, 4, 5, 6))
    tree.record_match((1, 2, 9, 10, 11, 12), matched_tokens=2)
    tree.publish((1, 2, 9, 10, 11, 12))
    slow = HotPrefixBlockEvictionSelector(defer_for_host=False)
    fast_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    fast = HotPrefixBlockProjection(fast_selector)

    def reconcile(token_ids: tuple[int, ...], block_ids: tuple[int, ...]) -> None:
        _slow_reconcile(slow, tree, token_ids, block_ids, 4)
        fast.reconcile(
            namespace=tree.namespace,
            cached_tokens=len(token_ids),
            path=tree.path_snapshot(token_ids),
            physical_block_ids=block_ids,
            block_size=4,
            aging_epoch=tree.aging_epoch,
            total_tree_nodes=tree.node_count,
        )
        assert _group_facts(fast_selector) == _group_facts(slow)
        candidates = tuple(KVCacheBlock(block_id) for block_id in sorted(block_ids))
        assert fast_selector.select_blocks(candidates, 1) == slow.select_blocks(
            candidates, 1
        )
        for block_id in block_ids:
            assert fast_selector.collateral_block_ids(
                (block_id,)
            ) == slow.collateral_block_ids((block_id,))

    reconcile((1, 2, 3, 4, 5, 6), (10, 11))
    reconcile((1, 2, 9, 10, 11, 12), (12, 13))
    tree.record_match((1, 2, 3, 4, 5, 6), matched_tokens=6)
    slow.age_all()
    fast.age()
    reconcile((1, 2, 3, 4, 5, 6), (10, 11))
    slow.discard((10, 11))
    fast_selector.discard((10, 11))
    reconcile((1, 2, 3, 4, 5, 6), (20, 21))


def test_phase_a_matches_slow_oracle_for_deterministic_random_trace() -> None:
    rng = random.Random(20260901)
    tree = LocalHotPrefixTree(
        hotness_store=ExactHotnessStore(),
        namespace=b"random-trace",
        aging_interval=5,
    )
    slow = HotPrefixBlockEvictionSelector(defer_for_host=False)
    fast_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    fast = HotPrefixBlockProjection(fast_selector)
    fast_selector.set_discard_observer(fast.discard)
    paths: list[tuple[int, ...]] = []
    bindings: dict[tuple[int, ...], tuple[int, ...]] = {}
    next_block = 1_000
    previous_aging_epoch = tree.aging_epoch
    path: tuple[int, ...]

    for step in range(60):
        if not paths or rng.random() < 0.35:
            branch = step + 1
            path = (1, 2, branch, branch + 100, branch + 200, branch + 300)
            tree.publish(path)
            paths.append(path)
            bindings[path] = (100, next_block)
            next_block += 1
        else:
            path = rng.choice(paths)
            if rng.random() < 0.75:
                tree.record_match(path, matched_tokens=len(path))
            else:
                old_binding = bindings[path]
                slow.discard((old_binding[-1],))
                fast_selector.discard((old_binding[-1],))
                bindings[path] = (100, next_block)
                next_block += 1
        if tree.aging_epoch != previous_aging_epoch:
            slow.age_all()
            fast.age()
            previous_aging_epoch = tree.aging_epoch

        binding = bindings[path]
        _slow_reconcile(slow, tree, path, binding, 4)
        fast.reconcile(
            namespace=tree.namespace,
            cached_tokens=len(path),
            path=tree.path_snapshot(path),
            physical_block_ids=binding,
            block_size=4,
            aging_epoch=tree.aging_epoch,
            total_tree_nodes=tree.node_count,
        )

        assert _group_facts(fast_selector) == _group_facts(slow)
        for block_id in binding:
            assert fast_selector.collateral_block_ids(
                (block_id,)
            ) == slow.collateral_block_ids((block_id,))
