# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

import vllm.v1.core.hotprefix_projection as projection_module
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hotprefix import (
    EvictionGroup,
    EvictionNode,
    ExactHotnessStore,
    HotnessRecord,
    HotPrefixBlockEvictionSelector,
    HotPrefixEvictionDeferred,
    HotPrefixNodeSnapshot,
    LocalHotPrefixTree,
)
from vllm.v1.core.hotprefix_projection import HotPrefixBlockProjection
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request

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


def test_projection_component_size_excludes_unrelated_selector_groups() -> None:
    selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    selector.update_groups(
        (
            EvictionGroup(
                nodes=(EvictionNode(b"unrelated", 1, 1),),
                block_ids=tuple(range(100, 120)),
                block_size=4,
            ),
        )
    )
    projection = HotPrefixBlockProjection(selector)
    path = (
        HotPrefixNodeSnapshot(
            prefix_id=b"target",
            full_prefix=(1, 2, 3, 4),
            segment=(1, 2, 3, 4),
            parent=None,
            children=(),
            record=HotnessRecord(frequency=3, clock=8, depth=1),
        ),
    )

    result = projection.reconcile(
        namespace=b"model\0tenant",
        cached_tokens=4,
        path=path,
        physical_block_ids=(10,),
        block_size=4,
        aging_epoch=0,
    )

    assert result.max_component_blocks == 1


def test_projection_discard_uses_block_reverse_index() -> None:
    selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    projection = HotPrefixBlockProjection(selector)
    path = (
        HotPrefixNodeSnapshot(
            prefix_id=b"target",
            full_prefix=(1, 2, 3, 4),
            segment=(1, 2, 3, 4),
            parent=None,
            children=(),
            record=HotnessRecord(frequency=3, clock=8, depth=1),
        ),
    )
    for index in range(32):
        projection.reconcile(
            namespace=f"namespace-{index}".encode(),
            cached_tokens=4,
            path=path,
            physical_block_ids=(100 + index,),
            block_size=4,
            aging_epoch=0,
        )

    result = projection.discard((110,))

    assert result.invalidated_signatures == 1
    assert result.signature_keys_examined == 1


def test_projection_off_mode_skips_discard_timing_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    projection = HotPrefixBlockProjection(selector, collect_work=False)
    path = (
        HotPrefixNodeSnapshot(
            prefix_id=b"target",
            full_prefix=(1, 2, 3, 4),
            segment=(1, 2, 3, 4),
            parent=None,
            children=(),
            record=HotnessRecord(frequency=3, clock=8, depth=1),
        ),
    )
    assert (
        projection.reconcile(
            namespace=b"namespace",
            cached_tokens=4,
            path=path,
            physical_block_ids=(10,),
            block_size=4,
            aging_epoch=0,
        )
        is None
    )

    def fail_timing() -> int:
        raise AssertionError("off mode must not time discard normalization")

    monkeypatch.setattr(projection_module.time, "monotonic_ns", fail_timing)

    assert projection.discard((10,)) is None


def test_overlap_merge_preserves_pending_and_terminal_group_state() -> None:
    def group(prefix_id: bytes, block_ids: tuple[int, ...]) -> EvictionGroup:
        return EvictionGroup(
            nodes=(EvictionNode(prefix_id, 1, 1),),
            block_ids=block_ids,
            block_size=4,
        )

    pending = HotPrefixBlockEvictionSelector()
    pending.update_groups((group(b"a", (1, 2)),))
    pending.mark_group_pending((1, 2))
    pending.update_groups((group(b"b", (2, 3)),))
    candidates = tuple(KVCacheBlock(block_id) for block_id in (1, 2, 3))

    with pytest.raises(HotPrefixEvictionDeferred):
        pending.select_blocks(candidates, 1)
    assert pending.deferred_groups() == ()

    terminal = HotPrefixBlockEvictionSelector()
    terminal.update_groups((group(b"a", (1, 2)),))
    terminal.mark_group_terminal((1, 2))
    terminal.update_groups((group(b"b", (2, 3)),))

    assert terminal.select_blocks(candidates, 2) == (1, 2)


def test_store_deferred_pending_terminal_preserves_multi_victim_order() -> None:
    selector = HotPrefixBlockEvictionSelector()
    selector.update_groups(
        (
            EvictionGroup(
                nodes=(EvictionNode(b"cold", 1, 0),),
                block_ids=(1, 2),
                block_size=4,
            ),
        )
    )
    selector.update_priorities({3: 100.0})
    candidates = tuple(KVCacheBlock(block_id) for block_id in (1, 2, 3))

    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks(candidates, 2)
    assert [group.block_ids for group in selector.deferred_groups()] == [(1, 2)]

    selector.mark_group_pending((1, 2))
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks(candidates, 2)
    assert selector.deferred_groups() == ()

    selector.mark_group_terminal((1, 2))
    assert selector.select_blocks(candidates, 2) == (1, 2)


def test_block_pool_evict_invalidates_projection_signature() -> None:
    init_none_hash(sha256)
    selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    projection = HotPrefixBlockProjection(selector)
    selector.set_discard_observer(projection.discard)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=4,
        eviction_selector=selector,
    )
    request = Request(
        "block-pool-discard",
        [1, 2, 3, 4],
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    block = pool.get_new_blocks(1)[0]
    pool.cache_full_blocks(request, [block], 0, 1, 4, 0)
    pool.free_blocks([block])
    path = (
        HotPrefixNodeSnapshot(
            prefix_id=b"target",
            full_prefix=(1, 2, 3, 4),
            segment=(1, 2, 3, 4),
            parent=None,
            children=(),
            record=HotnessRecord(frequency=3, clock=8, depth=1),
        ),
    )
    first = projection.reconcile(
        namespace=b"namespace",
        cached_tokens=4,
        path=path,
        physical_block_ids=(block.block_id,),
        block_size=4,
        aging_epoch=0,
    )
    assert first is not None and not first.skipped

    pool.evict_blocks({block.block_id})
    second = projection.reconcile(
        namespace=b"namespace",
        cached_tokens=4,
        path=path,
        physical_block_ids=(block.block_id,),
        block_size=4,
        aging_epoch=0,
    )

    assert second is not None and not second.skipped
    assert second.discard_calls == 1
    assert second.invalidated_signatures == 1


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
        for requested in range(1, min(3, len(candidates)) + 1):
            assert fast_selector.select_blocks(
                candidates, requested
            ) == slow.select_blocks(candidates, requested)
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
