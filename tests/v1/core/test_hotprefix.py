# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import sys
import types

import pytest
import torch

if sys.platform == "win32":
    uvloop = types.ModuleType("uvloop")
    uvloop.__dict__["run"] = asyncio.run
    sys.modules.setdefault("uvloop", uvloop)

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hotprefix import (
    CuckooHotnessStore,
    CuckooInsertionError,
    EvictionGroup,
    EvictionNode,
    ExactHotnessStore,
    HotnessRecord,
    HotPrefixBlockEvictionSelector,
    LocalHotPrefixTree,
    PromotionManager,
    PromotionState,
    make_hotprefix_namespace,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def test_hotprefix_namespace_includes_revision_and_layout() -> None:
    common = {
        "model": "model-a",
        "group_specs": ("FullAttentionSpec(block_size=16,page_size=4096)",),
    }

    base = make_hotprefix_namespace(revision="r1", kv_layout="NHD", **common)
    assert base != make_hotprefix_namespace(revision="r2", kv_layout="NHD", **common)
    assert base != make_hotprefix_namespace(revision="r1", kv_layout="HND", **common)


def test_exact_hotness_store_saturates_and_ages_records() -> None:
    store = ExactHotnessStore(max_value=3, max_age=3)
    prefix_id = b"prefix-a"

    store.insert(prefix_id, depth=7)
    for _ in range(5):
        store.access(prefix_id)

    assert store.get(prefix_id) == HotnessRecord(frequency=3, clock=3, depth=3)

    store.age_all()
    store.age_all()
    store.age_all()
    store.age_all()

    assert store.get(prefix_id) == HotnessRecord(frequency=3, clock=0, depth=3)


def test_cuckoo_hotness_store_matches_exact_record_semantics() -> None:
    store = CuckooHotnessStore(
        num_buckets=8,
        slots_per_bucket=2,
        max_kicks=8,
        max_value=7,
        max_age=7,
    )

    assert store.insert(b"prefix-a", depth=2) == HotnessRecord(1, 7, 2)
    assert store.access(b"prefix-a") == HotnessRecord(2, 7, 2)

    store.age_all()

    assert store.get(b"prefix-a") == HotnessRecord(2, 6, 2)


def test_cuckoo_insertion_failure_preserves_existing_records() -> None:
    store = CuckooHotnessStore(
        num_buckets=1,
        slots_per_bucket=1,
        max_kicks=1,
    )
    original = store.insert(b"prefix-a", depth=1)

    with pytest.raises(CuckooInsertionError):
        store.insert(b"prefix-b", depth=1)

    assert store.get(b"prefix-a") == original


def test_local_tree_splits_history_and_initializes_a_new_branch() -> None:
    tree = LocalHotPrefixTree(
        hotness_store=ExactHotnessStore(),
        namespace=b"model-a",
        aging_interval=100,
    )
    tree.publish((1, 2, 3, 4))

    matched = tree.record_match((1, 2, 9, 10), matched_tokens=2)
    tree.publish((1, 2, 9, 10))

    by_prefix = {node.full_prefix: node for node in tree.snapshot()}
    assert [node.full_prefix for node in matched] == [(1, 2)]
    assert by_prefix[(1, 2)].record == HotnessRecord(2, 255, 1)
    assert by_prefix[(1, 2, 3, 4)].record == HotnessRecord(1, 255, 2)
    assert by_prefix[(1, 2, 9, 10)].record == HotnessRecord(1, 255, 2)
    assert by_prefix[(1, 2)].children == (
        by_prefix[(1, 2, 3, 4)].prefix_id,
        by_prefix[(1, 2, 9, 10)].prefix_id,
    )


def test_eviction_group_uses_reclaimable_length_and_hottest_affected_node() -> None:
    group = EvictionGroup(
        nodes=(
            EvictionNode(b"cold-leaf", frequency=2, clock=16),
            EvictionNode(b"hot-parent", frequency=7, clock=8),
        ),
        block_ids=(10, 11),
        block_size=16,
    )

    assert group.reclaimable_length == 32
    assert group.priority == pytest.approx(7.25)


class _ReverseEvictionSelector:
    def select_blocks(
        self, candidates: tuple[KVCacheBlock, ...], num_blocks: int
    ) -> tuple[int, ...]:
        return tuple(block.block_id for block in reversed(candidates))[:num_blocks]


class _InvalidEvictionSelector:
    def select_blocks(
        self, candidates: tuple[KVCacheBlock, ...], num_blocks: int
    ) -> tuple[int, ...]:
        return (10_000,) * num_blocks


def test_block_pool_uses_configured_selector_for_cached_free_blocks() -> None:
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
        eviction_selector=_ReverseEvictionSelector(),
    )
    allocated = pool.get_new_blocks(3)
    for index, block in enumerate(allocated):
        block.set_block_hash(make_block_hash_with_group_id(bytes([index]), 0), 16)
    pool.free_blocks(allocated)

    selected = pool.get_new_blocks(1)

    assert selected[0].block_id == allocated[-1].block_id


def test_block_pool_restores_uncached_blocks_when_selector_is_invalid() -> None:
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=4,
        eviction_selector=_InvalidEvictionSelector(),
    )
    cached = pool.get_new_blocks(2)
    for index, block in enumerate(cached):
        block.set_block_hash(make_block_hash_with_group_id(bytes([index]), 0), 4)
    pool.free_blocks(cached)

    with pytest.raises(ValueError, match="outside the free queue"):
        pool.get_new_blocks(3)

    assert pool.get_num_free_blocks() == 4


def test_overlapping_logical_nodes_form_one_physical_eviction_group() -> None:
    selector = HotPrefixBlockEvictionSelector()
    selector.update_groups(
        (
            EvictionGroup((EvictionNode(b"node-a", 1, 4),), (1, 2), 4),
            EvictionGroup((EvictionNode(b"node-b", 2, 4),), (2, 3), 4),
        )
    )

    assert selector.collateral_block_ids((1,)) == (2, 3)
    selector.age_all()
    assert {node.clock for node in selector.eviction_groups()[0].nodes} == {3}


def test_block_pool_invalidates_collateral_hotprefix_blocks() -> None:
    selector = HotPrefixBlockEvictionSelector()
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=4,
        eviction_selector=selector,
    )
    allocated = pool.get_new_blocks(4)
    for index, block in enumerate(allocated):
        block.set_block_hash(make_block_hash_with_group_id(bytes([index]), 0), 4)
    pool.free_blocks(allocated)
    grouped_ids = tuple(block.block_id for block in allocated[:2])
    selector.update_groups(
        (EvictionGroup((EvictionNode(b"node", 1, 0),), grouped_ids, 4),)
    )
    selector.mark_group_terminal(grouped_ids)
    selector.update_priorities({block.block_id: 100.0 for block in allocated[2:]})

    selected = pool.get_new_blocks(1)

    assert selected[0].block_id in grouped_ids
    collateral_id = next(
        block_id for block_id in grouped_ids if block_id != selected[0].block_id
    )
    assert pool.blocks[collateral_id].block_hash is None
    assert pool.blocks[collateral_id].ref_cnt == 0


def test_promotion_is_published_only_after_all_budgeted_chunks_complete() -> None:
    manager = PromotionManager(max_inflight=1)
    transaction = manager.start(
        prefix_id=b"prefix-a",
        total_bytes=300,
        target_block_ids=(10, 11),
    )

    assert manager.coalesce("request-a", b"prefix-a") is True
    assert manager.advance(b"prefix-a", budget_bytes=128) == 128
    assert transaction.state is PromotionState.COPYING
    assert transaction.published is False

    assert manager.advance(b"prefix-a", budget_bytes=256) == 172
    assert transaction.state is PromotionState.READY
    assert transaction.published is True
    assert manager.take_waiters(b"prefix-a") == ("request-a",)


def test_pending_promotion_does_not_coalesce_requests() -> None:
    manager = PromotionManager(max_inflight=1)
    transaction = manager.plan(
        prefix_id=b"prefix-a",
        total_bytes=100,
        target_block_ids=(10,),
    )

    assert transaction.state is PromotionState.PENDING
    assert manager.coalesce("request-a", b"prefix-a") is False


def test_kv_cache_manager_enables_hotprefix_without_changing_default_lru() -> None:
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    hotprefix = KVCacheManager(
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[group],
            prefix_cache_eviction_policy="hotprefix",
            hotprefix_num_buckets=8,
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )
    lru = KVCacheManager(
        KVCacheConfig(8, [], [group]),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )

    assert hotprefix.hotprefix_tree is not None
    assert hotprefix.block_pool.eviction_selector is not None
    assert lru.hotprefix_tree is None
    assert lru.block_pool.eviction_selector is None


def test_hotprefix_promotion_keeps_one_transfer_worth_of_decode_headroom() -> None:
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=6,
            kv_cache_tensors=[],
            kv_cache_groups=[group],
            prefix_cache_eviction_policy="hotprefix",
            hotprefix_num_buckets=8,
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )
    occupied = manager.block_pool.get_new_blocks(3)
    page_size = group.kv_cache_spec.page_size_bytes

    assert (
        manager.reserve_hotprefix_promotion(
            prefix_id=b"starvation-guard",
            token_ids=range(8),
            total_bytes=2 * page_size,
            min_free_blocks=0,
        )
        is None
    )

    manager.block_pool.free_blocks(occupied)
    assert (
        manager.reserve_hotprefix_promotion(
            prefix_id=b"starvation-guard",
            token_ids=range(8),
            total_bytes=2 * page_size,
            min_free_blocks=0,
        )
        is not None
    )


def test_local_hotprefix_namespaces_match_lmcache_and_isolate_cache_salts() -> None:
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[group],
            prefix_cache_eviction_policy="hotprefix",
            hotprefix_num_buckets=8,
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
        hotprefix_namespace=b"model-a\0",
    )
    tenant_a = manager.get_hotprefix_tree("tenant-a")
    tenant_b = manager.get_hotprefix_tree("tenant-b")
    assert tenant_a is not None and tenant_b is not None

    tenant_a.publish((1, 2, 3, 4))
    tenant_b.publish((1, 2, 3, 4))
    prefix_a = tenant_a.snapshot()[0].prefix_id
    prefix_b = tenant_b.snapshot()[0].prefix_id

    expected = hashlib.blake2b(digest_size=16)
    namespace = b"model-a\0tenant-a"
    expected.update(len(namespace).to_bytes(4, "little"))
    expected.update(namespace)
    for token_id in (1, 2, 3, 4):
        expected.update(token_id.to_bytes(8, "little", signed=False))

    assert prefix_a == expected.digest()
    assert prefix_a != prefix_b


def test_kv_cache_manager_updates_hotprefix_from_native_apc_hits() -> None:
    init_none_hash(sha256)
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[group],
            prefix_cache_eviction_policy="hotprefix",
            hotprefix_num_buckets=8,
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )
    tokens = [1, 2, 3, 4, 5]
    first = Request(
        "first",
        tokens,
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    assert manager.allocate_slots(first, num_new_tokens=len(tokens)) is not None
    manager.free(first)
    second = Request(
        "second",
        tokens,
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(4, sha256),
    )

    _, matched_tokens, _ = manager.get_computed_blocks(second)

    assert matched_tokens == 4
    assert manager.hotprefix_tree is not None
    snapshot = manager.hotprefix_tree.snapshot()
    assert snapshot[0].full_prefix == (1, 2, 3, 4)
    assert snapshot[0].record.frequency == 2

    manager.get_computed_blocks(second)
    assert manager.hotprefix_tree.snapshot()[0].record.frequency == 2


def test_kv_cache_manager_marks_only_overflow_repetition_invalid() -> None:
    init_none_hash(sha256)
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=7,
            kv_cache_tensors=[],
            kv_cache_groups=[group],
            prefix_cache_eviction_policy="hotprefix",
            hotprefix_num_buckets=1,
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )

    for index in range(5):
        request = Request(
            f"request-{index}",
            [index + 1, 10, 11, 12],
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(4, sha256),
        )
        assert manager.allocate_slots(request, num_new_tokens=4) is not None
        manager.free(request)

    assert manager.hotprefix_tree is not None
    snapshots = manager.hotprefix_tree.snapshot()
    assert len(snapshots) == 5
    assert sum(not node.valid for node in snapshots) == 1
    assert manager.block_pool.eviction_selector is not None


def test_hotprefix_selector_keeps_frequently_reused_prefix_over_lru_order() -> None:
    init_none_hash(sha256)
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=3,
            kv_cache_tensors=[],
            kv_cache_groups=[group],
            prefix_cache_eviction_policy="hotprefix",
            hotprefix_num_buckets=8,
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )

    def request(request_id: str, tokens: list[int]) -> Request:
        return Request(
            request_id,
            tokens,
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(4, sha256),
        )

    hot_tokens = [1, 2, 3, 4, 9]
    hot = request("hot-store", hot_tokens)
    hot_blocks = manager.allocate_slots(hot, num_new_tokens=4)
    assert hot_blocks is not None
    hot_block_id = hot_blocks.get_block_ids()[0][0]
    manager.free(hot)
    for index in range(3):
        manager.get_computed_blocks(request(f"hot-hit-{index}", hot_tokens))

    cold_tokens = [5, 6, 7, 8, 9]
    cold = request("cold-store", cold_tokens)
    cold_blocks = manager.allocate_slots(cold, num_new_tokens=4)
    assert cold_blocks is not None
    cold_block_id = cold_blocks.get_block_ids()[0][0]
    manager.free(cold)

    assert manager.hotprefix_tree is not None
    by_prefix = {
        node.full_prefix: node.record for node in manager.hotprefix_tree.snapshot()
    }
    assert by_prefix[tuple(hot_tokens[:4])].frequency == 4
    assert by_prefix[tuple(cold_tokens[:4])].frequency == 1

    pressure = request("pressure", [9, 10, 11, 12, 13])
    assert manager.allocate_slots(pressure, num_new_tokens=4) is None
    reserved = manager.reserve_hotprefix_eviction_store(
        transfer_chunk_tokens=4,
        min_free_blocks=1,
    )
    assert reserved is not None
    assert reserved.token_ids == tuple(cold_tokens[:4])
    assert reserved.block_ids == (cold_block_id,)
    assert manager.block_pool.get_num_free_blocks() == 1
    manager.release_hotprefix_eviction_store(reserved)
    assert manager.block_pool.get_num_free_blocks() == 2

    selected_blocks = manager.allocate_slots(pressure, num_new_tokens=4)
    assert selected_blocks is not None
    selected = [selected_blocks.blocks[0][0]]

    assert selected[0].block_id == cold_block_id
    assert selected[0].block_id != hot_block_id
    promotion_nodes = manager.get_hotprefix_promotion_nodes()
    assert len(promotion_nodes) == 1
    assert [node.full_prefix for node in promotion_nodes[0][1]] == [
        tuple(cold_tokens[:4])
    ]

    manager.block_pool.evict_blocks({selected[0].block_id})
    manager.free(pressure)
    cold_node = promotion_nodes[0][1][0]
    transaction = manager.reserve_hotprefix_promotion(
        prefix_id=cold_node.prefix_id,
        token_ids=cold_node.full_prefix,
        total_bytes=group.kv_cache_spec.page_size_bytes,
        min_free_blocks=0,
    )
    assert transaction is not None
    promoted = request("cold-promotion", list(cold_node.full_prefix))
    assert manager.advance_hotprefix_promotion(
        cold_node.prefix_id,
        copied_bytes=transaction.total_bytes,
    )
    assert manager.publish_hotprefix_promotion(promoted, cold_node.prefix_id) == ()
    assert manager.hotprefix_promotion_manager is not None
    assert manager.hotprefix_promotion_manager.get(cold_node.prefix_id) is None

    _, promoted_hit_tokens, _ = manager.get_computed_blocks(
        request("cold-after-promotion", cold_tokens)
    )
    assert promoted_hit_tokens == 4
