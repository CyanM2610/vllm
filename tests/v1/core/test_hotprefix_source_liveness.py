# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full-prefix STORE sources can outlive their ancestor eviction group."""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.hotprefix import (
    EvictionGroup,
    EvictionNode,
    HotPrefixBlockEvictionSelector,
    HotPrefixEvictionDeferred,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def projected_prefix():
    init_none_hash(sha256)
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=4,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
            prefix_cache_eviction_policy="hotprefix",
        ),
        max_model_len=32,
        scheduler_block_size=4,
        hash_block_size=4,
    )
    tokens = tuple(range(1, 9))
    request = Request(
        "seed",
        list(tokens),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    allocation = manager.allocate_slots(request, num_new_tokens=8)
    assert allocation is not None
    parent, child = allocation.get_block_ids()[0]
    manager.free(request)
    # Model the physical projection after a radix split at token four.
    # Both source blocks carry real APC hashes from the same request.
    selector = HotPrefixBlockEvictionSelector()
    selector.update_groups(
        (
            EvictionGroup(
                (
                    EvictionNode(
                        b"parent",
                        0,
                        0,
                        b"namespace",
                        tokens[:4],
                        (parent,),
                    ),
                ),
                (parent,),
                4,
            ),
            EvictionGroup(
                (
                    EvictionNode(
                        b"child",
                        1,
                        0,
                        b"namespace",
                        tokens,
                        (parent, child),
                    ),
                ),
                (child,),
                4,
            ),
        )
    )
    manager.hotprefix_eviction_selector = selector
    manager.block_pool.eviction_selector = selector
    return manager, selector, parent, child


@pytest.mark.parametrize("reuse", [False, True])
def test_evicted_ancestor_cannot_become_a_descendant_store_source(
    projected_prefix,
    reuse,
):
    manager, selector, parent, child = projected_prefix
    replacement_hash = manager.block_pool.blocks[child].block_hash
    manager.block_pool.evict_blocks({parent})
    if reuse:
        # A new occupant with any non-None hash must not revive the old binding.
        assert replacement_hash is not None
        manager.block_pool.blocks[parent].set_block_hash(replacement_hash)
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    assert (
        manager.reserve_hotprefix_eviction_store(
            transfer_chunk_tokens=4,
            min_free_blocks=0,
        )
        is None
    )
    assert selector.select_blocks((manager.block_pool.blocks[child],), 1) == (child,)
    assert manager.block_pool.get_num_free_blocks() == 3


@pytest.mark.parametrize("unavailable", ["hash_missing", "unaligned"])
def test_unstorable_victims_have_a_terminal_reclaim_path(projected_prefix, unavailable):
    manager, selector, parent, child = projected_prefix
    if unavailable == "hash_missing":
        # Defensive case for existing stale snapshots / missing callback paths.
        manager.block_pool.blocks[parent].reset_hash()
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    assert (
        manager.reserve_hotprefix_eviction_store(
            transfer_chunk_tokens=16 if unavailable == "unaligned" else 4,
            min_free_blocks=0,
        )
        is None
    )
    assert selector.select_blocks((manager.block_pool.blocks[child],), 1) == (child,)


def test_invalidating_a_source_does_not_release_an_inflight_store(projected_prefix):
    manager, selector, parent, child = projected_prefix
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    candidate = manager.reserve_hotprefix_eviction_store(
        transfer_chunk_tokens=4,
        min_free_blocks=0,
    )
    assert candidate is not None
    assert candidate.block_ids == (parent, child)
    manager.block_pool.evict_blocks({parent})
    assert (
        manager.reserve_hotprefix_eviction_store(
            transfer_chunk_tokens=4,
            min_free_blocks=0,
        )
        is None
    )
    assert manager.block_pool.blocks[parent].ref_cnt == 1
    assert manager.block_pool.blocks[child].ref_cnt == 1
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    manager.release_hotprefix_eviction_store(candidate)
    assert manager.block_pool.get_num_free_blocks() == 3
    assert selector.select_blocks((manager.block_pool.blocks[child],), 1) == (child,)


def test_fresh_projection_can_replace_an_invalid_source_binding(projected_prefix):
    manager, selector, parent, child = projected_prefix
    manager.block_pool.evict_blocks({parent})
    node = selector.eviction_groups()[0].nodes[0]
    assert node.full_block_ids == ()
    selector.update_groups(
        (
            EvictionGroup(
                (
                    EvictionNode(
                        b"child",
                        1,
                        0,
                        b"namespace",
                        (5, 6, 7, 8),
                        (child,),
                    ),
                ),
                (child,),
                4,
            ),
        )
    )
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    candidate = manager.reserve_hotprefix_eviction_store(
        transfer_chunk_tokens=4,
        min_free_blocks=0,
    )
    assert candidate is not None
    assert candidate.block_ids == (child,)
    manager.release_hotprefix_eviction_store(candidate)


def test_headroom_shortage_remains_retryable(projected_prefix):
    manager, selector, parent, child = projected_prefix
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    assert (
        manager.reserve_hotprefix_eviction_store(
            transfer_chunk_tokens=4,
            min_free_blocks=2,
        )
        is None
    )
    with pytest.raises(HotPrefixEvictionDeferred):
        selector.select_blocks((manager.block_pool.blocks[child],), 1)
    candidate = manager.reserve_hotprefix_eviction_store(
        transfer_chunk_tokens=4,
        min_free_blocks=0,
    )
    assert candidate is not None
    manager.release_hotprefix_eviction_store(candidate)


def test_moving_apc_hashes_invalidates_the_old_source(projected_prefix):
    manager, selector, parent, child = projected_prefix
    pool = manager.block_pool
    destination = next(
        block for block in pool.blocks if not block.is_null and block.block_hash is None
    )
    pool.move_block_hashes(pool.blocks[parent], destination)
    assert selector.eviction_groups()[0].nodes[0].full_block_ids == ()


def test_source_reverse_index_follows_overlapping_group_merges(projected_prefix):
    manager, selector, parent, child = projected_prefix
    selector.update_groups(
        (
            EvictionGroup(
                (
                    EvictionNode(
                        b"overlap",
                        2,
                        0,
                        b"namespace",
                        tuple(range(12)),
                        (parent, child, 3),
                    ),
                ),
                (child, 3),
                4,
            ),
        )
    )
    manager.block_pool.evict_blocks({parent})
    groups = selector.eviction_groups()
    assert len(groups) == 1
    assert groups[0].block_ids == (child, 3)
    assert all(node.full_block_ids == () for node in groups[0].nodes)


def test_reset_removes_all_projected_source_bindings(projected_prefix):
    manager, selector, parent, child = projected_prefix
    assert manager.block_pool.reset_prefix_cache()
    assert selector.eviction_groups() == ()
