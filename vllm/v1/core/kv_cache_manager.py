# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.hotprefix import (
    CuckooHotnessStore,
    EvictionGroup,
    EvictionNode,
    EvictionStoreCandidate,
    HotPrefixBlockEvictionSelector,
    HotPrefixEvictionDeferred,
    HotPrefixNodeSnapshot,
    LocalHotPrefixTree,
    PromotionManager,
    PromotionState,
    PromotionTransaction,
)
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        num_prefill_lookahead: int = 0,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
        hotprefix_namespace: bytes = b"vllm-local",
    ) -> None:
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        self.hotprefix_tree: LocalHotPrefixTree | None = None
        self.hotprefix_trees: dict[str, LocalHotPrefixTree] = {}
        self._hotprefix_hotness_store: CuckooHotnessStore | None = None
        self._hotprefix_namespace = hotprefix_namespace
        self._hotprefix_aging_interval = kv_cache_config.hotprefix_aging_interval
        self._hotprefix_reserved_stores: set[bytes] = set()
        self._hotprefix_observed_request_ids: set[str] = set()
        self.hotprefix_eviction_selector: HotPrefixBlockEvictionSelector | None = None
        self.hotprefix_promotion_manager: PromotionManager | None = None
        self._hotprefix_block_size = scheduler_block_size
        self.hotprefix_hash_block_size = hash_block_size
        if kv_cache_config.prefix_cache_eviction_policy == "hotprefix":
            if not enable_caching:
                raise ValueError("hotprefix eviction requires prefix caching")
            if len(kv_cache_config.kv_cache_groups) != 1:
                raise ValueError("hotprefix currently requires one KV cache group")
            self._hotprefix_hotness_store = CuckooHotnessStore(
                num_buckets=kv_cache_config.hotprefix_num_buckets
            )
            self.hotprefix_tree = self.get_hotprefix_tree("")
            self.hotprefix_eviction_selector = HotPrefixBlockEvictionSelector()
            self.hotprefix_promotion_manager = PromotionManager(max_inflight=1)
            self.block_pool.eviction_selector = self.hotprefix_eviction_selector
        elif kv_cache_config.prefix_cache_eviction_policy != "lru":
            raise ValueError(
                "prefix_cache_eviction_policy must be 'lru' or 'hotprefix'"
            )

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def hotprefix_block_size(self) -> int:
        """Return the physical scheduler block size used by HotPrefix."""
        return self._hotprefix_block_size

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        """Whether a local prefix cache lookup may be run for this request."""
        return self.enable_caching and not request.skip_reading_prefix_cache

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        # Don't count a request that skipped the cache lookup.
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
                - ``shared_prefix_boundary``: the block-aligned token position of
                  a shared prefix that a sparse-retention group (Mamba / sliding
                  window) has not cached yet (Marconi-style APC), or 0 if none.
                  Pinned so sparse prefix-cache retention does not drop
                  the junction and defeat cross-request reuse.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # When kv_cache_report_mode is "full", emit BlockStored events
        # for the reused prefix cache blocks so that external consumers
        # (e.g. gateway) can learn about them.
        if (
            num_new_computed_tokens > 0
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
            for group_idx, group_blocks in enumerate(computed_blocks):
                num_blocks = len(group_blocks)
                if num_blocks > 0:
                    group = self.kv_cache_config.kv_cache_groups[group_idx]
                    block_size = group.kv_cache_spec.block_size
                    self.block_pool.emit_cached_block_events(
                        request,
                        num_blocks,
                        block_size,
                        group_idx,
                    )

        # The junction to pin is where the lagging sparse-retention group stops
        # (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        # the longest single-group hit. Sub-block gaps are left to the mask,
        # which floors to the alignment boundary (a no-op there).
        shared_prefix_boundary = (
            num_new_computed_tokens + num_uncached if num_uncached else 0
        )

        blocks = self.create_kv_cache_blocks(computed_blocks)
        hotprefix_tree = self.get_hotprefix_tree(request.cache_salt)
        if (
            hotprefix_tree is not None
            and request.request_id not in self._hotprefix_observed_request_ids
        ):
            aging_epoch = hotprefix_tree.aging_epoch
            hotprefix_tree.record_match(
                request.all_token_ids,
                matched_tokens=num_new_computed_tokens,
            )
            if (
                hotprefix_tree.aging_epoch != aging_epoch
                and self.hotprefix_eviction_selector is not None
            ):
                self.hotprefix_eviction_selector.age_all()
            self._update_hotprefix_priorities(
                request.all_token_ids,
                computed_blocks,
                tree=hotprefix_tree,
            )
            self._hotprefix_observed_request_ids.add(request.request_id)
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    def get_computed_blocks_for_connector(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        """Local prefix-cache lookup for a request scheduled with a KV connector.

        Hybrid (Mamba + full-attention) models can have per-group prefix hits
        diverge under block pressure: the full-attention tail may be evicted
        while a deeper Mamba state survives, or vice versa. Report the
        full-attention hit as the local prefix - the connector transfers the
        remaining suffix and the Mamba state is transferred unconditionally by
        nixl's ``_apply_prefix_caching`` - and flag when that hit ran deeper
        than a lagging group. Such a hit only has a valid Mamba state at its
        boundary if the connector supplies it, so the caller must fall back to
        ``get_computed_blocks`` to reconcile when no external tokens are found.

        Non-hybrid models and already-convergent hits use ``get_computed_blocks``.

        Returns:
            The ``get_computed_blocks`` triple (blocks, number of local computed
            tokens, shared-prefix boundary) plus ``hit_diverged``.
        """
        coordinator = self.coordinator
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.full_attention_group_id is not None
        ):
            return *self.get_computed_blocks(request), False

        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0, False

        fa_group_id = coordinator.full_attention_group_id
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1
        )
        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
            # A lagging group hit deeper than full attention means its
            # full-attention blocks were evicted; use the reconciled boundary
            # that every group agrees on.
            return *self.get_computed_blocks(request), False

        num_local = per_group_hits[fa_group_id]
        blocks = self.create_kv_cache_blocks(computed)
        # Per-group lookups do not detect an uncached shared prefix (boundary 0).
        return blocks, num_local, 0, min(per_group_hits) < num_local

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Number of free blocks that must be left available for
                other in-flight sequences to complete. The actual allocation is only
                made if it fits within (free blocks - reserved_blocks). Used to gate
                async KV-connector loads so their initial allocation cannot consume
                blocks an already in-flight (prefilling) sequence is relying on.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        try:
            self.block_pool.ensure_allocation_ready(num_blocks_to_allocate)
        except HotPrefixEvictionDeferred:
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        try:
            new_blocks = self.coordinator.allocate_new_blocks(
                request.request_id,
                num_tokens_need_slot,
                num_tokens_main_model,
                num_encoder_tokens,
            )
        except HotPrefixEvictionDeferred:
            return None

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)
        if request.is_finished():
            self._hotprefix_observed_request_ids.discard(request.request_id)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for R-SWA gap eviction.
        """
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them in reverse order (so that tail blocks are evicted first).

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in allocation order.
        """
        return self.coordinator.pop_blocks_for_free(request.request_id)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        self._hotprefix_observed_request_ids.clear()
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def estimate_cached_tokens(self, request: Request) -> int:
        """Estimate the number of tokens cached by the request."""
        cached_tokens: int | None = None
        for group, blocks in zip(
            self.kv_cache_config.kv_cache_groups,
            self.get_blocks(request.request_id).blocks,
        ):
            if isinstance(
                group.kv_cache_spec,
                (CrossAttentionSpec, EncoderOnlyAttentionSpec),
            ):
                # Cross-attention and encoder-only groups are not prefix cached.
                continue

            group_cached_tokens = 0
            for block in blocks:
                group_cached_tokens = max(
                    group_cached_tokens,
                    block.block_hash_num_tokens or 0,
                )

            cached_tokens = (
                group_cached_tokens
                if cached_tokens is None
                else min(cached_tokens, group_cached_tokens)
            )

        return cached_tokens or 0

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)
            self._publish_hotprefix_blocks(request, num_computed_tokens)

    def _publish_hotprefix_blocks(
        self, request: Request, num_computed_tokens: int
    ) -> None:
        hotprefix_tree = self.get_hotprefix_tree(request.cache_salt)
        if hotprefix_tree is None:
            return
        cached_tokens = min(
            num_computed_tokens,
            self.estimate_cached_tokens(request),
        )
        token_ids = request.all_token_ids[:cached_tokens]
        if not token_ids:
            return
        hotprefix_tree.publish(token_ids)
        self._update_hotprefix_priorities(
            token_ids,
            self.coordinator.get_blocks(request.request_id),
            tree=hotprefix_tree,
        )

    def _update_hotprefix_priorities(
        self,
        token_ids: Sequence[int],
        blocks: tuple[list[KVCacheBlock], ...],
        *,
        tree: LocalHotPrefixTree,
    ) -> None:
        selector = self.hotprefix_eviction_selector
        if selector is None or not blocks:
            return
        request_tokens = tuple(token_ids)
        path = [
            node
            for node in tree.snapshot()
            if node.valid
            if len(node.full_prefix) <= len(request_tokens)
            and request_tokens[: len(node.full_prefix)] == node.full_prefix
        ]
        groups = []
        for node in path:
            node_end = len(node.full_prefix)
            node_start = node_end - len(node.segment)
            block_ids = tuple(
                block.block_id
                for block_index, block in enumerate(blocks[0])
                if not block.is_null
                and block_index * self._hotprefix_block_size < node_end
                and (block_index + 1) * self._hotprefix_block_size > node_start
            )
            if block_ids:
                groups.append(
                    EvictionGroup(
                        (
                            EvictionNode(
                                node.prefix_id,
                                node.record.frequency,
                                node.record.clock,
                                tree.namespace,
                                node.full_prefix,
                                tuple(
                                    block.block_id
                                    for block in blocks[0][
                                        : cdiv(
                                            len(node.full_prefix),
                                            self._hotprefix_block_size,
                                        )
                                    ]
                                    if not block.is_null
                                ),
                            ),
                        ),
                        block_ids,
                        self._hotprefix_block_size,
                    )
                )
        selector.update_groups(groups)

    def reserve_hotprefix_eviction_store(
        self,
        *,
        transfer_chunk_tokens: int,
        min_free_blocks: int = 1,
    ) -> EvictionStoreCandidate | None:
        """Pin one cold, chunk-aligned full prefix for selective Host admission.

        This runs after foreground scheduling. It never consumes the requested
        free-block headroom and leaves native APC hashes live until STORE reaches
        a terminal state.
        """
        selector = self.hotprefix_eviction_selector
        if selector is None:
            return None
        if transfer_chunk_tokens <= 0 or min_free_blocks < 0:
            raise ValueError("invalid HotPrefix eviction reservation limits")
        free_ids = {
            block.block_id
            for block in self.block_pool.free_block_queue.get_all_free_blocks()
        }
        page_size_bytes = self.kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec.page_size_bytes
        for group in selector.deferred_groups():
            nodes = sorted(
                (
                    node
                    for node in group.nodes
                    if node.full_prefix
                    and node.full_block_ids
                    and len(node.full_prefix) % transfer_chunk_tokens == 0
                    and node.prefix_id not in self._hotprefix_reserved_stores
                ),
                key=lambda node: (-len(node.full_prefix), node.prefix_id),
            )
            for node in nodes:
                source_ids = node.full_block_ids
                source_free_ids = set(source_ids) & free_ids
                if len(free_ids) - len(source_free_ids) < min_free_blocks:
                    continue
                source_blocks = [
                    self.block_pool.blocks[block_id] for block_id in source_ids
                ]
                if any(block.block_hash is None for block in source_blocks):
                    continue
                self.block_pool.touch(source_blocks)
                self._hotprefix_reserved_stores.add(node.prefix_id)
                selector.mark_group_pending(group.block_ids)
                return EvictionStoreCandidate(
                    node.namespace,
                    node.prefix_id,
                    node.full_prefix,
                    source_ids,
                    group.block_ids,
                    len(source_ids) * page_size_bytes,
                    node.frequency,
                    node.clock,
                )
        return None

    def release_hotprefix_eviction_store(
        self, candidate: EvictionStoreCandidate
    ) -> None:
        """Release the source pin after DEDUP, rejection, publish, or failure."""
        if candidate.prefix_id not in self._hotprefix_reserved_stores:
            raise RuntimeError("HotPrefix eviction STORE source is not reserved")
        self._hotprefix_reserved_stores.remove(candidate.prefix_id)
        selector = self.hotprefix_eviction_selector
        if selector is None:
            raise RuntimeError("HotPrefix eviction selector disappeared")
        selector.mark_group_terminal(candidate.eviction_group_block_ids)
        self.block_pool.free_blocks(
            [self.block_pool.blocks[block_id] for block_id in candidate.block_ids]
        )

    def reserve_hotprefix_promotion(
        self,
        *,
        prefix_id: bytes,
        token_ids: Sequence[int],
        total_bytes: int,
        min_free_blocks: int = 1,
    ) -> PromotionTransaction | None:
        """Reserve detached HBM target blocks after foreground scheduling."""
        manager = self.hotprefix_promotion_manager
        if manager is None:
            return None
        num_blocks = cdiv(len(token_ids), self._hotprefix_block_size)
        if len(token_ids) % self._hotprefix_block_size != 0:
            return None
        page_size_bytes = self.kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec.page_size_bytes
        if total_bytes != num_blocks * page_size_bytes:
            return None
        if self.block_pool.get_num_free_blocks() - num_blocks < min_free_blocks:
            return None
        if manager.get(prefix_id) is not None:
            return None
        try:
            target_blocks = self.block_pool.get_new_blocks(num_blocks)
        except HotPrefixEvictionDeferred:
            return None
        return manager.start(
            prefix_id=prefix_id,
            total_bytes=total_bytes,
            target_block_ids=tuple(block.block_id for block in target_blocks),
        )

    def publish_hotprefix_promotion(
        self,
        request: Request,
        prefix_id: bytes,
    ) -> tuple[str, ...]:
        """Atomically publish a completed detached promotion into native APC."""
        manager = self.hotprefix_promotion_manager
        if manager is None:
            raise RuntimeError("HotPrefix promotion is disabled")
        transaction = manager.get(prefix_id)
        if transaction is None or transaction.state is not PromotionState.READY:
            raise RuntimeError("HotPrefix promotion copy is not complete")
        blocks = [
            self.block_pool.blocks[block_id]
            for block_id in transaction.target_block_ids
        ]
        tree = self.get_hotprefix_tree(request.cache_salt)
        if tree is None:
            raise RuntimeError("HotPrefix local tree disappeared")
        token_ids = request.all_token_ids[: len(blocks) * self._hotprefix_block_size]
        node_by_id = {node.prefix_id: node for node in tree.snapshot()}
        node = node_by_id.get(prefix_id)
        if node is None or node.full_prefix != tuple(token_ids):
            raise RuntimeError("promotion prefix no longer matches Local tree")
        self.block_pool.cache_full_blocks(
            request,
            blocks,
            num_cached_blocks=0,
            num_full_blocks=len(blocks),
            block_size=self._hotprefix_block_size,
            kv_cache_group_id=0,
        )
        tree.publish(token_ids)
        self._update_hotprefix_priorities(
            token_ids,
            (blocks,),
            tree=tree,
        )
        self.block_pool.free_blocks(blocks)
        waiters = manager.take_waiters(prefix_id)
        manager.retire(prefix_id)
        return waiters

    def advance_hotprefix_promotion(
        self, prefix_id: bytes, *, copied_bytes: int
    ) -> bool:
        """Account one completed transfer chunk without partial publication."""
        manager = self.hotprefix_promotion_manager
        if manager is None:
            raise RuntimeError("HotPrefix promotion is disabled")
        manager.advance(prefix_id, budget_bytes=copied_bytes)
        transaction = manager.get(prefix_id)
        assert transaction is not None
        return transaction.state is PromotionState.READY

    def fail_hotprefix_promotion(self, prefix_id: bytes) -> tuple[str, ...]:
        """Release unpublished target blocks after a failed Host retrieve."""
        manager = self.hotprefix_promotion_manager
        if manager is None:
            raise RuntimeError("HotPrefix promotion is disabled")
        transaction = manager.get(prefix_id)
        if transaction is None:
            return ()
        manager.fail(prefix_id)
        self.block_pool.evict_blocks(set(transaction.target_block_ids))
        self.block_pool.free_blocks(
            [
                self.block_pool.blocks[block_id]
                for block_id in transaction.target_block_ids
            ]
        )
        waiters = manager.take_waiters(prefix_id)
        manager.retire(prefix_id)
        return waiters

    def get_hotprefix_tree(
        self, cache_salt: str | None = None
    ) -> LocalHotPrefixTree | None:
        """Return the Local tree for one vLLM cache namespace.

        vLLM cache salts are correctness boundaries.  Trees share the bounded
        hotness store but never share radix topology or prefix identities, and
        the namespace byte string matches LMCache's ``model\0cache_salt`` key.
        """
        store = self._hotprefix_hotness_store
        if store is None:
            return None
        salt = cache_salt or ""
        tree = self.hotprefix_trees.get(salt)
        if tree is None:
            tree = LocalHotPrefixTree(
                hotness_store=store,
                namespace=self._hotprefix_namespace + salt.encode(),
                aging_interval=self._hotprefix_aging_interval,
            )
            self.hotprefix_trees[salt] = tree
        return tree

    def get_hotprefix_promotion_nodes(
        self,
    ) -> tuple[tuple[bytes, tuple[HotPrefixNodeSnapshot, ...]], ...]:
        """Return locally hot logical nodes that are currently absent from HBM."""
        selector = self.hotprefix_eviction_selector
        if selector is None:
            return ()
        resident = selector.resident_prefix_ids()
        by_namespace: list[tuple[bytes, tuple[HotPrefixNodeSnapshot, ...]]] = []
        for tree in self.hotprefix_trees.values():
            missing = tuple(
                sorted(
                    (
                        node
                        for node in tree.snapshot()
                        if node.valid and node.prefix_id not in resident
                    ),
                    key=lambda node: (
                        -(
                            node.record.frequency
                            + node.record.clock / max(1, len(node.segment))
                        ),
                        node.prefix_id,
                    ),
                )
            )
            if missing:
                by_namespace.append((tree.namespace, missing))
        return tuple(sorted(by_namespace, key=lambda item: item[0]))

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def truncate_computed_blocks(
        self, blocks: KVCacheBlocks, num_computed_tokens: int
    ) -> KVCacheBlocks:
        """Return a lookup-result view truncated at an aligned token endpoint.

        An external hit can supply the final Mamba state even when the local
        Mamba group ends before this endpoint. Other groups must cover it.
        Pure slicing: refcounts are untouched and ``blocks`` is not mutated.
        """
        truncated: list[list[KVCacheBlock]] = []
        for group_blocks, manager, group in zip(
            blocks.blocks,
            self.coordinator.single_type_managers,
            self.kv_cache_config.kv_cache_groups,
            strict=True,
        ):
            if not group.kv_cache_spec.prefix_cacheable:
                # Scratch groups (e.g. the CSA compressor ring) hold a fixed
                # block covering no token range, so a lookup result never
                # carries computed blocks for them and their block size does
                # not divide the endpoint.
                assert not group_blocks
                truncated.append([])
                continue
            assert num_computed_tokens % manager.block_size == 0
            num_blocks = num_computed_tokens // manager.block_size
            if isinstance(group.kv_cache_spec, MambaSpec):
                num_blocks = min(num_blocks, len(group_blocks))
            else:
                assert num_blocks <= len(group_blocks)
            truncated.append(list(group_blocks[:num_blocks]))
        return self.create_kv_cache_blocks(tuple(truncated))

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def get_zeroing_block_ids_in_range(
        self, request_id: str, start_token: int, end_token: int
    ) -> list[int]:
        """The request's block ids covering [start_token, end_token), from
        the groups whose new blocks are zeroed by the worker."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                start_idx = start_token // mgr.block_size
                end_idx = cdiv(end_token, mgr.block_size)
                blocks = mgr.req_to_blocks[request_id]
                ids.extend(blk.block_id for blk in blocks[start_idx:end_idx])
        return ids

    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        """Re-record the request's blocks from start_token onwards for
        zeroing, e.g. blocks a failed async KV load left unwritten.

        start_token must be block-aligned: zeroing a partially-valid block
        would wipe its valid prefix.
        """
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                assert start_token % mgr.block_size == 0
                start_idx = start_token // mgr.block_size
                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending copies and return their retained endpoints."""
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks

    def take_boundary_state_offloads(
        self,
    ) -> dict[str, list[tuple[int, int, int]]]:
        """Drain this step's boundary-state hand-offs for a KV connector.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``
        for mamba "align" boundary states: the
        request's committed boundary-state snapshots and, on a sub-block partial
        hit, the CoW copy of its last-prompt-boundary state. Only mamba "align"
        groups contribute; empty otherwise. A connector reads the referenced
        blocks — never resolving them positionally — and offloads them so a
        later request can hit that prefix.
        """
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_boundary_state_offloads():
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        self.coordinator.new_step_starts()
