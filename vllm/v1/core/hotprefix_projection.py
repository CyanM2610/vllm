# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incrementally reconcile token-level HotPrefix paths onto physical blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.hotprefix import (
    EvictionGroup,
    EvictionNode,
    HotPrefixBlockEvictionSelector,
    HotPrefixNodeSnapshot,
)


@dataclass(frozen=True)
class ProjectionResult:
    """Normalized work and invalidation facts for one reconciliation."""

    skipped: bool
    reason: str
    tree_nodes: int
    path_nodes: int
    request_blocks: int
    projected_blocks: int
    groups_rebuilt: int
    max_component_blocks: int
    topology_changed: bool
    hotness_changed: bool
    binding_changed: bool


@dataclass(frozen=True)
class _ProjectionSignature:
    cached_tokens: int
    physical_block_ids: tuple[int | None, ...]
    aging_epoch: int
    topology: tuple[tuple[bytes, tuple[int, ...], tuple[int, ...], bool], ...]
    hotness: tuple[tuple[bytes, int, int], ...]


class HotPrefixBlockProjection:
    """Hide projection signatures, arithmetic block mapping, and invalidation."""

    def __init__(self, selector: HotPrefixBlockEvictionSelector) -> None:
        self._selector = selector
        self._signatures: dict[tuple[bytes, bytes], _ProjectionSignature] = {}

    def reconcile(
        self,
        *,
        namespace: bytes,
        cached_tokens: int,
        path: Sequence[HotPrefixNodeSnapshot],
        physical_block_ids: Sequence[int | None],
        block_size: int,
        aging_epoch: int,
        total_tree_nodes: int | None = None,
    ) -> ProjectionResult:
        """Project one immutable radix path and update the eviction selector.

        Args:
            namespace: Cache-salt-scoped Local tree namespace.
            cached_tokens: Cacheable token prefix length.
            path: Immutable logical nodes on that token path only.
            physical_block_ids: Position-preserving request block binding.
            block_size: Token capacity of one physical block.
            aging_epoch: Local clock-aging epoch.
            total_tree_nodes: Current non-root tree node count.

        Returns:
            Normalized work and change classification for observability.

        Raises:
            ValueError: If lengths or block size are invalid.
        """
        if cached_tokens < 0:
            raise ValueError("cached_tokens must be non-negative")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        blocks = tuple(physical_block_ids)
        nodes = tuple(
            node
            for node in path
            if node.valid and len(node.full_prefix) <= cached_tokens
        )
        signature = _ProjectionSignature(
            cached_tokens=cached_tokens,
            physical_block_ids=blocks,
            aging_epoch=aging_epoch,
            topology=tuple(
                (node.prefix_id, node.full_prefix, node.segment, node.valid)
                for node in nodes
            ),
            hotness=tuple(
                (node.prefix_id, node.record.frequency, node.record.clock)
                for node in nodes
            ),
        )
        signature_key = (namespace, nodes[-1].prefix_id if nodes else b"")
        previous = self._signatures.get(signature_key)
        tree_nodes = len(nodes) if total_tree_nodes is None else total_tree_nodes
        if signature == previous:
            return ProjectionResult(
                skipped=True,
                reason="identical",
                tree_nodes=tree_nodes,
                path_nodes=len(nodes),
                request_blocks=len(blocks),
                projected_blocks=0,
                groups_rebuilt=0,
                max_component_blocks=0,
                topology_changed=False,
                hotness_changed=False,
                binding_changed=False,
            )

        topology_changed = previous is None or previous.topology != signature.topology
        hotness_changed = (
            previous is None
            or previous.hotness != signature.hotness
            or previous.aging_epoch != signature.aging_epoch
        )
        binding_changed = previous is None or (
            previous.cached_tokens != signature.cached_tokens
            or previous.physical_block_ids != signature.physical_block_ids
        )
        reason = (
            "topology"
            if topology_changed
            else "binding"
            if binding_changed
            else "hotness"
        )
        groups = self._build_groups(
            namespace=namespace,
            nodes=nodes,
            physical_block_ids=blocks,
            block_size=block_size,
        )
        self._selector.update_groups(groups)
        components = self._selector.eviction_groups()
        self._signatures[signature_key] = signature
        return ProjectionResult(
            skipped=False,
            reason=reason,
            tree_nodes=tree_nodes,
            path_nodes=len(nodes),
            request_blocks=len(blocks),
            projected_blocks=sum(len(group.block_ids) for group in groups),
            groups_rebuilt=len(groups),
            max_component_blocks=max(
                (len(group.block_ids) for group in components), default=0
            ),
            topology_changed=topology_changed,
            hotness_changed=hotness_changed,
            binding_changed=binding_changed,
        )

    def discard(self, block_ids: Sequence[int]) -> None:
        """Invalidate signatures affected by discarded physical bindings.

        Args:
            block_ids: Physical bindings removed by BlockPool.
        """
        discarded = set(block_ids)
        self._signatures = {
            signature_key: signature
            for signature_key, signature in self._signatures.items()
            if discarded.isdisjoint(
                block_id
                for block_id in signature.physical_block_ids
                if block_id is not None
            )
        }

    def age(self) -> None:
        """Age selector priorities; tree epochs invalidate reconciliation."""
        self._selector.age_all()

    @staticmethod
    def _build_groups(
        *,
        namespace: bytes,
        nodes: Sequence[HotPrefixNodeSnapshot],
        physical_block_ids: tuple[int | None, ...],
        block_size: int,
    ) -> tuple[EvictionGroup, ...]:
        groups: list[EvictionGroup] = []
        for node in nodes:
            node_end = len(node.full_prefix)
            node_start = node_end - len(node.segment)
            first_block = node_start // block_size
            last_block = cdiv(node_end, block_size)
            block_ids = tuple(
                block_id
                for block_id in physical_block_ids[first_block:last_block]
                if block_id is not None
            )
            if not block_ids:
                continue
            full_block_ids = tuple(
                block_id
                for block_id in physical_block_ids[:last_block]
                if block_id is not None
            )
            groups.append(
                EvictionGroup(
                    nodes=(
                        EvictionNode(
                            node.prefix_id,
                            node.record.frequency,
                            node.record.clock,
                            namespace,
                            node.full_prefix,
                            full_block_ids,
                        ),
                    ),
                    block_ids=block_ids,
                    block_size=block_size,
                )
            )
        return tuple(groups)
