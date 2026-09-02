# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HotPrefix policy primitives for vLLM's prefix cache."""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


PrefixId = bytes


class HotPrefixEvictionDeferred(RuntimeError):
    """Raised when a cached victim must finish Host admission before reuse."""


def make_hotprefix_namespace(
    *,
    model: str,
    revision: str | None,
    kv_layout: str | None,
    group_specs: Sequence[str],
) -> bytes:
    """Build the shared LogicalPrefix namespace before appending cache salt.

    Args:
        model: Canonical model identifier.
        revision: Resolved model revision or commit hash when available.
        kv_layout: Resolved KV tensor layout.
        group_specs: Complete stable representations of KV cache group specs.

    Returns:
        Namespace prefix shared by vLLM and LMCache.
    """
    digest = hashlib.blake2b(digest_size=16)
    for group_spec in group_specs:
        encoded_spec = group_spec.encode()
        digest.update(len(encoded_spec).to_bytes(4, "little"))
        digest.update(encoded_spec)
    identity = "\0".join(
        ("hotprefix-v1", model, revision or "", kv_layout or "")
    ).encode()
    return identity + b"\0" + digest.digest() + b"\0"


@dataclass(frozen=True)
class HotnessRecord:
    """Saturating frequency, recency clock, and radix depth for one prefix."""

    frequency: int
    clock: int
    depth: int


class HotnessStore(Protocol):
    """Storage seam for exact and approximate HotPrefix metadata."""

    def insert(
        self,
        prefix_id: PrefixId,
        *,
        depth: int,
        record: HotnessRecord | None = None,
    ) -> HotnessRecord: ...

    def access(self, prefix_id: PrefixId) -> HotnessRecord: ...

    def age_all(self) -> None: ...

    def get(self, prefix_id: PrefixId) -> HotnessRecord | None: ...

    def set_depth(self, prefix_id: PrefixId, depth: int) -> HotnessRecord: ...


class CuckooInsertionError(RuntimeError):
    """Raised when a CuckooHotnessStore cannot insert without data loss."""


@dataclass(frozen=True)
class _CuckooEntry:
    fingerprint: int
    record: HotnessRecord


class ExactHotnessStore:
    """Store exact HotPrefix records for conformance and debugging.

    Args:
        max_value: Largest representable frequency and depth.
        max_age: Clock value assigned on insertion and access.
    """

    def __init__(self, *, max_value: int = 255, max_age: int = 255) -> None:
        if max_value <= 0:
            raise ValueError("max_value must be positive")
        if max_age < 0 or max_age > max_value:
            raise ValueError("max_age must be between zero and max_value")
        self._max_value = max_value
        self._max_age = max_age
        self._records: dict[PrefixId, HotnessRecord] = {}

    def insert(
        self,
        prefix_id: PrefixId,
        *,
        depth: int,
        record: HotnessRecord | None = None,
    ) -> HotnessRecord:
        """Insert a prefix or return its existing record.

        Args:
            prefix_id: Stable logical prefix identity.
            depth: Token-radix depth for a new record.
            record: Optional inherited record used after a radix split.

        Returns:
            The stored record.
        """
        existing = self._records.get(prefix_id)
        if existing is not None:
            return existing
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if record is None:
            stored = HotnessRecord(1, self._max_age, min(depth, self._max_value))
        else:
            stored = HotnessRecord(
                min(record.frequency, self._max_value),
                min(record.clock, self._max_age),
                min(depth, self._max_value),
            )
        self._records[prefix_id] = stored
        return stored

    def access(self, prefix_id: PrefixId) -> HotnessRecord:
        """Increment one record and reset its recency clock.

        Args:
            prefix_id: Existing stable logical prefix identity.

        Returns:
            The updated record.

        Raises:
            KeyError: If the prefix has not been inserted.
        """
        record = self._records[prefix_id]
        updated = HotnessRecord(
            min(record.frequency + 1, self._max_value),
            self._max_age,
            record.depth,
        )
        self._records[prefix_id] = updated
        return updated

    def age_all(self) -> None:
        """Decrease every positive clock by one."""
        for prefix_id, record in self._records.items():
            self._records[prefix_id] = HotnessRecord(
                record.frequency,
                max(0, record.clock - 1),
                record.depth,
            )

    def get(self, prefix_id: PrefixId) -> HotnessRecord | None:
        """Return the record for a prefix when present.

        Args:
            prefix_id: Stable logical prefix identity.

        Returns:
            The record or ``None`` when absent.
        """
        return self._records.get(prefix_id)

    def set_depth(self, prefix_id: PrefixId, depth: int) -> HotnessRecord:
        """Set the saturated radix depth for an existing prefix.

        Args:
            prefix_id: Existing stable logical prefix identity.
            depth: New token-radix depth.

        Returns:
            The updated record.
        """
        if depth < 0:
            raise ValueError("depth must be non-negative")
        record = self._records[prefix_id]
        updated = HotnessRecord(
            record.frequency,
            record.clock,
            min(depth, self._max_value),
        )
        self._records[prefix_id] = updated
        return updated


class CuckooHotnessStore:
    """Store approximate HotPrefix records in a deterministic cuckoo filter.

    Args:
        num_buckets: Power-of-two number of cuckoo buckets.
        slots_per_bucket: Number of entries in each bucket.
        max_kicks: Maximum relocations attempted for one insertion.
        fingerprint_bits: Number of bits retained from a prefix fingerprint.
        max_value: Largest representable frequency and depth.
        max_age: Clock value assigned on insertion and access.

    Raises:
        ValueError: If the filter configuration is invalid.
    """

    def __init__(
        self,
        *,
        num_buckets: int,
        slots_per_bucket: int = 4,
        max_kicks: int = 500,
        fingerprint_bits: int = 8,
        max_value: int = 255,
        max_age: int = 255,
    ) -> None:
        if num_buckets <= 0 or num_buckets & (num_buckets - 1):
            raise ValueError("num_buckets must be a positive power of two")
        if slots_per_bucket <= 0:
            raise ValueError("slots_per_bucket must be positive")
        if max_kicks <= 0:
            raise ValueError("max_kicks must be positive")
        if fingerprint_bits <= 0 or fingerprint_bits > 32:
            raise ValueError("fingerprint_bits must be between 1 and 32")
        if max_value <= 0:
            raise ValueError("max_value must be positive")
        if max_age < 0 or max_age > max_value:
            raise ValueError("max_age must be between zero and max_value")
        self._num_buckets = num_buckets
        self._bucket_mask = num_buckets - 1
        self._slots_per_bucket = slots_per_bucket
        self._max_kicks = max_kicks
        self._fingerprint_mask = (1 << fingerprint_bits) - 1
        self._max_value = max_value
        self._max_age = max_age
        self._buckets: list[list[_CuckooEntry | None]] = [
            [None] * slots_per_bucket for _ in range(num_buckets)
        ]

    def insert(
        self,
        prefix_id: PrefixId,
        *,
        depth: int,
        record: HotnessRecord | None = None,
    ) -> HotnessRecord:
        """Insert a prefix without corrupting existing entries on failure.

        Args:
            prefix_id: Stable logical prefix identity.
            depth: Token-radix depth for a new record.
            record: Optional inherited record used after a radix split.

        Returns:
            The stored record. A fingerprint false positive returns the
            matching approximate record.

        Raises:
            CuckooInsertionError: If relocation reaches ``max_kicks``.
            ValueError: If ``depth`` is negative.
        """
        if depth < 0:
            raise ValueError("depth must be non-negative")
        fingerprint, first, second, digest = self._locations(prefix_id)
        found = self._find(fingerprint, first, second)
        if found is not None:
            return found.record
        stored = (
            HotnessRecord(1, self._max_age, min(depth, self._max_value))
            if record is None
            else HotnessRecord(
                min(record.frequency, self._max_value),
                min(record.clock, self._max_age),
                min(depth, self._max_value),
            )
        )
        entry = _CuckooEntry(fingerprint, stored)
        working = [bucket.copy() for bucket in self._buckets]
        for bucket_index in (first, second):
            slot = self._empty_slot(working[bucket_index])
            if slot is not None:
                working[bucket_index][slot] = entry
                self._buckets = working
                return stored

        current = entry
        bucket_index = first
        for kick in range(self._max_kicks):
            slot = self._kick_slot(digest, kick)
            displaced = working[bucket_index][slot]
            working[bucket_index][slot] = current
            if displaced is None:
                self._buckets = working
                return stored
            current = displaced
            bucket_index = self._alternate_bucket(bucket_index, current.fingerprint)
            empty = self._empty_slot(working[bucket_index])
            if empty is not None:
                working[bucket_index][empty] = current
                self._buckets = working
                return stored
        raise CuckooInsertionError("cuckoo insertion exceeded max_kicks")

    def access(self, prefix_id: PrefixId) -> HotnessRecord:
        """Increment one approximate record and reset its recency clock.

        Args:
            prefix_id: Existing stable logical prefix identity.

        Returns:
            The updated record.

        Raises:
            KeyError: If neither candidate bucket contains the fingerprint.
        """
        fingerprint, first, second, _ = self._locations(prefix_id)
        for bucket_index in (first, second):
            bucket = self._buckets[bucket_index]
            for slot, entry in enumerate(bucket):
                if entry is not None and entry.fingerprint == fingerprint:
                    updated = HotnessRecord(
                        min(entry.record.frequency + 1, self._max_value),
                        self._max_age,
                        entry.record.depth,
                    )
                    bucket[slot] = _CuckooEntry(fingerprint, updated)
                    return updated
        raise KeyError(prefix_id)

    def age_all(self) -> None:
        """Decrease every positive clock by one."""
        for bucket in self._buckets:
            for slot, entry in enumerate(bucket):
                if entry is not None:
                    bucket[slot] = _CuckooEntry(
                        entry.fingerprint,
                        HotnessRecord(
                            entry.record.frequency,
                            max(0, entry.record.clock - 1),
                            entry.record.depth,
                        ),
                    )

    def get(self, prefix_id: PrefixId) -> HotnessRecord | None:
        """Return the approximate record matching a prefix fingerprint.

        Args:
            prefix_id: Stable logical prefix identity.

        Returns:
            The matching approximate record or ``None``.
        """
        fingerprint, first, second, _ = self._locations(prefix_id)
        entry = self._find(fingerprint, first, second)
        return entry.record if entry is not None else None

    def set_depth(self, prefix_id: PrefixId, depth: int) -> HotnessRecord:
        """Set the saturated radix depth for an approximate record.

        Args:
            prefix_id: Existing stable logical prefix identity.
            depth: New token-radix depth.

        Returns:
            The updated approximate record.

        Raises:
            KeyError: If neither candidate bucket contains the fingerprint.
            ValueError: If ``depth`` is negative.
        """
        if depth < 0:
            raise ValueError("depth must be non-negative")
        fingerprint, first, second, _ = self._locations(prefix_id)
        for bucket_index in (first, second):
            bucket = self._buckets[bucket_index]
            for slot, entry in enumerate(bucket):
                if entry is not None and entry.fingerprint == fingerprint:
                    updated = HotnessRecord(
                        entry.record.frequency,
                        entry.record.clock,
                        min(depth, self._max_value),
                    )
                    bucket[slot] = _CuckooEntry(fingerprint, updated)
                    return updated
        raise KeyError(prefix_id)

    def _locations(self, prefix_id: PrefixId) -> tuple[int, int, int, bytes]:
        digest = hashlib.blake2b(prefix_id, digest_size=16).digest()
        fingerprint = int.from_bytes(digest[:4], "little") & self._fingerprint_mask
        fingerprint = fingerprint or 1
        first = int.from_bytes(digest[4:8], "little") & self._bucket_mask
        second = self._alternate_bucket(first, fingerprint)
        return fingerprint, first, second, digest

    def _alternate_bucket(self, bucket_index: int, fingerprint: int) -> int:
        encoded = fingerprint.to_bytes(4, "little")
        hashed = hashlib.blake2b(encoded, digest_size=4).digest()
        return bucket_index ^ (int.from_bytes(hashed, "little") & self._bucket_mask)

    def _find(self, fingerprint: int, first: int, second: int) -> _CuckooEntry | None:
        for bucket_index in (first, second):
            for entry in self._buckets[bucket_index]:
                if entry is not None and entry.fingerprint == fingerprint:
                    return entry
        return None

    def _empty_slot(self, bucket: list[_CuckooEntry | None]) -> int | None:
        return next((slot for slot, entry in enumerate(bucket) if entry is None), None)

    def _kick_slot(self, digest: bytes, kick: int) -> int:
        kick_digest = hashlib.blake2b(
            digest + kick.to_bytes(4, "little"), digest_size=4
        ).digest()
        return int.from_bytes(kick_digest, "little") % self._slots_per_bucket


@dataclass
class _RadixNode:
    segment: tuple[int, ...]
    full_prefix: tuple[int, ...]
    prefix_id: PrefixId
    depth: int
    parent: "_RadixNode | None"
    children: dict[int, "_RadixNode"] = field(default_factory=dict)


@dataclass(frozen=True)
class HotPrefixNodeSnapshot:
    """Immutable public view of one token-level HotPrefix radix node."""

    prefix_id: PrefixId
    full_prefix: tuple[int, ...]
    segment: tuple[int, ...]
    parent: PrefixId | None
    children: tuple[PrefixId, ...]
    record: HotnessRecord
    valid: bool = True


@dataclass(frozen=True)
class EvictionNode:
    """Hotness facts for one logical node invalidated by block eviction."""

    prefix_id: PrefixId
    frequency: int
    clock: int
    namespace: bytes = b""
    full_prefix: tuple[int, ...] = ()
    full_block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.frequency < 0 or self.clock < 0:
            raise ValueError("eviction-node hotness must be non-negative")


@dataclass(frozen=True)
class EvictionGroup:
    """Physical vLLM blocks and logical nodes invalidated together.

    Args:
        nodes: Every logical node invalidated by reclaiming the blocks.
        block_ids: Complete physical blocks reclaimed by the action.
        block_size: Token capacity of each physical block.
    """

    nodes: tuple[EvictionNode, ...]
    block_ids: tuple[int, ...]
    block_size: int

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("an eviction group must invalidate at least one node")
        if not self.block_ids:
            raise ValueError("an eviction group must reclaim at least one block")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("eviction-group block IDs must be unique")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    @property
    def reclaimable_length(self) -> int:
        """Return the token capacity of all exclusively reclaimed blocks."""
        return len(self.block_ids) * self.block_size

    @property
    def priority(self) -> float:
        """Return the hottest affected node's block-aware Eq. 5 priority."""
        length = self.reclaimable_length
        return max(node.frequency + node.clock / length for node in self.nodes)


@dataclass(frozen=True)
class EvictionGroupUpdate:
    """Affected connected-component work from one selector update."""

    groups_updated: int = 0
    max_component_blocks: int = 0


@dataclass(frozen=True)
class EvictionStoreCandidate:
    """Pinned full-prefix HBM source for one selective Host STORE."""

    namespace: bytes
    prefix_id: PrefixId
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    eviction_group_block_ids: tuple[int, ...]
    size_bytes: int
    frequency: int
    clock: int

    def __post_init__(self) -> None:
        if not self.namespace or not self.prefix_id:
            raise ValueError("eviction STORE identity must not be empty")
        if not self.token_ids or not self.block_ids:
            raise ValueError("eviction STORE source must not be empty")
        if self.size_bytes <= 0:
            raise ValueError("eviction STORE size must be positive")


class HotPrefixBlockEvictionSelector:
    """Translate logical HotPrefix priorities into BlockPool selections."""

    def __init__(
        self,
        *,
        defer_for_host: bool = True,
        apply_hotprefix_priority: bool = True,
    ) -> None:
        self._defer_for_host = defer_for_host
        self._apply_hotprefix_priority = apply_hotprefix_priority
        self._priorities: dict[int, float] = {}
        self._groups_by_block: dict[int, EvictionGroup] = {}
        self._pending_group_keys: set[frozenset[int]] = set()
        self._terminal_group_keys: set[frozenset[int]] = set()
        self._deferred_group_keys: set[frozenset[int]] = set()
        self._discard_observer: Callable[[Sequence[int]], object] | None = None
        self._state_epoch = 0

    @property
    def state_epoch(self) -> int:
        """Return the monotonic projection/group feasibility epoch."""
        return self._state_epoch

    def set_discard_observer(self, observer: Callable[[Sequence[int]], object]) -> None:
        """Notify projection state whenever physical bindings are discarded.

        Args:
            observer: Callback receiving discarded physical block IDs.
        """
        self._discard_observer = observer

    def update_priorities(self, priorities: dict[int, float]) -> None:
        """Replace priorities for the supplied physical block IDs.

        Args:
            priorities: Block ID to block-projected HotPrefix priority.
        """
        if any(block_id < 0 for block_id in priorities):
            raise ValueError("block IDs must be non-negative")
        self._priorities.update(priorities)
        if priorities:
            self._state_epoch += 1

    def update_groups(self, groups: Sequence[EvictionGroup]) -> EvictionGroupUpdate:
        """Merge block-overlapping logical groups and publish their priorities.

        A token-level radix boundary can fall inside a physical block.  Such
        nodes cannot be invalidated independently, so overlapping groups form
        one connected component and every block receives the component score.
        """
        changed = False
        groups_updated = 0
        max_component_blocks = 0
        for incoming in groups:
            changed = True
            groups_updated += 1
            nodes = {node.prefix_id: node for node in incoming.nodes}
            block_ids = set(incoming.block_ids)
            block_size = incoming.block_size
            was_pending = False
            was_terminal = False
            was_deferred = False
            while True:
                overlapping = {
                    id(group): group
                    for block_id in tuple(block_ids)
                    if (group := self._groups_by_block.get(block_id)) is not None
                }
                if not overlapping:
                    break
                grew = False
                for group in overlapping.values():
                    if group.block_size != block_size:
                        raise ValueError(
                            "overlapping eviction groups disagree on block size"
                        )
                    previous_size = len(block_ids)
                    block_ids.update(group.block_ids)
                    for node in group.nodes:
                        nodes.setdefault(node.prefix_id, node)
                    old_key = frozenset(group.block_ids)
                    was_pending = was_pending or old_key in self._pending_group_keys
                    was_terminal = was_terminal or old_key in self._terminal_group_keys
                    was_deferred = was_deferred or old_key in self._deferred_group_keys
                    self._pending_group_keys.discard(old_key)
                    self._terminal_group_keys.discard(old_key)
                    self._deferred_group_keys.discard(old_key)
                    grew = grew or len(block_ids) != previous_size
                    for old_block_id in group.block_ids:
                        self._groups_by_block.pop(old_block_id, None)
                if not grew:
                    break
            merged = EvictionGroup(
                tuple(sorted(nodes.values(), key=lambda node: node.prefix_id)),
                tuple(sorted(block_ids)),
                block_size,
            )
            for block_id in merged.block_ids:
                self._groups_by_block[block_id] = merged
                self._priorities[block_id] = merged.priority
            merged_key = frozenset(merged.block_ids)
            if was_pending:
                self._pending_group_keys.add(merged_key)
            elif was_terminal:
                self._terminal_group_keys.add(merged_key)
            elif was_deferred:
                self._deferred_group_keys.add(merged_key)
            max_component_blocks = max(max_component_blocks, len(merged.block_ids))
        if changed:
            self._state_epoch += 1
        return EvictionGroupUpdate(groups_updated, max_component_blocks)

    def collateral_block_ids(
        self, selected_block_ids: Sequence[int]
    ) -> tuple[int, ...]:
        """Return blocks whose APC entries share a logical eviction group."""
        if not self._apply_hotprefix_priority:
            return ()
        selected = set(selected_block_ids)
        collateral: set[int] = set()
        for block_id in selected:
            group = self._groups_by_block.get(block_id)
            if group is not None:
                collateral.update(group.block_ids)
        collateral.difference_update(selected)
        return tuple(sorted(collateral))

    def resident_prefix_ids(self) -> frozenset[PrefixId]:
        """Return logical nodes still represented by cached HBM blocks."""
        unique_groups = {id(group): group for group in self._groups_by_block.values()}
        return frozenset(
            node.prefix_id for group in unique_groups.values() for node in group.nodes
        )

    def eviction_groups(self) -> tuple[EvictionGroup, ...]:
        """Return unique physical groups in ascending eviction priority."""
        unique_groups = {id(group): group for group in self._groups_by_block.values()}
        return tuple(
            sorted(
                unique_groups.values(),
                key=lambda group: (group.priority, group.block_ids),
            )
        )

    def age_all(self) -> None:
        """Apply one Local clock decay pass to every physical group score."""
        groups = self.eviction_groups()
        self._groups_by_block.clear()
        for group in groups:
            aged = EvictionGroup(
                tuple(
                    EvictionNode(
                        node.prefix_id,
                        node.frequency,
                        max(0, node.clock - 1),
                        node.namespace,
                        node.full_prefix,
                        node.full_block_ids,
                    )
                    for node in group.nodes
                ),
                group.block_ids,
                group.block_size,
            )
            for block_id in aged.block_ids:
                self._groups_by_block[block_id] = aged
                self._priorities[block_id] = aged.priority
        if groups:
            self._state_epoch += 1

    def deferred_groups(self) -> tuple[EvictionGroup, ...]:
        """Return actual allocation victims awaiting Host admission."""
        groups = {
            frozenset(group.block_ids): group
            for group in self._groups_by_block.values()
        }
        return tuple(
            sorted(
                (groups[key] for key in self._deferred_group_keys if key in groups),
                key=lambda group: (group.priority, group.block_ids),
            )
        )

    def mark_group_pending(self, block_ids: Sequence[int]) -> None:
        """Mark a deferred physical group as pinned for STORE."""
        key = frozenset(block_ids)
        self._deferred_group_keys.discard(key)
        self._pending_group_keys.add(key)
        self._state_epoch += 1

    def mark_group_terminal(self, block_ids: Sequence[int]) -> None:
        """Allow reuse after DEDUP, rejection, publication, or failure."""
        key = frozenset(block_ids)
        self._deferred_group_keys.discard(key)
        self._pending_group_keys.discard(key)
        self._terminal_group_keys.add(key)
        self._state_epoch += 1

    def discard(self, block_ids: Sequence[int]) -> None:
        """Forget priorities for blocks whose cached contents were removed."""
        affected_groups = {
            id(group): group
            for block_id in block_ids
            if (group := self._groups_by_block.get(block_id)) is not None
        }
        for group in affected_groups.values():
            key = frozenset(group.block_ids)
            self._pending_group_keys.discard(key)
            self._terminal_group_keys.discard(key)
            self._deferred_group_keys.discard(key)
            for grouped_block_id in group.block_ids:
                self._groups_by_block.pop(grouped_block_id, None)
                self._priorities.pop(grouped_block_id, None)
        for block_id in block_ids:
            self._priorities.pop(block_id, None)
        if affected_groups or block_ids:
            self._state_epoch += 1
        if self._discard_observer is not None:
            self._discard_observer(block_ids)

    def select_blocks(
        self,
        candidates: tuple["KVCacheBlock", ...],
        num_blocks: int,
    ) -> tuple[int, ...]:
        """Return the lowest-priority cached blocks with LRU tie-breaking."""
        if num_blocks < 0 or num_blocks > len(candidates):
            raise ValueError("num_blocks is outside the candidate range")
        lru_position = {
            block.block_id: position for position, block in enumerate(candidates)
        }
        ordered = (
            sorted(
                candidates,
                key=lambda block: (
                    self._priorities.get(block.block_id, 0.0),
                    lru_position[block.block_id],
                ),
            )
            if self._apply_hotprefix_priority
            else candidates
        )
        eligible: list[int] = []
        deferred: set[frozenset[int]] = set()
        for block in ordered:
            group = self._groups_by_block.get(block.block_id)
            if group is None:
                eligible.append(block.block_id)
            else:
                key = frozenset(group.block_ids)
                if (
                    not self._apply_hotprefix_priority
                    or not self._defer_for_host
                    or key in self._terminal_group_keys
                ):
                    eligible.append(block.block_id)
                elif key not in self._pending_group_keys:
                    deferred.add(key)
            if len(eligible) == num_blocks:
                return tuple(eligible)
        self._deferred_group_keys.update(deferred)
        raise HotPrefixEvictionDeferred(
            "cached HotPrefix victims require Host admission"
        )


class PromotionState(Enum):
    """Lifecycle state of one logical-node promotion."""

    PENDING = "pending"
    COPYING = "copying"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PromotionTransaction:
    """Node-atomic promotion whose copy may span scheduler steps."""

    prefix_id: PrefixId
    total_bytes: int
    target_block_ids: tuple[int, ...]
    state: PromotionState = PromotionState.PENDING
    copied_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.prefix_id:
            raise ValueError("prefix_id must not be empty")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if not self.target_block_ids:
            raise ValueError("target_block_ids must not be empty")
        if len(set(self.target_block_ids)) != len(self.target_block_ids):
            raise ValueError("target_block_ids must be unique")

    @property
    def published(self) -> bool:
        """Return whether the complete logical node is available to APC."""
        return self.state is PromotionState.READY

    @property
    def remaining_bytes(self) -> int:
        """Return bytes not yet copied."""
        return self.total_bytes - self.copied_bytes


class PromotionManager:
    """Bound in-flight promotion and coalesce requests onto active copies."""

    def __init__(self, *, max_inflight: int) -> None:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self._max_inflight = max_inflight
        self._transactions: dict[PrefixId, PromotionTransaction] = {}
        self._waiters: dict[PrefixId, list[str]] = {}

    def plan(
        self,
        *,
        prefix_id: PrefixId,
        total_bytes: int,
        target_block_ids: tuple[int, ...],
    ) -> PromotionTransaction:
        """Create a PENDING promotion without reserving transfer concurrency."""
        if prefix_id in self._transactions:
            raise ValueError("promotion already exists for prefix")
        transaction = PromotionTransaction(
            prefix_id,
            total_bytes,
            target_block_ids,
        )
        self._transactions[prefix_id] = transaction
        return transaction

    def start(
        self,
        *,
        prefix_id: PrefixId,
        total_bytes: int,
        target_block_ids: tuple[int, ...],
    ) -> PromotionTransaction:
        """Create and begin a promotion immediately."""
        transaction = self.plan(
            prefix_id=prefix_id,
            total_bytes=total_bytes,
            target_block_ids=target_block_ids,
        )
        self.begin(prefix_id)
        return transaction

    def begin(self, prefix_id: PrefixId) -> PromotionTransaction:
        """Move a PENDING promotion into COPYING when budget is available."""
        transaction = self._transactions[prefix_id]
        if transaction.state is not PromotionState.PENDING:
            raise RuntimeError("only a PENDING promotion can begin")
        inflight = sum(
            item.state is PromotionState.COPYING for item in self._transactions.values()
        )
        if inflight >= self._max_inflight:
            raise RuntimeError("maximum in-flight promotions reached")
        transaction.state = PromotionState.COPYING
        return transaction

    def advance(self, prefix_id: PrefixId, *, budget_bytes: int) -> int:
        """Consume one step's copy budget without publishing partial KV."""
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        transaction = self._transactions[prefix_id]
        if transaction.state is not PromotionState.COPYING:
            raise RuntimeError("only a COPYING promotion can advance")
        copied = min(budget_bytes, transaction.remaining_bytes)
        transaction.copied_bytes += copied
        if transaction.remaining_bytes == 0:
            transaction.state = PromotionState.READY
        return copied

    def coalesce(self, request_id: str, prefix_id: PrefixId) -> bool:
        """Join a request only to a copy that has actually started."""
        transaction = self._transactions.get(prefix_id)
        if transaction is None or transaction.state is not PromotionState.COPYING:
            return False
        waiters = self._waiters.setdefault(prefix_id, [])
        if request_id not in waiters:
            waiters.append(request_id)
        return True

    def take_waiters(self, prefix_id: PrefixId) -> tuple[str, ...]:
        """Drain requests after a promotion reaches a terminal state."""
        transaction = self._transactions[prefix_id]
        if transaction.state in (PromotionState.PENDING, PromotionState.COPYING):
            raise RuntimeError("promotion is not terminal")
        return tuple(self._waiters.pop(prefix_id, ()))

    def fail(self, prefix_id: PrefixId) -> None:
        """Fail an active promotion without publishing partial KV."""
        transaction = self._transactions[prefix_id]
        if transaction.state not in (PromotionState.PENDING, PromotionState.COPYING):
            raise RuntimeError("terminal promotion cannot fail again")
        transaction.state = PromotionState.FAILED

    def cancel(self, prefix_id: PrefixId) -> None:
        """Cancel a PENDING promotion that has not copied bytes."""
        transaction = self._transactions[prefix_id]
        if transaction.state is not PromotionState.PENDING:
            raise RuntimeError("only a PENDING promotion can be cancelled")
        transaction.state = PromotionState.CANCELLED

    def get(self, prefix_id: PrefixId) -> PromotionTransaction | None:
        """Return an active or terminal transaction before retirement."""
        return self._transactions.get(prefix_id)

    def reserved_block_count(self) -> int:
        """Return target blocks reserved by all promotion transactions."""
        return sum(
            len(transaction.target_block_ids)
            for transaction in self._transactions.values()
        )

    def retire(self, prefix_id: PrefixId) -> PromotionTransaction:
        """Remove a terminal transaction after blocks and waiters are handled."""
        transaction = self._transactions[prefix_id]
        if transaction.state in (PromotionState.PENDING, PromotionState.COPYING):
            raise RuntimeError("cannot retire an active promotion")
        if self._waiters.get(prefix_id):
            raise RuntimeError("promotion waiters must be drained before retirement")
        return self._transactions.pop(prefix_id)


class LocalHotPrefixTree:
    """Maintain a token-level Local HotPrefix Tree as a policy shadow view.

    Args:
        hotness_store: Exact or cuckoo metadata store.
        namespace: Stable model/cache namespace included in prefix identities.
        aging_interval: Number of initial request lookups between aging passes.
    """

    def __init__(
        self,
        *,
        hotness_store: HotnessStore,
        namespace: bytes,
        aging_interval: int,
    ) -> None:
        if aging_interval <= 0:
            raise ValueError("aging_interval must be positive")
        self._hotness_store = hotness_store
        self._namespace = namespace
        self._aging_interval = aging_interval
        self._requests_since_aging = 0
        self._aging_epoch = 0
        self._node_count = 0
        self._root = _RadixNode((), (), b"", 0, None)
        self._invalid_records: dict[PrefixId, HotnessRecord] = {}

    @property
    def namespace(self) -> bytes:
        """Return the model/cache-salt namespace used for prefix identities."""
        return self._namespace

    @property
    def aging_epoch(self) -> int:
        """Return the number of completed Local aging passes."""
        return self._aging_epoch

    @property
    def node_count(self) -> int:
        """Return the number of non-root radix nodes without traversing."""
        return self._node_count

    def publish(self, token_ids: Sequence[int]) -> tuple[HotPrefixNodeSnapshot, ...]:
        """Publish a computed token path into the logical radix tree.

        Args:
            token_ids: Complete token path whose KV became cacheable.

        Returns:
            Snapshots of nodes created by this publication.
        """
        remaining = self._validate_tokens(token_ids)
        node = self._root
        created: list[_RadixNode] = []
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                new_node = self._new_node(node, remaining)
                node.children[remaining[0]] = new_node
                created.append(new_node)
                break
            common = self._common_prefix_length(child.segment, remaining)
            if common < len(child.segment):
                split_parent = self._split_node(child, common)
                created.append(split_parent)
                node = split_parent
                remaining = remaining[common:]
                if remaining:
                    new_node = self._new_node(node, remaining)
                    node.children[remaining[0]] = new_node
                    created.append(new_node)
                break
            node = child
            remaining = remaining[common:]
        return tuple(self._snapshot_node(item) for item in created)

    def record_match(
        self, token_ids: Sequence[int], *, matched_tokens: int
    ) -> tuple[HotPrefixNodeSnapshot, ...]:
        """Record one initial request's actual native-APC matched path.

        Args:
            token_ids: Complete request token sequence.
            matched_tokens: Number of tokens vLLM native APC actually reused.

        Returns:
            Updated snapshots for every matched logical node.

        Raises:
            ValueError: If the reported native match is not represented by the
                Local HotPrefix Tree.
        """
        tokens = self._validate_tokens(token_ids)
        if matched_tokens < 0 or matched_tokens > len(tokens):
            raise ValueError("matched_tokens is outside the request token range")
        remaining = tokens[:matched_tokens]
        node = self._root
        matched_nodes: list[_RadixNode] = []
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                raise ValueError("native APC match is absent from LocalHotPrefixTree")
            common = self._common_prefix_length(child.segment, remaining)
            if common == 0:
                raise ValueError("native APC match diverges from LocalHotPrefixTree")
            if common < len(child.segment):
                if common != len(remaining):
                    raise ValueError(
                        "native APC match crosses a divergent logical segment"
                    )
                child = self._split_node(child, common)
            if child.prefix_id not in self._invalid_records:
                self._hotness_store.access(child.prefix_id)
            matched_nodes.append(child)
            node = child
            remaining = remaining[common:]
        self._requests_since_aging += 1
        if self._requests_since_aging == self._aging_interval:
            self._hotness_store.age_all()
            self._requests_since_aging = 0
            self._aging_epoch += 1
        return tuple(self._snapshot_node(item) for item in matched_nodes)

    def snapshot(self) -> tuple[HotPrefixNodeSnapshot, ...]:
        """Return all logical nodes in deterministic prefix order."""
        nodes: list[_RadixNode] = []
        stack = list(self._root.children.values())
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        nodes.sort(key=lambda item: item.full_prefix)
        return tuple(self._snapshot_node(item) for item in nodes)

    def path_snapshot(
        self, token_ids: Sequence[int]
    ) -> tuple[HotPrefixNodeSnapshot, ...]:
        """Return immutable nodes on one token path without scanning branches.

        Args:
            token_ids: Token path to follow from the radix root.

        Returns:
            Fully matched logical nodes in root-to-leaf order.
        """
        remaining = self._validate_tokens(token_ids)
        node = self._root
        path: list[HotPrefixNodeSnapshot] = []
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break
            common = self._common_prefix_length(child.segment, remaining)
            if common < len(child.segment):
                break
            path.append(self._snapshot_node(child))
            node = child
            remaining = remaining[common:]
        return tuple(path)

    def _new_node(self, parent: _RadixNode, segment: tuple[int, ...]) -> _RadixNode:
        full_prefix = parent.full_prefix + segment
        prefix_id = self._make_prefix_id(full_prefix)
        depth = parent.depth + 1
        try:
            self._hotness_store.insert(prefix_id, depth=depth)
        except CuckooInsertionError:
            self._invalid_records[prefix_id] = HotnessRecord(0, 0, depth)
        self._node_count += 1
        return _RadixNode(segment, full_prefix, prefix_id, depth, parent)

    def _split_node(self, child: _RadixNode, split_length: int) -> _RadixNode:
        if split_length <= 0 or split_length >= len(child.segment):
            raise ValueError("split_length must be inside a node segment")
        parent = child.parent
        if parent is None:
            raise RuntimeError("root node cannot be split")
        inherited = self._invalid_records.get(child.prefix_id)
        child_is_invalid = inherited is not None
        if inherited is None:
            inherited = self._hotness_store.get(child.prefix_id)
        if inherited is None:
            raise RuntimeError("radix node has no hotness record")
        parent_segment = child.segment[:split_length]
        parent_prefix = parent.full_prefix + parent_segment
        split_parent = _RadixNode(
            parent_segment,
            parent_prefix,
            self._make_prefix_id(parent_prefix),
            child.depth,
            parent,
        )
        self._node_count += 1
        if child_is_invalid:
            self._invalid_records[split_parent.prefix_id] = inherited
        else:
            try:
                self._hotness_store.insert(
                    split_parent.prefix_id,
                    depth=split_parent.depth,
                    record=inherited,
                )
            except CuckooInsertionError:
                self._invalid_records[split_parent.prefix_id] = HotnessRecord(
                    0, 0, split_parent.depth
                )
        parent.children[parent_segment[0]] = split_parent
        child.segment = child.segment[split_length:]
        child.parent = split_parent
        split_parent.children[child.segment[0]] = child
        self._update_depths(child, split_parent.depth + 1)
        return split_parent

    def _update_depths(self, node: _RadixNode, depth: int) -> None:
        node.depth = depth
        if node.prefix_id in self._invalid_records:
            record = self._invalid_records[node.prefix_id]
            self._invalid_records[node.prefix_id] = HotnessRecord(
                record.frequency,
                record.clock,
                depth,
            )
        else:
            self._hotness_store.set_depth(node.prefix_id, depth)
        for child in node.children.values():
            self._update_depths(child, depth + 1)

    def _snapshot_node(self, node: _RadixNode) -> HotPrefixNodeSnapshot:
        record = self._invalid_records.get(node.prefix_id)
        valid = record is None
        if record is None:
            record = self._hotness_store.get(node.prefix_id)
        if record is None:
            raise RuntimeError("radix node has no hotness record")
        children = tuple(
            child.prefix_id
            for child in sorted(
                node.children.values(), key=lambda item: item.full_prefix
            )
        )
        parent = node.parent
        parent_prefix_id = (
            parent.prefix_id
            if parent is not None and parent is not self._root
            else None
        )
        return HotPrefixNodeSnapshot(
            node.prefix_id,
            node.full_prefix,
            node.segment,
            parent_prefix_id,
            children,
            record,
            valid,
        )

    def _make_prefix_id(self, full_prefix: tuple[int, ...]) -> PrefixId:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(len(self._namespace).to_bytes(4, "little"))
        digest.update(self._namespace)
        for token_id in full_prefix:
            digest.update(token_id.to_bytes(8, "little", signed=False))
        return digest.digest()

    def _validate_tokens(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        tokens = tuple(token_ids)
        if any(token_id < 0 for token_id in tokens):
            raise ValueError("token IDs must be non-negative")
        return tokens

    def _common_prefix_length(
        self, left: tuple[int, ...], right: tuple[int, ...]
    ) -> int:
        common = 0
        for left_token, right_token in zip(left, right):
            if left_token != right_token:
                break
            common += 1
        return common
