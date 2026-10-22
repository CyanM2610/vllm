"""Same-step allocation rollback for the frozen single-group P-only scheduler.

The scheduler owns selection lists, budgets, native victim choice and request
states. This module only remembers actual allocations, invalidates their
unexecuted cache publications, and cancels dependency-invalidated retrieves
which have NOT yet been placed in connector metadata. It never probes an
allocator or an asynchronous result and never cancels submitted GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationRecord:
    request: object
    bucket: str
    matched_tokens: int
    quota: int
    sequence: int
    consumed_prefix_blocks: frozenset[int]
    published_tail: tuple[tuple[int, object], ...]
    dependencies: frozenset[str]
    restore_target_blocks: frozenset[int]

    @property
    def request_id(self):
        return self.request.request_id

    @property
    def unsubmitted_restore(self):
        return self.quota == 0


def _pending_restore_tracker(connector, request):
    if connector is None:
        raise RuntimeError("unsubmitted restore has no connector")
    tracker = connector.request_trackers.get(request.request_id)
    if tracker is None or tracker.state.name != "WAITING_FOR_LOAD":
        raise RuntimeError("restore has already been submitted or is not pending")
    if request.status.name != "WAITING_FOR_REMOTE_KVS":
        raise RuntimeError("pending restore request has an unexpected native state")
    if request.num_in_flight_tokens or request.num_stale_output_tokens:
        raise RuntimeError("pending restore has prior submitted compute ownership")
    if tracker.num_scheduled_tokens != 0:
        raise RuntimeError("pending restore unexpectedly has emitted compute metadata")
    if connector.lazy_offload:
        raise RuntimeError("rollback supports the frozen non-lazy connector only")
    adapter = connector.scheduler_adapter
    # This is the existing generation-bound owner, not a new Future/result
    # check. Its end_session path retains cleanup on disconnect and shutdown.
    if adapter._session_cleanup is None:
        raise RuntimeError("restore rollback requires Hybrid owned session cleanup")
    if request.request_id not in adapter._owned_lookup_ids:
        raise RuntimeError("pending restore lost its lookup cleanup owner")
    return tracker, adapter


def cancel_unsubmitted_restore(connector, request):
    """Retire a not-yet-emitted retrieve and keep this request on recompute.

    WAITING_FOR_LOAD is changed to READY when native build_connector_meta
    consumes a retrieve. Requiring the former excludes all already-submitted
    loads. END releases logical lookup ownership; it does not invent physical
    completion or release DMA pins. The existing Hybrid cleanup owner is
    responsible for eventual acknowledged END, including transport failure.

    The caller must then free the unsubmitted target, remove skipped/inflight
    membership, reset computed progress and requeue as PREEMPTED. Its separate
    worker-init marker determines NewRequestData on the next actual dispatch.
    Do not call this for a
    normal compute allocation (its tracker is already READY).
    """
    tracker, adapter = _pending_restore_tracker(connector, request)
    adapter.end_session(request.request_id)
    # The request ID has been retired on the lookup server. Recompute must not
    # submit a fresh LOOKUP with that ID. Native BYPASS transitions to READY
    # on the next actual compute allocation and still permits normal STORE.
    tracker.allocated_block_ids.clear()
    tracker.num_scheduled_tokens = 0
    tracker.num_stored_tokens = 0
    tracker.num_vllm_hit_tokens = 0
    tracker.num_lmcache_hit_tokens = 0
    tracker.state = type(tracker.state).BYPASS_LMCACHE
    return {
        "request_id": request.request_id,
        "cleanup_state": "ownership_handed_to_generation_bound_cleanup",
        "lease_release_confirmed": False,
        "retrieve_submitted": False,
        "requeue_status": "PREEMPTED",
    }


class StepAllocationLedger:
    """One schedule call; use before connector build_meta/update_after_schedule.

    record(request, bucket, matched_tokens, quota): called exactly once after
    successful native allocate_slots and connector.update_state_after_alloc.
    For compute, call after selected-list registration but before progress is
    advanced. matched_tokens is the actual pre-quota computed/matched boundary
    used by that allocation, NOT current/future total prompt length. Record
    quota-zero async LOAD after its WAITING_FOR_REMOTE_KVS state is installed.

    rollback_closure(victim): preserves the native victim chosen by the caller.
    Returns that victim and dependency-invalidated consumers in reverse
    allocation order. Before returning, it retires unsubmitted dependent LOADs
    and removes exactly the rolled-back full-block tail cache hashes. It does
    NOT free targets, change request status, alter queues/lists or restore
    token budgets. The caller MUST do those native operations for every result.
    last_rollback records which removals were dependency invalidations.
    """

    def __init__(self, scheduler, emit=None):
        self.scheduler = scheduler
        self.manager = scheduler.kv_cache_manager
        self.pool = self.manager.block_pool
        self.block_size = scheduler.block_size
        managers = self.manager.coordinator.single_type_managers
        if (
            len(managers) != 1
            or self.manager.num_kv_cache_groups != 1
            or type(managers[0]).__name__ != "FullAttentionManager"
            or managers[0].block_size != self.block_size
            or self.pool.hash_block_size != self.block_size
        ):
            raise RuntimeError(
                "rollback requires one dense group with equal hash/block geometry"
            )
        if managers[0]._pending_cow_copies:
            raise RuntimeError("rollback does not support pending CoW copies")
        self.records = {}
        self.producers = {}
        self.sequence = 0
        self.last_rollback = None
        self.emit = emit

    def record(self, request, bucket, matched_tokens, quota):
        rid = request.request_id
        if rid in self.records:
            raise RuntimeError("request allocated twice within one ledger step")
        if matched_tokens < 0 or quota < 0:
            raise ValueError("negative allocation boundary")
        if quota == 0 and request.status.name != "WAITING_FOR_REMOTE_KVS":
            raise RuntimeError(
                "zero-quota ledger record is only an unsubmitted async restore"
            )
        group = self.manager.coordinator.single_type_managers[0]
        if group._pending_cow_copies:
            raise RuntimeError(
                "same-step CoW is outside dense equal-block rollback scope"
            )
        groups = self.manager.get_blocks(rid).blocks
        if len(groups) != 1:
            raise RuntimeError("request block table is not single-group")
        blocks = groups[0]
        b = self.block_size
        # Async external targets are unhashed because delay_cache_blocks=True.
        # Only hashed local prefix entries can consume earlier publications.
        consumed = frozenset(
            blk.block_id
            for blk in blocks[: (matched_tokens + b - 1) // b]
            if not blk.is_null and blk.block_hash is not None
        )
        deps = frozenset(self.producers[i] for i in consumed if i in self.producers)
        restore_targets = (
            frozenset(
                blk.block_id
                for blk in blocks[: (matched_tokens + b - 1) // b]
                if not blk.is_null and blk.block_hash is None
            )
            if quota == 0
            else frozenset()
        )
        tail = []
        if self.manager.enable_caching and quota:
            last = (matched_tokens + quota) // b
            if last > len(blocks):
                raise RuntimeError(
                    "scheduled quota exceeds actual allocated block table"
                )
            for i in range(matched_tokens // b, last):
                blk = blocks[i]
                if blk.is_null or blk.block_hash is None:
                    raise RuntimeError(
                        "native scheduled full tail was not published as expected"
                    )
                if blk.block_hash_num_tokens != (i + 1) * b:
                    raise RuntimeError("unexpected partial/aliased tail publication")
                if self.pool.cached_block_hashes_by_block.get(blk.block_id):
                    raise RuntimeError(
                        "rollback does not support multiple hashes per block"
                    )
                if blk.block_id in self.producers:
                    raise RuntimeError(
                        "multiple unsubmitted writers own the same full block"
                    )
                tail.append((blk.block_id, blk.block_hash))
        self.sequence += 1
        entry = AllocationRecord(
            request,
            bucket,
            matched_tokens,
            quota,
            self.sequence,
            consumed,
            tuple(tail),
            deps,
            restore_targets,
        )
        self.records[rid] = entry
        self.producers.update({i: rid for i, _ in tail})
        return entry

    def rollback_closure(self, victim):
        root = victim.request_id
        ids = {root}
        while True:
            additional = {
                rid
                for rid, record in self.records.items()
                if record.dependencies.intersection(ids)
            } - ids
            if not additional:
                break
            ids.update(additional)
        entries = sorted(
            (r for rid, r in self.records.items() if rid in ids),
            key=lambda r: r.sequence,
            reverse=True,
        )
        # Validate all local state before releasing anything. No async query,
        # result(), trial allocation or synchronization occurs here.
        for entry in entries:
            if entry.unsubmitted_restore:
                _pending_restore_tracker(self.scheduler.connector, entry.request)
            for block_id, block_hash in entry.published_tail:
                blk = self.pool.blocks[block_id]
                if (
                    blk.block_hash != block_hash
                    or self.pool.cached_block_hashes_by_block.get(block_id)
                ):
                    raise RuntimeError(
                        "rollback tail changed ownership before withdrawal"
                    )
        cleanup_receipts = []
        for entry in entries:
            if entry.unsubmitted_restore:
                cleanup_receipts.append(
                    cancel_unsubmitted_restore(self.scheduler.connector, entry.request)
                )
        invalid = {i for entry in entries for i, _ in entry.published_tail}
        if invalid:
            # Native evict_blocks drops hashes even for ref_cnt>0, but never
            # frees the blocks. Subsequent native preempt/free owns references.
            self.manager.evict_blocks(invalid)
        ordered = [r.request for r in entries]
        if root not in self.records:
            ordered.append(victim)
        self.last_rollback = {
            "native_victim": root,
            "rollback_order": [r.request_id for r in ordered],
            "dependency_invalidated": [
                r.request_id for r in ordered if r.request_id != root
            ],
            "invalidated_tail_blocks": sorted(invalid),
            "unsubmitted_restores": [
                r.request_id for r in entries if r.unsubmitted_restore
            ],
            "restore_cleanup": cleanup_receipts,
            "restore_target_blocks": {
                r.request_id: sorted(r.restore_target_blocks)
                for r in entries
                if r.unsubmitted_restore
            },
            "restore_zeroing_skip_ids": sorted(
                {
                    i
                    for r in entries
                    if r.unsubmitted_restore
                    for i in r.restore_target_blocks
                }
                & getattr(self.scheduler, "_skip_zero_block_ids", set())
            ),
            "entries": [
                {
                    "request_id": r.request_id,
                    "bucket": r.bucket,
                    "quota": r.quota,
                    "matched_tokens": r.matched_tokens,
                    "dependencies": sorted(r.dependencies),
                    "published_tail_blocks": [i for i, _ in r.published_tail],
                    "restore_target_blocks": sorted(r.restore_target_blocks),
                }
                for r in entries
            ],
        }
        for entry in entries:
            self.records.pop(entry.request_id)
            for i, _ in entry.published_tail:
                self.producers.pop(i, None)
        if self.emit is not None:
            self.emit("global_sjf_rollback", root, **self.last_rollback)
        return ordered
