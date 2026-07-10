# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.cxl_numa.allocator import NumaMemoryRegion
from vllm.v1.kv_offload.tiering.cxl_numa.copy_engine import (
    CopyDirection,
    CopyJobResult,
    CopyOperation,
    NumaCopyEngine,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class CXLNumaMetrics:
    TRANSFER_BYTES = "vllm:kv_offload_cxl_numa_transfer_bytes"
    TRANSFER_TIME_SECONDS = "vllm:kv_offload_cxl_numa_transfer_time_seconds"
    TRANSFER_SIZE_BYTES = "vllm:kv_offload_cxl_numa_transfer_size_bytes"
    LOOKUPS = "vllm:kv_offload_cxl_numa_lookups"
    EVICTIONS = "vllm:kv_offload_cxl_numa_evictions"
    ALLOCATION_FAILURES = "vllm:kv_offload_cxl_numa_allocation_failures"
    CACHE_USAGE_PERC = "vllm:kv_offload_cxl_numa_cache_usage_perc"
    INFLIGHT_JOBS = "vllm:kv_offload_cxl_numa_inflight_jobs"


class _SlotState(Enum):
    STORING = "storing"
    READY = "ready"


@dataclass
class _SlotRecord:
    slot_id: int
    state: _SlotState
    pin_count: int = 0
    owner_job_id: int | None = None


@dataclass
class _PendingTierJob:
    job_metadata: JobMetadata
    direction: CopyDirection
    records: Sequence[_SlotRecord]
    newly_stored_keys: Sequence[OffloadKey] = ()


class CXLNumaSecondaryTierManager(SecondaryTierManager):
    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        transfer_labels = ("direction", "numa_node")
        node_label = ("numa_node",)
        return {
            CXLNumaMetrics.TRANSFER_BYTES: OffloadingCounterMetadata(
                documentation=(
                    "Total bytes copied between CPU primary and CXL-like NUMA memory."
                ),
                labelnames=transfer_labels,
            ),
            CXLNumaMetrics.TRANSFER_TIME_SECONDS: OffloadingCounterMetadata(
                documentation=(
                    "Total whole-job copy time between CPU primary and CXL-like "
                    "NUMA memory, in seconds."
                ),
                labelnames=transfer_labels,
            ),
            CXLNumaMetrics.TRANSFER_SIZE_BYTES: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of whole-job CXL-like NUMA copy sizes, in bytes."
                ),
                labelnames=transfer_labels,
            ),
            CXLNumaMetrics.LOOKUPS: OffloadingCounterMetadata(
                documentation="Number of CXL-like NUMA cache lookups by result.",
                labelnames=("result", "numa_node"),
            ),
            CXLNumaMetrics.EVICTIONS: OffloadingCounterMetadata(
                documentation="Number of blocks evicted from CXL-like NUMA memory.",
                labelnames=node_label,
            ),
            CXLNumaMetrics.ALLOCATION_FAILURES: OffloadingCounterMetadata(
                documentation=(
                    "Number of CXL-like NUMA store jobs rejected for lack of space."
                ),
                labelnames=node_label,
            ),
            CXLNumaMetrics.CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CXL-like NUMA KV-cache slots currently allocated."
                ),
                labelnames=node_label,
            ),
            CXLNumaMetrics.INFLIGHT_JOBS: OffloadingGaugeMetadata(
                documentation="Number of in-flight CXL-like NUMA copy jobs.",
                labelnames=node_label,
            ),
        }

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        numa_node: int,
        numa_bytes_to_use: int,
        n_load_threads: int = 4,
        n_store_threads: int = 2,
        prefault: bool = True,
        verify_placement: bool = True,
        copy_fn: Callable[[int, int, int], object] | None = None,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self._validate_config(
            primary_kv_view,
            numa_node,
            numa_bytes_to_use,
            n_load_threads,
            n_store_threads,
            prefault,
            verify_placement,
        )
        assert primary_kv_view.strides is not None
        assert primary_kv_view.shape is not None
        self._block_size_bytes = primary_kv_view.strides[0]
        self._num_primary_blocks = primary_kv_view.shape[0]
        actual_size = (
            numa_bytes_to_use // self._block_size_bytes * self._block_size_bytes
        )
        if actual_size == 0:
            raise ValueError("numa_bytes_to_use must hold at least one KV block")

        self.numa_node = numa_node
        self.capacity_blocks = actual_size // self._block_size_bytes
        self._primary_address = ctypes.addressof(
            ctypes.c_char.from_buffer(primary_kv_view)
        )
        self._region = NumaMemoryRegion.allocate(
            actual_size,
            numa_node,
            prefault=prefault,
            verify_placement=verify_placement,
        )
        try:
            self._copy_engine = NumaCopyEngine(
                n_load_threads,
                n_store_threads,
                copy_fn=copy_fn,
            )
        except BaseException:
            self._region.close()
            raise

        self._records: dict[OffloadKey, _SlotRecord] = {}
        self._free_slots: deque[int] = deque(range(self.capacity_blocks))
        self._lru: OrderedDict[OffloadKey, None] = OrderedDict()
        self._pending_jobs: dict[int, _PendingTierJob] = {}
        self._finished_jobs: deque[JobResult] = deque()
        self._state_lock = threading.RLock()
        self._stats = OffloadingConnectorStats()
        self._shutdown = False
        logger.info(
            "Created CXL-like NUMA secondary tier on node %d: %d bytes, "
            "%d blocks, %d bytes per block, %d load threads, %d store "
            "threads, prefault=%s, verify_placement=%s",
            numa_node,
            actual_size,
            self.capacity_blocks,
            self._block_size_bytes,
            n_load_threads,
            n_store_threads,
            prefault,
            verify_placement,
        )

    @staticmethod
    def _validate_config(
        primary_kv_view: memoryview,
        numa_node: int,
        numa_bytes_to_use: int,
        n_load_threads: int,
        n_store_threads: int,
        prefault: bool,
        verify_placement: bool,
    ) -> None:
        if primary_kv_view.readonly:
            raise ValueError("primary_kv_view must be writable")
        if primary_kv_view.ndim != 2 or not primary_kv_view.c_contiguous:
            raise ValueError("primary_kv_view must be a C-contiguous 2-D view")
        if primary_kv_view.strides is None or primary_kv_view.strides[0] <= 0:
            raise ValueError("primary_kv_view must have a positive row stride")
        if not isinstance(numa_node, int) or isinstance(numa_node, bool):
            raise TypeError("numa_node must be an int")
        if numa_node < 0:
            raise ValueError("numa_node must be non-negative")
        if not isinstance(numa_bytes_to_use, int) or isinstance(
            numa_bytes_to_use, bool
        ):
            raise TypeError("numa_bytes_to_use must be an int")
        if numa_bytes_to_use <= 0:
            raise ValueError("numa_bytes_to_use must be positive")
        for name, value in (
            ("n_load_threads", n_load_threads),
            ("n_store_threads", n_store_threads),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(prefault, bool):
            raise TypeError("prefault must be a bool")
        if not isinstance(verify_placement, bool):
            raise TypeError("verify_placement must be a bool")

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        with self._state_lock:
            record = self._records.get(key)
            if record is None:
                result = LookupResult.MISS
            elif record.state is _SlotState.STORING:
                result = LookupResult.RETRY
            else:
                result = LookupResult.HIT
            self._stats.increase_counter(
                CXLNumaMetrics.LOOKUPS,
                labelvalues=(result.name.lower(), str(self.numa_node)),
            )
            return result

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        keys, block_ids = self._normalize_job(job_metadata)
        with self._state_lock:
            existing_ready: list[OffloadKey] = []
            new_items: list[tuple[OffloadKey, int]] = []
            seen: set[OffloadKey] = set()
            for key, block_id in zip(keys, block_ids):
                if key in seen:
                    continue
                seen.add(key)
                record = self._records.get(key)
                if record is None:
                    new_items.append((key, block_id))
                elif record.state is _SlotState.READY:
                    existing_ready.append(key)

            free_slot_ids = list(islice(self._free_slots, len(new_items)))
            slots_needed = len(new_items) - len(free_slot_ids)
            evictions: list[tuple[OffloadKey, _SlotRecord]] = []
            protected = set(keys)
            if slots_needed > 0:
                for candidate_key in self._lru:
                    record = self._records[candidate_key]
                    if candidate_key not in protected and record.pin_count == 0:
                        evictions.append((candidate_key, record))
                        if len(evictions) == slots_needed:
                            break

            if len(evictions) < slots_needed:
                self._stats.increase_counter(
                    CXLNumaMetrics.ALLOCATION_FAILURES,
                    labelvalues=(str(self.numa_node),),
                )
                self._finished_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=False)
                )
                return

            for key in existing_ready:
                self._lru.move_to_end(key)
            for _ in free_slot_ids:
                self._free_slots.popleft()
            for key, _ in evictions:
                del self._records[key]
                del self._lru[key]
            if evictions:
                self._stats.increase_counter(
                    CXLNumaMetrics.EVICTIONS,
                    len(evictions),
                    labelvalues=(str(self.numa_node),),
                )

            slot_ids = [*free_slot_ids, *(record.slot_id for _, record in evictions)]
            new_records: list[_SlotRecord] = []
            new_keys: list[OffloadKey] = []
            operations: list[CopyOperation] = []
            for (key, block_id), slot_id in zip(new_items, slot_ids):
                record = _SlotRecord(
                    slot_id=slot_id,
                    state=_SlotState.STORING,
                    owner_job_id=job_metadata.job_id,
                )
                self._records[key] = record
                new_records.append(record)
                new_keys.append(key)
                operations.append(
                    CopyOperation(
                        src_address=self._primary_block_address(block_id),
                        dst_address=self._remote_slot_address(slot_id),
                        size_bytes=self._block_size_bytes,
                    )
                )

            if not operations:
                self._finished_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=True)
                )
                return
            self._pending_jobs[job_metadata.job_id] = _PendingTierJob(
                job_metadata=job_metadata,
                direction=CopyDirection.STORE,
                records=tuple(new_records),
                newly_stored_keys=tuple(new_keys),
            )
            try:
                self._copy_engine.submit(
                    job_metadata.job_id,
                    CopyDirection.STORE,
                    operations,
                )
            except BaseException:
                del self._pending_jobs[job_metadata.job_id]
                self._rollback_store(new_keys, new_records)
                self._finished_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=False)
                )

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        keys, block_ids = self._normalize_job(job_metadata)
        with self._state_lock:
            records = [self._records.get(key) for key in keys]
            if any(
                record is None or record.state is not _SlotState.READY
                for record in records
            ):
                self._finished_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=False)
                )
                return

            ready_records = [record for record in records if record is not None]
            for key, record in zip(keys, ready_records):
                record.pin_count += 1
                self._lru.move_to_end(key)
            operations = [
                CopyOperation(
                    src_address=self._remote_slot_address(record.slot_id),
                    dst_address=self._primary_block_address(block_id),
                    size_bytes=self._block_size_bytes,
                )
                for record, block_id in zip(ready_records, block_ids)
            ]
            self._pending_jobs[job_metadata.job_id] = _PendingTierJob(
                job_metadata=job_metadata,
                direction=CopyDirection.LOAD,
                records=tuple(ready_records),
            )
            try:
                self._copy_engine.submit(
                    job_metadata.job_id,
                    CopyDirection.LOAD,
                    operations,
                )
            except BaseException:
                del self._pending_jobs[job_metadata.job_id]
                for record in ready_records:
                    record.pin_count -= 1
                self._finished_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=False)
                )

    def _normalize_job(
        self, job_metadata: JobMetadata
    ) -> tuple[tuple[OffloadKey, ...], tuple[int, ...]]:
        keys = tuple(job_metadata.keys)
        block_ids = tuple(int(block_id) for block_id in job_metadata.block_ids)
        if len(keys) != len(block_ids):
            raise ValueError("job keys and block_ids must have equal length")
        if any(
            block_id < 0 or block_id >= self._num_primary_blocks
            for block_id in block_ids
        ):
            raise ValueError("primary block_id is out of range")
        return keys, block_ids

    def _primary_block_address(self, block_id: int) -> int:
        return self._primary_address + block_id * self._block_size_bytes

    def _remote_slot_address(self, slot_id: int) -> int:
        return self._region.address + slot_id * self._block_size_bytes

    def _rollback_store(
        self, keys: Sequence[OffloadKey], records: Sequence[_SlotRecord]
    ) -> None:
        for key, record in zip(keys, records):
            if self._records.get(key) is record:
                del self._records[key]
                self._lru.pop(key, None)
                self._free_slots.append(record.slot_id)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        with self._state_lock:
            results = list(self._finished_jobs)
            self._finished_jobs.clear()
            for copy_result in self._copy_engine.get_finished():
                pending = self._pending_jobs.pop(copy_result.job_id)
                if pending.direction is CopyDirection.STORE:
                    self._complete_store(pending, copy_result.success)
                else:
                    for record in pending.records:
                        record.pin_count -= 1
                self._record_transfer(copy_result)
                results.append(
                    JobResult(
                        job_id=copy_result.job_id,
                        success=copy_result.success,
                    )
                )
            return results

    def _complete_store(self, pending: _PendingTierJob, success: bool) -> None:
        if not success:
            self._rollback_store(pending.newly_stored_keys, pending.records)
            return
        for key, record in zip(pending.newly_stored_keys, pending.records):
            if self._records.get(key) is not record:
                continue
            record.state = _SlotState.READY
            record.owner_job_id = None
            self._lru[key] = None

    def _record_transfer(self, result: CopyJobResult) -> None:
        labels = (result.direction.value, str(self.numa_node))
        self._stats.increase_counter(
            CXLNumaMetrics.TRANSFER_BYTES,
            result.num_bytes,
            labelvalues=labels,
        )
        self._stats.increase_counter(
            CXLNumaMetrics.TRANSFER_TIME_SECONDS,
            result.elapsed_seconds,
            labelvalues=labels,
        )
        self._stats.observe_histogram(
            CXLNumaMetrics.TRANSFER_SIZE_BYTES,
            result.num_bytes,
            labelvalues=labels,
        )

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        with self._state_lock:
            for key in keys:
                record = self._records.get(key)
                if record is not None and record.state is _SlotState.READY:
                    self._lru.move_to_end(key)

    @override
    def drain_jobs(self) -> None:
        self._copy_engine.drain()

    @override
    def has_pending_work(self) -> bool:
        return self._copy_engine.inflight_jobs > 0

    @override
    def get_stats(self) -> OffloadingConnectorStats:
        with self._state_lock:
            stats = self._stats
            self._stats = OffloadingConnectorStats()
            node_label = (str(self.numa_node),)
            stats.set_gauge(
                CXLNumaMetrics.CACHE_USAGE_PERC,
                len(self._records) / self.capacity_blocks,
                labelvalues=node_label,
            )
            stats.set_gauge(
                CXLNumaMetrics.INFLIGHT_JOBS,
                self._copy_engine.inflight_jobs,
                labelvalues=node_label,
            )
            return stats

    @override
    def shutdown(self) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._copy_engine.shutdown()
        list(self.get_finished_jobs())
        self._region.close()
