# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from collections import OrderedDict, deque
from collections.abc import Collection, Iterable, Sequence
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
from vllm.v1.kv_offload.tiering.cxl_memsim.client import CxlMemSimClient
from vllm.v1.kv_offload.tiering.cxl_memsim.copy_engine import (
    CxlMemSimCopyDirection,
    CxlMemSimCopyEngine,
    CxlMemSimCopyJobResult,
    CxlMemSimCopyOperation,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class CxlMemSimMetrics:
    TRANSFER_BYTES = "vllm:kv_offload_cxl_memsim_transfer_bytes"
    WALL_TIME_SECONDS = "vllm:kv_offload_cxl_memsim_wall_time_seconds"
    MODEL_TIME_SECONDS = "vllm:kv_offload_cxl_memsim_model_time_seconds"
    HOST_COPY_TIME_SECONDS = "vllm:kv_offload_cxl_memsim_host_copy_time_seconds"
    SERIALIZATION_TIME_SECONDS = "vllm:kv_offload_cxl_memsim_serialization_time_seconds"
    CACHELINE_COUNT = "vllm:kv_offload_cxl_memsim_cacheline_count"
    TRANSFER_SIZE_BYTES = "vllm:kv_offload_cxl_memsim_transfer_size_bytes"
    LOOKUPS = "vllm:kv_offload_cxl_memsim_lookups"
    EVICTIONS = "vllm:kv_offload_cxl_memsim_evictions"
    ALLOCATION_FAILURES = "vllm:kv_offload_cxl_memsim_allocation_failures"
    CACHE_USAGE_PERC = "vllm:kv_offload_cxl_memsim_cache_usage_perc"
    INFLIGHT_JOBS = "vllm:kv_offload_cxl_memsim_inflight_jobs"


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
    direction: CxlMemSimCopyDirection
    records: Sequence[_SlotRecord]
    newly_stored_keys: Sequence[OffloadKey] = ()


class CxlMemSimSecondaryTierManager(SecondaryTierManager):
    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        direction = ("direction",)
        return {
            CxlMemSimMetrics.TRANSFER_BYTES: OffloadingCounterMetadata(
                documentation="Total bytes transferred through CXLMemSim.",
                labelnames=direction,
            ),
            CxlMemSimMetrics.WALL_TIME_SECONDS: OffloadingCounterMetadata(
                documentation="Total wall-clock CXLMemSim job time in seconds.",
                labelnames=direction,
            ),
            CxlMemSimMetrics.MODEL_TIME_SECONDS: OffloadingCounterMetadata(
                documentation="Total CXL-modeled completion time in seconds.",
                labelnames=direction,
            ),
            CxlMemSimMetrics.HOST_COPY_TIME_SECONDS: OffloadingCounterMetadata(
                documentation="Total bulk host-copy time in seconds.",
                labelnames=direction,
            ),
            CxlMemSimMetrics.SERIALIZATION_TIME_SECONDS: (
                OffloadingCounterMetadata(
                    documentation="Total modeled link serialization time in seconds.",
                    labelnames=direction,
                )
            ),
            CxlMemSimMetrics.CACHELINE_COUNT: OffloadingCounterMetadata(
                documentation="Total logical 64-byte CXL transactions.",
                labelnames=direction,
            ),
            CxlMemSimMetrics.TRANSFER_SIZE_BYTES: OffloadingHistogramMetadata(
                documentation="Histogram of whole CXLMemSim job sizes in bytes.",
                labelnames=direction,
            ),
            CxlMemSimMetrics.LOOKUPS: OffloadingCounterMetadata(
                documentation="Number of CXLMemSim cache lookups by result.",
                labelnames=("result",),
            ),
            CxlMemSimMetrics.EVICTIONS: OffloadingCounterMetadata(
                documentation="Number of blocks evicted from CXLMemSim."
            ),
            CxlMemSimMetrics.ALLOCATION_FAILURES: OffloadingCounterMetadata(
                documentation="Number of CXLMemSim stores rejected for lack of space."
            ),
            CxlMemSimMetrics.CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation="Fraction of CXLMemSim KV-cache slots allocated."
            ),
            CxlMemSimMetrics.INFLIGHT_JOBS: OffloadingGaugeMetadata(
                documentation="Number of in-flight CXLMemSim copy jobs."
            ),
        }

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        client_library: str,
        control_shm_name: str,
        cxl_bytes_to_use: int,
        cxl_offset_bytes: int = 0,
        n_load_threads: int = 4,
        n_store_threads: int = 2,
        request_timeout_ms: int = 30000,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self._validate_config(
            primary_kv_view,
            client_library,
            control_shm_name,
            cxl_bytes_to_use,
            cxl_offset_bytes,
            n_load_threads,
            n_store_threads,
            request_timeout_ms,
        )
        assert primary_kv_view.strides is not None
        assert primary_kv_view.shape is not None
        self._block_size_bytes = primary_kv_view.strides[0]
        self._num_primary_blocks = primary_kv_view.shape[0]
        actual_size = (
            cxl_bytes_to_use // self._block_size_bytes * self._block_size_bytes
        )
        if actual_size == 0:
            raise ValueError("cxl_bytes_to_use must hold at least one KV block")

        client = CxlMemSimClient.open(
            client_library, control_shm_name, request_timeout_ms
        )
        try:
            if (
                cxl_offset_bytes > client.capacity_bytes
                or actual_size > client.capacity_bytes - cxl_offset_bytes
            ):
                raise ValueError(
                    "configured CXLMemSim range exceeds server capacity "
                    f"{client.capacity_bytes}"
                )
            copy_engine = CxlMemSimCopyEngine(client, n_load_threads, n_store_threads)
        except BaseException:
            client.close()
            raise

        self.cxl_offset_bytes = cxl_offset_bytes
        self.cxl_capacity_bytes = actual_size
        self.capacity_blocks = actual_size // self._block_size_bytes
        self._primary_address = ctypes.addressof(
            ctypes.c_char.from_buffer(primary_kv_view)
        )
        self._client = client
        self._copy_engine = copy_engine
        self._records: dict[OffloadKey, _SlotRecord] = {}
        self._free_slots: deque[int] = deque(range(self.capacity_blocks))
        self._lru: OrderedDict[OffloadKey, None] = OrderedDict()
        self._pending_jobs: dict[int, _PendingTierJob] = {}
        self._finished_jobs: deque[JobResult] = deque()
        self._state_lock = threading.RLock()
        self._stats = OffloadingConnectorStats()
        self._shutdown = False
        logger.info(
            "Connected CXLMemSim secondary tier: offset %d, %d bytes, "
            "%d blocks, %d bytes per block, %d load threads, %d store threads",
            cxl_offset_bytes,
            actual_size,
            self.capacity_blocks,
            self._block_size_bytes,
            n_load_threads,
            n_store_threads,
        )

    @staticmethod
    def _validate_config(
        primary_kv_view: memoryview,
        client_library: str,
        control_shm_name: str,
        cxl_bytes_to_use: int,
        cxl_offset_bytes: int,
        n_load_threads: int,
        n_store_threads: int,
        request_timeout_ms: int,
    ) -> None:
        if primary_kv_view.readonly:
            raise ValueError("primary_kv_view must be writable")
        if primary_kv_view.ndim != 2 or not primary_kv_view.c_contiguous:
            raise ValueError("primary_kv_view must be a C-contiguous 2-D view")
        if primary_kv_view.strides is None or primary_kv_view.strides[0] <= 0:
            raise ValueError("primary_kv_view must have a positive row stride")
        for name, value in (
            ("client_library", client_library),
            ("control_shm_name", control_shm_name),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str")
            if not value:
                raise ValueError(f"{name} must be non-empty")
        for name, value, allow_zero in (
            ("cxl_bytes_to_use", cxl_bytes_to_use, False),
            ("cxl_offset_bytes", cxl_offset_bytes, True),
            ("n_load_threads", n_load_threads, False),
            ("n_store_threads", n_store_threads, False),
            ("request_timeout_ms", request_timeout_ms, False),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value < 0 or (value == 0 and not allow_zero):
                requirement = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {requirement}")

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
                CxlMemSimMetrics.LOOKUPS,
                labelvalues=(result.name.lower(),),
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
                self._stats.increase_counter(CxlMemSimMetrics.ALLOCATION_FAILURES)
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
                self._stats.increase_counter(CxlMemSimMetrics.EVICTIONS, len(evictions))

            slot_ids = [*free_slot_ids, *(record.slot_id for _, record in evictions)]
            new_records: list[_SlotRecord] = []
            new_keys: list[OffloadKey] = []
            operations: list[CxlMemSimCopyOperation] = []
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
                    CxlMemSimCopyOperation(
                        host_address=self._primary_block_address(block_id),
                        cxl_offset=self._slot_offset(slot_id),
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
                direction=CxlMemSimCopyDirection.STORE,
                records=tuple(new_records),
                newly_stored_keys=tuple(new_keys),
            )
            try:
                self._copy_engine.submit(
                    job_metadata.job_id,
                    CxlMemSimCopyDirection.STORE,
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
                CxlMemSimCopyOperation(
                    host_address=self._primary_block_address(block_id),
                    cxl_offset=self._slot_offset(record.slot_id),
                    size_bytes=self._block_size_bytes,
                )
                for record, block_id in zip(ready_records, block_ids)
            ]
            self._pending_jobs[job_metadata.job_id] = _PendingTierJob(
                job_metadata=job_metadata,
                direction=CxlMemSimCopyDirection.LOAD,
                records=tuple(ready_records),
            )
            try:
                self._copy_engine.submit(
                    job_metadata.job_id,
                    CxlMemSimCopyDirection.LOAD,
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

    def _slot_offset(self, slot_id: int) -> int:
        return self.cxl_offset_bytes + slot_id * self._block_size_bytes

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
                if pending.direction is CxlMemSimCopyDirection.STORE:
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

    def _record_transfer(self, result: CxlMemSimCopyJobResult) -> None:
        labels = (result.direction.value,)
        for metric, value in (
            (CxlMemSimMetrics.TRANSFER_BYTES, result.num_bytes),
            (CxlMemSimMetrics.WALL_TIME_SECONDS, result.elapsed_seconds),
            (CxlMemSimMetrics.MODEL_TIME_SECONDS, result.model_seconds),
            (CxlMemSimMetrics.HOST_COPY_TIME_SECONDS, result.host_copy_seconds),
            (
                CxlMemSimMetrics.SERIALIZATION_TIME_SECONDS,
                result.serialization_seconds,
            ),
            (CxlMemSimMetrics.CACHELINE_COUNT, result.cacheline_count),
        ):
            self._stats.increase_counter(metric, value, labelvalues=labels)
        self._stats.observe_histogram(
            CxlMemSimMetrics.TRANSFER_SIZE_BYTES,
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
            stats.set_gauge(
                CxlMemSimMetrics.CACHE_USAGE_PERC,
                len(self._records) / self.capacity_blocks,
            )
            stats.set_gauge(
                CxlMemSimMetrics.INFLIGHT_JOBS,
                self._copy_engine.inflight_jobs,
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
        self._client.close()
