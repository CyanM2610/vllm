# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.cxl_numa import manager as manager_module
from vllm.v1.kv_offload.tiering.cxl_numa.manager import (
    CXLNumaMetrics,
    CXLNumaSecondaryTierManager,
)

_CTX = ReqContext(req_id="test")


class FakeNumaRegion:
    def __init__(self, size_bytes: int, numa_node: int) -> None:
        self.buffer = ctypes.create_string_buffer(size_bytes)
        self.address = ctypes.addressof(self.buffer)
        self.size_bytes = size_bytes
        self.numa_node = numa_node
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.address = 0


class IterationLimitedDeque(deque[int]):
    def __init__(self, values: Iterator[int], max_items: int) -> None:
        super().__init__(values)
        self.max_items = max_items

    def __iter__(self) -> Iterator[int]:
        for index, value in enumerate(super().__iter__()):
            if index >= self.max_items:
                raise AssertionError("free-slot selection scanned excess entries")
            yield value


def _primary_view(num_blocks: int = 8, block_size: int = 16) -> memoryview:
    return memoryview(bytearray(num_blocks * block_size)).cast(
        "B", shape=(num_blocks, block_size)
    )


def _key(value: int) -> OffloadKey:
    return make_offload_key(str(value).encode(), 0)


def _job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int],
    *,
    is_promotion: bool = False,
) -> JobMetadata:
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def _write_primary_block(
    tier: CXLNumaSecondaryTierManager, block_id: int, data: bytes
) -> None:
    assert len(data) == tier._block_size_bytes
    ctypes.memmove(
        tier._primary_address + block_id * tier._block_size_bytes,
        data,
        len(data),
    )


def _read_primary_block(tier: CXLNumaSecondaryTierManager, block_id: int) -> bytes:
    return ctypes.string_at(
        tier._primary_address + block_id * tier._block_size_bytes,
        tier._block_size_bytes,
    )


def _finish(tier: CXLNumaSecondaryTierManager) -> list:
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


@contextmanager
def _tier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    numa_bytes_to_use: int = 64,
    block_size: int = 16,
    copy_fn: Any = None,
    n_load_threads: int = 1,
    n_store_threads: int = 1,
) -> Iterator[tuple[CXLNumaSecondaryTierManager, FakeNumaRegion, MagicMock]]:
    region = FakeNumaRegion(
        size_bytes=numa_bytes_to_use // block_size * block_size,
        numa_node=1,
    )
    allocate = MagicMock(return_value=region)
    monkeypatch.setattr(manager_module.NumaMemoryRegion, "allocate", allocate)
    tier = CXLNumaSecondaryTierManager(
        offloading_spec=MagicMock(),
        primary_kv_view=_primary_view(block_size=block_size),
        tier_type="cxl_numa",
        numa_node=1,
        numa_bytes_to_use=numa_bytes_to_use,
        n_load_threads=n_load_threads,
        n_store_threads=n_store_threads,
        copy_fn=copy_fn,
    )
    try:
        yield tier, region, allocate
    finally:
        tier.shutdown()


def test_constructor_rounds_capacity_down_to_whole_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, numa_bytes_to_use=39, block_size=16) as (
        tier,
        region,
        allocate,
    ):
        allocate.assert_called_once_with(
            32,
            1,
            prefault=True,
            verify_placement=True,
        )
        assert tier.capacity_blocks == 2
        assert region.size_bytes == 32


def test_constructor_logs_pool_and_copy_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = MagicMock()
    monkeypatch.setattr(manager_module, "logger", log)

    with _tier(
        monkeypatch,
        numa_bytes_to_use=39,
        block_size=16,
        n_load_threads=3,
        n_store_threads=2,
    ):
        log.info.assert_called_once()
        message, *arguments = log.info.call_args.args
        assert "CXL-like NUMA secondary tier" in message
        assert arguments == [1, 32, 2, 16, 3, 2, True, True]


def test_constructor_rejects_capacity_smaller_than_one_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocate = MagicMock()
    monkeypatch.setattr(manager_module.NumaMemoryRegion, "allocate", allocate)

    with pytest.raises(ValueError, match="one KV block"):
        CXLNumaSecondaryTierManager(
            offloading_spec=MagicMock(),
            primary_kv_view=_primary_view(block_size=16),
            tier_type="cxl_numa",
            numa_node=1,
            numa_bytes_to_use=15,
        )

    allocate.assert_not_called()


@pytest.mark.parametrize("field", ["n_load_threads", "n_store_threads"])
def test_constructor_validates_threads_before_remote_allocation(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    allocate = MagicMock()
    monkeypatch.setattr(manager_module.NumaMemoryRegion, "allocate", allocate)
    kwargs = {"n_load_threads": 1, "n_store_threads": 1, field: 0}

    with pytest.raises(ValueError, match=field):
        CXLNumaSecondaryTierManager(
            offloading_spec=MagicMock(),
            primary_kv_view=_primary_view(),
            tier_type="cxl_numa",
            numa_node=1,
            numa_bytes_to_use=64,
            **kwargs,
        )

    allocate.assert_not_called()


def test_empty_lookup_is_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    with _tier(monkeypatch) as (tier, _, _):
        assert tier.lookup(_key(1), _CTX) is LookupResult.MISS


def test_store_selects_only_the_required_free_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, numa_bytes_to_use=64) as (tier, _, _):
        tier._free_slots = IterationLimitedDeque(iter(range(4)), max_items=1)
        tier.submit_store(_job(1, [_key(1)], [0]))
        assert _finish(tier)[0].success


def test_lookup_is_retry_while_store_is_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def copy(dst: int, src: int, size: int) -> None:
        started.set()
        assert release.wait(5)
        ctypes.memmove(dst, src, size)

    with _tier(monkeypatch, copy_fn=copy) as (tier, _, _):
        try:
            tier.submit_store(_job(1, [_key(1)], [0]))
            assert started.wait(5)
            assert tier.lookup(_key(1), _CTX) is LookupResult.RETRY
        finally:
            release.set()
        tier.drain_jobs()
        list(tier.get_finished_jobs())


def test_lookup_is_hit_after_store_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch) as (tier, _, _):
        key = _key(1)
        tier.submit_store(_job(1, [key], [0]))
        tier.drain_jobs()

        assert list(tier.get_finished_jobs())[0].success
        assert tier.lookup(key, _CTX) is LookupResult.HIT


def test_store_load_roundtrip_is_byte_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch) as (tier, _, _):
        key = _key(1)
        expected = b"remote-numa-data"
        _write_primary_block(tier, 0, expected)
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success

        _write_primary_block(tier, 1, bytes(16))
        tier.submit_load(_job(2, [key], [1], is_promotion=True))
        assert _finish(tier)[0].success
        assert _read_primary_block(tier, 1) == expected


def test_duplicate_ready_key_does_not_copy_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch) as (tier, _, _):
        key = _key(1)
        original = b"original-content"
        _write_primary_block(tier, 0, original)
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success

        _write_primary_block(tier, 1, b"changed-content!")
        tier.submit_store(_job(2, [key], [1]))
        assert list(tier.get_finished_jobs())[0].success

        _write_primary_block(tier, 2, bytes(16))
        tier.submit_load(_job(3, [key], [2], is_promotion=True))
        assert _finish(tier)[0].success
        assert _read_primary_block(tier, 2) == original


def test_capacity_evicts_oldest_unpinned_ready_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, numa_bytes_to_use=32) as (tier, _, _):
        keys = [_key(index) for index in range(3)]
        for job_id, key in enumerate(keys, start=1):
            _write_primary_block(tier, job_id - 1, bytes([job_id]) * 16)
            tier.submit_store(_job(job_id, [key], [job_id - 1]))
            assert _finish(tier)[0].success

        assert tier.lookup(keys[0], _CTX) is LookupResult.MISS
        assert tier.lookup(keys[1], _CTX) is LookupResult.HIT
        assert tier.lookup(keys[2], _CTX) is LookupResult.HIT


def test_pinned_key_is_not_evictable(monkeypatch: pytest.MonkeyPatch) -> None:
    load_started = threading.Event()
    release_load = threading.Event()
    primary_range = [0, 0]

    def copy(dst: int, src: int, size: int) -> None:
        if primary_range[0] <= dst < primary_range[1]:
            load_started.set()
            assert release_load.wait(5)
        ctypes.memmove(dst, src, size)

    with _tier(monkeypatch, numa_bytes_to_use=16, copy_fn=copy) as (tier, _, _):
        primary_range[:] = [
            tier._primary_address,
            tier._primary_address + tier._num_primary_blocks * 16,
        ]
        first, second = _key(1), _key(2)
        _write_primary_block(tier, 0, b"first-key-value!")
        tier.submit_store(_job(1, [first], [0]))
        assert _finish(tier)[0].success

        try:
            tier.submit_load(_job(2, [first], [1], is_promotion=True))
            assert load_started.wait(5)
            tier.submit_store(_job(3, [second], [2]))
            [failed_store] = list(tier.get_finished_jobs())
            assert (failed_store.job_id, failed_store.success) == (3, False)
            assert tier.lookup(first, _CTX) is LookupResult.HIT
            assert tier.lookup(second, _CTX) is LookupResult.MISS
        finally:
            release_load.set()
        assert _finish(tier)[0].success


def test_store_rolls_back_all_new_reservations_when_job_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_started = threading.Event()
    release_load = threading.Event()
    primary_range = [0, 0]

    def copy(dst: int, src: int, size: int) -> None:
        if primary_range[0] <= dst < primary_range[1]:
            load_started.set()
            assert release_load.wait(5)
        ctypes.memmove(dst, src, size)

    with _tier(monkeypatch, numa_bytes_to_use=32, copy_fn=copy) as (tier, _, _):
        primary_range[:] = [
            tier._primary_address,
            tier._primary_address + tier._num_primary_blocks * 16,
        ]
        existing, new_a, new_b = _key(1), _key(2), _key(3)
        _write_primary_block(tier, 0, b"existing-content")
        tier.submit_store(_job(1, [existing], [0]))
        assert _finish(tier)[0].success

        try:
            tier.submit_load(_job(2, [existing], [1], is_promotion=True))
            assert load_started.wait(5)
            tier.submit_store(_job(3, [new_a, new_b], [2, 3]))
            [failed_store] = list(tier.get_finished_jobs())
            assert not failed_store.success
            assert tier.lookup(new_a, _CTX) is LookupResult.MISS
            assert tier.lookup(new_b, _CTX) is LookupResult.MISS
            assert tier.lookup(existing, _CTX) is LookupResult.HIT
        finally:
            release_load.set()
        assert _finish(tier)[0].success


def test_failed_store_releases_reserved_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail_next = [True]

    def copy(dst: int, src: int, size: int) -> None:
        if fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("injected store failure")
        ctypes.memmove(dst, src, size)

    with _tier(monkeypatch, numa_bytes_to_use=16, copy_fn=copy) as (tier, _, _):
        first, second = _key(1), _key(2)
        tier.submit_store(_job(1, [first], [0]))
        assert not _finish(tier)[0].success
        assert tier.lookup(first, _CTX) is LookupResult.MISS

        tier.submit_store(_job(2, [second], [1]))
        assert _finish(tier)[0].success
        assert tier.lookup(second, _CTX) is LookupResult.HIT


def test_load_requires_every_key_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    with _tier(monkeypatch, numa_bytes_to_use=32) as (tier, _, _):
        ready, missing = _key(1), _key(2)
        tier.submit_store(_job(1, [ready], [0]))
        assert _finish(tier)[0].success

        tier.submit_load(_job(2, [ready, missing], [1, 2], is_promotion=True))
        [failed_load] = list(tier.get_finished_jobs())
        assert not failed_load.success

        replacements = [_key(3), _key(4)]
        tier.submit_store(_job(3, replacements, [3, 4]))
        assert _finish(tier)[0].success
        assert tier.lookup(ready, _CTX) is LookupResult.MISS
        assert all(tier.lookup(key, _CTX) is LookupResult.HIT for key in replacements)


def test_two_loads_can_read_the_same_ready_key_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    two_loads_started = threading.Event()
    primary_range = [0, 0]
    started = 0
    started_lock = threading.Lock()

    def copy(dst: int, src: int, size: int) -> None:
        nonlocal started
        if primary_range[0] <= dst < primary_range[1]:
            with started_lock:
                started += 1
                if started == 2:
                    two_loads_started.set()
            assert release.wait(5)
        ctypes.memmove(dst, src, size)

    with _tier(
        monkeypatch,
        numa_bytes_to_use=16,
        copy_fn=copy,
        n_load_threads=2,
    ) as (tier, _, _):
        primary_range[:] = [
            tier._primary_address,
            tier._primary_address + tier._num_primary_blocks * 16,
        ]
        key, replacement = _key(1), _key(2)
        _write_primary_block(tier, 0, b"concurrent-load!")
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success

        try:
            tier.submit_load(_job(2, [key], [1], is_promotion=True))
            tier.submit_load(_job(3, [key], [2], is_promotion=True))
            assert two_loads_started.wait(5)
            tier.submit_store(_job(4, [replacement], [3]))
            assert not list(tier.get_finished_jobs())[0].success
        finally:
            release.set()
        assert all(result.success for result in _finish(tier))
        assert _read_primary_block(tier, 1) == b"concurrent-load!"
        assert _read_primary_block(tier, 2) == b"concurrent-load!"

        tier.submit_store(_job(5, [replacement], [3]))
        assert _finish(tier)[0].success


def test_failed_load_unpins_every_entry_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_range = [0, 0]

    def copy(dst: int, src: int, size: int) -> None:
        if primary_range[0] <= dst < primary_range[1]:
            raise RuntimeError("injected load failure")
        ctypes.memmove(dst, src, size)

    with _tier(monkeypatch, numa_bytes_to_use=16, copy_fn=copy) as (tier, _, _):
        primary_range[:] = [
            tier._primary_address,
            tier._primary_address + tier._num_primary_blocks * 16,
        ]
        key, replacement = _key(1), _key(2)
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success

        tier.submit_load(_job(2, [key], [1], is_promotion=True))
        [failed_load] = _finish(tier)
        assert (failed_load.job_id, failed_load.success) == (2, False)

        tier.submit_store(_job(3, [replacement], [2]))
        assert _finish(tier)[0].success
        assert tier.lookup(key, _CTX) is LookupResult.MISS
        assert tier.lookup(replacement, _CTX) is LookupResult.HIT


def test_touch_updates_lru_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, numa_bytes_to_use=32) as (tier, _, _):
        first, second, third = (_key(index) for index in range(3))
        tier.submit_store(_job(1, [first], [0]))
        assert _finish(tier)[0].success
        tier.submit_store(_job(2, [second], [1]))
        assert _finish(tier)[0].success

        tier.touch([first], _CTX)
        tier.submit_store(_job(3, [third], [2]))
        assert _finish(tier)[0].success

        assert tier.lookup(first, _CTX) is LookupResult.HIT
        assert tier.lookup(second, _CTX) is LookupResult.MISS
        assert tier.lookup(third, _CTX) is LookupResult.HIT


def test_metric_definitions_have_exact_labels() -> None:
    definitions = CXLNumaSecondaryTierManager.build_metric_definitions({})

    assert definitions[CXLNumaMetrics.TRANSFER_BYTES].labelnames == (
        "direction",
        "numa_node",
    )
    assert definitions[CXLNumaMetrics.TRANSFER_TIME_SECONDS].labelnames == (
        "direction",
        "numa_node",
    )
    assert definitions[CXLNumaMetrics.TRANSFER_SIZE_BYTES].labelnames == (
        "direction",
        "numa_node",
    )
    assert definitions[CXLNumaMetrics.LOOKUPS].labelnames == (
        "result",
        "numa_node",
    )


def test_stats_record_transfers_lookups_and_reset_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, numa_bytes_to_use=32) as (tier, _, _):
        key = _key(1)
        assert tier.lookup(key, _CTX) is LookupResult.MISS
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success
        assert tier.lookup(key, _CTX) is LookupResult.HIT
        tier.submit_load(_job(2, [key], [1], is_promotion=True))
        assert _finish(tier)[0].success

        stats = tier.get_stats().data["data"]
        assert stats[CXLNumaMetrics.TRANSFER_BYTES][("store", "1")] == 16
        assert stats[CXLNumaMetrics.TRANSFER_BYTES][("load", "1")] == 16
        assert len(stats[CXLNumaMetrics.TRANSFER_SIZE_BYTES][("store", "1")]) == 1
        assert stats[CXLNumaMetrics.LOOKUPS][("miss", "1")] == 1
        assert stats[CXLNumaMetrics.LOOKUPS][("hit", "1")] == 1
        assert stats[CXLNumaMetrics.CACHE_USAGE_PERC][("1",)] == 0.5
        assert stats[CXLNumaMetrics.INFLIGHT_JOBS][("1",)] == 0

        reset_stats = tier.get_stats().data["data"]
        assert CXLNumaMetrics.TRANSFER_BYTES not in reset_stats
        assert CXLNumaMetrics.TRANSFER_SIZE_BYTES not in reset_stats
        assert reset_stats[CXLNumaMetrics.CACHE_USAGE_PERC][("1",)] == 0.5


def test_inflight_gauge_and_shutdown_wait_before_region_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy_started = threading.Event()
    release_copy = threading.Event()
    shutdown_returned = threading.Event()

    def copy(dst: int, src: int, size: int) -> None:
        copy_started.set()
        assert release_copy.wait(5)
        ctypes.memmove(dst, src, size)

    with _tier(monkeypatch, copy_fn=copy) as (tier, region, _):
        tier.submit_store(_job(1, [_key(1)], [0]))
        assert copy_started.wait(5)
        assert tier.has_pending_work()
        stats = tier.get_stats().data["data"]
        assert stats[CXLNumaMetrics.INFLIGHT_JOBS][("1",)] == 1

        def shutdown() -> None:
            tier.shutdown()
            shutdown_returned.set()

        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()
        try:
            assert not shutdown_returned.wait(0.05)
            assert not region.closed
        finally:
            release_copy.set()
        assert shutdown_returned.wait(5)
        shutdown_thread.join(5)
        assert region.closed
        tier.shutdown()


def test_on_new_request_uses_block_level_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch) as (tier, _, _):
        assert tier.on_new_request(_CTX).policy is OffloadPolicy.BLOCK_LEVEL
