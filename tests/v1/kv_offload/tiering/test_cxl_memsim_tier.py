# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.cxl_memsim import manager as manager_module
from vllm.v1.kv_offload.tiering.cxl_memsim.client import (
    CxlMemSimTransferResult,
)
from vllm.v1.kv_offload.tiering.cxl_memsim.manager import (
    CxlMemSimMetrics,
    CxlMemSimSecondaryTierManager,
)

_CTX = ReqContext(req_id="test")


class _FakeClient:
    def __init__(self, capacity_bytes: int = 256) -> None:
        self.capacity_bytes = capacity_bytes
        self.storage = bytearray(capacity_bytes)
        self.closed = False
        self.calls: list[tuple[str, int, int]] = []
        self.read_hook: Callable[[], None] | None = None
        self.write_hook: Callable[[], None] | None = None
        self.fail_next_write = False

    def write(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult:
        if self.write_hook is not None:
            self.write_hook()
        if self.fail_next_write:
            self.fail_next_write = False
            raise RuntimeError("injected write failure")
        self.calls.append(("write", offset, size_bytes))
        self.storage[offset : offset + size_bytes] = ctypes.string_at(
            host_address, size_bytes
        )
        return self._result(size_bytes)

    def read(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult:
        if self.read_hook is not None:
            self.read_hook()
        self.calls.append(("read", offset, size_bytes))
        source = (ctypes.c_char * size_bytes).from_buffer(self.storage, offset)
        ctypes.memmove(host_address, ctypes.addressof(source), size_bytes)
        return self._result(size_bytes)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _result(size_bytes: int) -> CxlMemSimTransferResult:
        return CxlMemSimTransferResult(
            num_bytes=size_bytes,
            host_copy_seconds=1e-6,
            model_seconds=2e-6,
            serialization_seconds=0.5e-6,
            cacheline_count=2,
        )


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


def _write_primary(
    tier: CxlMemSimSecondaryTierManager, block_id: int, data: bytes
) -> None:
    ctypes.memmove(
        tier._primary_address + block_id * tier._block_size_bytes,
        data,
        len(data),
    )


def _read_primary(tier: CxlMemSimSecondaryTierManager, block_id: int) -> bytes:
    return ctypes.string_at(
        tier._primary_address + block_id * tier._block_size_bytes,
        tier._block_size_bytes,
    )


def _finish(tier: CxlMemSimSecondaryTierManager) -> list[Any]:
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


@contextmanager
def _tier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: _FakeClient | None = None,
    cxl_bytes_to_use: int = 64,
    cxl_offset_bytes: int = 0,
    block_size: int = 16,
    n_load_threads: int = 1,
    n_store_threads: int = 1,
) -> Iterator[tuple[CxlMemSimSecondaryTierManager, _FakeClient, MagicMock]]:
    fake_client = client or _FakeClient()
    open_client = MagicMock(return_value=fake_client)
    monkeypatch.setattr(manager_module.CxlMemSimClient, "open", open_client)
    tier = CxlMemSimSecondaryTierManager(
        offloading_spec=MagicMock(),
        primary_kv_view=_primary_view(block_size=block_size),
        tier_type="cxl_memsim",
        client_library="/fake/libcxlmemsim_client.so",
        control_shm_name="/test_bulk",
        cxl_bytes_to_use=cxl_bytes_to_use,
        cxl_offset_bytes=cxl_offset_bytes,
        n_load_threads=n_load_threads,
        n_store_threads=n_store_threads,
        request_timeout_ms=250,
    )
    try:
        yield tier, fake_client, open_client
    finally:
        tier.shutdown()


def test_constructor_connects_and_rounds_configured_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, cxl_bytes_to_use=39, cxl_offset_bytes=16) as (
        tier,
        client,
        open_client,
    ):
        open_client.assert_called_once_with(
            "/fake/libcxlmemsim_client.so", "/test_bulk", 250
        )
        assert tier.capacity_blocks == 2
        assert tier.cxl_offset_bytes == 16
        assert tier.cxl_capacity_bytes == 32
        assert not client.closed

    assert client.closed


def test_constructor_rejects_range_beyond_server_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(capacity_bytes=64)
    open_client = MagicMock(return_value=client)
    monkeypatch.setattr(manager_module.CxlMemSimClient, "open", open_client)

    with pytest.raises(ValueError, match="server capacity"):
        CxlMemSimSecondaryTierManager(
            offloading_spec=MagicMock(),
            primary_kv_view=_primary_view(),
            tier_type="cxl_memsim",
            client_library="/fake/lib.so",
            control_shm_name="/bulk",
            cxl_bytes_to_use=64,
            cxl_offset_bytes=16,
        )

    assert client.closed


def test_store_load_roundtrip_uses_offsets_and_is_byte_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, cxl_offset_bytes=32) as (tier, client, _):
        key = _key(1)
        expected = b"cxl-memsim-data!"
        _write_primary(tier, 0, expected)
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success
        assert tier.lookup(key, _CTX) is LookupResult.HIT

        _write_primary(tier, 1, bytes(16))
        tier.submit_load(_job(2, [key], [1], is_promotion=True))
        assert _finish(tier)[0].success

        assert _read_primary(tier, 1) == expected
        assert client.calls == [("write", 32, 16), ("read", 32, 16)]


def test_lookup_is_retry_while_store_is_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    client = _FakeClient()

    def wait_for_release() -> None:
        started.set()
        assert release.wait(5)

    client.write_hook = wait_for_release
    with _tier(monkeypatch, client=client) as (tier, _, _):
        try:
            key = _key(1)
            tier.submit_store(_job(1, [key], [0]))
            assert started.wait(5)
            assert tier.lookup(key, _CTX) is LookupResult.RETRY
        finally:
            release.set()
        assert _finish(tier)[0].success


def test_failed_store_rolls_back_slot_for_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    client.fail_next_write = True
    with _tier(monkeypatch, client=client, cxl_bytes_to_use=16) as (tier, _, _):
        first, second = _key(1), _key(2)
        tier.submit_store(_job(1, [first], [0]))
        assert not _finish(tier)[0].success
        assert tier.lookup(first, _CTX) is LookupResult.MISS

        tier.submit_store(_job(2, [second], [1]))
        assert _finish(tier)[0].success
        assert tier.lookup(second, _CTX) is LookupResult.HIT


def test_capacity_evicts_oldest_ready_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _tier(monkeypatch, cxl_bytes_to_use=32) as (tier, _, _):
        keys = [_key(index) for index in range(3)]
        for job_id, key in enumerate(keys, start=1):
            _write_primary(tier, job_id - 1, bytes([job_id]) * 16)
            tier.submit_store(_job(job_id, [key], [job_id - 1]))
            assert _finish(tier)[0].success

        assert tier.lookup(keys[0], _CTX) is LookupResult.MISS
        assert tier.lookup(keys[1], _CTX) is LookupResult.HIT
        assert tier.lookup(keys[2], _CTX) is LookupResult.HIT


def test_active_load_pins_only_slot_against_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    client = _FakeClient()

    def wait_for_release() -> None:
        started.set()
        assert release.wait(5)

    with _tier(monkeypatch, client=client, cxl_bytes_to_use=16) as (tier, _, _):
        first, second = _key(1), _key(2)
        _write_primary(tier, 0, b"first-key-value!")
        tier.submit_store(_job(1, [first], [0]))
        assert _finish(tier)[0].success
        client.read_hook = wait_for_release

        try:
            tier.submit_load(_job(2, [first], [1], is_promotion=True))
            assert started.wait(5)
            tier.submit_store(_job(3, [second], [2]))
            [failed] = list(tier.get_finished_jobs())
            assert (failed.job_id, failed.success) == (3, False)
        finally:
            release.set()
        assert _finish(tier)[0].success


def test_metrics_keep_wall_host_model_and_cachelines_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definitions = CxlMemSimSecondaryTierManager.build_metric_definitions({})
    assert definitions[CxlMemSimMetrics.MODEL_TIME_SECONDS].labelnames == ("direction",)
    assert definitions[CxlMemSimMetrics.HOST_COPY_TIME_SECONDS].labelnames == (
        "direction",
    )

    with _tier(monkeypatch, cxl_bytes_to_use=32) as (tier, _, _):
        key = _key(1)
        tier.submit_store(_job(1, [key], [0]))
        assert _finish(tier)[0].success
        stats = tier.get_stats().data["data"]

        assert stats[CxlMemSimMetrics.TRANSFER_BYTES][("store",)] == 16
        assert stats[CxlMemSimMetrics.HOST_COPY_TIME_SECONDS][("store",)] == 1e-6
        assert stats[CxlMemSimMetrics.MODEL_TIME_SECONDS][("store",)] == 2e-6
        assert stats[CxlMemSimMetrics.CACHELINE_COUNT][("store",)] == 2
        assert stats[CxlMemSimMetrics.CACHE_USAGE_PERC][()] == 0.5
        assert stats[CxlMemSimMetrics.INFLIGHT_JOBS][()] == 0


def test_shutdown_waits_for_transfer_before_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    shutdown_returned = threading.Event()
    client = _FakeClient()

    def wait_for_release() -> None:
        started.set()
        assert release.wait(5)

    client.write_hook = wait_for_release
    with _tier(monkeypatch, client=client) as (tier, _, _):
        tier.submit_store(_job(1, [_key(1)], [0]))
        assert started.wait(5)

        thread = threading.Thread(
            target=lambda: (tier.shutdown(), shutdown_returned.set())
        )
        thread.start()
        try:
            assert not shutdown_returned.wait(0.05)
            assert not client.closed
        finally:
            release.set()
        assert shutdown_returned.wait(5)
        thread.join(5)
        assert client.closed
