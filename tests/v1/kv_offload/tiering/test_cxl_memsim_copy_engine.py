# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from typing import Any

import pytest

from vllm.v1.kv_offload.tiering.cxl_memsim.client import (
    CxlMemSimTransferResult,
)
from vllm.v1.kv_offload.tiering.cxl_memsim.copy_engine import (
    CxlMemSimCopyDirection,
    CxlMemSimCopyEngine,
    CxlMemSimCopyOperation,
)


class _FakeClient:
    def __init__(self, capacity_bytes: int = 4096) -> None:
        self.capacity_bytes = capacity_bytes
        self.storage = bytearray(capacity_bytes)
        self.calls: list[tuple[str, int, int]] = []
        self.fail = False
        self.write_hook: Any = None

    def write(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult:
        if self.write_hook is not None:
            self.write_hook()
        if self.fail:
            raise RuntimeError("injected write failure")
        self.calls.append(("write", offset, size_bytes))
        self.storage[offset : offset + size_bytes] = ctypes.string_at(
            host_address, size_bytes
        )
        return self._result(size_bytes)

    def read(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult:
        if self.fail:
            raise RuntimeError("injected read failure")
        self.calls.append(("read", offset, size_bytes))
        source = (ctypes.c_char * size_bytes).from_buffer(self.storage, offset)
        ctypes.memmove(host_address, ctypes.addressof(source), size_bytes)
        return self._result(size_bytes)

    @staticmethod
    def _result(size_bytes: int) -> CxlMemSimTransferResult:
        return CxlMemSimTransferResult(
            num_bytes=size_bytes,
            host_copy_seconds=1e-6,
            model_seconds=2e-6,
            serialization_seconds=0.5e-6,
            cacheline_count=(size_bytes + 63) // 64,
        )


def _operation(buffer: ctypes.Array[Any], offset: int) -> CxlMemSimCopyOperation:
    return CxlMemSimCopyOperation(
        host_address=ctypes.addressof(buffer),
        cxl_offset=offset,
        size_bytes=len(buffer),
    )


def test_store_and_load_use_native_directions_and_aggregate_metrics() -> None:
    client = _FakeClient()
    first = ctypes.create_string_buffer(b"first", 5)
    second = ctypes.create_string_buffer(b"second", 6)
    destination = ctypes.create_string_buffer(5)
    engine = CxlMemSimCopyEngine(client, 1, 1)
    try:
        engine.submit(
            1,
            CxlMemSimCopyDirection.STORE,
            [_operation(first, 0), _operation(second, 64)],
        )
        engine.drain()
        [store_result] = engine.get_finished()

        engine.submit(
            2,
            CxlMemSimCopyDirection.LOAD,
            [_operation(destination, 0)],
        )
        engine.drain()
        [load_result] = engine.get_finished()
    finally:
        engine.shutdown()

    assert destination.raw == b"first"
    assert client.calls == [
        ("write", 0, 5),
        ("write", 64, 6),
        ("read", 0, 5),
    ]
    assert store_result.num_bytes == 11
    assert store_result.host_copy_seconds == 2e-6
    assert store_result.model_seconds == 4e-6
    assert store_result.serialization_seconds == 1e-6
    assert store_result.cacheline_count == 2
    assert load_result.direction is CxlMemSimCopyDirection.LOAD


def test_native_failure_marks_job_failed() -> None:
    client = _FakeClient()
    client.fail = True
    buffer = ctypes.create_string_buffer(b"fail", 4)
    engine = CxlMemSimCopyEngine(client, 1, 1)
    try:
        engine.submit(
            3,
            CxlMemSimCopyDirection.STORE,
            [_operation(buffer, 0)],
        )
        engine.drain()
        [result] = engine.get_finished()
    finally:
        engine.shutdown()

    assert not result.success
    assert result.num_bytes == 0
    assert result.host_copy_seconds == 0
    assert result.model_seconds == 0


def test_zero_operation_job_completes_immediately() -> None:
    engine = CxlMemSimCopyEngine(_FakeClient(), 1, 1)
    try:
        engine.submit(4, CxlMemSimCopyDirection.LOAD, [])
        [result] = engine.get_finished()
    finally:
        engine.shutdown()

    assert result.success
    assert result.num_bytes == 0
    assert engine.inflight_jobs == 0


def test_duplicate_id_is_rejected_while_drain_waits_for_active_job() -> None:
    started = threading.Event()
    release = threading.Event()
    drain_returned = threading.Event()
    client = _FakeClient()

    def wait_for_release() -> None:
        started.set()
        assert release.wait(5)

    client.write_hook = wait_for_release
    buffer = ctypes.create_string_buffer(b"wait", 4)
    engine = CxlMemSimCopyEngine(client, 2, 3)
    try:
        assert len(engine._threads) == 5
        engine.submit(5, CxlMemSimCopyDirection.STORE, [_operation(buffer, 0)])
        assert started.wait(5)
        with pytest.raises(ValueError, match="duplicate job_id"):
            engine.submit(5, CxlMemSimCopyDirection.STORE, [_operation(buffer, 64)])
        drain_thread = threading.Thread(
            target=lambda: (engine.drain(), drain_returned.set())
        )
        drain_thread.start()
        assert not drain_returned.wait(0.05)
        release.set()
        assert drain_returned.wait(5)
        drain_thread.join(5)
    finally:
        release.set()
        engine.shutdown()


def test_shutdown_is_idempotent_and_rejects_new_work() -> None:
    engine = CxlMemSimCopyEngine(_FakeClient(), 1, 1)
    engine.shutdown()
    engine.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        engine.submit(6, CxlMemSimCopyDirection.LOAD, [])


@pytest.mark.parametrize(
    ("arguments", "error_type"),
    [
        ((0, 0, 1), ValueError),
        ((1, -1, 1), ValueError),
        ((1, 0, 0), ValueError),
        ((True, 0, 1), TypeError),
    ],
)
def test_operation_validates_fields(
    arguments: tuple[Any, Any, Any], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        CxlMemSimCopyOperation(*arguments)
