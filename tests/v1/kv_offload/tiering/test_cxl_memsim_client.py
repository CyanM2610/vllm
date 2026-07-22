# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from collections.abc import Callable
from typing import Any

import pytest

from vllm.v1.kv_offload.tiering.cxl_memsim import client as client_module
from vllm.v1.kv_offload.tiering.cxl_memsim.client import CxlMemSimClient


class _FakeFunction:
    def __init__(self, implementation: Callable[..., Any]) -> None:
        self._implementation = implementation
        self.argtypes: list[Any] | None = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        return self._implementation(*args)


class _FakeLibrary:
    def __init__(self, open_result: int = 0) -> None:
        self.open_result = open_result
        self.closed_handles: list[int] = []
        self.calls: list[tuple[str, int, int, int]] = []
        self.cxl_bulk_client_open = _FakeFunction(self._open)
        self.cxl_bulk_client_close = _FakeFunction(self._close)
        self.cxl_bulk_client_capacity = _FakeFunction(lambda handle: 8192)
        self.cxl_bulk_read = _FakeFunction(self._read)
        self.cxl_bulk_write = _FakeFunction(self._write)
        self.cxl_bulk_error_string = _FakeFunction(
            lambda error_code: b"injected native error"
        )

    def _open(self, name: bytes, timeout_ms: int, out_handle: Any) -> int:
        assert name == b"/test_bulk"
        assert timeout_ms == 250
        if self.open_result == 0:
            pointer = ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_void_p))
            pointer[0] = ctypes.c_void_p(1234)
        return self.open_result

    def _close(self, handle: ctypes.c_void_p) -> None:
        self.closed_handles.append(handle.value)

    def _read(
        self,
        handle: ctypes.c_void_p,
        offset: int,
        address: ctypes.c_void_p,
        size: int,
        result: Any,
    ) -> int:
        self.calls.append(("read", offset, address.value, size))
        self._set_result(result, size)
        return 0

    def _write(
        self,
        handle: ctypes.c_void_p,
        offset: int,
        address: ctypes.c_void_p,
        size: int,
        result: Any,
    ) -> int:
        self.calls.append(("write", offset, address.value, size))
        self._set_result(result, size)
        return 0

    @staticmethod
    def _set_result(result: Any, size: int) -> None:
        native = ctypes.cast(
            result, ctypes.POINTER(client_module._NativeTransferResult)
        ).contents
        native.bytes = size
        native.host_copy_ns = 1000
        native.model_latency_ns = 2500
        native.serialization_ns = 500
        native.cacheline_count = 3


def _open_client(
    monkeypatch: pytest.MonkeyPatch, library: _FakeLibrary
) -> CxlMemSimClient:
    monkeypatch.setattr(client_module.ctypes, "CDLL", lambda path: library)
    return CxlMemSimClient.open("/fake/lib.so", "/test_bulk", 250)


def test_open_configures_abi_and_exposes_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _FakeLibrary()
    client = _open_client(monkeypatch, library)
    try:
        assert client.capacity_bytes == 8192
        assert library.cxl_bulk_read.argtypes is not None
        assert library.cxl_bulk_read.restype is ctypes.c_int
    finally:
        client.close()

    assert library.closed_handles == [1234]
    client.close()
    assert library.closed_handles == [1234]


def test_read_and_write_translate_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _FakeLibrary()
    client = _open_client(monkeypatch, library)
    buffer = ctypes.create_string_buffer(128)
    address = ctypes.addressof(buffer)
    try:
        write_result = client.write(64, address, 128)
        read_result = client.read(64, address, 128)
    finally:
        client.close()

    assert library.calls == [
        ("write", 64, address, 128),
        ("read", 64, address, 128),
    ]
    assert write_result == read_result
    assert write_result.num_bytes == 128
    assert write_result.host_copy_seconds == 1e-6
    assert write_result.model_seconds == 2.5e-6
    assert write_result.serialization_seconds == 5e-7
    assert write_result.cacheline_count == 3


def test_native_open_error_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _FakeLibrary(open_result=6)
    monkeypatch.setattr(client_module.ctypes, "CDLL", lambda path: library)

    with pytest.raises(RuntimeError, match="injected native error"):
        CxlMemSimClient.open("/fake/lib.so", "/test_bulk", 250)


@pytest.mark.parametrize(
    ("arguments", "error_type"),
    [
        (("", "/test_bulk", 250), ValueError),
        (("/fake/lib.so", "", 250), ValueError),
        (("/fake/lib.so", "/test_bulk", 0), ValueError),
        (("/fake/lib.so", "/test_bulk", True), TypeError),
    ],
)
def test_open_rejects_invalid_configuration(
    arguments: tuple[Any, Any, Any], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        CxlMemSimClient.open(*arguments)


def test_transfer_after_close_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _open_client(monkeypatch, _FakeLibrary())
    client.close()

    with pytest.raises(RuntimeError, match="closed"):
        client.read(0, 1, 1)
