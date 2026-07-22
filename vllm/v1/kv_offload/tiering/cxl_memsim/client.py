# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from dataclasses import dataclass
from typing import Any


class _NativeTransferResult(ctypes.Structure):
    _fields_ = [
        ("bytes", ctypes.c_uint64),
        ("host_copy_ns", ctypes.c_uint64),
        ("model_latency_ns", ctypes.c_uint64),
        ("serialization_ns", ctypes.c_uint64),
        ("cacheline_count", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class CxlMemSimTransferResult:
    num_bytes: int
    host_copy_seconds: float
    model_seconds: float
    serialization_seconds: float
    cacheline_count: int


class CxlMemSimClient:
    def __init__(self, library: Any, handle: ctypes.c_void_p, capacity: int) -> None:
        self._library = library
        self._handle = handle
        self.capacity_bytes = capacity
        self._condition = threading.Condition()
        self._active_calls = 0
        self._closing = False
        self._closed = False

    @classmethod
    def open(
        cls,
        library_path: str,
        control_shm_name: str,
        timeout_ms: int,
    ) -> "CxlMemSimClient":
        for name, value in (
            ("library_path", library_path),
            ("control_shm_name", control_shm_name),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str")
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool):
            raise TypeError("timeout_ms must be an int")
        if timeout_ms <= 0 or timeout_ms > 0xFFFFFFFF:
            raise ValueError("timeout_ms must be between 1 and 2^32 - 1")

        try:
            library = ctypes.CDLL(library_path)
        except OSError as exc:
            raise RuntimeError(
                f"failed to load CXLMemSim client library {library_path!r}: {exc}"
            ) from exc
        cls._configure_abi(library)
        handle = ctypes.c_void_p()
        error_code = library.cxl_bulk_client_open(
            control_shm_name.encode(), timeout_ms, ctypes.byref(handle)
        )
        if error_code != 0:
            message = cls._error_message(library, error_code)
            raise RuntimeError(f"failed to connect to CXLMemSim: {message}")
        if not handle.value:
            raise RuntimeError("CXLMemSim returned an empty client handle")

        capacity = int(library.cxl_bulk_client_capacity(handle))
        if capacity <= 0:
            library.cxl_bulk_client_close(handle)
            raise RuntimeError("CXLMemSim reported zero bulk capacity")
        return cls(library, handle, capacity)

    @staticmethod
    def _configure_abi(library: Any) -> None:
        library.cxl_bulk_client_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.cxl_bulk_client_open.restype = ctypes.c_int
        library.cxl_bulk_client_close.argtypes = [ctypes.c_void_p]
        library.cxl_bulk_client_close.restype = None
        library.cxl_bulk_client_capacity.argtypes = [ctypes.c_void_p]
        library.cxl_bulk_client_capacity.restype = ctypes.c_uint64
        transfer_args = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_NativeTransferResult),
        ]
        library.cxl_bulk_read.argtypes = transfer_args
        library.cxl_bulk_read.restype = ctypes.c_int
        library.cxl_bulk_write.argtypes = transfer_args
        library.cxl_bulk_write.restype = ctypes.c_int
        library.cxl_bulk_error_string.argtypes = [ctypes.c_int]
        library.cxl_bulk_error_string.restype = ctypes.c_char_p

    @staticmethod
    def _error_message(library: Any, error_code: int) -> str:
        raw_message = library.cxl_bulk_error_string(error_code)
        if not raw_message:
            return f"native error {error_code}"
        return raw_message.decode(errors="replace")

    def read(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult:
        return self._transfer("read", offset, host_address, size_bytes)

    def write(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult:
        return self._transfer("write", offset, host_address, size_bytes)

    def _transfer(
        self,
        direction: str,
        offset: int,
        host_address: int,
        size_bytes: int,
    ) -> CxlMemSimTransferResult:
        for name, value in (
            ("offset", offset),
            ("host_address", host_address),
            ("size_bytes", size_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if host_address <= 0:
            raise ValueError("host_address must be positive")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if offset > self.capacity_bytes or size_bytes > self.capacity_bytes - offset:
            raise ValueError("CXLMemSim transfer is out of range")

        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("CXLMemSim client is closed")
            self._active_calls += 1
        try:
            native_result = _NativeTransferResult()
            native_function = (
                self._library.cxl_bulk_read
                if direction == "read"
                else self._library.cxl_bulk_write
            )
            error_code = native_function(
                self._handle,
                offset,
                ctypes.c_void_p(host_address),
                size_bytes,
                ctypes.byref(native_result),
            )
            if error_code != 0:
                message = self._error_message(self._library, error_code)
                raise RuntimeError(f"CXLMemSim {direction} failed: {message}")
            if native_result.bytes != size_bytes:
                raise RuntimeError(
                    f"CXLMemSim {direction} completed {native_result.bytes} of "
                    f"{size_bytes} bytes"
                )
            return CxlMemSimTransferResult(
                num_bytes=int(native_result.bytes),
                host_copy_seconds=native_result.host_copy_ns / 1e9,
                model_seconds=native_result.model_latency_ns / 1e9,
                serialization_seconds=native_result.serialization_ns / 1e9,
                cacheline_count=int(native_result.cacheline_count),
            )
        finally:
            with self._condition:
                self._active_calls -= 1
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.wait_for(lambda: self._active_calls == 0)
        try:
            self._library.cxl_bulk_client_close(self._handle)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
