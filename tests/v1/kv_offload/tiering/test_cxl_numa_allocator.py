# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import mmap
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.tiering.cxl_numa import allocator
from vllm.v1.kv_offload.tiering.cxl_numa.allocator import (
    NumaMemoryRegion,
    NumaPlacementError,
)


class FakeFunction:
    def __init__(self, implementation: Callable[..., Any]) -> None:
        self.implementation = implementation
        self.argtypes: list[Any] | None = None
        self.restype: Any = None
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.implementation(*args)


class FakeLibNuma:
    def __init__(
        self,
        *,
        available: int = 0,
        max_node: int = 1,
        node_size: int = 32 << 20,
        node_free: int = 16 << 20,
        placement_node: int = 0,
        move_pages_result: int = 0,
    ) -> None:
        self.buffers: list[ctypes.Array[Any]] = []
        self.allocated_addresses: list[int] = []
        self.queried_pages: list[int] = []

        def numa_node_size64(node: int, free_bytes: Any) -> int:
            free_bytes._obj.value = node_free
            return node_size

        def numa_alloc_onnode(size: int, node: int) -> int:
            buffer = ctypes.create_string_buffer(size)
            address = ctypes.addressof(buffer)
            ctypes.memset(address, 0xA5, size)
            self.buffers.append(buffer)
            self.allocated_addresses.append(address)
            return address

        def numa_move_pages(
            pid: int,
            count: int,
            pages: Any,
            nodes: Any,
            status: Any,
            flags: int,
        ) -> int:
            self.queried_pages = [pages[index] for index in range(count)]
            if move_pages_result == 0:
                for index in range(count):
                    status[index] = placement_node
            return move_pages_result

        self.numa_available = FakeFunction(lambda: available)
        self.numa_max_node = FakeFunction(lambda: max_node)
        self.numa_node_size64 = FakeFunction(numa_node_size64)
        self.numa_alloc_onnode = FakeFunction(numa_alloc_onnode)
        self.numa_move_pages = FakeFunction(numa_move_pages)
        self.numa_free = FakeFunction(lambda address, size: None)


class FakeLibC:
    def __init__(self) -> None:
        self.memset = FakeFunction(ctypes.memset)


def _patch_numa_environment(
    monkeypatch: pytest.MonkeyPatch,
    libnuma: FakeLibNuma,
    *,
    allowed_nodes: set[int] | None = None,
) -> None:
    if allowed_nodes is None:
        allowed_nodes = {0, 1}
    monkeypatch.setattr(allocator.sys, "platform", "linux")
    monkeypatch.setattr(allocator, "get_libnuma", lambda: libnuma)
    monkeypatch.setattr(allocator, "_read_allowed_memory_nodes", lambda: allowed_nodes)


def test_allocate_rejects_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(allocator.sys, "platform", "darwin")
    monkeypatch.setattr(
        allocator,
        "get_libnuma",
        lambda: pytest.fail("libnuma must not be loaded on non-Linux platforms"),
    )

    with pytest.raises(RuntimeError, match="Linux"):
        NumaMemoryRegion.allocate(4096, 0)


def test_allocate_rejects_missing_libnuma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(allocator.sys, "platform", "linux")
    monkeypatch.setattr(allocator, "get_libnuma", lambda: None)

    with pytest.raises(RuntimeError, match="libnuma"):
        NumaMemoryRegion.allocate(4096, 0)


@pytest.mark.parametrize(
    ("numa_node", "error_type"),
    [
        (-1, ValueError),
        (2, ValueError),
        (True, TypeError),
        (0.0, TypeError),
    ],
)
def test_allocate_rejects_invalid_node(
    monkeypatch: pytest.MonkeyPatch,
    numa_node: Any,
    error_type: type[Exception],
) -> None:
    libnuma = FakeLibNuma(max_node=1)
    _patch_numa_environment(monkeypatch, libnuma)

    with pytest.raises(error_type, match="numa_node"):
        NumaMemoryRegion.allocate(4096, numa_node)


@pytest.mark.parametrize(
    ("size_bytes", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
    ],
)
def test_allocate_rejects_invalid_size(
    monkeypatch: pytest.MonkeyPatch,
    size_bytes: Any,
    error_type: type[Exception],
) -> None:
    libnuma = FakeLibNuma()
    _patch_numa_environment(monkeypatch, libnuma)

    with pytest.raises(error_type, match="size_bytes"):
        NumaMemoryRegion.allocate(size_bytes, 0)


@pytest.mark.parametrize(
    ("sample_stride_bytes", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
    ],
)
def test_allocate_rejects_invalid_sample_stride(
    monkeypatch: pytest.MonkeyPatch,
    sample_stride_bytes: Any,
    error_type: type[Exception],
) -> None:
    libnuma = FakeLibNuma()
    _patch_numa_environment(monkeypatch, libnuma)

    with pytest.raises(error_type, match="sample_stride_bytes"):
        NumaMemoryRegion.allocate(
            4096,
            0,
            sample_stride_bytes=sample_stride_bytes,
        )


def test_allocate_rejects_unavailable_numa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma(available=-1)
    _patch_numa_environment(monkeypatch, libnuma)

    with pytest.raises(RuntimeError, match="NUMA is unavailable"):
        NumaMemoryRegion.allocate(4096, 0)


def test_allocate_rejects_size_larger_than_node_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma(node_free=4096)
    _patch_numa_environment(monkeypatch, libnuma)

    with pytest.raises(MemoryError, match="free memory"):
        NumaMemoryRegion.allocate(4097, 0)


def test_allocate_rejects_node_outside_mems_allowed_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma()
    _patch_numa_environment(monkeypatch, libnuma, allowed_nodes={1})

    with pytest.raises(ValueError, match="not allowed"):
        NumaMemoryRegion.allocate(4096, 0)


def test_prefault_zeroes_the_allocated_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma()
    libc = FakeLibC()
    _patch_numa_environment(monkeypatch, libnuma)
    monkeypatch.setattr(allocator, "_get_libc", lambda: libc)
    size_bytes = mmap.PAGESIZE * 2

    region = NumaMemoryRegion.allocate(
        size_bytes, 0, prefault=True, verify_placement=False
    )
    try:
        assert ctypes.string_at(region.address, size_bytes) == bytes(size_bytes)
        assert libc.memset.calls == [(region.address, 0, size_bytes)]
    finally:
        region.close()


def test_verified_allocation_logs_sampled_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma(placement_node=1)
    log = MagicMock()
    _patch_numa_environment(monkeypatch, libnuma)
    monkeypatch.setattr(allocator, "logger", log, raising=False)

    region = NumaMemoryRegion.allocate(mmap.PAGESIZE * 3, 1)
    try:
        log.info.assert_called_once()
        message, *arguments = log.info.call_args.args
        assert "placement verified" in message
        assert arguments[-2:] == [2, [1]]
    finally:
        region.close()


def test_query_samples_first_stride_and_last_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma()
    _patch_numa_environment(monkeypatch, libnuma)
    stride = mmap.PAGESIZE * 2
    size_bytes = mmap.PAGESIZE * 3 + 123

    region = NumaMemoryRegion.allocate(
        size_bytes,
        0,
        prefault=False,
        verify_placement=False,
        sample_stride_bytes=stride,
    )
    try:
        assert region.query_sample_nodes() == [0, 0, 0]
        assert libnuma.queried_pages == [
            region.address,
            region.address + stride,
            region.address + mmap.PAGESIZE * 3,
        ]
        pid, count, _, nodes, _, flags = libnuma.numa_move_pages.calls[0]
        assert (pid, count, nodes, flags) == (0, 3, None, 0)
    finally:
        region.close()


def test_placement_mismatch_frees_region_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma(placement_node=1)
    _patch_numa_environment(monkeypatch, libnuma)
    size_bytes = mmap.PAGESIZE * 2

    with pytest.raises(NumaPlacementError, match="NUMA node 0"):
        NumaMemoryRegion.allocate(size_bytes, 0)

    assert libnuma.numa_free.calls == [(libnuma.allocated_addresses[0], size_bytes)]


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    libnuma = FakeLibNuma()
    _patch_numa_environment(monkeypatch, libnuma)
    size_bytes = mmap.PAGESIZE
    region = NumaMemoryRegion.allocate(
        size_bytes, 0, prefault=False, verify_placement=False
    )
    address = region.address

    assert region.close() is None
    assert region.address == 0
    assert region.close() is None
    assert libnuma.numa_free.calls == [(address, size_bytes)]


def test_verify_without_prefault_faults_only_sampled_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libnuma = FakeLibNuma()
    _patch_numa_environment(monkeypatch, libnuma)
    stride = mmap.PAGESIZE * 2
    size_bytes = mmap.PAGESIZE * 4

    region = NumaMemoryRegion.allocate(
        size_bytes,
        0,
        prefault=False,
        verify_placement=True,
        sample_stride_bytes=stride,
    )
    try:
        contents = ctypes.string_at(region.address, size_bytes)
        assert contents[0] == 0
        assert contents[stride] == 0
        assert contents[mmap.PAGESIZE * 3] == 0
        assert contents[mmap.PAGESIZE : stride] == b"\xa5" * mmap.PAGESIZE
        assert contents != bytes(size_bytes)
    finally:
        region.close()


def test_move_pages_failure_frees_region(monkeypatch: pytest.MonkeyPatch) -> None:
    libnuma = FakeLibNuma(move_pages_result=-1)
    _patch_numa_environment(monkeypatch, libnuma)
    size_bytes = mmap.PAGESIZE

    with pytest.raises(NumaPlacementError, match="numa_move_pages"):
        NumaMemoryRegion.allocate(size_bytes, 0)

    assert libnuma.numa_free.calls == [(libnuma.allocated_addresses[0], size_bytes)]
