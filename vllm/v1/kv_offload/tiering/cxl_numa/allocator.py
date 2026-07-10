# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import mmap
import sys
from functools import cache
from typing import Any

from vllm.logger import init_logger
from vllm.utils.numa_utils import get_libnuma

logger = init_logger(__name__)


class NumaPlacementError(RuntimeError):
    pass


@cache
def _get_libc() -> Any:
    return ctypes.CDLL(None)


def _configure_libc(libc: Any) -> None:
    libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    libc.memset.restype = ctypes.c_void_p


def _parse_node_list(raw: str) -> set[int]:
    nodes: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            nodes.update(range(start, end + 1))
        else:
            nodes.add(int(item))
    return nodes


def _read_allowed_memory_nodes() -> set[int]:
    with open("/proc/self/status") as status_file:
        for line in status_file:
            if line.startswith("Mems_allowed_list:"):
                return _parse_node_list(line.partition(":")[2].strip())
    return set()


def _configure_libnuma(libnuma: Any) -> None:
    libnuma.numa_available.argtypes = []
    libnuma.numa_available.restype = ctypes.c_int

    libnuma.numa_max_node.argtypes = []
    libnuma.numa_max_node.restype = ctypes.c_int

    libnuma.numa_node_size64.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    libnuma.numa_node_size64.restype = ctypes.c_longlong

    libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_alloc_onnode.restype = ctypes.c_void_p

    libnuma.numa_move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    libnuma.numa_move_pages.restype = ctypes.c_int

    libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libnuma.numa_free.restype = None


class NumaMemoryRegion:
    address: int
    size_bytes: int
    numa_node: int

    def __init__(
        self,
        address: int,
        size_bytes: int,
        numa_node: int,
        libnuma: Any,
        libc: Any,
        sample_stride_bytes: int,
    ) -> None:
        self.address = address
        self.size_bytes = size_bytes
        self.numa_node = numa_node
        self._libnuma = libnuma
        self._libc = libc
        self._sample_stride_bytes = sample_stride_bytes

    @classmethod
    def allocate(
        cls,
        size_bytes: int,
        numa_node: int,
        *,
        prefault: bool = True,
        verify_placement: bool = True,
        sample_stride_bytes: int = 2 << 20,
    ) -> "NumaMemoryRegion":
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise TypeError("size_bytes must be an int")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if not isinstance(numa_node, int) or isinstance(numa_node, bool):
            raise TypeError("numa_node must be an int")
        if numa_node < 0:
            raise ValueError("numa_node must be non-negative")
        if not isinstance(sample_stride_bytes, int) or isinstance(
            sample_stride_bytes, bool
        ):
            raise TypeError("sample_stride_bytes must be an int")
        if sample_stride_bytes <= 0:
            raise ValueError("sample_stride_bytes must be positive")
        if sys.platform != "linux":
            raise RuntimeError("NUMA allocation is supported only on Linux")

        libnuma = get_libnuma()
        if libnuma is None:
            raise RuntimeError("libnuma is not installed")
        _configure_libnuma(libnuma)
        if libnuma.numa_available() < 0:
            raise RuntimeError("NUMA is unavailable on this system")

        max_node = libnuma.numa_max_node()
        if numa_node > max_node:
            raise ValueError(
                f"numa_node {numa_node} exceeds maximum NUMA node {max_node}"
            )

        allowed_nodes = _read_allowed_memory_nodes()
        if numa_node not in allowed_nodes:
            raise ValueError(f"numa_node {numa_node} is not allowed for this process")

        free_bytes = ctypes.c_longlong()
        libnuma.numa_node_size64(numa_node, ctypes.byref(free_bytes))
        if size_bytes > free_bytes.value:
            raise MemoryError(
                f"requested {size_bytes} bytes exceeds {free_bytes.value} bytes "
                f"of free memory on NUMA node {numa_node}"
            )

        address = libnuma.numa_alloc_onnode(size_bytes, numa_node)
        if isinstance(address, ctypes.c_void_p):
            address = address.value
        if not address:
            raise MemoryError(
                f"failed to allocate {size_bytes} bytes on NUMA node {numa_node}"
            )

        libc = _get_libc()
        _configure_libc(libc)
        region = cls(
            address=int(address),
            size_bytes=size_bytes,
            numa_node=numa_node,
            libnuma=libnuma,
            libc=libc,
            sample_stride_bytes=sample_stride_bytes,
        )
        try:
            if prefault:
                region._libc.memset(region.address, 0, region.size_bytes)
            if verify_placement:
                if not prefault:
                    region._fault_sample_pages()
                sample_nodes = region.query_sample_nodes()
                if any(node != numa_node for node in sample_nodes):
                    raise NumaPlacementError(
                        f"NUMA placement verification failed for NUMA node "
                        f"{numa_node}: sampled nodes {sample_nodes}"
                    )
                logger.info(
                    "Allocated CXL-like NUMA region on node %d: %d bytes; "
                    "placement verified with %d samples on nodes %s",
                    numa_node,
                    size_bytes,
                    len(sample_nodes),
                    sorted(set(sample_nodes)),
                )
            return region
        except BaseException:
            region.close()
            raise

    def _sample_addresses(self) -> list[int]:
        offsets = list(range(0, self.size_bytes, self._sample_stride_bytes))
        last_page_offset = (self.size_bytes - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        if last_page_offset not in offsets:
            offsets.append(last_page_offset)
        return [self.address + offset for offset in offsets]

    def _fault_sample_pages(self) -> None:
        for address in self._sample_addresses():
            self._libc.memset(address, 0, 1)

    def query_sample_nodes(self) -> list[int]:
        if self.address == 0:
            raise RuntimeError("NUMA memory region is closed")

        sample_addresses = self._sample_addresses()
        pages = (ctypes.c_void_p * len(sample_addresses))(*sample_addresses)
        status = (ctypes.c_int * len(sample_addresses))()
        result = self._libnuma.numa_move_pages(
            0, len(sample_addresses), pages, None, status, 0
        )
        if result != 0:
            raise NumaPlacementError(
                f"numa_move_pages failed with return code {result}"
            )
        return list(status)

    def close(self) -> None:
        if self.address == 0:
            return
        self._libnuma.numa_free(self.address, self.size_bytes)
        self.address = 0
