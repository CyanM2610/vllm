# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.utils.numa_utils import get_libnuma
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.cxl_numa.allocator import (
    _read_allowed_memory_nodes,
)
from vllm.v1.kv_offload.tiering.cxl_numa.manager import (
    CXLNumaSecondaryTierManager,
)

_POOL_BYTES = 64 << 20
_BLOCK_BYTES = 1 << 20
_CTX = ReqContext(req_id="cxl-numa-integration")


def _target_node() -> int:
    raw = os.getenv("VLLM_TEST_CXL_NUMA_NODE")
    if raw is None:
        pytest.skip("VLLM_TEST_CXL_NUMA_NODE is not set")
    assert raw is not None
    try:
        node = int(raw)
    except ValueError:
        pytest.fail("VLLM_TEST_CXL_NUMA_NODE must be an integer")
    if node < 0:
        pytest.fail("VLLM_TEST_CXL_NUMA_NODE must be non-negative")
    return node


def _node_free_bytes(node: int) -> int | None:
    path = Path(f"/sys/devices/system/node/node{node}/meminfo")
    try:
        for line in path.read_text().splitlines():
            if "MemFree:" in line:
                return int(line.split()[-2]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _job(job_id: int, keys: list, block_ids: list[int], promotion: bool):
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=promotion,
        req_context=_CTX,
    )


@pytest.mark.optional
def test_real_remote_numa_placement_and_roundtrip() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("real NUMA placement requires Linux")
    if get_libnuma() is None:
        pytest.skip("libnuma is unavailable")

    node = _target_node()
    if node not in _read_allowed_memory_nodes():
        pytest.skip(f"NUMA node {node} is outside Mems_allowed_list")
    free_bytes = _node_free_bytes(node)
    if free_bytes is None or free_bytes < _POOL_BYTES:
        pytest.skip(f"NUMA node {node} lacks {_POOL_BYTES} free bytes")

    primary = memoryview(bytearray(4 * _BLOCK_BYTES)).cast("B", shape=(4, _BLOCK_BYTES))
    tier = CXLNumaSecondaryTierManager(
        offloading_spec=MagicMock(),
        primary_kv_view=primary,
        tier_type="cxl_numa",
        numa_node=node,
        numa_bytes_to_use=_POOL_BYTES,
        n_load_threads=2,
        n_store_threads=2,
        prefault=True,
        verify_placement=True,
    )
    try:
        expected_node = int(os.getenv("VLLM_TEST_CXL_NUMA_EXPECTED_NODE", str(node)))
        assert set(tier._region.query_sample_nodes()) == {expected_node}

        expected = (bytes([0x31]) * _BLOCK_BYTES, bytes([0xA7]) * _BLOCK_BYTES)
        ctypes.memmove(tier._primary_address, expected[0], _BLOCK_BYTES)
        ctypes.memmove(
            tier._primary_address + _BLOCK_BYTES,
            expected[1],
            _BLOCK_BYTES,
        )
        keys = [make_offload_key(b"real-a", 0), make_offload_key(b"real-b", 0)]
        tier.submit_store(_job(1, keys, [0, 1], promotion=False))
        tier.drain_jobs()
        [store_result] = tier.get_finished_jobs()
        assert store_result.success

        ctypes.memset(tier._primary_address, 0, 4 * _BLOCK_BYTES)
        tier.submit_load(_job(2, keys, [2, 3], promotion=True))
        tier.drain_jobs()
        [load_result] = tier.get_finished_jobs()
        assert load_result.success
        assert (
            ctypes.string_at(tier._primary_address + 2 * _BLOCK_BYTES, _BLOCK_BYTES)
            == expected[0]
        )
        assert (
            ctypes.string_at(tier._primary_address + 3 * _BLOCK_BYTES, _BLOCK_BYTES)
            == expected[1]
        )
    finally:
        tier.shutdown()
