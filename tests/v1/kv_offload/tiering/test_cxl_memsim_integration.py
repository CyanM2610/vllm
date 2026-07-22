# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.cxl_memsim.client import CxlMemSimClient
from vllm.v1.kv_offload.tiering.cxl_memsim.manager import (
    CxlMemSimMetrics,
    CxlMemSimSecondaryTierManager,
)

_LIBRARY = os.getenv("VLLM_TEST_CXLMEMSIM_LIBRARY")
_CONTROL = os.getenv("VLLM_TEST_CXLMEMSIM_CONTROL")
pytestmark = pytest.mark.skipif(
    not _LIBRARY or not _CONTROL,
    reason="requires an external CXLMemSim bulk-shm server",
)


def test_native_bulk_roundtrip_preserves_bytes_and_cacheline_footprint() -> None:
    assert _LIBRARY is not None
    assert _CONTROL is not None
    client = CxlMemSimClient.open(_LIBRARY, _CONTROL, 5000)
    source = ctypes.create_string_buffer(
        bytes((index * 17) % 256 for index in range(257)), 257
    )
    destination = ctypes.create_string_buffer(257)
    try:
        write_result = client.write(17, ctypes.addressof(source), 257)
        read_result = client.read(17, ctypes.addressof(destination), 257)
    finally:
        client.close()

    assert destination.raw == source.raw
    assert write_result.cacheline_count == 5
    assert read_result.cacheline_count == 5
    assert write_result.serialization_seconds > 0
    assert read_result.serialization_seconds > 0


def test_secondary_tier_roundtrip_uses_real_native_transport() -> None:
    assert _LIBRARY is not None
    assert _CONTROL is not None
    block_size = 128
    primary = memoryview(bytearray(4 * block_size)).cast("B", shape=(4, block_size))
    tier = CxlMemSimSecondaryTierManager(
        offloading_spec=MagicMock(),
        primary_kv_view=primary,
        tier_type="cxl_memsim",
        client_library=_LIBRARY,
        control_shm_name=_CONTROL,
        cxl_bytes_to_use=2 * block_size,
        cxl_offset_bytes=4096,
        n_load_threads=1,
        n_store_threads=1,
        request_timeout_ms=5000,
    )
    key = make_offload_key(b"native-integration", 0)
    context = ReqContext(req_id="native-integration")
    expected = bytes((index * 29) % 256 for index in range(block_size))
    try:
        ctypes.memmove(tier._primary_address, expected, block_size)
        tier.submit_store(
            JobMetadata(
                job_id=1,
                keys=[key],
                block_ids=np.array([0], dtype=np.int64),
                is_promotion=False,
                req_context=context,
            )
        )
        tier.drain_jobs()
        assert list(tier.get_finished_jobs())[0].success

        destination_address = tier._primary_address + block_size
        ctypes.memset(destination_address, 0, block_size)
        tier.submit_load(
            JobMetadata(
                job_id=2,
                keys=[key],
                block_ids=np.array([1], dtype=np.int64),
                is_promotion=True,
                req_context=context,
            )
        )
        tier.drain_jobs()
        assert list(tier.get_finished_jobs())[0].success
        stats = tier.get_stats().data["data"]
        actual = ctypes.string_at(destination_address, block_size)
    finally:
        tier.shutdown()

    assert actual == expected
    assert stats[CxlMemSimMetrics.CACHELINE_COUNT][("store",)] == 2
    assert stats[CxlMemSimMetrics.CACHELINE_COUNT][("load",)] == 2
