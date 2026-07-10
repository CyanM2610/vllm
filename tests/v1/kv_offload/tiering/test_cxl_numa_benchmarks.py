# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from benchmarks.kv_offload import cxl_numa_e2e, cxl_numa_microbench


def test_microbenchmark_binds_to_allowed_cpus_on_local_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cxl_numa_microbench.Path,
        "read_text",
        lambda _: "0-2,4\n",
    )
    get_affinity = MagicMock(side_effect=[{0, 2, 8}, {0, 2}])
    set_affinity = MagicMock()
    monkeypatch.setattr(cxl_numa_microbench.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(cxl_numa_microbench.os, "sched_setaffinity", set_affinity)

    report = cxl_numa_microbench._bind_to_numa_node(0)

    set_affinity.assert_called_once_with(0, {0, 2})
    assert report == {
        "numa_node": 0,
        "affinity_before": [0, 2, 8],
        "affinity_effective": [0, 2],
    }


@pytest.mark.parametrize(
    ("load_bytes", "lookup_hits", "expected_verdict"),
    [
        (0.0, 0.0, "NOT_TRIGGERED"),
        (1.0, 0.0, "FAIL"),
        (-1.0, 1.0, "FAIL"),
        (1.0, 1.0, "PASS"),
    ],
)
def test_e2e_verdict_distinguishes_missing_loads_from_broken_metrics(
    load_bytes: float,
    lookup_hits: float,
    expected_verdict: str,
) -> None:
    verdict, _ = cxl_numa_e2e._classify_scenario(
        tokens_equal=True,
        metric_delta={
            "load_bytes": load_bytes,
            "lookup_hits": lookup_hits,
            "allocation_failures": 0.0,
        },
        final_inflight_jobs=0.0,
    )

    assert verdict == expected_verdict
