# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.overheads.benchmark_hotprefix_projection import run_benchmarks

pytestmark = pytest.mark.cpu_test


def test_projection_benchmark_reports_repeated_timed_path_lookups() -> None:
    results = run_benchmarks(
        iterations=2,
        warmup_iterations=1,
        repetitions=2,
    )

    assert results
    for result in results:
        assert len(result.slow_samples_seconds) == 2
        assert len(result.phase_a_samples_seconds) == 2
        assert result.slow_path_lookups_timed == 4
        assert result.phase_a_path_lookups_timed == 4
        assert result.slow_median_seconds >= 0
        assert result.phase_a_median_seconds >= 0
        assert result.slow_mad_seconds >= 0
        assert result.phase_a_mad_seconds >= 0
    hotness_only = next(item for item in results if item.scenario == "hotness_only")
    assert hotness_only.binding_changes == 0
    assert hotness_only.discard_calls == 0
    binding_only = next(item for item in results if item.scenario == "binding_only")
    assert binding_only.discard_calls > 0
    assert binding_only.discard_signature_keys_examined > 0
