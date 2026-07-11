# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _workload_module():
    return importlib.import_module("benchmarks.kv_offload.cxl_numa_workload")


def _analysis_module():
    return importlib.import_module("benchmarks.kv_offload.analyze_cxl_numa_results")


class _AsyncChunks:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def test_stream_consumer_records_ttft_and_token_ids() -> None:
    workload = _workload_module()
    chunks = _AsyncChunks(
        [
            b'data: {"choices":[{"text":"a","token_ids":[11]}]}\n\n',
            b'data: {"choices":[{"text":"b","token_ids":[12],',
            b'"finish_reason":"length"}]}\n\n',
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
            b'data: [DONE]\n\n',
        ]
    )
    ticks = iter([10.25, 10.50, 10.75])

    result = asyncio.run(
        workload._consume_sse_stream(
            chunks,
            started=10.0,
            clock=lambda: next(ticks),
        )
    )

    assert result["ttft_seconds"] == pytest.approx(0.25)
    assert result["e2e_seconds"] == pytest.approx(0.75)
    assert result["token_ids"] == [11, 12]
    assert result["finish_reason"] == "length"
    assert result["usage"]["completion_tokens"] == 2


@pytest.mark.parametrize(
    ("expected_tier", "delta", "expected_verdict"),
    [
        (
            "cold",
            {
                "local_prefix_hits": 0,
                "external_prefix_hits": 0,
                "gpu_load_bytes": 0,
                "cxl_load_bytes": 0,
                "cxl_lookup_hits": 0,
                "allocation_failures": 0,
            },
            "PASS",
        ),
        (
            "cpu",
            {
                "local_prefix_hits": 0,
                "external_prefix_hits": 8192,
                "gpu_load_bytes": 1,
                "cxl_load_bytes": 0,
                "cxl_lookup_hits": 0,
                "allocation_failures": 0,
            },
            "PASS",
        ),
        (
            "cxl",
            {
                "local_prefix_hits": 0,
                "external_prefix_hits": 8192,
                "gpu_load_bytes": 1,
                "cxl_load_bytes": 1,
                "cxl_lookup_hits": 1,
                "allocation_failures": 0,
            },
            "PASS",
        ),
        (
            "cxl",
            {
                "local_prefix_hits": 0,
                "external_prefix_hits": 0,
                "gpu_load_bytes": 0,
                "cxl_load_bytes": 0,
                "cxl_lookup_hits": 0,
                "allocation_failures": 0,
            },
            "NOT_TRIGGERED",
        ),
    ],
)
def test_expected_tier_classification(
    expected_tier: str,
    delta: dict[str, float],
    expected_verdict: str,
) -> None:
    workload = _workload_module()

    verdict, _ = workload._classify_expected_tier(
        expected_tier=expected_tier,
        tokens_equal=True,
        metric_delta=delta,
        final_inflight_jobs=0,
    )

    assert verdict == expected_verdict


def test_expected_tier_rejects_wrong_tokens_and_unfinished_jobs() -> None:
    workload = _workload_module()
    clean_delta = {
        "local_prefix_hits": 0,
        "external_prefix_hits": 0,
        "gpu_load_bytes": 0,
        "cxl_load_bytes": 0,
        "cxl_lookup_hits": 0,
        "allocation_failures": 0,
    }

    assert workload._classify_expected_tier(
        expected_tier="cold",
        tokens_equal=False,
        metric_delta=clean_delta,
        final_inflight_jobs=0,
    )[0] == "FAIL"
    assert workload._classify_expected_tier(
        expected_tier="cold",
        tokens_equal=True,
        metric_delta=clean_delta,
        final_inflight_jobs=1,
    )[0] == "FAIL"


def test_hierarchical_bootstrap_preserves_constant_runs() -> None:
    analysis = _analysis_module()

    low, high = analysis._hierarchical_bootstrap_ci(
        [[2.0, 2.0], [2.0], [2.0, 2.0, 2.0]],
        iterations=100,
        seed=7,
    )

    assert low == 2.0
    assert high == 2.0


def _proof_artifact(mode: str, ttft: float, rep: int = 1) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scenario": "proof",
        "experiment_category": "proof",
        "mode": mode,
        "profile": "standard",
        "prompt_length": 8192,
        "output_tokens": 32,
        "churn_prompts": 6,
        "run_id": f"standard_{mode}_rep{rep}",
        "second_a": {
            "ttft_seconds": ttft,
            "e2e_seconds": ttft + 0.1,
            "token_ids": [1, 2],
        },
        "verdict": "PASS",
    }


def test_summary_reports_speedup_against_matching_cold_baseline() -> None:
    analysis = _analysis_module()

    summaries = analysis.summarize_artifacts(
        [
            _proof_artifact("no_offload", 2.0),
            _proof_artifact("no_offload", 2.0),
            _proof_artifact("remote_secondary", 1.0),
            _proof_artifact("remote_secondary", 1.0),
        ],
        bootstrap_iterations=100,
        seed=11,
    )

    by_mode = {summary["mode"]: summary for summary in summaries}
    assert by_mode["no_offload"]["ttft_median_seconds"] == 2.0
    assert by_mode["remote_secondary"]["ttft_median_seconds"] == 1.0
    assert by_mode["remote_secondary"]["speedup_vs_no_offload"] == 2.0
    assert by_mode["remote_secondary"]["speedup_ci95"] == [2.0, 2.0]


def test_online_summary_reports_restart_level_cxl_hit_fraction() -> None:
    analysis = _analysis_module()
    artifact = {
        "schema_version": 1,
        "scenario": "online",
        "experiment_category": "online",
        "mode": "remote_secondary",
        "profile": "standard",
        "prompt_length": 8192,
        "output_tokens": 32,
        "working_set_size": 8,
        "concurrency": 8,
        "requests": [
            {"success": True, "ttft_seconds": 1.0, "e2e_seconds": 1.1},
            {"success": True, "ttft_seconds": 1.2, "e2e_seconds": 1.3},
        ],
        "summary": {
            "request_throughput_per_second": 10.0,
            "cxl_hit_fraction_estimate": 0.875,
        },
        "request_metric_delta": {
            "cxl_load_bytes": 1_000_000_000,
            "cxl_load_time_seconds": 1.0,
            "gpu_load_bytes": 2_000_000_000,
            "gpu_load_time_seconds": 1.0,
        },
        "verdict": "PASS",
    }

    [summary] = analysis.summarize_artifacts(
        [artifact], bootstrap_iterations=20, seed=3
    )

    assert summary["cxl_hit_fraction_median"] == 0.875
    assert summary["cxl_effective_gbps_median"] == 1.0
    assert summary["gpu_load_effective_gbps_median"] == 2.0


def test_acceptance_requires_five_passing_and_faster_remote_runs() -> None:
    analysis = _analysis_module()
    artifacts = []
    for rep in range(1, 6):
        artifacts.append(_proof_artifact("no_offload", 2.0, rep))
        artifacts.append(_proof_artifact("remote_secondary", 1.0, rep))
    summaries = analysis.summarize_artifacts(
        artifacts, bootstrap_iterations=20, seed=5
    )

    acceptance = analysis.evaluate_acceptance(artifacts, summaries)

    standard = acceptance["profiles"]["standard"]
    assert standard["mechanism_status"] == "PASS"
    assert standard["proof_performance_status"] == "PASS"
    assert standard["all_remote_restarts_faster_than_cold"] is True
    assert standard["online_status"] == "INCOMPLETE"


def test_analysis_cli_writes_json_and_csv(tmp_path: Path) -> None:
    analysis = _analysis_module()
    input_root = tmp_path / "results"
    input_root.mkdir()
    (input_root / "cold.json").write_text(
        json.dumps(_proof_artifact("no_offload", 2.0)), encoding="utf-8"
    )
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"

    exit_code = analysis.main(
        [
            "--input-root",
            str(input_root),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--bootstrap-iterations",
            "20",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["groups"]
    assert "ttft_median_seconds" in output_csv.read_text(encoding="utf-8")


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="matrix runner syntax test requires a native Linux bash",
)
def test_matrix_runner_has_valid_bash_syntax() -> None:
    script = Path("benchmarks/kv_offload/run_cxl_numa_matrix.sh")

    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="matrix runner dry-run requires a native Linux bash",
)
def test_matrix_runner_dry_run_renders_all_proof_modes(tmp_path: Path) -> None:
    script = Path("benchmarks/kv_offload/run_cxl_numa_matrix.sh")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--profile",
            "standard",
            "--phases",
            "proof",
            "--repetitions",
            "1",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "standard_no_offload_rep1" in result.stdout
    assert "standard_cpu_only_rep1" in result.stdout
    assert "standard_local_secondary_rep1" in result.stdout
    assert "standard_remote_secondary_rep1" in result.stdout
