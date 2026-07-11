# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize CXL-NUMA experiment artifacts with restart-aware bootstrap CIs."""

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_statistic(run_samples: list[list[float]], rng: random.Random) -> float:
    sampled: list[float] = []
    for _ in range(len(run_samples)):
        run = run_samples[rng.randrange(len(run_samples))]
        sampled.extend(run[rng.randrange(len(run))] for _ in range(len(run)))
    return statistics.median(sampled)


def _hierarchical_bootstrap_ci(
    run_samples: list[list[float]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if not run_samples or any(not run for run in run_samples):
        raise ValueError("bootstrap requires at least one sample in every run")
    rng = random.Random(seed)
    estimates = [_bootstrap_statistic(run_samples, rng) for _ in range(iterations)]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _speedup_bootstrap_ci(
    baseline_runs: list[list[float]],
    candidate_runs: list[list[float]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        baseline = _bootstrap_statistic(baseline_runs, rng)
        candidate = _bootstrap_statistic(candidate_runs, rng)
        estimates.append(baseline / candidate)
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


_GROUP_FIELDS = (
    "scenario",
    "experiment_category",
    "mode",
    "profile",
    "prompt_length",
    "output_tokens",
    "churn_prompts",
    "working_set_size",
    "concurrency",
    "connector_block_size",
    "n_load_threads",
    "n_store_threads",
    "cxl_capacity_bytes",
)
_COMPARISON_FIELDS = (
    "scenario",
    "experiment_category",
    "profile",
    "prompt_length",
    "output_tokens",
    "churn_prompts",
    "working_set_size",
    "concurrency",
)


def _identity(artifact: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(artifact.get(field) for field in fields)


def _run_values(artifact: dict[str, Any], key: str) -> list[float]:
    if artifact.get("scenario") == "proof":
        value = artifact.get("second_a", {}).get(key)
        return [] if value is None else [float(value)]
    return [
        float(request[key])
        for request in artifact.get("requests", [])
        if request.get("success", True) and request.get(key) is not None
    ]


def _metric_delta(artifact: dict[str, Any]) -> dict[str, float]:
    key = (
        "second_a_metric_delta"
        if artifact.get("scenario") == "proof"
        else "request_metric_delta"
    )
    return artifact.get(key, {})


def _effective_gbps(
    artifacts: list[dict[str, Any]], bytes_key: str, time_key: str
) -> list[float]:
    values: list[float] = []
    for artifact in artifacts:
        delta = _metric_delta(artifact)
        elapsed = float(delta.get(time_key, 0))
        if elapsed > 0:
            values.append(float(delta.get(bytes_key, 0)) / elapsed / 1e9)
    return values


def summarize_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        if artifact.get("scenario") in ("proof", "online"):
            grouped[_identity(artifact, _GROUP_FIELDS)].append(artifact)

    summaries: list[dict[str, Any]] = []
    samples_by_group: dict[tuple[Any, ...], list[list[float]]] = {}
    for group_key, runs in grouped.items():
        ttft_runs = [
            values
            for run in runs
            if (values := _run_values(run, "ttft_seconds"))
        ]
        if not ttft_runs:
            continue
        e2e_runs = [
            values
            for run in runs
            if (values := _run_values(run, "e2e_seconds"))
        ]
        ttfts = [value for run in ttft_runs for value in run]
        e2es = [value for run in e2e_runs for value in run]
        ci_low, ci_high = _hierarchical_bootstrap_ci(
            ttft_runs, iterations=bootstrap_iterations, seed=seed
        )
        cxl_hit_fractions = [
            float(run["summary"]["cxl_hit_fraction_estimate"])
            for run in runs
            if run.get("summary", {}).get("cxl_hit_fraction_estimate") is not None
        ]
        cxl_effective_gbps = _effective_gbps(
            runs, "cxl_load_bytes", "cxl_load_time_seconds"
        )
        gpu_effective_gbps = _effective_gbps(
            runs, "gpu_load_bytes", "gpu_load_time_seconds"
        )
        summary = dict(zip(_GROUP_FIELDS, group_key))
        summary.update(
            {
                "num_runs": len(runs),
                "num_requests": len(ttfts),
                "ttft_p50_seconds": _percentile(ttfts, 0.50),
                "ttft_median_seconds": statistics.median(ttfts),
                "ttft_p95_seconds": _percentile(ttfts, 0.95),
                "ttft_p99_seconds": _percentile(ttfts, 0.99),
                "ttft_ci95_seconds": [ci_low, ci_high],
                "e2e_median_seconds": statistics.median(e2es) if e2es else None,
                "throughput_median_requests_per_second": statistics.median(
                    throughput_values
                )
                if (
                    throughput_values := [
                        float(run["summary"]["request_throughput_per_second"])
                        for run in runs
                        if run.get("summary", {}).get("request_throughput_per_second")
                        is not None
                    ]
                )
                else None,
                "cxl_hit_fraction_median": statistics.median(cxl_hit_fractions)
                if cxl_hit_fractions
                else None,
                "cxl_effective_gbps_median": statistics.median(
                    cxl_effective_gbps
                )
                if cxl_effective_gbps
                else None,
                "gpu_load_effective_gbps_median": statistics.median(
                    gpu_effective_gbps
                )
                if gpu_effective_gbps
                else None,
                "verdict_counts": dict(
                    Counter(run.get("verdict", "UNKNOWN") for run in runs)
                ),
                "speedup_vs_no_offload": None,
                "speedup_ci95": None,
            }
        )
        summaries.append(summary)
        samples_by_group[group_key] = ttft_runs

    cold_by_comparison: dict[
        tuple[Any, ...], tuple[dict[str, Any], list[list[float]]]
    ] = {}
    for summary in summaries:
        if summary["mode"] != "no_offload":
            continue
        comparison = tuple(summary.get(field) for field in _COMPARISON_FIELDS)
        group_key = tuple(summary.get(field) for field in _GROUP_FIELDS)
        cold_by_comparison[comparison] = (summary, samples_by_group[group_key])

    for summary in summaries:
        if summary["mode"] == "no_offload":
            continue
        comparison = tuple(summary.get(field) for field in _COMPARISON_FIELDS)
        baseline_entry = cold_by_comparison.get(comparison)
        if baseline_entry is None:
            continue
        baseline, baseline_runs = baseline_entry
        group_key = tuple(summary.get(field) for field in _GROUP_FIELDS)
        candidate_runs = samples_by_group[group_key]
        low, high = _speedup_bootstrap_ci(
            baseline_runs,
            candidate_runs,
            iterations=bootstrap_iterations,
            seed=seed + 1,
        )
        summary["speedup_vs_no_offload"] = (
            baseline["ttft_median_seconds"] / summary["ttft_median_seconds"]
        )
        summary["speedup_ci95"] = [low, high]

    return sorted(
        summaries,
        key=lambda summary: tuple(str(summary.get(field)) for field in _GROUP_FIELDS),
    )


def _replicate_id(artifact: dict[str, Any]) -> int | None:
    match = re.search(r"rep(\d+)$", str(artifact.get("run_id", "")))
    return int(match.group(1)) if match else None


def evaluate_acceptance(
    artifacts: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    profiles = sorted(
        {
            str(artifact.get("profile"))
            for artifact in artifacts
            if artifact.get("profile") is not None
        }
    )
    report: dict[str, Any] = {"profiles": {}}
    for profile in profiles:
        proof_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.get("experiment_category") == "proof"
            and artifact.get("scenario") == "proof"
            and artifact.get("profile") == profile
        ]
        remote_runs = [
            artifact
            for artifact in proof_artifacts
            if artifact.get("mode") == "remote_secondary"
        ]
        remote_verdicts = [artifact.get("verdict") for artifact in remote_runs]
        if any(verdict in ("FAIL", "ERROR") for verdict in remote_verdicts):
            mechanism_status = "FAIL"
        elif len(remote_runs) >= 5 and all(
            verdict == "PASS" for verdict in remote_verdicts
        ):
            mechanism_status = "PASS"
        else:
            mechanism_status = "INCOMPLETE"

        cold_by_rep = {
            rep: float(artifact["second_a"]["ttft_seconds"])
            for artifact in proof_artifacts
            if artifact.get("mode") == "no_offload"
            and (rep := _replicate_id(artifact)) is not None
            and artifact.get("second_a", {}).get("ttft_seconds") is not None
        }
        remote_by_rep = {
            rep: float(artifact["second_a"]["ttft_seconds"])
            for artifact in remote_runs
            if (rep := _replicate_id(artifact)) is not None
            and artifact.get("second_a", {}).get("ttft_seconds") is not None
        }
        paired_reps = sorted(cold_by_rep.keys() & remote_by_rep.keys())
        all_remote_faster: bool | None = None
        if len(paired_reps) >= 5:
            all_remote_faster = all(
                remote_by_rep[rep] < cold_by_rep[rep] for rep in paired_reps
            )

        proof_summary = next(
            (
                summary
                for summary in summaries
                if summary.get("experiment_category") == "proof"
                and summary.get("scenario") == "proof"
                and summary.get("profile") == profile
                and summary.get("mode") == "remote_secondary"
                and summary.get("prompt_length") == 8192
                and summary.get("connector_block_size") == 64
            ),
            None,
        )
        speedup = proof_summary.get("speedup_vs_no_offload") if proof_summary else None
        ttft_reduction = 1 - 1 / speedup if speedup else None
        if mechanism_status == "FAIL" or all_remote_faster is False:
            proof_performance_status = "FAIL"
        elif (
            mechanism_status == "PASS"
            and all_remote_faster is True
            and ttft_reduction is not None
            and ttft_reduction >= 0.10
        ):
            proof_performance_status = "PASS"
        elif mechanism_status == "PASS" and ttft_reduction is not None:
            proof_performance_status = "FAIL"
        else:
            proof_performance_status = "INCOMPLETE"

        online_summaries = [
            summary
            for summary in summaries
            if summary.get("experiment_category") == "online"
            and summary.get("scenario") == "online"
            and summary.get("profile") == profile
        ]
        cold_online = {
            summary.get("concurrency"): summary
            for summary in online_summaries
            if summary.get("mode") == "no_offload"
        }
        remote_online = {
            summary.get("concurrency"): summary
            for summary in online_summaries
            if summary.get("mode") == "remote_secondary"
        }
        online_checks: list[dict[str, Any]] = []
        for concurrency in (1, 4, 8, 16):
            cold = cold_online.get(concurrency)
            remote = remote_online.get(concurrency)
            if cold is None or remote is None:
                continue
            cold_throughput = cold.get("throughput_median_requests_per_second")
            remote_throughput = remote.get("throughput_median_requests_per_second")
            throughput_ratio = (
                remote_throughput / cold_throughput
                if cold_throughput and remote_throughput is not None
                else None
            )
            hit_fraction = remote.get("cxl_hit_fraction_median")
            complete = (
                cold.get("num_runs", 0) >= 3 and remote.get("num_runs", 0) >= 3
            )
            passed = (
                complete
                and hit_fraction is not None
                and hit_fraction >= 0.80
                and throughput_ratio is not None
                and throughput_ratio >= 0.90
            )
            online_checks.append(
                {
                    "concurrency": concurrency,
                    "cxl_hit_fraction": hit_fraction,
                    "throughput_vs_cold": throughput_ratio,
                    "complete": complete,
                    "pass": passed,
                }
            )
        if len(online_checks) < 4 or not all(
            check["complete"] for check in online_checks
        ):
            online_status = "INCOMPLETE"
        elif all(check["pass"] for check in online_checks):
            online_status = "PASS"
        else:
            online_status = "FAIL"

        component_statuses = (
            mechanism_status,
            proof_performance_status,
            online_status,
        )
        overall_status = (
            "FAIL"
            if "FAIL" in component_statuses
            else "PASS"
            if all(status == "PASS" for status in component_statuses)
            else "INCOMPLETE"
        )
        report["profiles"][profile] = {
            "overall_status": overall_status,
            "mechanism_status": mechanism_status,
            "remote_proof_runs": len(remote_runs),
            "remote_proof_verdicts": dict(Counter(remote_verdicts)),
            "all_remote_restarts_faster_than_cold": all_remote_faster,
            "proof_speedup_vs_no_offload": speedup,
            "proof_ttft_reduction_fraction": ttft_reduction,
            "proof_performance_status": proof_performance_status,
            "online_status": online_status,
            "online_checks": online_checks,
        }
    return report


def _load_artifacts(input_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("scenario") in ("proof", "online"):
            artifacts.append(data)
    return artifacts


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_GROUP_FIELDS) + [
        "num_runs",
        "num_requests",
        "ttft_p50_seconds",
        "ttft_median_seconds",
        "ttft_p95_seconds",
        "ttft_p99_seconds",
        "ttft_ci95_seconds",
        "e2e_median_seconds",
        "throughput_median_requests_per_second",
        "cxl_hit_fraction_median",
        "cxl_effective_gbps_median",
        "gpu_load_effective_gbps_median",
        "verdict_counts",
        "speedup_vs_no_offload",
        "speedup_ci95",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in summary.items()
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=_positive_int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = _load_artifacts(args.input_root)
    summaries = summarize_artifacts(
        artifacts,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    acceptance = evaluate_acceptance(artifacts, summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_root": str(args.input_root),
                "bootstrap_iterations": args.bootstrap_iterations,
                "groups": summaries,
                "acceptance": acceptance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_csv, summaries)
    print(f"wrote {args.output_json} and {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
