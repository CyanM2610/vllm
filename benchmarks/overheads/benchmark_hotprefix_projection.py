# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU microbenchmark for HotPrefix token-to-block projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm.v1.core.hotprefix import (
    EvictionGroup,
    EvictionNode,
    ExactHotnessStore,
    HotPrefixBlockEvictionSelector,
    LocalHotPrefixTree,
)
from vllm.v1.core.hotprefix_projection import HotPrefixBlockProjection


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    iterations: int
    warmup_iterations: int
    repetitions: int
    tree_nodes: int
    path_nodes: int
    request_blocks: int
    slow_samples_seconds: tuple[float, ...]
    phase_a_samples_seconds: tuple[float, ...]
    slow_median_seconds: float
    phase_a_median_seconds: float
    slow_mad_seconds: float
    phase_a_mad_seconds: float
    reduction_percent: float
    phase_a_skipped: int
    groups_rebuilt: int
    slow_path_lookups_timed: int
    phase_a_path_lookups_timed: int
    binding_changes: int
    discard_calls: int
    discarded_blocks: int
    discard_signature_keys_examined: int
    invalidated_signatures: int
    discard_seconds: float


def _median_absolute_deviation(samples: Sequence[float]) -> float:
    center = statistics.median(samples)
    return statistics.median(abs(sample - center) for sample in samples)


def _result(
    *,
    scenario: str,
    iterations: int,
    warmup_iterations: int,
    repetitions: int,
    tree_nodes: int,
    path_nodes: int,
    request_blocks: int,
    slow_samples: list[float],
    phase_a_samples: list[float],
    phase_a_skipped: int,
    groups_rebuilt: int,
    slow_path_lookups_timed: int,
    phase_a_path_lookups_timed: int,
    binding_changes: int,
    discard_calls: int,
    discarded_blocks: int,
    discard_signature_keys_examined: int,
    invalidated_signatures: int,
    discard_seconds: float,
) -> BenchmarkResult:
    slow_median = statistics.median(slow_samples)
    phase_a_median = statistics.median(phase_a_samples)
    reduction = (
        100 * (slow_median - phase_a_median) / slow_median if slow_median else 0.0
    )
    return BenchmarkResult(
        scenario=scenario,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        repetitions=repetitions,
        tree_nodes=tree_nodes,
        path_nodes=path_nodes,
        request_blocks=request_blocks,
        slow_samples_seconds=tuple(slow_samples),
        phase_a_samples_seconds=tuple(phase_a_samples),
        slow_median_seconds=slow_median,
        phase_a_median_seconds=phase_a_median,
        slow_mad_seconds=_median_absolute_deviation(slow_samples),
        phase_a_mad_seconds=_median_absolute_deviation(phase_a_samples),
        reduction_percent=reduction,
        phase_a_skipped=phase_a_skipped,
        groups_rebuilt=groups_rebuilt,
        slow_path_lookups_timed=slow_path_lookups_timed,
        phase_a_path_lookups_timed=phase_a_path_lookups_timed,
        binding_changes=binding_changes,
        discard_calls=discard_calls,
        discarded_blocks=discarded_blocks,
        discard_signature_keys_examined=discard_signature_keys_examined,
        invalidated_signatures=invalidated_signatures,
        discard_seconds=discard_seconds,
    )


def _slow_reconcile(
    selector: HotPrefixBlockEvictionSelector,
    tree: LocalHotPrefixTree,
    token_ids: tuple[int, ...],
    physical_block_ids: tuple[int, ...],
    block_size: int,
) -> tuple[int, int]:
    path = [
        node
        for node in tree.snapshot()
        if node.valid
        and len(node.full_prefix) <= len(token_ids)
        and token_ids[: len(node.full_prefix)] == node.full_prefix
    ]
    groups: list[EvictionGroup] = []
    for node in path:
        node_end = len(node.full_prefix)
        node_start = node_end - len(node.segment)
        block_ids = tuple(
            block_id
            for index, block_id in enumerate(physical_block_ids)
            if index * block_size < node_end and (index + 1) * block_size > node_start
        )
        if not block_ids:
            continue
        groups.append(
            EvictionGroup(
                nodes=(
                    EvictionNode(
                        node.prefix_id,
                        node.record.frequency,
                        node.record.clock,
                        tree.namespace,
                        node.full_prefix,
                        physical_block_ids[: (node_end + block_size - 1) // block_size],
                    ),
                ),
                block_ids=block_ids,
                block_size=block_size,
            )
        )
    selector.update_groups(groups)
    return len(path), len(groups)


def _build_tree(
    *, tree_size: int, prefix_blocks: int, block_size: int
) -> tuple[LocalHotPrefixTree, tuple[int, ...]]:
    tree = LocalHotPrefixTree(
        hotness_store=ExactHotnessStore(),
        namespace=b"projection-benchmark",
        aging_interval=10_000_000,
    )
    shared = tuple(range(1, block_size + 1))
    tail_length = prefix_blocks * block_size - len(shared)
    target = shared + tuple(range(10_000, 10_000 + tail_length))
    tree.publish(target)
    for branch in range(1, tree_size):
        branch_token = 20_000 + branch
        branch_tokens = (
            shared
            + (branch_token,)
            + tuple(
                range(
                    30_000 + branch * tail_length,
                    30_000 + branch * tail_length + max(0, tail_length - 1),
                )
            )
        )
        tree.publish(branch_tokens)
    return tree, target


def _benchmark_repeated(
    *,
    scenario: str,
    iterations: int,
    warmup_iterations: int,
    repetitions: int,
    tree_size: int,
    prefix_blocks: int,
) -> BenchmarkResult:
    block_size = 16
    slow_samples: list[float] = []
    phase_a_samples: list[float] = []
    path_nodes = 0
    groups_rebuilt = 0
    phase_a_skipped = 0
    tree_nodes = 0
    slow_path_lookups = 0
    phase_a_path_lookups = 0
    for _repetition in range(repetitions):
        tree, target = _build_tree(
            tree_size=tree_size,
            prefix_blocks=prefix_blocks,
            block_size=block_size,
        )
        tree_nodes = tree.node_count
        block_ids = tuple(range(1_000, 1_000 + prefix_blocks))
        slow_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
        for _ in range(warmup_iterations):
            _slow_reconcile(slow_selector, tree, target, block_ids, block_size)
        started_ns = time.perf_counter_ns()
        for _ in range(iterations):
            path_nodes, _ = _slow_reconcile(
                slow_selector, tree, target, block_ids, block_size
            )
            slow_path_lookups += 1
        slow_samples.append((time.perf_counter_ns() - started_ns) / 1e9)

        fast_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
        projection = HotPrefixBlockProjection(fast_selector)
        for _ in range(warmup_iterations):
            projection.reconcile(
                namespace=tree.namespace,
                cached_tokens=len(target),
                path=tree.path_snapshot(target),
                physical_block_ids=block_ids,
                block_size=block_size,
                aging_epoch=tree.aging_epoch,
                total_tree_nodes=tree.node_count,
            )
        started_ns = time.perf_counter_ns()
        for _ in range(iterations):
            result = projection.reconcile(
                namespace=tree.namespace,
                cached_tokens=len(target),
                path=tree.path_snapshot(target),
                physical_block_ids=block_ids,
                block_size=block_size,
                aging_epoch=tree.aging_epoch,
                total_tree_nodes=tree.node_count,
            )
            assert result is not None
            phase_a_path_lookups += 1
            phase_a_skipped += int(result.skipped)
            groups_rebuilt += result.groups_rebuilt
        phase_a_samples.append((time.perf_counter_ns() - started_ns) / 1e9)
    return _result(
        scenario=scenario,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        repetitions=repetitions,
        tree_nodes=tree_nodes,
        path_nodes=path_nodes,
        request_blocks=prefix_blocks,
        slow_samples=slow_samples,
        phase_a_samples=phase_a_samples,
        phase_a_skipped=phase_a_skipped,
        groups_rebuilt=groups_rebuilt,
        slow_path_lookups_timed=slow_path_lookups,
        phase_a_path_lookups_timed=phase_a_path_lookups,
        binding_changes=0,
        discard_calls=0,
        discarded_blocks=0,
        discard_signature_keys_examined=0,
        invalidated_signatures=0,
        discard_seconds=0.0,
    )


def _benchmark_mutating(
    *, mode: str, iterations: int, warmup_iterations: int, repetitions: int
) -> BenchmarkResult:
    block_size = 16
    prefix_blocks = 32
    first_binding = tuple(range(1_000, 1_000 + prefix_blocks))
    second_binding = tuple(range(2_000, 2_000 + prefix_blocks))
    slow_samples: list[float] = []
    phase_a_samples: list[float] = []
    path_nodes = 0
    groups_rebuilt = 0
    phase_a_skipped = 0
    tree_nodes = 0
    slow_path_lookups = 0
    phase_a_path_lookups = 0
    binding_changes = 0
    discard_calls = 0
    discarded_blocks = 0
    discard_signature_keys_examined = 0
    invalidated_signatures = 0
    discard_duration_ns = 0

    def binding_for(index: int) -> tuple[int, ...]:
        if mode == "hotness":
            return first_binding
        return first_binding if index % 2 == 0 else second_binding

    for _repetition in range(repetitions):
        slow_tree, slow_target = _build_tree(
            tree_size=16,
            prefix_blocks=prefix_blocks,
            block_size=block_size,
        )
        tree_nodes = slow_tree.node_count
        slow_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
        slow_previous = first_binding
        for index in range(warmup_iterations):
            binding = binding_for(index)
            if mode == "hotness":
                slow_tree.record_match(slow_target, matched_tokens=len(slow_target))
            else:
                slow_selector.discard(slow_previous)
            _slow_reconcile(slow_selector, slow_tree, slow_target, binding, block_size)
            slow_previous = binding
        started_ns = time.perf_counter_ns()
        for index in range(iterations):
            binding = binding_for(warmup_iterations + index)
            if mode == "hotness":
                slow_tree.record_match(slow_target, matched_tokens=len(slow_target))
            else:
                slow_selector.discard(slow_previous)
                binding_changes += int(binding != slow_previous)
            path_nodes, _ = _slow_reconcile(
                slow_selector, slow_tree, slow_target, binding, block_size
            )
            slow_path_lookups += 1
            slow_previous = binding
        slow_samples.append((time.perf_counter_ns() - started_ns) / 1e9)

        fast_tree, fast_target = _build_tree(
            tree_size=16,
            prefix_blocks=prefix_blocks,
            block_size=block_size,
        )
        fast_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
        projection = HotPrefixBlockProjection(fast_selector)
        fast_selector.set_discard_observer(projection.discard)
        fast_previous = first_binding
        for index in range(warmup_iterations):
            binding = binding_for(index)
            if mode == "hotness":
                fast_tree.record_match(fast_target, matched_tokens=len(fast_target))
            else:
                fast_selector.discard(fast_previous)
            projection.reconcile(
                namespace=fast_tree.namespace,
                cached_tokens=len(fast_target),
                path=fast_tree.path_snapshot(fast_target),
                physical_block_ids=binding,
                block_size=block_size,
                aging_epoch=fast_tree.aging_epoch,
                total_tree_nodes=fast_tree.node_count,
            )
            fast_previous = binding
        started_ns = time.perf_counter_ns()
        for index in range(iterations):
            binding = binding_for(warmup_iterations + index)
            if mode == "hotness":
                fast_tree.record_match(fast_target, matched_tokens=len(fast_target))
            else:
                fast_selector.discard(fast_previous)
            result = projection.reconcile(
                namespace=fast_tree.namespace,
                cached_tokens=len(fast_target),
                path=fast_tree.path_snapshot(fast_target),
                physical_block_ids=binding,
                block_size=block_size,
                aging_epoch=fast_tree.aging_epoch,
                total_tree_nodes=fast_tree.node_count,
            )
            assert result is not None
            phase_a_path_lookups += 1
            phase_a_skipped += int(result.skipped)
            groups_rebuilt += result.groups_rebuilt
            discard_calls += result.discard_calls
            discarded_blocks += result.discarded_blocks
            discard_signature_keys_examined += result.discard_signature_keys_examined
            invalidated_signatures += result.invalidated_signatures
            discard_duration_ns += result.discard_duration_ns
            fast_previous = binding
        phase_a_samples.append((time.perf_counter_ns() - started_ns) / 1e9)
    return _result(
        scenario=f"{mode}_only",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        repetitions=repetitions,
        tree_nodes=tree_nodes,
        path_nodes=path_nodes,
        request_blocks=prefix_blocks,
        slow_samples=slow_samples,
        phase_a_samples=phase_a_samples,
        phase_a_skipped=phase_a_skipped,
        groups_rebuilt=groups_rebuilt,
        slow_path_lookups_timed=slow_path_lookups,
        phase_a_path_lookups_timed=phase_a_path_lookups,
        binding_changes=binding_changes,
        discard_calls=discard_calls,
        discarded_blocks=discarded_blocks,
        discard_signature_keys_examined=discard_signature_keys_examined,
        invalidated_signatures=invalidated_signatures,
        discard_seconds=discard_duration_ns / 1e9,
    )


def run_benchmarks(
    iterations: int,
    *,
    warmup_iterations: int = 100,
    repetitions: int = 5,
) -> list[BenchmarkResult]:
    """Run repeated, tree-size, prefix-length, and mutation probes.

    Args:
        iterations: Reconciliation calls per scenario.
        warmup_iterations: Untimed calls before each sample.
        repetitions: Independent scenario rebuilds per distribution.

    Returns:
        Run-level CPU and normalized-work results.
    """
    cases: Sequence[tuple[str, int, int]] = (
        ("w1_repeated", 16, 32),
        ("tree_64", 64, 32),
        ("tree_256", 256, 32),
        ("prefix_8_blocks", 16, 8),
        ("prefix_128_blocks", 16, 128),
        ("forked_tree", 128, 64),
    )
    repeated = [
        _benchmark_repeated(
            scenario=name,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            repetitions=repetitions,
            tree_size=tree_size,
            prefix_blocks=prefix_blocks,
        )
        for name, tree_size, prefix_blocks in cases
    ]
    return [
        *repeated,
        _benchmark_mutating(
            mode="hotness",
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            repetitions=repetitions,
        ),
        _benchmark_mutating(
            mode="binding",
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            repetitions=repetitions,
        ),
    ]


def _git_metadata() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip(),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
    }


def _run_process_repetitions(
    *, iterations: int, warmup_iterations: int, repetitions: int
) -> tuple[list[BenchmarkResult], list[int]]:
    script = Path(__file__).resolve()
    payloads: list[dict[str, object]] = []
    for _ in range(repetitions):
        command = [
            sys.executable,
            str(script),
            "--iterations",
            str(iterations),
            "--warmup-iterations",
            str(warmup_iterations),
            "--repetitions",
            "1",
            "--single-process",
        ]
        completed = subprocess.run(
            command,
            cwd=script.parents[2],
            text=True,
            capture_output=True,
            check=True,
        )
        payloads.append(json.loads(completed.stdout))

    first_results = payloads[0]["results"]
    assert isinstance(first_results, list)
    combined: list[BenchmarkResult] = []
    for index, first in enumerate(first_results):
        assert isinstance(first, dict)
        samples = []
        for payload in payloads:
            results = payload["results"]
            assert isinstance(results, list)
            sample = results[index]
            assert isinstance(sample, dict)
            samples.append(sample)
        combined.append(
            _result(
                scenario=str(first["scenario"]),
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                repetitions=repetitions,
                tree_nodes=int(first["tree_nodes"]),
                path_nodes=int(first["path_nodes"]),
                request_blocks=int(first["request_blocks"]),
                slow_samples=[float(item["slow_median_seconds"]) for item in samples],
                phase_a_samples=[
                    float(item["phase_a_median_seconds"]) for item in samples
                ],
                phase_a_skipped=sum(int(item["phase_a_skipped"]) for item in samples),
                groups_rebuilt=sum(int(item["groups_rebuilt"]) for item in samples),
                slow_path_lookups_timed=sum(
                    int(item["slow_path_lookups_timed"]) for item in samples
                ),
                phase_a_path_lookups_timed=sum(
                    int(item["phase_a_path_lookups_timed"]) for item in samples
                ),
                binding_changes=sum(int(item["binding_changes"]) for item in samples),
                discard_calls=sum(int(item["discard_calls"]) for item in samples),
                discarded_blocks=sum(int(item["discarded_blocks"]) for item in samples),
                discard_signature_keys_examined=sum(
                    int(item["discard_signature_keys_examined"]) for item in samples
                ),
                invalidated_signatures=sum(
                    int(item["invalidated_signatures"]) for item in samples
                ),
                discard_seconds=sum(float(item["discard_seconds"]) for item in samples),
            )
        )
    process_ids = []
    for payload in payloads:
        environment = payload["environment"]
        assert isinstance(environment, dict)
        process_ids.append(int(environment["pid"]))
    return combined, process_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=str)
    parser.add_argument("--single-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.iterations, args.warmup_iterations, args.repetitions) <= 0:
        raise ValueError(
            "iterations, warmup iterations, and repetitions must be positive"
        )
    if args.single_process or args.repetitions == 1:
        results = run_benchmarks(
            args.iterations,
            warmup_iterations=args.warmup_iterations,
            repetitions=args.repetitions,
        )
        process_ids = [os.getpid()]
        repetition_scope = "single_python_process"
    else:
        results, process_ids = _run_process_repetitions(
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            repetitions=args.repetitions,
        )
        repetition_scope = "fresh_python_process"
    payload = {
        "schema_version": 2,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "pid": os.getpid(),
            "sample_process_ids": process_ids,
        },
        "repository": _git_metadata(),
        "config": {
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "repetitions": args.repetitions,
            "repetition_scope": repetition_scope,
        },
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
