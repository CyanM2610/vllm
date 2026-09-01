# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU microbenchmark for HotPrefix token-to-block projection."""

from __future__ import annotations

import argparse
import json
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
    tree_nodes: int
    path_nodes: int
    request_blocks: int
    slow_seconds: float
    phase_a_seconds: float
    reduction_percent: float
    phase_a_skipped: int
    groups_rebuilt: int


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
    *, scenario: str, iterations: int, tree_size: int, prefix_blocks: int
) -> BenchmarkResult:
    block_size = 16
    tree, target = _build_tree(
        tree_size=tree_size,
        prefix_blocks=prefix_blocks,
        block_size=block_size,
    )
    block_ids = tuple(range(1_000, 1_000 + prefix_blocks))
    slow_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    started_ns = time.perf_counter_ns()
    path_nodes = 0
    slow_groups = 0
    for _ in range(iterations):
        path_nodes, slow_groups = _slow_reconcile(
            slow_selector, tree, target, block_ids, block_size
        )
    slow_seconds = (time.perf_counter_ns() - started_ns) / 1e9

    fast_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    projection = HotPrefixBlockProjection(fast_selector)
    path = tree.path_snapshot(target)
    skipped = 0
    groups_rebuilt = 0
    started_ns = time.perf_counter_ns()
    for _ in range(iterations):
        result = projection.reconcile(
            namespace=tree.namespace,
            cached_tokens=len(target),
            path=path,
            physical_block_ids=block_ids,
            block_size=block_size,
            aging_epoch=tree.aging_epoch,
            total_tree_nodes=tree.node_count,
        )
        skipped += int(result.skipped)
        groups_rebuilt += result.groups_rebuilt
    phase_a_seconds = (time.perf_counter_ns() - started_ns) / 1e9
    reduction = (
        100 * (slow_seconds - phase_a_seconds) / slow_seconds if slow_seconds else 0.0
    )
    return BenchmarkResult(
        scenario=scenario,
        iterations=iterations,
        tree_nodes=tree.node_count,
        path_nodes=path_nodes,
        request_blocks=len(block_ids),
        slow_seconds=slow_seconds,
        phase_a_seconds=phase_a_seconds,
        reduction_percent=reduction,
        phase_a_skipped=skipped,
        groups_rebuilt=groups_rebuilt or slow_groups,
    )


def _benchmark_mutating(*, mode: str, iterations: int) -> BenchmarkResult:
    block_size = 16
    prefix_blocks = 32
    slow_tree, slow_target = _build_tree(
        tree_size=16,
        prefix_blocks=prefix_blocks,
        block_size=block_size,
    )
    fast_tree, fast_target = _build_tree(
        tree_size=16,
        prefix_blocks=prefix_blocks,
        block_size=block_size,
    )
    first_binding = tuple(range(1_000, 1_000 + prefix_blocks))
    second_binding = tuple(range(2_000, 2_000 + prefix_blocks))
    slow_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    slow_previous = first_binding
    started_ns = time.perf_counter_ns()
    path_nodes = 0
    slow_groups = 0
    for index in range(iterations):
        binding = first_binding if index % 2 == 0 else second_binding
        if mode == "hotness":
            slow_tree.record_match(slow_target, matched_tokens=len(slow_target))
        else:
            slow_selector.discard(slow_previous)
        path_nodes, slow_groups = _slow_reconcile(
            slow_selector,
            slow_tree,
            slow_target,
            binding,
            block_size,
        )
        slow_previous = binding
    slow_seconds = (time.perf_counter_ns() - started_ns) / 1e9

    fast_selector = HotPrefixBlockEvictionSelector(defer_for_host=False)
    projection = HotPrefixBlockProjection(fast_selector)
    fast_selector.set_discard_observer(projection.discard)
    fast_previous = first_binding
    skipped = 0
    groups_rebuilt = 0
    started_ns = time.perf_counter_ns()
    for index in range(iterations):
        binding = first_binding if index % 2 == 0 else second_binding
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
        skipped += int(result.skipped)
        groups_rebuilt += result.groups_rebuilt
        fast_previous = binding
    phase_a_seconds = (time.perf_counter_ns() - started_ns) / 1e9
    reduction = (
        100 * (slow_seconds - phase_a_seconds) / slow_seconds if slow_seconds else 0.0
    )
    return BenchmarkResult(
        scenario=f"{mode}_only",
        iterations=iterations,
        tree_nodes=fast_tree.node_count,
        path_nodes=path_nodes,
        request_blocks=prefix_blocks,
        slow_seconds=slow_seconds,
        phase_a_seconds=phase_a_seconds,
        reduction_percent=reduction,
        phase_a_skipped=skipped,
        groups_rebuilt=groups_rebuilt or slow_groups,
    )


def run_benchmarks(iterations: int) -> list[BenchmarkResult]:
    """Run repeated, tree-size, prefix-length, and mutation probes.

    Args:
        iterations: Reconciliation calls per scenario.

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
            tree_size=tree_size,
            prefix_blocks=prefix_blocks,
        )
        for name, tree_size, prefix_blocks in cases
    ]
    return [
        *repeated,
        _benchmark_mutating(mode="hotness", iterations=iterations),
        _benchmark_mutating(mode="binding", iterations=iterations),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    payload = {
        "schema_version": 1,
        "results": [asdict(result) for result in run_benchmarks(args.iterations)],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
