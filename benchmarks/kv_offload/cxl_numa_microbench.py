# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import ctypes
import json
import os
import socket
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm.v1.kv_offload.tiering.cxl_numa.allocator import NumaMemoryRegion
from vllm.v1.kv_offload.tiering.cxl_numa.copy_engine import (
    CopyDirection,
    CopyOperation,
    NumaCopyEngine,
)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_cpu_list(raw: str) -> set[int]:
    cpus: set[int] = set()
    for item in raw.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", maxsplit=1)
            cpus.update(range(int(start_raw), int(end_raw) + 1))
        else:
            cpus.add(int(item))
    return cpus


def _bind_to_numa_node(numa_node: int) -> dict[str, Any]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is unavailable on this platform")

    cpulist_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
    try:
        node_cpus = _parse_cpu_list(cpulist_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to read CPU list for NUMA node {numa_node}"
        ) from exc

    affinity_before = set(os.sched_getaffinity(0))
    requested_affinity = node_cpus & affinity_before
    if not requested_affinity:
        raise RuntimeError(f"NUMA node {numa_node} has no CPUs in the allowed affinity")
    os.sched_setaffinity(0, requested_affinity)
    affinity_effective = set(os.sched_getaffinity(0))
    if affinity_effective != requested_affinity:
        raise RuntimeError(
            "effective CPU affinity does not match the requested local-node CPUs"
        )
    return {
        "numa_node": numa_node,
        "affinity_before": sorted(affinity_before),
        "affinity_effective": sorted(affinity_effective),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _operations(
    source: NumaMemoryRegion,
    destination: NumaMemoryRegion,
    block_size_bytes: int,
    blocks_per_job: int,
) -> tuple[CopyOperation, ...]:
    return tuple(
        CopyOperation(
            src_address=source.address + index * block_size_bytes,
            dst_address=destination.address + index * block_size_bytes,
            size_bytes=block_size_bytes,
        )
        for index in range(blocks_per_job)
    )


def _measure(
    engine: NumaCopyEngine,
    direction: CopyDirection,
    operations: tuple[CopyOperation, ...],
    warmup: int,
    repetitions: int,
    next_job_id: int,
) -> tuple[list[float], int]:
    durations: list[float] = []
    for repetition in range(warmup + repetitions):
        engine.submit(next_job_id, direction, operations)
        next_job_id += 1
        engine.drain()
        [result] = engine.get_finished()
        if not result.success:
            raise RuntimeError(f"copy job {result.job_id} failed")
        if repetition >= warmup:
            durations.append(result.elapsed_seconds)
    return durations, next_job_id


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cpu_binding = _bind_to_numa_node(args.local_node)
    pool_bytes = args.block_size_bytes * args.blocks_per_job
    local_source = NumaMemoryRegion.allocate(
        pool_bytes, args.local_node, prefault=True, verify_placement=True
    )
    local_destination: NumaMemoryRegion | None = None
    remote: NumaMemoryRegion | None = None
    try:
        local_destination = NumaMemoryRegion.allocate(
            pool_bytes, args.local_node, prefault=True, verify_placement=True
        )
        remote = NumaMemoryRegion.allocate(
            pool_bytes, args.remote_node, prefault=True, verify_placement=True
        )
        ctypes.memset(local_source.address, 0x31, pool_bytes)
        ctypes.memset(local_destination.address, 0xA7, pool_bytes)
        ctypes.memset(remote.address, 0x5C, pool_bytes)

        paths = {
            ("local", CopyDirection.STORE): _operations(
                local_source,
                local_destination,
                args.block_size_bytes,
                args.blocks_per_job,
            ),
            ("local", CopyDirection.LOAD): _operations(
                local_destination,
                local_source,
                args.block_size_bytes,
                args.blocks_per_job,
            ),
            ("remote", CopyDirection.STORE): _operations(
                local_source,
                remote,
                args.block_size_bytes,
                args.blocks_per_job,
            ),
            ("remote", CopyDirection.LOAD): _operations(
                remote,
                local_destination,
                args.block_size_bytes,
                args.blocks_per_job,
            ),
        }

        results: list[dict[str, Any]] = []
        next_job_id = 0
        for thread_count in args.thread_counts:
            engine = NumaCopyEngine(thread_count, thread_count)
            try:
                for (path, direction), operations in paths.items():
                    durations, next_job_id = _measure(
                        engine,
                        direction,
                        operations,
                        args.warmup,
                        args.repetitions,
                        next_job_id,
                    )
                    bandwidths = [pool_bytes / duration / 1e9 for duration in durations]
                    results.append(
                        {
                            "path": path,
                            "direction": direction.value,
                            "thread_count_per_priority_group": thread_count,
                            "n_load_threads": thread_count,
                            "n_store_threads": thread_count,
                            "median_job_latency_seconds": statistics.median(durations),
                            "p95_job_latency_seconds": _percentile(durations, 0.95),
                            "median_gbps": statistics.median(bandwidths),
                            "durations_seconds": durations,
                        }
                    )
                    print(
                        f"path={path} direction={direction.value} "
                        f"threads={thread_count} complete",
                        flush=True,
                    )
            finally:
                engine.shutdown()

        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "local_node": args.local_node,
            "remote_node": args.remote_node,
            "cpu_binding": cpu_binding,
            "local_sample_nodes": sorted(set(local_source.query_sample_nodes())),
            "remote_sample_nodes": sorted(set(remote.query_sample_nodes())),
            "block_size_bytes": args.block_size_bytes,
            "blocks_per_job": args.blocks_per_job,
            "job_size_bytes": pool_bytes,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "thread_counts": args.thread_counts,
            "results": results,
        }
    finally:
        if remote is not None:
            remote.close()
        if local_destination is not None:
            local_destination.close()
        local_source.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local and remote-NUMA KV block copies."
    )
    parser.add_argument("--local-node", type=_non_negative_int, required=True)
    parser.add_argument("--remote-node", type=_non_negative_int, required=True)
    parser.add_argument("--block-size-bytes", type=_positive_int, required=True)
    parser.add_argument("--blocks-per-job", type=_positive_int, default=16)
    parser.add_argument(
        "--thread-counts", type=_positive_int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--repetitions", type=_positive_int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
