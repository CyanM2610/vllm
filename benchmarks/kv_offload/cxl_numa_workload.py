# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Controlled proof and online workloads for CXL-NUMA KV offloading."""

import argparse
import codecs
import json
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ExpectedTier = Literal["cold", "cpu", "cxl"]

CXL_TRANSFER_BYTES = "vllm:kv_offload_cxl_numa_transfer_bytes"
CXL_TRANSFER_TIME = "vllm:kv_offload_cxl_numa_transfer_time_seconds"
CXL_LOOKUPS = "vllm:kv_offload_cxl_numa_lookups"
CXL_ALLOCATION_FAILURES = "vllm:kv_offload_cxl_numa_allocation_failures"
CXL_INFLIGHT_JOBS = "vllm:kv_offload_cxl_numa_inflight_jobs"
GPU_LOAD_BYTES = "vllm:kv_offload_load_bytes"
GPU_LOAD_TIME = "vllm:kv_offload_load_time"
GPU_STORE_BYTES = "vllm:kv_offload_store_bytes"
GPU_STORE_TIME = "vllm:kv_offload_store_time"
LOCAL_PREFIX_HITS = "vllm:prefix_cache_hits"
EXTERNAL_PREFIX_HITS = "vllm:external_prefix_cache_hits"

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s|$)"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


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


class _SSEDecoder:
    def __init__(self) -> None:
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def add_chunk(self, chunk: bytes) -> list[str]:
        self._buffer += self._decoder.decode(chunk).replace("\r\n", "\n")
        messages: list[str] = []
        while "\n\n" in self._buffer:
            message, self._buffer = self._buffer.split("\n\n", maxsplit=1)
            if message.strip():
                messages.append(message.strip())
        return messages

    def finish(self) -> list[str]:
        self._buffer += self._decoder.decode(b"", final=True)
        if not self._buffer.strip():
            return []
        message = self._buffer.strip()
        self._buffer = ""
        return [message]


class _StreamAccumulator:
    def __init__(self, started: float) -> None:
        self.started = started
        self.first_token_at: float | None = None
        self.token_ids: list[int] = []
        self.text = ""
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def add_message(self, message: str, clock: Callable[[], float]) -> None:
        data_lines = [
            line.removeprefix("data:").strip()
            for line in message.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            return
        data = json.loads(payload)
        if data.get("usage") is not None:
            self.usage = data["usage"]
        choices = data.get("choices") or []
        if not choices:
            return
        observed_at = clock()
        choice = choices[0]
        delta_ids = choice.get("token_ids") or []
        delta_text = choice.get("text") or ""
        if self.first_token_at is None and (delta_ids or delta_text):
            self.first_token_at = observed_at
        self.token_ids.extend(int(token_id) for token_id in delta_ids)
        self.text += delta_text
        if choice.get("finish_reason") is not None:
            self.finish_reason = choice["finish_reason"]

    def result(self, finished_at: float) -> dict[str, Any]:
        if self.first_token_at is None:
            raise RuntimeError("stream completed without a generated token")
        return {
            "token_ids": self.token_ids,
            "text": self.text,
            "ttft_seconds": self.first_token_at - self.started,
            "e2e_seconds": finished_at - self.started,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


async def _consume_sse_stream(
    chunks: AsyncIterable[bytes],
    *,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    decoder = _SSEDecoder()
    accumulator = _StreamAccumulator(started)
    async for chunk in chunks:
        for message in decoder.add_chunk(chunk):
            if not message.startswith(":"):
                accumulator.add_message(message, clock)
    for message in decoder.finish():
        accumulator.add_message(message, clock)
    return accumulator.result(clock())


def _consume_sse_response(
    chunks: Iterable[bytes],
    *,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    decoder = _SSEDecoder()
    accumulator = _StreamAccumulator(started)
    for chunk in chunks:
        for message in decoder.add_chunk(chunk):
            if not message.startswith(":"):
                accumulator.add_message(message, clock)
    for message in decoder.finish():
        accumulator.add_message(message, clock)
    return accumulator.result(clock())


def request_completion_streaming(
    base_url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return _consume_sse_response(response, started=started)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _http_get(url: str, timeout_seconds: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        key: bytes(value, "utf-8").decode("unicode_escape")
        for key, value in _LABEL_RE.findall(raw)
    }


def _metric_total(
    text: str,
    metric_name: str,
    required_labels: dict[str, str] | None = None,
) -> float:
    names = {metric_name, f"{metric_name}_total"}
    total = 0.0
    for line in text.splitlines():
        match = _SAMPLE_RE.match(line)
        if match is None or match.group("name") not in names:
            continue
        labels = _parse_labels(match.group("labels"))
        if required_labels and any(
            labels.get(key) != value for key, value in required_labels.items()
        ):
            continue
        total += float(match.group("value"))
    return total


def snapshot_metrics(base_url: str, numa_node: int | None) -> dict[str, float]:
    text = _http_get(f"{base_url.rstrip('/')}/metrics")
    node_labels = None if numa_node is None else {"numa_node": str(numa_node)}
    load_labels = {"direction": "load"}
    if node_labels:
        load_labels.update(node_labels)
    lookup_labels = {"result": "hit"}
    if node_labels:
        lookup_labels.update(node_labels)
    return {
        "local_prefix_hits": _metric_total(text, LOCAL_PREFIX_HITS),
        "external_prefix_hits": _metric_total(text, EXTERNAL_PREFIX_HITS),
        "gpu_load_bytes": _metric_total(text, GPU_LOAD_BYTES),
        "gpu_load_time_seconds": _metric_total(text, GPU_LOAD_TIME),
        "gpu_store_bytes": _metric_total(text, GPU_STORE_BYTES),
        "gpu_store_time_seconds": _metric_total(text, GPU_STORE_TIME),
        "cxl_load_bytes": _metric_total(text, CXL_TRANSFER_BYTES, load_labels),
        "cxl_load_time_seconds": _metric_total(text, CXL_TRANSFER_TIME, load_labels),
        "cxl_lookup_hits": _metric_total(text, CXL_LOOKUPS, lookup_labels),
        "allocation_failures": _metric_total(
            text, CXL_ALLOCATION_FAILURES, node_labels
        ),
        "inflight_jobs": _metric_total(text, CXL_INFLIGHT_JOBS, node_labels),
    }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def wait_for_quiescence(
    base_url: str,
    numa_node: int | None,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    consecutive_stable = 0
    polls = 0
    previous_store_bytes: float | None = None
    while time.monotonic() - started < timeout_seconds:
        snapshot = snapshot_metrics(base_url, numa_node)
        polls += 1
        store_bytes = snapshot["gpu_store_bytes"]
        if snapshot["inflight_jobs"] == 0 and store_bytes == previous_store_bytes:
            consecutive_stable += 1
            if consecutive_stable == 3:
                return {
                    "elapsed_seconds": time.monotonic() - started,
                    "polls": polls,
                    "snapshot": snapshot,
                }
        else:
            consecutive_stable = 0
        previous_store_bytes = store_bytes
        time.sleep(poll_seconds)
    raise TimeoutError("offload jobs did not quiesce before timeout")


def _classify_expected_tier(
    *,
    expected_tier: ExpectedTier,
    tokens_equal: bool,
    metric_delta: dict[str, float],
    final_inflight_jobs: float,
) -> tuple[str, str | None]:
    if not tokens_equal:
        return "FAIL", "greedy token IDs differ"
    if metric_delta["allocation_failures"] != 0:
        return "FAIL", "CXL NUMA allocation failures increased"
    if final_inflight_jobs != 0:
        return "FAIL", "in-flight jobs did not return to zero"
    if any(value < 0 for value in metric_delta.values()):
        return "FAIL", "a cumulative metric decreased"

    external_hits = metric_delta["external_prefix_hits"]
    gpu_load = metric_delta["gpu_load_bytes"]
    cxl_load = metric_delta["cxl_load_bytes"]
    cxl_hits = metric_delta["cxl_lookup_hits"]
    if expected_tier == "cold":
        if external_hits or gpu_load or cxl_load:
            return "FAIL", "cold request unexpectedly loaded external KV"
        return "PASS", None
    if expected_tier == "cpu":
        if external_hits <= 0 or gpu_load <= 0:
            return "NOT_TRIGGERED", "CPU offload hit was not observed"
        if cxl_load or cxl_hits:
            return "FAIL", "CPU-only request unexpectedly loaded CXL NUMA KV"
        return "PASS", None
    if cxl_load == 0 and cxl_hits == 0:
        return "NOT_TRIGGERED", "CXL NUMA hit was not observed"
    if external_hits <= 0 or gpu_load <= 0 or cxl_load <= 0 or cxl_hits <= 0:
        return "FAIL", "CXL metrics do not form a complete promotion path"
    return "PASS", None


def _seed_token_ids(tokenizer: Any, count: int = 256) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    candidates: list[int] = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        if tokenizer.convert_ids_to_tokens(token_id) is None:
            continue
        candidates.append(token_id)
        if len(candidates) == count:
            return candidates
    raise RuntimeError("tokenizer does not expose enough non-special tokens")


def build_working_set(
    tokenizer: Any,
    prompt_length: int,
    working_set_size: int,
    seed: int,
) -> list[list[int]]:
    seed_ids = _seed_token_ids(tokenizer)
    prompts: list[list[int]] = []
    first_blocks: set[tuple[int, ...]] = set()
    for index in range(working_set_size):
        header = tokenizer.encode(
            f"CXL NUMA working-set prompt {index}: ", add_special_tokens=False
        )
        header = [
            token_id
            for token_id in header
            if token_id not in tokenizer.all_special_ids
        ]
        rng = random.Random(seed + index)
        prompt = header[:prompt_length]
        prompt.extend(rng.choice(seed_ids) for _ in range(prompt_length - len(prompt)))
        first_block = tuple(prompt[:64])
        if first_block in first_blocks:
            raise AssertionError("working-set prompts must differ in the first block")
        first_blocks.add(first_block)
        prompts.append(prompt)
    return prompts


def _request(
    args: argparse.Namespace, prompt: list[int], request_seed: int
) -> dict[str, Any]:
    return request_completion_streaming(
        args.base_url,
        args.model,
        prompt,
        args.output_tokens,
        request_seed,
        args.timeout_seconds,
    )


def _artifact_base(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "scenario": args.scenario,
        "experiment_category": args.experiment_category,
        "mode": args.mode,
        "expected_tier": args.expected_tier,
        "expected_numa_node": args.expected_numa_node,
        "profile": args.profile,
        "run_id": args.run_id,
        "base_url": args.base_url,
        "model": args.model,
        "prompt_length": args.prompt_length,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "connector_block_size": args.connector_block_size,
        "n_load_threads": args.n_load_threads,
        "n_store_threads": args.n_store_threads,
        "cxl_capacity_bytes": args.cxl_capacity_bytes,
        "latency_measurement": "streaming_client_seconds",
    }


def run_proof(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    prompts = build_working_set(
        tokenizer, args.prompt_length, args.churn_prompts + 1, args.seed
    )
    artifact = _artifact_base(args)
    artifact["churn_prompts"] = args.churn_prompts
    artifact["first_a"] = _request(args, prompts[0], args.seed)
    artifact["quiescence_after_first_a"] = wait_for_quiescence(
        args.base_url,
        args.expected_numa_node,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    churn_results: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts[1:]):
        result = _request(args, prompt, args.seed + index + 1)
        churn_results.append(
            {
                "index": index,
                "ttft_seconds": result["ttft_seconds"],
                "e2e_seconds": result["e2e_seconds"],
            }
        )
        print(f"churn request {index + 1}/{args.churn_prompts} complete", flush=True)
    artifact["churn"] = churn_results
    artifact["quiescence_after_churn"] = wait_for_quiescence(
        args.base_url,
        args.expected_numa_node,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )

    before = snapshot_metrics(args.base_url, args.expected_numa_node)
    artifact["metrics_before_second_a"] = before
    artifact["second_a"] = _request(args, prompts[0], args.seed)
    artifact["quiescence_after_second_a"] = wait_for_quiescence(
        args.base_url,
        args.expected_numa_node,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    after = snapshot_metrics(args.base_url, args.expected_numa_node)
    delta = _delta(after, before)
    artifact["metrics_after_second_a"] = after
    artifact["second_a_metric_delta"] = delta
    tokens_equal = artifact["first_a"]["token_ids"] == artifact["second_a"]["token_ids"]
    artifact["tokens_equal"] = tokens_equal
    verdict, reason = _classify_expected_tier(
        expected_tier=args.expected_tier,
        tokens_equal=tokens_equal,
        metric_delta=delta,
        final_inflight_jobs=after["inflight_jobs"],
    )
    expected_blocks = args.prompt_length // args.connector_block_size
    artifact["expected_external_blocks"] = expected_blocks
    if (
        verdict == "PASS"
        and args.expected_tier == "cold"
        and delta["local_prefix_hits"] > 0
    ):
        verdict = "FAIL"
        reason = "cold baseline retained part of the prefix in GPU KV cache"
    if (
        verdict == "PASS"
        and args.expected_tier != "cold"
        and delta["external_prefix_hits"] < args.prompt_length
    ):
        verdict = "FAIL"
        reason = "external prefix hits do not cover the complete prompt"
    if (
        verdict == "PASS"
        and args.expected_tier == "cxl"
        and delta["cxl_lookup_hits"] < expected_blocks
    ):
        verdict = "FAIL"
        reason = "CXL lookup hits do not cover the complete aligned prefix"
    artifact["verdict"] = verdict
    if reason:
        artifact["reason"] = reason
    return artifact


def run_online(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    prompts = build_working_set(
        tokenizer, args.prompt_length, args.working_set_size, args.seed
    )
    artifact = _artifact_base(args)
    artifact.update(
        {
            "working_set_size": args.working_set_size,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
        }
    )

    reference_tokens: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompts):
        result = _request(args, prompt, args.seed + index)
        reference_tokens[index] = result["token_ids"]
        print(f"warmup request {index + 1}/{len(prompts)} complete", flush=True)
    artifact["quiescence_after_warmup"] = wait_for_quiescence(
        args.base_url,
        args.expected_numa_node,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    before = snapshot_metrics(args.base_url, args.expected_numa_node)
    started = time.perf_counter()
    request_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for request_index in range(args.num_requests):
            prompt_index = request_index % len(prompts)
            future = pool.submit(
                _request,
                args,
                prompts[prompt_index],
                args.seed + prompt_index,
            )
            futures[future] = (request_index, prompt_index)
        for completed, future in enumerate(as_completed(futures), start=1):
            request_index, prompt_index = futures[future]
            try:
                result = future.result()
                result.update(
                    {
                        "request_index": request_index,
                        "prompt_index": prompt_index,
                        "tokens_equal": result["token_ids"]
                        == reference_tokens[prompt_index],
                        "success": True,
                    }
                )
            except BaseException as exc:
                result = {
                    "request_index": request_index,
                    "prompt_index": prompt_index,
                    "tokens_equal": False,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            request_results.append(result)
            if completed % 10 == 0 or completed == args.num_requests:
                print(
                    f"online requests {completed}/{args.num_requests} complete",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    request_results.sort(key=lambda item: item["request_index"])
    artifact["requests"] = request_results
    artifact["quiescence_after_requests"] = wait_for_quiescence(
        args.base_url,
        args.expected_numa_node,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    after = snapshot_metrics(args.base_url, args.expected_numa_node)
    delta = _delta(after, before)
    successful_results = [
        result for result in request_results if result.get("success")
    ]
    total_tokens = sum(len(result["token_ids"]) for result in successful_results)
    expected_cxl_lookups = (
        args.num_requests * (args.prompt_length // args.connector_block_size)
    )
    artifact["metrics_before_requests"] = before
    artifact["metrics_after_requests"] = after
    artifact["request_metric_delta"] = delta
    artifact["summary"] = {
        "elapsed_seconds": elapsed,
        "request_throughput_per_second": len(request_results) / elapsed,
        "output_token_throughput_per_second": total_tokens / elapsed,
        "tokens_equal_fraction": sum(
            bool(result["tokens_equal"]) for result in request_results
        )
        / len(request_results),
        "success_fraction": len(successful_results) / len(request_results),
        "cxl_hit_fraction_estimate": min(
            1.0,
            delta["cxl_lookup_hits"] / expected_cxl_lookups,
        )
        if expected_cxl_lookups
        else 0.0,
    }
    tokens_equal = len(successful_results) == len(request_results) and all(
        result["tokens_equal"] for result in request_results
    )
    verdict, reason = _classify_expected_tier(
        expected_tier=args.expected_tier,
        tokens_equal=tokens_equal,
        metric_delta=delta,
        final_inflight_jobs=after["inflight_jobs"],
    )
    artifact["verdict"] = verdict
    if reason:
        artifact["reason"] = reason
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("proof", "online"), required=True)
    parser.add_argument("--experiment-category", default="manual")
    parser.add_argument(
        "--mode",
        choices=("no_offload", "cpu_only", "local_secondary", "remote_secondary"),
        required=True,
    )
    parser.add_argument(
        "--expected-tier", choices=("cold", "cpu", "cxl"), required=True
    )
    parser.add_argument("--expected-numa-node", type=_non_negative_int)
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt-length", type=_positive_int, default=8192)
    parser.add_argument("--churn-prompts", type=_positive_int, default=6)
    parser.add_argument("--working-set-size", type=_positive_int, default=8)
    parser.add_argument("--num-requests", type=_positive_int, default=200)
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    parser.add_argument("--output-tokens", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--connector-block-size", type=_positive_int, default=64)
    parser.add_argument("--n-load-threads", type=_positive_int, default=4)
    parser.add_argument("--n-store-threads", type=_positive_int, default=2)
    parser.add_argument("--cxl-capacity-bytes", type=_non_negative_int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        artifact = (
            run_proof(args, tokenizer)
            if args.scenario == "proof"
            else run_online(args, tokenizer)
        )
    except BaseException as exc:
        artifact = {
            **_artifact_base(args),
            "verdict": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_artifact(args.output, artifact)
        raise
    _write_artifact(args.output, artifact)
    print(f"verdict={artifact['verdict']} artifact={args.output}", flush=True)
    if artifact["verdict"] == "PASS":
        return 0
    return 2 if artifact["verdict"] == "NOT_TRIGGERED" else 1


if __name__ == "__main__":
    sys.exit(main())
