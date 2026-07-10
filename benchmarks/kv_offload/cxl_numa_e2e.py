# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex as re
from transformers import AutoTokenizer

TRANSFER_BYTES = "vllm:kv_offload_cxl_numa_transfer_bytes"
LOOKUPS = "vllm:kv_offload_cxl_numa_lookups"
ALLOCATION_FAILURES = "vllm:kv_offload_cxl_numa_allocation_failures"
INFLIGHT_JOBS = "vllm:kv_offload_cxl_numa_inflight_jobs"

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


def _http_get(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


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
    total = 0.0
    names = {metric_name, f"{metric_name}_total"}
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


def snapshot_metrics(base_url: str) -> dict[str, float]:
    text = _http_get(f"{base_url.rstrip('/')}/metrics")
    return {
        "load_bytes": _metric_total(text, TRANSFER_BYTES, {"direction": "load"}),
        "store_bytes": _metric_total(text, TRANSFER_BYTES, {"direction": "store"}),
        "lookup_hits": _metric_total(text, LOOKUPS, {"result": "hit"}),
        "allocation_failures": _metric_total(text, ALLOCATION_FAILURES),
        "inflight_jobs": _metric_total(text, INFLIGHT_JOBS),
    }


def wait_for_quiescence(
    base_url: str,
    timeout_seconds: float = 600,
    poll_seconds: float = 1,
) -> dict[str, Any]:
    started = time.monotonic()
    consecutive_zero = 0
    polls = 0
    while time.monotonic() - started < timeout_seconds:
        snapshot = snapshot_metrics(base_url)
        polls += 1
        if snapshot["inflight_jobs"] == 0:
            consecutive_zero += 1
            if consecutive_zero == 3:
                return {
                    "elapsed_seconds": time.monotonic() - started,
                    "polls": polls,
                    "snapshot": snapshot,
                }
        else:
            consecutive_zero = 0
        time.sleep(poll_seconds)
    raise TimeoutError("CXL NUMA jobs did not quiesce within 600 seconds")


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
    raise RuntimeError(f"tokenizer exposed only {len(candidates)} non-special tokens")


def build_prompts(
    tokenizer: Any,
    prompt_length: int,
    churn_prompts: int,
    seed: int,
) -> tuple[list[int], list[list[int]]]:
    seed_ids = _seed_token_ids(tokenizer)
    prefix = (seed_ids * ((prompt_length + len(seed_ids) - 1) // len(seed_ids)))[
        :prompt_length
    ]
    churn: list[list[int]] = []
    first_blocks = {tuple(prefix[:64])}
    for index in range(churn_prompts):
        header = tokenizer.encode(
            f"CXL NUMA churn prompt {index}: ", add_special_tokens=False
        )
        header = [
            token_id for token_id in header if token_id not in tokenizer.all_special_ids
        ]
        if not header:
            header = [seed_ids[index % len(seed_ids)]]
        rng = random.Random(seed + index)
        prompt = [*header[:prompt_length]]
        prompt.extend(rng.choice(seed_ids) for _ in range(prompt_length - len(prompt)))
        if len(prompt) != prompt_length:
            raise AssertionError("churn prompt has incorrect token length")
        first_block = tuple(prompt[:64])
        if first_block in first_blocks:
            raise AssertionError("churn prompts must differ within their first block")
        first_blocks.add(first_block)
        churn.append(prompt)
    if len(prefix) != prompt_length:
        raise AssertionError("prefix prompt has incorrect token length")
    return prefix, churn


def request_completion(
    base_url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": False,
        "seed": seed,
    }
    started = time.perf_counter()
    response = _http_post_json(
        f"{base_url.rstrip('/')}/v1/completions", payload, timeout=600
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError("completion response did not contain token_ids")
    return {
        "token_ids": token_ids,
        "request_latency_seconds": elapsed,
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def _classify_scenario(
    *,
    tokens_equal: bool,
    metric_delta: dict[str, float],
    final_inflight_jobs: float,
) -> tuple[str, str | None]:
    if not tokens_equal:
        return "FAIL", "first and second A output token IDs differ"
    if metric_delta["allocation_failures"] != 0:
        return "FAIL", "allocation failures increased during second A"
    if metric_delta["load_bytes"] < 0:
        return "FAIL", "remote load-byte counter decreased during second A"
    if metric_delta["load_bytes"] == 0:
        return (
            "NOT_TRIGGERED",
            "second A did not increase remote load bytes; increase --churn-prompts",
        )
    if metric_delta["lookup_hits"] <= 0:
        return "FAIL", "remote loads increased without lookup-hit growth"
    if final_inflight_jobs != 0:
        return "FAIL", "CXL NUMA jobs did not return to zero"
    return "PASS", None


def run_scenario(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix, churn = build_prompts(
        tokenizer, args.prompt_length, args.churn_prompts, args.seed
    )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "base_url": args.base_url,
        "model": args.model,
        "prompt_length": args.prompt_length,
        "churn_prompts": args.churn_prompts,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "latency_measurement": "non_streaming_end_to_end_request_seconds",
    }

    artifact["first_a"] = request_completion(
        args.base_url, args.model, prefix, args.output_tokens, args.seed
    )
    artifact["quiescence_after_first_a"] = wait_for_quiescence(args.base_url)

    churn_results = []
    for index, prompt in enumerate(churn):
        result = request_completion(
            args.base_url,
            args.model,
            prompt,
            args.output_tokens,
            args.seed + index + 1,
        )
        churn_results.append(
            {
                "index": index,
                "request_latency_seconds": result["request_latency_seconds"],
            }
        )
        print(f"churn request {index + 1}/{len(churn)} complete", flush=True)
    artifact["churn"] = churn_results
    artifact["quiescence_after_churn"] = wait_for_quiescence(args.base_url)

    before = snapshot_metrics(args.base_url)
    artifact["metrics_before_second_a"] = before
    artifact["second_a"] = request_completion(
        args.base_url, args.model, prefix, args.output_tokens, args.seed
    )
    artifact["quiescence_after_second_a"] = wait_for_quiescence(args.base_url)
    after = snapshot_metrics(args.base_url)
    artifact["metrics_after_second_a"] = after
    delta = _delta(after, before)
    artifact["second_a_metric_delta"] = delta

    tokens_equal = artifact["first_a"]["token_ids"] == artifact["second_a"]["token_ids"]
    artifact["tokens_equal"] = tokens_equal
    verdict, reason = _classify_scenario(
        tokens_equal=tokens_equal,
        metric_delta=delta,
        final_inflight_jobs=after["inflight_jobs"],
    )
    artifact["verdict"] = verdict
    if reason is not None:
        artifact["reason"] = reason
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prove CXL NUMA KV promotion.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt-length", type=_positive_int, default=8192)
    parser.add_argument("--churn-prompts", type=_positive_int, default=12)
    parser.add_argument("--output-tokens", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--output", type=Path, default=Path("results/cxl_numa/e2e.json")
    )
    return parser.parse_args()


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    try:
        artifact = run_scenario(args)
    except BaseException as exc:
        artifact = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "base_url": args.base_url,
            "model": args.model,
            "verdict": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_artifact(args.output, artifact)
        raise
    _write_artifact(args.output, artifact)
    print(f"verdict={artifact['verdict']} artifact={args.output}", flush=True)
    if artifact["verdict"] != "PASS":
        print(artifact["reason"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
