"""External AsyncScheduler policy for the frozen text, one-token P-only study.

This module owns immutable input-length keys and a reference-only ordered view.
The common Scheduler driver still owns readiness, lookup, quotas, allocation,
preemption, queue transitions and output construction. An ordered candidate is
not claimed to be ready before that native branch actually checks it.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_wait_trace import emit

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True, order=True)
class InputKey:
    prompt_tokens: int
    arrival_time: float
    arrival_sequence: int


_MISSING = object()


def _field(value, dotted):
    for part in dotted.split("."):
        value = (
            value.get(part, _MISSING)
            if isinstance(value, dict)
            else getattr(value, part, _MISSING)
        )
        if value is _MISSING:
            break
    return value


def frozen_config_scope(vllm_config, block_size, *, expected_token_budget=2048):
    """Pure field checks; absent fields are rejected, never assumed supported."""
    expected = {
        "scheduler_config.async_scheduling": True,
        "scheduler_config.policy": "fcfs",
        "scheduler_config.runner_type": "generate",
        "scheduler_config.is_multimodal_model": False,
        "scheduler_config.enable_chunked_prefill": True,
        "scheduler_config.scheduler_reserve_full_isl": True,
        "scheduler_config.max_num_batched_tokens": expected_token_budget,
        "scheduler_config.max_num_seqs": 32,
        "scheduler_config.long_prefill_token_threshold": 0,
        "cache_config.enable_prefix_caching": True,
        "parallel_config.tensor_parallel_size": 1,
        "parallel_config.pipeline_parallel_size": 1,
        "parallel_config.data_parallel_size": 1,
        "parallel_config.decode_context_parallel_size": 1,
        "parallel_config.prefill_context_parallel_size": 1,
        "model_config.is_encoder_decoder": False,
        "model_config.is_diffusion": False,
        "model_config.enforce_eager": True,
        "lora_config": None,
        "speculative_config": None,
        "ec_transfer_config": None,
        "kv_transfer_config.kv_connector": "LMCacheMPConnector",
        "kv_transfer_config.kv_role": "kv_both",
    }
    observed, errors = {}, []
    for path, wanted in expected.items():
        value = _field(vllm_config, path)
        observed[path] = "<missing>" if value is _MISSING else value
        if value is _MISSING or value != wanted:
            errors.append(path)
    scheduled = _field(vllm_config, "scheduler_config.max_num_scheduled_tokens")
    observed["scheduler_config.max_num_scheduled_tokens"] = (
        "<missing>" if scheduled is _MISSING else scheduled
    )
    if scheduled is _MISSING or scheduled not in (None, expected_token_budget):
        errors.append("scheduler_config.max_num_scheduled_tokens")
    extra = _field(vllm_config, "kv_transfer_config.kv_connector_extra_config")
    for name, wanted in [
        ("lmcache.mp.lazy_offload", False),
        ("lmcache.mp.require_terminal_completion", True),
    ]:
        value = extra.get(name, _MISSING) if isinstance(extra, dict) else _MISSING
        observed[name] = "<missing>" if value is _MISSING else value
        if value is _MISSING or value != wanted:
            errors.append(name)
    observed["block_size"] = block_size
    if block_size != 16:
        errors.append("block_size")
    return observed, errors


def final_retrieve_observation(output):
    """Read the pinned LMCacheMPConnectorMetadata.requests, after construction.

    LMCacheMPRequestMetadata contains request_id/direction/op, and LoadStoreOp
    contains start/end/skip_first_n_tokens/block_ids. It has no byte-size field.
    These are operations in the scheduler output, not worker submits or DMA.
    """
    metadata = getattr(output, "kv_connector_metadata", None)
    requests = getattr(metadata, "requests", _MISSING)
    if requests is _MISSING:
        return {
            "observed": False,
            "requests": None,
            "request_ids": None,
            "bytes": None,
            "reason": "final LMCacheMPConnectorMetadata.requests not observed",
        }
    rows, errors = [], []
    for request in requests:
        direction = getattr(request, "direction", _MISSING)
        if direction is _MISSING:
            errors.append("request metadata direction missing")
            continue
        if direction != "RETRIEVE":
            continue
        op = getattr(request, "op", _MISSING)
        fields = {
            name: getattr(op, name, _MISSING)
            for name in ("start", "end", "skip_first_n_tokens", "block_ids")
        }
        rid = getattr(request, "request_id", _MISSING)
        if rid is _MISSING or any(value is _MISSING for value in fields.values()):
            errors.append("RETRIEVE metadata fields missing")
            continue
        groups = fields["block_ids"]
        # The pre-IPC form is nested; LoadStoreOp explicitly also supports the
        # single-group flattened wire representation. Neither reads token ids.
        blocks = (
            [list(groups)]
            if groups and isinstance(groups[0], int)
            else [list(g) for g in groups]
        )
        rows.append(
            {
                "request_id": rid,
                "start_token": fields["start"],
                "end_token": fields["end"],
                "skip_first_n_tokens": fields["skip_first_n_tokens"],
                "block_ids": blocks,
                "source_tokens": fields["end"] - fields["start"],
                "target_write_tokens": fields["end"]
                - fields["start"]
                - fields["skip_first_n_tokens"],
                "bytes": None,
                "bytes_reason": "not present in LoadStoreOp; measure actual transfer bytes downstream",
            }
        )
    return {
        "observed": not errors,
        "requests": rows,
        "request_ids": [r["request_id"] for r in rows],
        "bytes": None,
        "errors": errors,
        "semantics": "final RETRIEVE entries committed to scheduler output; not worker submission, GPU start or completion",
    }


class GlobalSJFInputScheduler(AsyncScheduler):
    """Select the common global driver, ordered by original full input length."""

    _global_sjf_input_enabled = True

    def _emit_policy(self, suffix, *args, **fields):
        if suffix == "enqueue_key_frozen":
            suffix = "key_frozen"
        emit("global_sjf_input_" + suffix, *args, **fields)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_sjf_keys = {}
        self._global_sjf_requests = {}
        self._global_sjf_seen_ids = set()
        self._global_sjf_arrival_sequence = 0
        # Means included in a committed NewRequestData, not a timed worker ack.
        self._global_sjf_worker_initialized = set()
        self._global_sjf_ordered_ids = []
        self._global_sjf_result_count = 0
        snapshot, errors = frozen_config_scope(
            self.vllm_config, self.block_size,
            expected_token_budget=getattr(self, "_global_sjf_expected_token_budget", 2048))
        for path, wanted in {
            "kv_cache_manager.num_kv_cache_groups": 1,
            "num_lookahead_tokens": 0,
            "num_prefill_lookahead": 0,
            "has_mamba_layers": False,
            "ec_connector": None,
            "num_sampled_tokens_per_step": 1,
            "num_spec_tokens": 0,
            "kv_cache_manager.block_pool.hash_block_size": 16,
        }.items():
            value = _field(self, path)
            snapshot[path] = "<missing>" if value is _MISSING else value
            if value is _MISSING or value != wanted:
                errors.append(path)
        managers = _field(self, "kv_cache_manager.coordinator.single_type_managers")
        if not isinstance(managers, (tuple, list)) or len(managers) != 1:
            errors.append("single_dense_manager")
        else:
            manager = managers[0]
            snapshot["single_manager_type"] = type(manager).__name__
            if type(manager).__name__ != "FullAttentionManager":
                errors.append("single_manager_type")
            if getattr(manager, "block_size", _MISSING) != 16:
                errors.append("single_manager.block_size")
            pending = getattr(manager, "_pending_cow_copies", _MISSING)
            if pending is _MISSING or pending:
                errors.append("single_manager._pending_cow_copies")
        connector = getattr(self, "connector", _MISSING)
        if connector is _MISSING or connector is None:
            errors.append("connector")
        if errors:
            self._global_sjf_reject("config", errors=errors, scope=snapshot)
        self._emit_policy(
            'enabled',
            scope=snapshot,
            key_definition=[
                "original_num_prompt_tokens",
                "original_arrival_time",
                "stable_add_request_sequence",
            ],
            external_scope="Qwen model identity, Hybrid L1, LRU, Promotion OFF, replay and lower chunk size are frozen by the launcher, not independently read from the LMCache service by this scheduler",
        )

    def _global_sjf_reject(self, reason, request=None, **fields):
        self._emit_policy(
            'scope_rejected',
            getattr(request, "request_id", None),
            reason=reason,
            **fields,
        )
        raise ValueError(
            f"Global-SJF-input outside frozen scope: {reason}: {fields.get('errors', [])}"
        )

    def add_request(self, request):
        rid = request.request_id
        errors = []
        for path, wanted in {
            "max_tokens": 1,
            "pooling_params": None,
            "lora_request": None,
            "prompt_embeds": None,
            "prompt_is_token_ids": None,
            "resumable": False,
            "structured_output_request": None,
            "priority": 0,
            "ec_transfer_params": None,
            "kv_transfer_params": None,
        }.items():
            value = _field(request, path)
            if value is _MISSING or value != wanted:
                errors.append(path)
        for path in ("mm_features", "spec_token_ids"):
            value = _field(request, path)
            if value is _MISSING or value:
                errors.append(path)
        sampling = getattr(request, "sampling_params", None)
        if sampling is None or getattr(sampling, "max_tokens", None) != 1:
            errors.append("sampling_params.max_tokens")
        prompt = getattr(request, "num_prompt_tokens", None)
        token_ids = getattr(request, "prompt_token_ids", None)
        if (
            not isinstance(prompt, int)
            or isinstance(prompt, bool)
            or token_ids is None
            or len(token_ids) != prompt
            or not 0 < prompt < self.max_model_len
        ):
            errors.append("text_input_length")
        arrival = getattr(request, "arrival_time", None)
        if not isinstance(arrival, (int, float)) or not math.isfinite(arrival):
            errors.append("arrival_time")
        if rid in self._global_sjf_seen_ids or rid in self.requests:
            errors.append("duplicate_or_streaming_request_id")
        if errors:
            self._global_sjf_reject("request", request, errors=errors)
        self._global_sjf_arrival_sequence += 1
        key = InputKey(prompt, float(arrival), self._global_sjf_arrival_sequence)
        self._global_sjf_keys[rid] = key
        self._global_sjf_requests[rid] = request
        self._global_sjf_seen_ids.add(rid)
        super().add_request(request)
        self._emit_policy(
            'enqueue_key_frozen',
            rid,
            prompt_tokens=key.prompt_tokens,
            arrival_time=key.arrival_time,
            arrival_sequence=key.arrival_sequence,
            freeze_point="first add_request, before native enqueue and lookup",
            semantics="full input length only; does not use matched KV, remaining output or reference answer length",
        )

    def _global_sjf_batch_id(self):
        return f"{os.getpid()}:{self.current_step}"

    def _global_sjf_retire_observer_references(self):
        """Keep only ID tombstones after native ownership really disappears.

        A finished request can remain in self.requests during async STORE or
        deferred block release, so finished_req_ids alone is not sufficient.
        """
        owned_ids = set(self.requests)
        for name in ("running", "waiting", "skipped_waiting"):
            owned_ids.update(r.request_id for r in getattr(self, name))
        for rid in self._global_sjf_requests.keys() - owned_ids:
            self._global_sjf_requests.pop(rid)
            self._global_sjf_keys.pop(rid)
            self._global_sjf_worker_initialized.discard(rid)

    def _global_sjf_candidates(self) -> list[Request]:
        started = time.monotonic_ns()
        self._global_sjf_retire_observer_references()
        requests, origins, seen = [], {}, set()
        for name in ("running", "skipped_waiting", "waiting"):
            for request in getattr(self, name):
                identity, rid = id(request), request.request_id
                if (
                    self._global_sjf_requests.get(rid) is not request
                    or rid not in self._global_sjf_keys
                ):
                    self._global_sjf_reject("unregistered_request_identity", request)
                if (
                    request.num_prompt_tokens
                    != self._global_sjf_keys[rid].prompt_tokens
                ):
                    self._global_sjf_reject(
                        "input_length_changed_after_freeze", request
                    )
                origins.setdefault(rid, []).append(name)
                if identity not in seen:
                    seen.add(identity)
                    requests.append(request)
        native_view = [r.request_id for r in requests]
        requests.sort(key=lambda r: self._global_sjf_keys[r.request_id])
        self._global_sjf_ordered_ids = [r.request_id for r in requests]
        self._global_sjf_result_count = 0
        rows = []
        for request in requests:
            rid = request.request_id
            key = self._global_sjf_keys[rid]
            rows.append(
                {
                    "request_id": rid,
                    "queue_origins": origins[rid],
                    "status": getattr(request.status, "name", str(request.status)),
                    "prompt_tokens": key.prompt_tokens,
                    "arrival_time": key.arrival_time,
                    "arrival_sequence": key.arrival_sequence,
                    "computed_tokens": request.num_computed_tokens,
                    "in_flight_tokens": request.num_in_flight_tokens,
                    "output_placeholders": request.num_output_placeholders,
                    "worker_new_committed": rid in self._global_sjf_worker_initialized,
                    "readiness": "unchecked; native branch remains authoritative",
                }
            )
        elapsed = time.monotonic_ns() - started
        self._emit_policy(
            'candidates',
            batch_id=self._global_sjf_batch_id(),
            step=self.current_step,
            native_bucket_view=native_view,
            ordered_candidates=self._global_sjf_ordered_ids,
            candidates=rows,
            elapsed_ns=elapsed,
            elapsed_scope="reference snapshot, fixed-key sort and event field construction; emit excluded",
        )
        return requests

    def _global_sjf_observe_result(
        self, request, *, outcome, branch, reason=None, quota=0, **fields
    ):
        self._global_sjf_result_count += 1
        self._emit_policy(
            'result',
            request.request_id,
            batch_id=self._global_sjf_batch_id(),
            step=self.current_step,
            outcome=outcome,
            branch=branch,
            reason=reason,
            quota=quota,
            native_state={
                "status": getattr(request.status, "name", str(request.status)),
                "computed_tokens": request.num_computed_tokens,
                "in_flight_tokens": request.num_in_flight_tokens,
                "output_placeholders": request.num_output_placeholders,
                "num_preemptions": request.num_preemptions,
            },
            **fields,
        )

    def _global_sjf_observe_step_end(self, output):
        """Called once after final output construction, before native update.

        A request tentatively admitted and then preempted in the same step is
        marked initialized only if it remains in final scheduled_new_reqs.
        This bookkeeping never means the worker has already received a batch.
        """
        newly_committed = [row.req_id for row in output.scheduled_new_reqs]
        self._global_sjf_worker_initialized.update(newly_committed)
        finished = set(output.finished_req_ids)
        self._global_sjf_worker_initialized.difference_update(finished)
        self._global_sjf_retire_observer_references()
        quotas = dict(output.num_scheduled_tokens)
        self._emit_policy(
            'step_end',
            batch_id=self._global_sjf_batch_id(),
            step=self.current_step,
            ordered_candidates=self._global_sjf_ordered_ids,
            selected=list(quotas),
            quotas=quotas,
            new_requests_committed=newly_committed,
            finished_requests=sorted(finished),
            result_observations=self._global_sjf_result_count,
            final_retrieves=final_retrieve_observation(output),
            semantics="final scheduler output commit; not worker arrival or GPU execution",
        )
