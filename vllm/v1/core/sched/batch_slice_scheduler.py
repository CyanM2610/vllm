"""One cold, fully admitted cohort: controlled one-versus-two prefill batches.

This is an experiment fixture, not an online batching policy. No target may
have a nonzero native prefix match. Native matches are all observed before
any cohort allocation, in the same synchronous schedule call; no stale HBM
block pointer is retained. Allocation/connector ownership are then performed
by the unchanged native WAITING branch and retained until normal completion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.global_sjf_input import frozen_config_scope
from vllm.v1.kv_wait_trace import emit
from vllm.v1.request import RequestStatus


class BatchSliceScheduler(AsyncScheduler):
    _batch_slice_enabled = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = os.environ.get("PREFILL_BATCH_SPEC")
        if not path:
            raise ValueError("PREFILL_BATCH_SPEC is required")
        workload = json.loads(Path(path).read_text(encoding="utf-8"))
        spec = workload["batch_experiment"]
        mode = os.environ["PREFILL_BATCH_MODE"]
        if ("merge" if spec["mode"] == "merged" else spec["mode"]) != ("merge" if mode == "merged" else mode):
            raise ValueError("Workload and environment batch mode differ")
        prompts = {r["id"]: len(r["input_tokens"]) for r in workload["requests"]}
        config = dict(scenario=spec["scenario"], mode=mode,
                      targets=[dict(logical_id=rid, prompt_tokens=prompts[rid],
                                    size_class="large" if rid == spec["large_id"] else "small")
                               for rid in spec["target_ids"]])
        self._batch_slice_initialize(config)

    def _batch_slice_initialize(self, config):
        scope, errors = frozen_config_scope(self.vllm_config, self.block_size)
        for name in ("num_lookahead_tokens", "num_prefill_lookahead", "num_spec_tokens"):
            if getattr(self, name, None) != 0:
                errors.append(name)
        if self.has_mamba_layers or self.ec_connector is not None:
            errors.append("dense_text_only")
        if self.connector is None or self.num_sampled_tokens_per_step != 1:
            errors.append("connector_or_one_token")
        if self.max_num_running_reqs != 32 or self.max_num_scheduled_tokens != 2048:
            errors.append("actual_sequence_or_token_budget")
        if errors:
            raise ValueError(f"Batch slice unsupported configuration: {errors}")
        self._slice_config = config
        self._slice_mode = "merge" if config["mode"] == "merged" else config["mode"]
        if self._slice_mode not in ("merge", "split"):
            raise ValueError("Batch slice mode must be merge or split")
        self._slice_specs = {str(x["logical_id"]): x for x in config["targets"]}
        if len(self._slice_specs) != len(config["targets"]):
            raise ValueError("Duplicate cohort target")
        sides = [x["size_class"] for x in config["targets"]]
        if sides.count("large") != 1 or sides.count("small") < 2:
            raise ValueError("Cohort needs multiple small and exactly one large request")
        if any(x not in ("small", "large") for x in sides):
            raise ValueError("Unknown size class")
        if len(sides) > self.max_num_running_reqs:
            raise ValueError("Cohort exceeds native sequence budget")
        if sum(int(x["prompt_tokens"]) for x in config["targets"]) > 2048:
            raise ValueError("Cold cohort exceeds token budget")
        self._slice_targets = {}
        self._slice_held = {}
        self._slice_submitted = set()
        self._slice_initialized = set()
        self._slice_match = {}
        self._slice_allowed = set()
        self._slice_ready = False
        self._slice_ready_step = None
        self._slice_submit_batch_count = 0
        emit("batch_slice_enabled", scenario=config["scenario"], mode=self._slice_mode,
             targets=config["targets"], scope=scope,
             load_page_scope_environment=os.environ.get("LMCACHE_LOAD_PAGE_SCOPE"),
             connector_class=type(self.connector).__module__ + "." + type(self.connector).__name__,
             scheduler_policy=str(self.policy),
             semantics="controlled cold cohort; match all, actually allocate all, then release one or two batches")

    @staticmethod
    def _slice_logical(request):
        extra = getattr(getattr(request, "sampling_params", None), "extra_args", None)
        return str(extra.get("native_token_replay_id", request.request_id)) if isinstance(extra, dict) else request.request_id

    def _slice_fail(self, reason, request=None, **fields):
        emit("batch_slice_failed", getattr(request, "request_id", None),
             scenario=self._slice_config["scenario"], mode=self._slice_mode,
             reason=reason, **fields)
        raise RuntimeError(f"Batch slice coverage failed: {reason}: {fields}")

    def add_request(self, request):
        logical = self._slice_logical(request)
        if logical in self._slice_specs:
            spec = self._slice_specs[logical]
            if logical in self._slice_targets:
                self._slice_fail("duplicate_target", request)
            if request.max_tokens != 1 or request.num_prompt_tokens != int(spec["prompt_tokens"]):
                self._slice_fail("target_input_or_output_contract", request)
            if request.num_computed_tokens != 0 or request.mm_features or request.has_encoder_inputs:
                self._slice_fail("target_not_fresh_text", request)
            self._slice_targets[logical] = request
        elif self._slice_targets:
            self._slice_fail("non_target_arrived_after_cohort", request)
        super().add_request(request)

    def _batch_slice_is_target(self, request):
        return self._slice_logical(request) in self._slice_specs

    def _batch_slice_begin(self):
        self._slice_match = {}
        self._slice_allowed = set()
        if self._slice_ready:
            pending = set(self._slice_held) - self._slice_submitted
            if self._slice_mode == "merge" or self._slice_submit_batch_count:
                self._slice_allowed = pending
            else:
                self._slice_allowed = {rid for rid in pending if self._slice_held[rid]["size_class"] == "small"}
            return
        if len(self._slice_targets) != len(self._slice_specs):
            return
        target_ids = {r.request_id for r in self._slice_targets.values()}
        if any(rid not in target_ids for rid in self.requests):
            # Prior real warmup/history must complete before the slice is formed.
            return
        if self._slice_held:
            self._slice_fail("partial_cohort_admission")
        matches = {}
        for logical in self._slice_specs:
            request = self._slice_targets[logical]
            if request.request_id not in self.requests or request.status != RequestStatus.WAITING:
                self._slice_fail("target_cancelled_or_not_waiting", request)
            if request.num_in_flight_tokens or request.num_output_placeholders:
                self._slice_fail("target_already_in_flight", request)
            match = self._match_waiting_prefix(request, phase="native")
            if match is None:
                continue
            if match.matched_tokens or match.external_tokens or match.load_kv_async or any(match.blocks.get_block_ids()):
                self._slice_fail("cold_slice_has_native_hit_or_load", request,
                                 matched=match.matched_tokens, external=match.external_tokens,
                                 async_load=match.load_kv_async)
            matches[request.request_id] = match
        if len(matches) == len(self._slice_specs):
            # All empty block results are fresh in this synchronous schedule.
            self._slice_match = matches
            emit("batch_slice_matching_complete", scenario=self._slice_config["scenario"],
                 mode=self._slice_mode, step=self.current_step, request_ids=list(matches))

    def _batch_slice_defer_waiting(self, request):
        return self._batch_slice_is_target(request) and request.request_id not in self._slice_match

    def _batch_slice_take_match(self, request):
        return self._slice_match.pop(request.request_id)

    def _batch_slice_hold(self, request, computed, quota):
        if computed != 0 or quota != request.num_prompt_tokens:
            self._slice_fail("not_complete_cold_allocation", request, computed=computed, quota=quota)
        rid = request.request_id
        if rid in self._slice_held:
            self._slice_fail("duplicate_hold", request)
        logical = self._slice_logical(request)
        blocks = self.kv_cache_manager.get_blocks(rid).get_block_ids()
        self.running.append(request)
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = computed
        row = dict(request_id=rid, logical_id=logical,
                   size_class=self._slice_specs[logical]["size_class"],
                   prompt_tokens=request.num_prompt_tokens, computed_tokens=computed,
                   remaining_tokens=quota, native_status=request.status.name,
                   native_checks_passed=True, allocation_succeeded=True,
                   external_tokens=0, physical_ready=True, block_ids=blocks,
                   readiness_evidence="native cold match before cohort allocations; actual full-prompt allocation and native connector ownership; no LOAD",
                   hold_step=self.current_step)
        self._slice_held[rid] = row
        emit("batch_slice_hold", rid, **{k: v for k, v in row.items() if k != "request_id"})

    def _batch_slice_defer_running(self, request):
        if not self._batch_slice_is_target(request):
            return False
        return request.request_id not in self._slice_allowed

    def _batch_slice_first_submit(self, request):
        return request.request_id in self._slice_held and request.request_id not in self._slice_initialized

    def _batch_slice_end(self, tokens, preempted):
        if preempted and self._slice_targets:
            self._slice_fail("unexpected_preemption")
        if not self._slice_ready and self._slice_match:
            self._slice_fail("native_admission_incomplete", unchecked=list(self._slice_match))
        if not self._slice_ready and self._slice_held:
            if len(self._slice_held) != len(self._slice_specs) or tokens:
                self._slice_fail("incomplete_common_admission", held=list(self._slice_held), tokens=tokens)
            self._slice_ready = True
            self._slice_ready_step = self.current_step
            emit("batch_slice_ready", scenario=self._slice_config["scenario"], mode=self._slice_mode,
                 targets=list(self._slice_held.values()), token_budget=self.max_num_scheduled_tokens,
                 seq_budget=self.max_num_running_reqs, ready_step=self.current_step,
                 t0_semantics="all native checks and actual ownership complete; no target computation submitted")
            return
        selected = {rid: count for rid, count in tokens.items() if rid in self._slice_held}
        if not selected:
            if self._slice_allowed:
                self._slice_fail("ready_group_not_scheduled", allowed=list(self._slice_allowed))
            return
        expected = {rid: self._slice_held[rid]["remaining_tokens"] for rid in self._slice_allowed}
        if selected != expected:
            self._slice_fail("actual_quota_differs_from_design", expected=expected, actual=selected)
        self._slice_initialized.update(selected)
        self._slice_submitted.update(selected)
        self._slice_submit_batch_count += 1
        emit("batch_slice_submit", scenario=self._slice_config["scenario"], mode=self._slice_mode,
             batch_id=f"{os.getpid()}:{self.current_step}", step=self.current_step,
             slice_batch_index=self._slice_submit_batch_count, tokens=selected,
             remaining_budget=self.max_num_scheduled_tokens - sum(tokens.values()))
