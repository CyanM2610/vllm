# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen compute-only quotas on native or existing global SJF traversal."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.compute_budget import BatchTimeModel, ComputeBatchBudget
from vllm.v1.core.sched.compute_cost import ComputeCostModel
from vllm.v1.core.sched.global_sjf_compute import GlobalSJFComputeScheduler
from vllm.v1.core.sched.global_sjf_input import frozen_config_scope
from vllm.v1.kv_wait_trace import emit


class ComputeBudgetMixin:
    """Bind quota decisions independently of operation/fetch scheduling."""

    @property
    def _global_sjf_expected_token_budget(self):
        # Only the explicit compute-budget subclass extends the parent scope.
        return self.scheduler_config.max_num_batched_tokens

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_budget_initialize(
            os.environ.get("PREFILL_COMPUTE_BUDGET_MODE", "off"),
            os.environ.get("PREFILL_COMPUTE_BUDGET_CONFIG"),
        )

    def _compute_budget_initialize(self, mode, config_path=None):
        hard_cap = self.scheduler_config.max_num_batched_tokens
        if hard_cap not in (2048, 4096):
            raise ValueError("unsupported compute hard budget")
        scope, errors = frozen_config_scope(
            self.vllm_config, self.block_size, expected_token_budget=hard_cap)
        config = self.vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        action_combination = (
            getattr(self, "_action_feedback", None) is not None
            and mode in ("off", "fixed") and config_path is None and hard_cap == 2048
            and not getattr(self, "_global_sjf_input_enabled", False)
            and not getattr(self, "_conditional_end_enabled", False)
        )
        if getattr(self, "_action_feedback", None) is not None and not action_combination:
            errors.append("unverified_action_feedback_combination")
        for name in (
            "decoupled_control",
            "prepare_lookup_operations",
            "decoupled_store",
        ):
            if config.get("lmcache.mp." + name, False):
                if not action_combination or name == "prepare_lookup_operations":
                    errors.append(name)
        if getattr(self, "_prefill_operations_enabled", False) or getattr(
            self, "_conditional_end_enabled", False
        ):
            if not action_combination:
                errors.append("operation_or_conditional_end_scheduler")
        if not self.use_v2_model_runner or self.max_num_scheduled_tokens != hard_cap:
            errors.append("V2_native_hard_budget")
        if (
            self.max_num_running_reqs != 32
            or self.scheduler_config.max_num_batched_tokens != hard_cap
        ):
            errors.append("native_input_or_sequence_budget")
        for name in (
            "num_lookahead_tokens",
            "num_prefill_lookahead",
            "num_spec_tokens",
        ):
            if getattr(self, name, None) != 0:
                errors.append(name)
        if self.has_mamba_layers or self.ec_connector is not None:
            errors.append("full_attention_text_only")
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if (
            self.kv_cache_manager.num_kv_cache_groups != 1
            or len(managers) != 1
            or type(managers[0]).__name__ != "FullAttentionManager"
            or managers[0].block_size != self.block_size
            or managers[0]._pending_cow_copies
            or self.kv_cache_manager.block_pool.hash_block_size != self.block_size
        ):
            errors.append("single_dense_equal_block_geometry")
        if self.connector is None or self.num_sampled_tokens_per_step != 1:
            errors.append("native_connector_one_token")
        if errors:
            raise ValueError("compute-only budget unsupported scope: " + repr(errors))
        # Reuse the frozen Qwen geometry validator; this structural model is
        # not the fitted forward-time model and does not affect quota choice.
        config = self.vllm_config.model_config
        ComputeCostModel.from_config(
            getattr(config, "hf_text_config", None) or config.hf_config
        )
        model = None
        identity = None
        if config_path is not None:
            data = Path(config_path).read_bytes()
            config = json.loads(data)
            if (
                config.get("schema") != "prefill-forward-time-budget-v1"
                or config.get("unit") != "ms"
            ):
                raise ValueError("unknown time model schema or unit")
            if config.get("hard_token_budget", 2048) != hard_cap:
                raise ValueError("time model hard budget differs from runtime")
            model = BatchTimeModel(**config["coefficients"], tau_ms=config["tau_ms"])
            identity = {"sha256": hashlib.sha256(data).hexdigest(), "config": config}
        self._compute_budget = ComputeBatchBudget(mode, model, hard_cap=hard_cap)
        self._compute_budget_retry_matches = set()
        emit(
            "compute_budget_enabled",
            mode=mode,
            fixed_cap=1024,
            hard_cap=hard_cap,
            minimum_legal_tokens=1,
            model=identity,
            scope=scope,
            order="existing_global_sjf_compute"
            if getattr(self, "_global_sjf_input_enabled", False)
            else "native",
            semantics="compute quota only; proposal is not allocation or physical readiness",
        )

    def add_request(self, request):
        if (
            request.max_tokens != 1
            or request.resumable
            or request.has_encoder_inputs
            or request.sampling_params is None
            or request.sampling_params.max_tokens != 1
            or request.pooling_params is not None
            or request.lora_request is not None
            or request.prompt_embeds is not None
            or request.structured_output_request is not None
            or request.spec_token_ids
            or request.priority != 0
        ):
            raise ValueError(
                "compute-budget baseline requires text non-resumable P-only max_tokens=1"
            )
        return super().add_request(request)

    def _compute_budget_begin(self):
        self._compute_budget.begin(self.current_step)
        self._compute_budget_retry_matches.intersection_update(
            rid
            for rid in self._compute_budget_retry_matches
            if rid in self.requests and not self.requests[rid].is_finished()
        )

    def _compute_budget_propose(self, request, p, native_tokens, branch):
        wall, cpu = time.monotonic_ns(), time.thread_time_ns()
        grant = self._compute_budget.propose(request.request_id, p, native_tokens)
        elapsed, cpu_elapsed = time.monotonic_ns() - wall, time.thread_time_ns() - cpu
        emit(
            "compute_budget_proposal",
            request.request_id,
            step=self.current_step,
            branch=branch,
            mode=self._compute_budget.mode,
            **{
                key: value
                for key, value in asdict(grant).items()
                if key != "request_id"
            },
            tau_ms=self._compute_budget.model.tau_ms
            if self._compute_budget.model
            else None,
            wall_ns=elapsed,
            thread_cpu_ns=cpu_elapsed,
            dependency_checks="native_passed",
            capacity="not_yet_acquired",
            physical_ready="not_inferred_from_computed_tokens",
        )
        return grant

    def _compute_budget_commit(self, grant, branch):
        self._compute_budget.commit(grant)
        self._compute_budget_retry_matches.discard(grant.request_id)
        Q, A, N = self._compute_budget.terms()
        emit(
            "compute_budget_committed",
            grant.request_id,
            step=self.current_step,
            branch=branch,
            p=grant.p,
            quota=grant.tokens,
            Q=Q,
            A=A,
            N=N,
            reason=grant.reason,
            end_batch=grant.end_after_commit,
            budget_exception=grant.budget_exception,
            predicted_ms=grant.predicted_after_ms,
        )

    def _compute_budget_revoke(self, request_id):
        record = self._compute_budget.revoke(request_id)
        if record is not None:
            Q, A, N = self._compute_budget.terms()
            emit(
                "compute_budget_revoked",
                request_id,
                step=self.current_step,
                p=record[0],
                quota=record[1],
                Q=Q,
                A=A,
                N=N,
                reason="native_same_step_rollback",
            )


class ComputeBudgetScheduler(ComputeBudgetMixin, AsyncScheduler):
    """A/B/C retain native traversal and vary only compute quota policy."""


class GlobalSJFComputeBudgetScheduler(ComputeBudgetMixin, GlobalSJFComputeScheduler):
    """D retains existing frozen first-match ordering/preparation."""
