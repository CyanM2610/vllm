# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in adapter; legacy compute-plugin exclusions remain intact."""

import json
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.joint_prefill import SRPTCoordinationPlan


class SRPTCoordinationScheduler(AsyncScheduler):
    schema = "srpt-coordination-v1"

    def make_policy(self, config):
        return SRPTCoordinationPlan(self.vllm_config.model_config.hf_config, config)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self._global_plan is None
            or self._first_match_preparation is None
            or self._first_match_preparation.limit != 32
            or self._native_compute_selection is not None
            or getattr(self, "_compute_budget", None) is not None
            or getattr(self, "_global_sjf_input_enabled", False)
            or getattr(self, "_conditional_end_enabled", False)
            or not self._global_plan.recovery_path_choice
            or self.max_num_scheduled_tokens != 2048
            or self.max_num_running_reqs != 32
        ):
            raise ValueError(
                "joint SRPT requires frozen global plan / prepare32 / budget2048"
            )
        config = json.loads(
            Path(os.environ["PREFILL_SRPT_COORDINATION_CONFIG"]).read_text()
        )
        if config.get("schema") != self.schema:
            raise ValueError("explicit frozen coordination configuration required")
        if config.get("scope") != "Qwen2.5-7B/A100/eager/P-only/block16/chunk256":
            raise ValueError("unsupported model or execution cost scope")
        policy = self.make_policy(config)
        self.set_prefill_plan_policy(policy)


class SRPTRollingScheduler(SRPTCoordinationScheduler):
    schema = "srpt-coordination-rolling-v2"

    def make_policy(self, config):
        from vllm.v1.core.sched.joint_prefill_rolling import SRPTRollingPlan

        if len(config["scenarios"]) != 8 or config.get("rounds") != 2:
            raise ValueError("rolling v2 requires frozen eight scenarios/two rounds")
        return SRPTRollingPlan(self.vllm_config.model_config.hf_config, config)

    def set_execution_window_reader(self, reader):
        from dataclasses import replace

        policy = self._global_plan.policy

        def read():
            window = reader()
            snapshot = policy.snapshot
            requests = (
                ()
                if snapshot is None
                else tuple(
                    r.request_id
                    for r in snapshot.requests
                    if r.lookup_submitted and not r.lookup_result_known
                )
            )
            # Bounded read of host future flags, no result consumption or STATUS.
            readiness = self.connector.lookup_reply_readiness(requests[:32])
            return replace(
                window,
                control_ready=readiness["ready"],
                control_checked=len(requests[:32]),
                control_omitted=max(0, len(requests) - 32),
            )

        policy.set_execution_window_reader(read)
