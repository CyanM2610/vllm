# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-local observations and decisions; native owners retain every resource."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationView:
    operation_id: str
    request_id: str
    kind: str
    start_token: int | None
    end_token: int | None
    stage: str
    dependencies: tuple[tuple[str, str], ...] = ()
    blocked_reason: str | None = None
    proposed_tokens: int | None = None
    granted_tokens: int | None = None
    owner: str = "native_runtime"
    native_phase: str | None = None
    source_ownership: bool | None = None
    physical_ready: str = "unknown"
    resources: tuple[tuple[str, int], ...] = ()
    request_status: str | None = None
    input_tokens: int | None = None
    accounted_computed_tokens: int | None = None
    in_flight_tokens: int | None = None


@dataclass(frozen=True)
class OperationDecision:
    # "attempt" requests native checks and acquisition, not a readiness promise.
    action: str = "attempt"
    compute_tokens: int | None = None


@dataclass(frozen=True)
class StepOperationView:
    step_id: int
    remaining_token_budget: int
    remaining_input_budget: int
    free_hbm_blocks: int
    inflight_prefill_reserved_blocks: int
    operations: tuple[OperationView, ...]
    connector_operations: tuple[OperationView, ...] = ()


class NativeOperationDecider:
    """Retain native traversal and quota while allowing operation advancement."""

    def decide(
        self, step: StepOperationView, candidate: OperationView
    ) -> OperationDecision:
        return OperationDecision()
