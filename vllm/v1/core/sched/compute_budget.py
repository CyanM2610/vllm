# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure compute-quota proposals; resource acquisition stays in native schedule."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchTimeModel:
    theta0: float
    thetaQ: float
    thetaA: float
    thetaN: float
    tau_ms: float

    def __post_init__(self) -> None:
        values = (self.theta0, self.thetaQ, self.thetaA, self.thetaN)
        if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("time-model coefficients must be finite and nonnegative")
        if (
            isinstance(self.tau_ms, bool)
            or not math.isfinite(self.tau_ms)
            or self.tau_ms <= 0
        ):
            raise ValueError("tau_ms must be finite and positive")

    def predict(self, Q: int, A: int, N: int) -> float:
        if N == 0:
            return 0.0
        value = self.theta0 + self.thetaQ * Q + self.thetaA * A + self.thetaN * N
        if not math.isfinite(value):
            raise ValueError("time prediction overflow")
        return value


@dataclass(frozen=True)
class ComputeGrant:
    request_id: str
    p: int
    native_tokens: int
    tokens: int
    end_after_commit: bool
    reason: str
    Q: int
    A: int
    N: int
    predicted_before_ms: float | None
    predicted_after_ms: float | None
    prediction_calls: int
    prediction_wall_ns: int
    prediction_thread_cpu_ns: int
    search_wall_ns: int
    search_thread_cpu_ns: int
    budget_exception: bool = False


class ComputeBatchBudget:
    """One step's successful compute allocations, never a capacity account.

    ``propose`` has no mutation. ``commit`` follows successful native allocation;
    ``revoke`` follows native same-step output/resource rollback. No method reads
    cache state, request queues, future arrivals or reference output lengths.
    """

    def __init__(self, mode: str, model: BatchTimeModel | None = None, *, hard_cap: int = 2048) -> None:
        if mode not in ("off", "fixed", "time"):
            raise ValueError("compute mode must be off, fixed or time")
        if mode == "time" and model is None:
            raise ValueError("time mode requires a frozen model and tau")
        if type(hard_cap) is not int or hard_cap not in (2048, 4096):
            raise ValueError("unsupported native hard budget")
        if hard_cap != 2048 and mode != "off":
            raise ValueError("4096 extension is native/off only")
        self.hard_cap = hard_cap
        self.mode = mode
        self.model = model
        self.records: dict[str, tuple[int, int]] = {}
        self.step_id = -1

    def begin(self, step_id: int) -> None:
        self.step_id = step_id
        self.records.clear()

    def terms(self) -> tuple[int, int, int]:
        return (
            sum(q for p, q in self.records.values()),
            sum(q * p + q * (q + 1) // 2 for p, q in self.records.values()),
            len(self.records),
        )

    def propose(self, request_id: str, p: int, native_tokens: int) -> ComputeGrant:
        if (
            type(p) is not int
            or p < 0
            or type(native_tokens) is not int
            or not 1 <= native_tokens <= self.hard_cap
            or request_id in self.records
        ):
            raise ValueError("invalid or duplicate native compute proposal")
        Q, A, N = self.terms()
        if Q + native_tokens > self.hard_cap:
            raise ValueError("native proposal exceeds remaining hard token budget")
        calls = 0
        prediction_wall_ns = prediction_thread_cpu_ns = 0

        def predict_terms(Q: int, A: int, N: int) -> float:
            nonlocal calls, prediction_wall_ns, prediction_thread_cpu_ns
            calls += 1
            assert self.model is not None
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            value = self.model.predict(Q, A, N)
            prediction_wall_ns += time.monotonic_ns() - wall
            prediction_thread_cpu_ns += time.thread_time_ns() - cpu
            return value

        def prediction(q: int) -> float:
            return predict_terms(Q + q, A + q * p + q * (q + 1) // 2, N + 1)

        before = predict_terms(Q, A, N) if self.model is not None else None
        exception = False
        search_wall, search_cpu = time.monotonic_ns(), time.thread_time_ns()
        if self.mode == "off":
            q, stop, reason = native_tokens, False, "native_identity"
        elif self.mode == "fixed":
            q = min(native_tokens, max(0, 1024 - Q))
            stop, reason = Q + q >= 1024, "fixed_compute_cap"
        else:
            assert self.model is not None
            if prediction(native_tokens) <= self.model.tau_ms:
                q, stop, reason = native_tokens, False, "full_fit"
            elif prediction(1) > self.model.tau_ms:
                if N:
                    q, stop, reason = 0, True, "nonempty_no_fit"
                else:
                    q, stop, reason = 1, True, "empty_minimum_progress"
                    exception = True
            else:
                low, high = 1, native_tokens - 1
                while low < high:
                    mid = (low + high + 1) // 2
                    if prediction(mid) <= self.model.tau_ms:
                        low = mid
                    else:
                        high = mid - 1
                q, stop, reason = low, True, "partial_fit"
        after = prediction(q) if q and self.model is not None else before
        return ComputeGrant(
            request_id,
            p,
            native_tokens,
            q,
            stop,
            reason,
            Q,
            A,
            N,
            before,
            after,
            calls,
            prediction_wall_ns,
            prediction_thread_cpu_ns,
            time.monotonic_ns() - search_wall,
            time.thread_time_ns() - search_cpu,
            exception,
        )

    def commit(self, grant: ComputeGrant) -> None:
        if grant.tokens <= 0 or grant.request_id in self.records:
            raise ValueError("commit must represent one new successful allocation")
        if self.terms() != (grant.Q, grant.A, grant.N):
            raise ValueError("proposal became stale after rollback; propose again")
        self.records[grant.request_id] = (grant.p, grant.tokens)

    def revoke(self, request_id: str) -> tuple[int, int] | None:
        return self.records.pop(request_id, None)

    def assert_output(self, quotas: dict[str, int]) -> None:
        if quotas != {rid: q for rid, (p, q) in self.records.items()}:
            raise AssertionError("compute budget records differ from actual output")
