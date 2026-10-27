# SPDX-License-Identifier: Apache-2.0
"""Visible unsubmitted prefill work; this is never a readiness predicate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RemainingEstimate:
    prefix_tokens: int
    remaining_tokens: int
    work: int
    source: str
    observed_ns: int
    inflight_tokens: int
    restore_reserved: bool


def remaining_estimate(request, model, observed_ns, first=None, *, recompute=False):
    if request.held_hbm_blocks or request.accounted_tokens:
        p = min(request.input_tokens, request.accounted_tokens)
        source, stamp = "native_accounted_unsubmitted", observed_ns
    elif request.last_match is not None:
        match = request.last_match
        p = (
            (
                match.local_only_tokens
                if match.local_only_tokens is not None
                else match.local_tokens
            )
            if recompute
            else match.matched_tokens
        )
        source, stamp = (
            "local_recompute" if recompute else "last_observed_match",
            match.observed_ns,
        )
    elif first is not None:
        p = first.prefix_tokens
        source = "first_observed_match"
        stamp = getattr(first, "observed_ns", getattr(first, "frozen_ns", observed_ns))
    else:
        return None
    if not 0 <= p <= request.input_tokens:
        return None
    r = request.input_tokens - p
    return RemainingEstimate(
        p,
        r,
        model.estimate(r, p),
        source,
        stamp,
        request.inflight_tokens,
        request.restore_reserved,
    )
