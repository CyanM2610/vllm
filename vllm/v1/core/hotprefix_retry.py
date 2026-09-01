# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feasibility-epoch backoff for repeated HotPrefix promotion rejection."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.hotprefix_observability import HotPrefixReason


@dataclass(frozen=True)
class PromotionFeasibilityState:
    """State whose changes can make a rejected promotion feasible."""

    free_block_epoch: int
    residency_epoch: int
    group_epoch: int
    hotness_version: tuple[int, int]


class PromotionRetryBackoff:
    """Cache one terminal rejection per prefix and feasibility state."""

    def __init__(self) -> None:
        self._rejections: dict[
            bytes, tuple[PromotionFeasibilityState, HotPrefixReason]
        ] = {}

    def record_rejection(
        self,
        prefix_id: bytes,
        state: PromotionFeasibilityState,
        reason: HotPrefixReason,
    ) -> None:
        """Remember why a prefix was infeasible in one exact state.

        Args:
            prefix_id: Stable logical prefix identity.
            state: Exact feasibility state used by the failed attempt.
            reason: Bounded terminal rejection reason.
        """
        self._rejections[prefix_id] = (state, reason)

    def rejection_for(
        self, prefix_id: bytes, state: PromotionFeasibilityState
    ) -> HotPrefixReason | None:
        """Return the cached reason only while feasibility is unchanged.

        Args:
            prefix_id: Stable logical prefix identity.
            state: Current feasibility state.

        Returns:
            Cached reason, or ``None`` when a retry is now allowed.
        """
        rejection = self._rejections.get(prefix_id)
        if rejection is None or rejection[0] != state:
            return None
        return rejection[1]

    def record_success(self, prefix_id: bytes) -> None:
        """Clear stale rejection state after a successful reservation.

        Args:
            prefix_id: Successfully reserved logical prefix.
        """
        self._rejections.pop(prefix_id, None)
