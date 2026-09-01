# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.hotprefix_observability import HotPrefixReason
from vllm.v1.core.hotprefix_retry import (
    PromotionFeasibilityState,
    PromotionRetryBackoff,
)

pytestmark = pytest.mark.cpu_test


def test_retry_backoff_reopens_immediately_after_feasibility_epoch_change() -> None:
    backoff = PromotionRetryBackoff()
    state = PromotionFeasibilityState(
        free_block_epoch=3,
        residency_epoch=7,
        group_epoch=11,
        hotness_version=(4, 255),
    )
    backoff.record_rejection(b"prefix", state, HotPrefixReason.CONFLICT)

    assert backoff.rejection_for(b"prefix", state) is HotPrefixReason.CONFLICT
    assert (
        backoff.rejection_for(
            b"prefix",
            PromotionFeasibilityState(
                free_block_epoch=4,
                residency_epoch=7,
                group_epoch=11,
                hotness_version=(4, 255),
            ),
        )
        is None
    )
