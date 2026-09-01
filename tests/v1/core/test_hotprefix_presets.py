# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.hotprefix_presets import resolve_hotprefix_capabilities

pytestmark = pytest.mark.cpu_test


def test_presets_expand_to_monotonic_immutable_capabilities() -> None:
    names = (
        "ablation_shadow_local",
        "ablation_local_drop",
        "ablation_access_only",
        "ablation_store_only",
        "ablation_on_demand",
        "hotprefix",
    )
    capabilities = [resolve_hotprefix_capabilities("hotprefix", name) for name in names]

    assert capabilities[0].shadow_local
    for previous, current in zip(capabilities, capabilities[1:]):
        assert previous.enabled_capabilities <= current.enabled_capabilities
    assert capabilities[-1].promotion
    assert capabilities[-1] == resolve_hotprefix_capabilities("hotprefix", None)


def test_lru_rejects_experiment_presets() -> None:
    assert not resolve_hotprefix_capabilities("lru", None).enabled_capabilities
    with pytest.raises(ValueError, match="requires.*hotprefix"):
        resolve_hotprefix_capabilities("lru", "ablation_shadow_local")
