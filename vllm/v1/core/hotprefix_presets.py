# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated immutable capability presets for HotPrefix cost experiments."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class HotPrefixCapabilities:
    """Immutable capabilities expanded from one named experiment preset."""

    shadow_local: bool = False
    hotprefix_eviction: bool = False
    global_access: bool = False
    selective_store: bool = False
    on_demand_fetch: bool = False
    promotion: bool = False

    @property
    def enabled_capabilities(self) -> frozenset[str]:
        """Return enabled capability names for validation and artifacts."""
        return frozenset(
            item.name for item in fields(self) if bool(getattr(self, item.name))
        )


_PRESETS: dict[str, HotPrefixCapabilities] = {
    "ablation_shadow_local": HotPrefixCapabilities(shadow_local=True),
    "ablation_local_drop": HotPrefixCapabilities(
        shadow_local=True,
        hotprefix_eviction=True,
    ),
    "ablation_access_only": HotPrefixCapabilities(
        shadow_local=True,
        hotprefix_eviction=True,
        global_access=True,
    ),
    "ablation_store_only": HotPrefixCapabilities(
        shadow_local=True,
        hotprefix_eviction=True,
        global_access=True,
        selective_store=True,
    ),
    "ablation_on_demand": HotPrefixCapabilities(
        shadow_local=True,
        hotprefix_eviction=True,
        global_access=True,
        selective_store=True,
        on_demand_fetch=True,
    ),
    "hotprefix": HotPrefixCapabilities(
        shadow_local=True,
        hotprefix_eviction=True,
        global_access=True,
        selective_store=True,
        on_demand_fetch=True,
        promotion=True,
    ),
}


def resolve_hotprefix_capabilities(
    eviction_policy: str,
    preset: str | None,
) -> HotPrefixCapabilities:
    """Resolve a policy and preset to validated immutable capabilities.

    Args:
        eviction_policy: Configured prefix-cache eviction policy.
        preset: Optional experiment-only HotPrefix preset.

    Returns:
        Capabilities for the process lifetime.

    Raises:
        ValueError: If a preset is combined with a non-HotPrefix policy.
    """
    if eviction_policy == "lru":
        if preset is not None:
            raise ValueError("HotPrefix experiment preset requires hotprefix policy")
        return HotPrefixCapabilities()
    if eviction_policy != "hotprefix":
        raise ValueError(f"unsupported prefix cache eviction policy: {eviction_policy}")
    if preset is not None and preset not in _PRESETS:
        raise ValueError(f"unknown HotPrefix experiment preset: {preset}")
    return _PRESETS[preset or "hotprefix"]
