"""Ephemeral native match result; never retained across a scheduler step."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixMatch:
    blocks: Any
    local_tokens: int
    external_tokens: int
    matched_tokens: int
    load_kv_async: bool
    shared_prefix_boundary: int
    connector_queries: int

