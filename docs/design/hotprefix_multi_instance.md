# Multi-instance HotPrefix integration

This branch adds the vLLM-local half of a faithful, multi-instance HotPrefix
baseline. Native automatic prefix caching remains the correctness authority;
HotPrefix is a policy shadow and never reports additional computed tokens.

## Enable the local policy

```bash
vllm serve MODEL \
  --enable-prefix-caching \
  --prefix-cache-eviction-policy hotprefix \
  --hotprefix-aging-interval 50 \
  --hotprefix-num-buckets 16384
```

`lru` remains the default. The first implementation supports one full-attention
KV-cache group. Unsupported hybrid/multi-group configurations fail at startup
instead of silently changing eviction semantics.

## Local behavior

- `KVCacheManager` maintains one token-level radix tree per
  `model\0cache_salt` namespace. All trees share the bounded cuckoo hotness
  store, while namespace-specific topology prevents tenant collisions.
- Native APC lookup updates the matched logical path. Cache publication updates
  topology and projects logical node segments onto actual vLLM blocks.
- Nodes sharing a boundary block become one `EvictionGroup`. The group score is
  based on reclaimable physical block capacity, and selecting any member
  invalidates collateral APC entries without allocating collateral blocks.
- The default BlockPool path is unchanged when the policy is `lru`.
- Connectors receive a `plan_background_transfers` callback after non-blocking
  model execution begins. This is the CPU overlap window for Host candidate
  lookup and later-step promotion planning; it cannot mutate the current
  `SchedulerOutput`.

## LMCache contract

The LMCache MP connector enables the shared half with
`lmcache.mp.hotprefix_enabled=true`. The connector validates that the local
eviction policy is also `hotprefix`, sends one initial access observation per
request, and queries READY Host candidates only for locally hot nodes absent
from HBM. Prefix IDs use the same BLAKE2b derivation on both sides.

An allocation that reaches an unadmitted cached victim is safely deferred. The
exact physical `EvictionGroup` is then admitted during the GPU overlap window;
its full-prefix source blocks stay pinned until STORE reaches DEDUP, REJECT,
READY, or failure, after which the original allocation may retry. Promotion
targets are also detached reservations and remain invisible to APC until every
worker reports RETRIEVE completion. Successful promotion computes ordinary vLLM
block hashes and publishes through BlockPool; failure removes any partial hashes
and frees the unpublished targets.

Promotion is node-atomic and limited to one in-flight transaction per instance.
The complete logical prefix is sliced into LMCache-aligned ranges bounded by
`lmcache.mp.hotprefix_promotion_budget_bytes`; at least one transfer chunk is
allowed as the starvation quantum. Chunks advance only on steps carrying decode
work, while all target blocks and the renewable source ticket remain reserved.
Admission also leaves at least one additional promotion-sized region free for
foreground decode growth; when that conservative HBM headroom is unavailable,
background promotion pauses instead of repeatedly preempting the foreground
request. Only the final successful chunk publishes native APC hashes.

## Cost observability and experiment presets

HotPrefix producers record immutable events through the two-method
`HotPrefixObservationCollector` interface (`record` and `drain`). The no-op
adapter is selected by `--hotprefix-observability-mode off`, independently of
general scheduler stats. `aggregate` emits interval deltas through
`SchedulerStats`; `trace` adds sampled high-cardinality spans. Off mode returns
before timing or event allocation. Prometheus labels contain only fixed
stage/action/outcome/reason enums. Request IDs and prefix digests are not
retained in aggregate stats.

`HotPrefixBlockProjection` owns per-path signatures, radix-path-to-block
mapping, arithmetic block spans, and binding invalidation behind
`reconcile/discard/age`. Identical path/hotness/binding signatures skip group
rebuilds. Projection metrics report invocation/skip reason, total/path nodes,
request/projected blocks, rebuilt groups, connected-component blocks, and aging
CPU. HBM gauges are pull snapshots for free, resident, promotion-reserved,
STORE-pinned, and decode-headroom blocks.

Promotion rejection is cached by the physical allocation epoch, residency
generation, eviction-group epoch, and candidate hotness. Unchanged feasibility
produces a backoff skip rather than another reservation attempt; any relevant
epoch change reopens the candidate. Promotion bytes are separated into copy,
publish, and fail phases. LMCache stream completion remains authoritative for
data-plane bytes.

The following immutable presets exist only for cost attribution. Normal
production behavior remains named `hotprefix` and is unchanged when no preset
is supplied:

| Preset | Enabled through this stage |
| --- | --- |
| `ablation_shadow_local` | Local tree/Cuckoo bookkeeping; LRU HBM eviction |
| `ablation_local_drop` | HotPrefix HBM victim selection without Host residency |
| `ablation_access_only` | Synchronous Global ACCESS, without STORE/fetch |
| `ablation_store_only` | Selective admission and eviction STORE |
| `ablation_on_demand` | Native Host lookup/retrieve |
| `hotprefix` | Background promotion and the faithful complete policy |

Configure them with `--hotprefix-experiment-preset` together with
`--prefix-cache-eviction-policy hotprefix`. Combining a preset with `lru` is
rejected at startup; native LRU is the P0 experiment baseline and uses no
preset.

STORE admission and promotion size a physical block as the per-layer page size
multiplied by the number of layers in its cache group. Using the per-layer page
alone undercounts Host capacity and transfer headroom for ordinary multi-layer
models.
