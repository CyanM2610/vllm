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
Only the final successful chunk publishes native APC hashes.
