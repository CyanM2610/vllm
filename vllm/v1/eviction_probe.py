"""Passive HBM cache/release observations, copied into the isolated vLLM source.

The observer never allocates, looks up L1, queries a Future, or waits for work.
It records native mutations after they occurred. Observation failures are
explicit events and cannot change a native result.
"""
from vllm.v1.kv_wait_trace import emit

emit("hbm_observer_enabled", schema="native-hbm-directory-release-v1")


def _hash(value):
    return value.hex() if isinstance(value, bytes) else repr(value)


def cached(request, blocks, start, end, block_size, group, mask):
    try:
        rows = []
        for index in range(start, end):
            block = blocks[index]
            if block.is_null or (mask is not None and not mask[index-start]):
                continue
            rows.append(dict(block_id=block.block_id, hash_id=_hash(block.block_hash),
                             prefix_end_tokens=(index+1)*block_size, ref_cnt=block.ref_cnt))
        if rows:
            emit("hbm_cache_published", request.request_id, group_id=group,
                 block_size=block_size, blocks=rows,
                 semantics="native directory hash publication; GPU completion is separately recorded")
    except Exception as error:
        emit("hbm_observation_error", getattr(request, "request_id", None), operation="cache_publish", error=repr(error))


def evicted(block_id, hashes, ref_cnt, num_tokens, reason):
    try:
        emit("hbm_hash_evicted", block_id=block_id, hash_ids=[_hash(h) for h in hashes],
             ref_cnt=ref_cnt, cached_prefix_end_tokens=num_tokens, reason=reason,
             semantics="native hash removal; no STORE is submitted by this action")
    except Exception as error:
        emit("hbm_observation_error", operation="hash_evict", error=repr(error))


def request_hold(scheduler, request, delay, connector_delay):
    try:
        emit("hbm_request_finish_hold", request.request_id, delay_free=bool(delay),
             connector_delay_free=bool(connector_delay),
             block_ids=[list(group) for group in scheduler.kv_cache_manager.get_block_ids(request.request_id)],
             free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
             last_sched_seq=request.last_sched_seq, processed_step_seq=scheduler.processed_step_seq,
             semantics="connector/native retention decision; not proof a physical STORE is still running")
    except Exception as error:
        emit("hbm_observation_error", getattr(request, "request_id", None), operation="request_hold", error=repr(error))


def request_released(manager, request):
    try:
        emit("hbm_request_blocks_released", request.request_id,
             free_blocks=manager.block_pool.get_num_free_blocks(),
             semantics="coordinator.free returned; shared blocks may retain other owners, cached hashes remain until reuse")
    except Exception as error:
        emit("hbm_observation_error", getattr(request, "request_id", None), operation="request_release", error=repr(error))
