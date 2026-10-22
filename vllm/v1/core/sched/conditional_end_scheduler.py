"""Fixed early-end experiment: preserve native traversal, quotas and ownership.

Both arms observe the same rule. Only ``on`` ends a not-yet-submitted step.
Capacity proof uses the native dense allocator's read-only block counter at
the actual allocation boundary. It neither allocates nor retains matched
blocks. Current submitted progress is not claimed to be physically complete.
"""
from __future__ import annotations
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.compute_cost import ComputeCostModel
from vllm.v1.core.sched.global_sjf_input import frozen_config_scope
from vllm.v1.kv_wait_trace import emit
from vllm.v1.request import RequestStatus

ALLOWED_COST_RATIOS = (2, 3, 4)  # User-authorized fixed arms, not a fitted score.


class ConditionalEndScheduler(AsyncScheduler):
    _conditional_end_enabled = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conditional_initialize(os.environ.get("PREFILL_BATCH_END_MODE"),
                                     os.environ.get("PREFILL_BATCH_END_RATIO", "4"))

    def _conditional_initialize(self, mode, ratio=4):
        scope, errors = frozen_config_scope(self.vllm_config, self.block_size)
        manager = self.kv_cache_manager
        dense = manager.coordinator.single_type_managers
        if mode not in ("off", "on"):
            errors.append("PREFILL_BATCH_END_MODE")
        if isinstance(ratio,bool) or str(ratio) not in ("2","3","4"):
            errors.append("PREFILL_BATCH_END_RATIO must be exactly 2, 3 or 4")
        if len(dense) != 1 or type(dense[0]).__name__ != "FullAttentionManager":
            errors.append("one_exact_full_attention_manager")
        elif dense[0]._max_admission_blocks_per_request is not None:
            errors.append("dense_admission_cap")
        for name in ("num_lookahead_tokens", "num_prefill_lookahead", "num_spec_tokens"):
            if getattr(self, name, None) != 0:
                errors.append(name)
        if self.has_mamba_layers or self.ec_connector is not None or self.connector is None:
            errors.append("dense_connector_only")
        if self.max_num_running_reqs != 32 or self.max_num_scheduled_tokens != 2048:
            errors.append("actual_sequence_or_token_budget")
        if errors:
            raise ValueError(f"Conditional-end unsupported scope: {errors}")
        self._conditional_mode = mode
        self._conditional_ratio = int(ratio)
        self._conditional_cost = ComputeCostModel.from_config(self.vllm_config.model_config.hf_config)
        if (self._conditional_cost.a, self._conditional_cost.b) != (13050576896,401408):
            raise ValueError("Conditional-end structural cost differs from frozen Qwen7B")
        self._conditional_deferred = set()
        self._conditional_retry_lookup = set()
        self._conditional_selected_rows = {}
        self._conditional_initial_observed = False
        emit("conditional_end_enabled", mode=mode, scope=scope,
             rule=dict(min_selected=2,min_tokens=512,max_tokens=1024,cost_ratio=self._conditional_ratio,total_tokens_max=2048,max_lifetime_deferrals=1),
             cost_model=self._conditional_cost.to_dict(),
             load_page_scope_environment=os.environ.get("LMCACHE_LOAD_PAGE_SCOPE"),
             connector_class=type(self.connector).__module__+"."+type(self.connector).__name__)

    @staticmethod
    def _conditional_logical(request):
        extra = getattr(getattr(request,"sampling_params",None),"extra_args",None)
        return extra.get("native_token_replay_id") if isinstance(extra,dict) else None

    def add_request(self, request):
        if not self._conditional_initial_observed and self._conditional_logical(request):
            self._conditional_initial_observed = True
            pool = self.kv_cache_manager.block_pool
            blocks = [b for b in pool.blocks if not b.is_null]
            emit("conditional_end_initial_hbm", request.request_id,
                 mode=self._conditional_mode, logical_id=self._conditional_logical(request),
                 hash_cache_entries=len(pool.cached_block_hash_to_block),
                 hash_bearing_physical_blocks=sum(b.block_hash is not None for b in blocks),
                 free_blocks=pool.get_num_free_blocks(),physical_blocks_excluding_null=len(blocks),
                 active_blocks=sum(b.ref_cnt>0 for b in blocks),active_refs=sum(b.ref_cnt for b in blocks),
                 existing_requests=list(self.requests),running_requests=[r.request_id for r in self.running],
                 warmup_completion="not inferred; join external warmup.end_ns and this timestamp",
                 semantics="read-only before first formal replay request is admitted; no reset or synchronization")
        super().add_request(request)

    def _conditional_begin(self):
        self._conditional_selected_rows = {}

    def _conditional_row(self, request, p, quota, branch):
        r = request.num_prompt_tokens - p
        return dict(request_id=request.request_id,logical_id=self._conditional_logical(request),
                    branch=branch,native_status=request.status.name,p=p,r=r,quota=quota,
                    W=self._conditional_cost.estimate(r,p) if r>=0 and p>=0 else None,
                    finishes_prefill=quota==r and r>0 and request.max_tokens==1,
                    num_in_flight_tokens=request.num_in_flight_tokens,
                    num_output_placeholders=request.num_output_placeholders,
                    readiness="native legal-submit dependency checks; not asserted physically complete")

    def _conditional_selected(self, request, p, quota, branch):
        wall=time.monotonic_ns(); cpu=time.thread_time_ns()
        row = self._conditional_row(request,p,quota,branch)
        row["actual_allocation_succeeded"] = True
        self._conditional_selected_rows[request.request_id] = row
        emit("conditional_end_selected",request.request_id,batch_id=f"{os.getpid()}:{self.current_step}",
             basis_wall_ns=time.monotonic_ns()-wall,basis_thread_cpu_ns=time.thread_time_ns()-cpu,
             **{k:v for k,v in row.items() if k!="request_id"})

    def _conditional_capacity(self, request, *, quota, local_tokens, external_tokens, blocks,
                              full_sequence, reserved_blocks, has_scheduled_reqs):
        """Exact native gates in the supported full-attention/no-lookahead case.

        Native remove_skipped_blocks returns immediately for FullAttention;
        both native block-count calls are pure. No native ownership, free-list,
        cache map or progress mutation takes place between this and allocation.
        """
        manager = self.kv_cache_manager
        computed = request.num_computed_tokens + local_tokens + external_tokens
        total = min(computed, self.max_model_len)
        local = request.num_computed_tokens + local_tokens
        block_lists = manager.empty_kv_cache_blocks.blocks if blocks is None else blocks.blocks
        watermark = manager.watermark_blocks if has_scheduled_reqs and request.status in (RequestStatus.WAITING,RequestStatus.PREEMPTED) else 0
        free = manager.block_pool.get_num_free_blocks()
        def count(tokens, admission):
            return manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,num_tokens=tokens,new_computed_blocks=block_lists,
                num_encoder_tokens=0,total_computed_tokens=total,num_local_computed_tokens=local,
                num_tokens_main_model=tokens,apply_admission_cap=admission)
        full = count(min(request.num_tokens,self.max_model_len),True) if full_sequence else None
        end = total + quota
        step = count(min(end,self.max_model_len),False)
        # Native max_model_len cap and full-step end are equal for P-only here.
        if end > self.max_model_len or external_tokens or reserved_blocks:
            return dict(known=False,reason="outside_dense_compute_only_boundary")
        full_fits = full is None or full + watermark <= free
        step_fits = step + watermark <= free - reserved_blocks
        return dict(known=True,passes=full_fits and step_fits,free_blocks=free,
                    full_sequence_required_blocks=full,step_required_blocks=step,
                    watermark_blocks=watermark,reserved_blocks=reserved_blocks,
                    full_sequence_fits=full_fits,step_fits=step_fits,
                    proof="native FullAttention coordinator count; remove_skipped_blocks is no-op; synchronous allocation boundary")

    def _conditional_should_stop(self, request, *, p, quota, branch, selected_tokens,
                                 local_tokens=0,external_tokens=0,blocks=None,
                                 full_sequence=False,reserved_blocks=0,has_scheduled_reqs=True,
                                 load_async=False):
        wall=time.monotonic_ns(); cpu=time.thread_time_ns()
        selected=[]
        for rid,n in selected_tokens.items():
            row=self._conditional_selected_rows.get(rid)
            if row is None or row["quota"]!=n:
                selected.append(dict(request_id=rid,quota=n,finishes_prefill=False,reason="selected_basis_unknown"))
            else:
                selected.append(dict(row))
        j=self._conditional_row(request,p,quota,branch)
        total=sum(x["quota"] for x in selected)
        reasons=[]
        if len(selected)<2: reasons.append("fewer_than_two_selected")
        if not all(x["finishes_prefill"] for x in selected): reasons.append("selected_prefill_not_complete")
        if not 512<=total<=1024: reasons.append("selected_tokens_outside_512_1024")
        if not j["finishes_prefill"]: reasons.append("candidate_not_complete_this_step")
        if load_async or external_tokens: reasons.append("candidate_requires_restore")
        if total+j["r"]>2048: reasons.append("combined_tokens_exceed_2048")
        if request.request_id in self._conditional_deferred: reasons.append("already_deferred_this_lifetime")
        max_cost=max((x.get("W") or 0 for x in selected),default=0)
        if j["W"] is None or j["W"]<self._conditional_ratio*max_cost: reasons.append("cost_ratio_below_frozen_threshold")
        capacity=dict(known=False,reason="size_or_state_rule_not_satisfied")
        if not reasons:
            capacity=self._conditional_capacity(request,quota=quota,local_tokens=local_tokens,
                external_tokens=external_tokens,blocks=blocks,full_sequence=full_sequence,
                reserved_blocks=reserved_blocks,has_scheduled_reqs=has_scheduled_reqs)
            if not capacity.get("known"): reasons.append("capacity_unknown")
            elif not capacity["passes"]: reasons.append("native_capacity_gate_fails")
        would_stop=not reasons
        stop=would_stop and self._conditional_mode=="on"
        if stop:
            self._conditional_deferred.add(request.request_id)
            if branch=="waiting" and request.num_computed_tokens==0:
                # Re-observe the same native future with idempotent absolute
                # prefix accounting, while re-matching HBM at actual takeover.
                self._conditional_retry_lookup.add(request.request_id)
        emit("conditional_end_evaluation",request.request_id,mode=self._conditional_mode,
             cost_ratio=self._conditional_ratio,
             batch_id=f"{os.getpid()}:{self.current_step}",step=self.current_step,
             S=selected,j=j,selected_tokens=total,remaining_budget=2048-total,
             native_dependency_checks_passed=True,capacity=capacity,
             would_stop=would_stop,triggered=stop,reasons=reasons,
             decision_wall_ns=time.monotonic_ns()-wall,decision_thread_cpu_ns=time.thread_time_ns()-cpu,
             timing_scope="rule and read-only capacity calculation; event serialization/emission excluded")
        return stop

    def _conditional_existing_lookup(self, request):
        retry=request.request_id in self._conditional_retry_lookup
        if retry:
            emit("conditional_end_retry_match",request.request_id,
                 batch_id=f"{os.getpid()}:{self.current_step}",
                 semantics="fresh native HBM match + same lower future, idempotent STORE prefix commit; no extra remote LOOKUP")
        return retry

    def _conditional_adopted(self, request):
        self._conditional_retry_lookup.discard(request.request_id)

    def _free_request(self, request, delay_free_blocks=False):
        result=super()._free_request(request,delay_free_blocks=delay_free_blocks)
        self._conditional_deferred.discard(request.request_id)
        self._conditional_retry_lookup.discard(request.request_id)
        self._conditional_selected_rows.pop(request.request_id,None)
        return result
