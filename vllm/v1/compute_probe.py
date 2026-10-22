"""Passive single-instance batch identity and native decision observations.

No cache lookup/allocation, CUDA event, query, or synchronization is added.
Only public request progress and native branch outcomes are recorded.
"""
from contextvars import ContextVar
from functools import wraps
import os

from vllm.v1.kv_wait_trace import emit, span

_context=ContextVar('compute_batch',default=None)
_installed=False


def batch_id(output):
    return getattr(output,'_compute_probe_id',None)


def state(request):
    return dict(id=request.request_id,status=request.status.name,
        prompt=request.num_prompt_tokens,computed=request.num_computed_tokens,
        tokens=request.num_tokens,with_spec=request.num_tokens_with_spec,
        placeholders=request.num_output_placeholders,inflight=request.num_in_flight_tokens,
        outputs=request.num_output_tokens,stale=request.num_stale_output_tokens,
        next_eligible=request.next_decode_eligible_step,
        prefill=bool(request.is_prefill_chunk))


def mark(scheduler,request,reason,**fields):
    scheduler._compute_probe_decisions[request.request_id]=dict(reason=reason,**fields)


def scheduled(function):
    @wraps(function)
    def call(self,*args,**kwargs):
        bid=f'{os.getpid()}:{self.current_step+1}'
        self._compute_probe_decisions={}
        before=[state(r) for r in self.requests.values()]
        pool=self.kv_cache_manager.block_pool
        emit('compute_decision_begin',batch_id=bid,step=self.current_step+1,requests=before,
             running=[r.request_id for r in self.running],
             waiting=[r.request_id for r in self.waiting],skipped=[r.request_id for r in self.skipped_waiting],
             token_budget=self.max_num_scheduled_tokens,input_budget=self.scheduler_config.max_num_batched_tokens,
             max_num_seqs=self.max_num_running_reqs,free_blocks=pool.get_num_free_blocks(),
             total_blocks=pool.num_gpu_blocks,chunk_threshold=self.scheduler_config.long_prefill_token_threshold,
             chunked_prefill=self.scheduler_config.enable_chunked_prefill,
             reserve_full_isl=self.scheduler_reserve_full_isl)
        output=function(self,*args,**kwargs)
        # Single-GPU UniProc passes this object by identity; verify at worker.
        output._compute_probe_id=bid
        chosen=[]
        for rid,n in output.num_scheduled_tokens.items():
            r=self.requests[rid];begin=r.num_computed_tokens-n
            chosen.append(dict(id=rid,quota=n,start=begin,end=r.num_computed_tokens,
                prompt=r.num_prompt_tokens,phase='prefill' if begin<r.num_prompt_tokens else 'decode'))
        emit('compute_decision_end',batch_id=bid,selected=chosen,decisions=self._compute_probe_decisions,
             requests=[state(r) for r in self.requests.values()],free_blocks=pool.get_num_free_blocks(),
             remaining_tokens=self.max_num_scheduled_tokens-output.total_num_scheduled_tokens,
             slots_remaining=self.max_num_running_reqs-len(self.running),
             preempted=list(output.preempted_req_ids or ()))
        return output
    return call


def observer_factory():
    identity=_context.get()
    if identity is None:return lambda *args,**kwargs:None
    bid=identity
    def observer(name,**fields):
        fields.pop('stream',None)  # Device stream identity is recorded by Nsight.
        emit('compute_phase',batch_id=bid,phase=name,**fields)
    return observer


def executor(kind):
    def decorate(function):
        @wraps(function)
        def call(self,*args,**kwargs):
            global _installed
            if not _installed:
                from vllm.v1.worker.gpu.phase_observation import set_phase_observer
                set_phase_observer(observer_factory);_installed=True
            if kind=='execute':
                bid=batch_id(args[0] if args else kwargs['scheduler_output'])
                self._compute_probe_last_bid=bid
            else:bid=getattr(self,'_compute_probe_last_bid',None)
            token=_context.set(bid)
            try:
                with span('batch_'+kind,bid,batch_id=bid):
                    return function(self,*args,**kwargs)
            finally:_context.reset(token)
        return call
    return decorate


def worker(function):
    @wraps(function)
    def call(self,*args,**kwargs):
        output=args[0] if args else kwargs['scheduler_output']
        bid=batch_id(output)
        emit('compute_worker_received',batch_id=bid,context_id=_context.get(),
             tokens=dict(output.num_scheduled_tokens),dummy=bool(kwargs.get('dummy_run',False)))
        return function(self,*args,**kwargs)
    return call


def current_batch():return _context.get()


def terminal():
    from vllm.v1.worker.gpu.phase_observation import phase_observation_errors
    emit('compute_observer_terminal',callback_errors=phase_observation_errors())


def adopted(function):
    @wraps(function)
    def call(self,output,model_output,*args,**kwargs):
        bid=batch_id(output)
        emit('compute_result_received',batch_id=bid,request_ids=list(model_output.req_ids),
             sampled_counts=[len(ids) for ids in (model_output.sampled_token_ids or [])])
        result=function(self,output,model_output,*args,**kwargs)
        emitted=[]
        for bundle in result.values():
            for r in bundle.outputs:
                emitted.append(dict(id=r.request_id,tokens=list(r.new_token_ids),
                    finish_reason=str(r.finish_reason) if r.finish_reason is not None else None))
        emit('compute_result_adopted',batch_id=bid,outputs=emitted,
             requests=[state(r) for r in self.requests.values()])
        return result
    return call
