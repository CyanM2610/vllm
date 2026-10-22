# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.compute_probe import scheduled as compute_scheduled, adopted as compute_adopted, mark as compute_mark
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.kv_wait_trace import emit as wait_emit, once as wait_once
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.retirement_observation import notify_retirement, retirement_call
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.prepared_prefix_match import PrefixMatch
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.model_uses_mrope = vllm_config.model_config.uses_mrope
        self.model_uses_xdrope = vllm_config.model_config.uses_xdrope
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.spec_decode_metrics_level = (
            self.observability_config.per_request_spec_decode_metrics
        )
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self.is_mm_encoder_only = vllm_config.is_mm_encoder_only

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
        if manager_cls_obj is None:
            manager_cls_obj = (
                EncoderDecoderCacheManager
                if self.is_encoder_decoder
                else EncoderCacheManager
            )
        self.encoder_cache_manager = manager_cls_obj.create_manager(
            cache_size=encoder_cache_size, vllm_config=vllm_config
        )
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
        # Positions past the computed tokens that the drafter reads mid-prefill.
        # Eagle-family drafters read 1 ahead, but multi-module MTP reads
        # num_spec_tokens ahead at chunked-prefill boundaries. Determines the
        # encoder scheduling shift, the deferred encoder free, the KV cache
        # manager's re-prefillable window (this minus 1), and how many tokens to
        # reserve between a chunk boundary and the prefill end.
        self.num_prefill_lookahead = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            self.use_eagle = speculative_config.use_eagle()
            if self.use_eagle:
                self.num_prefill_lookahead = (
                    self.num_spec_tokens
                    if speculative_config.use_multi_module_mtp()
                    else 1
                )

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            num_prefill_lookahead=self.num_prefill_lookahead,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        # Blocks that async KV loads will overwrite this step, skipped from
        # zeroing since the zeroing could race the out-of-band write.
        self._skip_zero_block_ids: set[int] = set()
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.mamba_has_prefill_checkpoint_blocks = (
            self.has_mamba_layers
            # TODO: support spec decoding
            and not self.use_eagle
            and all(
                not isinstance(group.kv_cache_spec, MambaSpec)
                or group.kv_cache_spec.num_prefill_checkpoint_blocks > 0
                for group in kv_cache_config.kv_cache_groups
            )
        )
        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )
        self.return_sampling_mask = vllm_config.model_config.return_sampling_mask

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()
        self._init_prefill_operations()

    def _init_prefill_operations(self) -> None:
        config = self.vllm_config.kv_transfer_config
        extra = (config.kv_connector_extra_config or {}) if config else {}
        enabled = extra.get("lmcache.mp.action_feedback", False)
        if type(enabled) is not bool:
            raise ValueError("lmcache.mp.action_feedback must be bool")
        self._action_feedback = None
        self._decoupled_control = bool(
            getattr(self.connector, "decoupled_control_enabled", False)
        )
        self._prefill_operations_enabled = enabled or self._decoupled_control or bool(
            getattr(self.connector, "decoupled_store_enabled", False)
        )
        if not self._prefill_operations_enabled:
            return
        from collections import OrderedDict

        from vllm.v1.core.sched.prefill_operations import NativeOperationDecider
        from vllm.v1.kv_cache_interface import (
            get_kv_cache_spec_kind,
            iter_layer_specs,
        )

        if (
            self.policy != SchedulingPolicy.FCFS
            or not self.scheduler_config.async_scheduling
            or self.scheduler_config.is_multimodal_model
            or self.vllm_config.speculative_config is not None
            or self.vllm_config.model_config.is_diffusion
            or not self.vllm_config.model_config.enforce_eager
            or not self.vllm_config.cache_config.enable_prefix_caching
            or self.lora_config is not None
            or self.ec_connector is not None
            or self.is_encoder_decoder
            or self.has_mamba_layers
            or not self.use_v2_model_runner
            or self.parallel_config.tensor_parallel_size != 1
            or self.parallel_config.pipeline_parallel_size != 1
            or self.parallel_config.data_parallel_size != 1
            or self.parallel_config.decode_context_parallel_size != 1
            or self.parallel_config.prefill_context_parallel_size != 1
            or not self.scheduler_reserve_full_isl
            or len(self.kv_cache_config.kv_cache_groups) != 1
        ):
            raise ValueError("decoupled control supports single-instance V2 text P-only")
        spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        layer_specs = iter_layer_specs(spec)
        if not layer_specs or any(
            get_kv_cache_spec_kind(layer).value != "full_attention"
            or layer.sliding_window is not None
            or layer.attention_chunk_size is not None
            or layer.non_causal
            for layer in layer_specs
        ):
            raise ValueError("decoupled control requires one full-attention KV group")
        self._operation_limit = self.connector.operation_step_limit
        if type(self._operation_limit) is not int or self._operation_limit <= 0:
            raise ValueError("operation_step_limit must be a positive integer")
        self._operation_requests = OrderedDict()
        self._step_operations = []
        self._operation_candidates = ()
        self._operation_budget = self.max_num_scheduled_tokens
        self._operation_input_budget = self.scheduler_config.max_num_batched_tokens
        self._operation_decider = NativeOperationDecider()
        if enabled:
            from vllm.v1.core.sched.prefill_operations import ActionFeedback
            from vllm.v1.core.sched.global_sjf_input import frozen_config_scope
            from vllm.v1.core.sched.compute_cost import ComputeCostModel

            _, errors = frozen_config_scope(self.vllm_config, self.block_size)
            if (errors or self.max_num_scheduled_tokens != 2048
                    or getattr(self, "_global_sjf_input_enabled", False)
                    or getattr(self, "_conditional_end_enabled", False)):
                raise ValueError("action feedback requires frozen P-only 2048 scope: " + repr(errors))
            model = self.vllm_config.model_config
            ComputeCostModel.from_config(getattr(model, "hf_text_config", None) or model.hf_config)
            self._action_feedback = ActionFeedback(self._operation_limit, self._emit_action_feedback)
            self._action_attempts = {}
            self._action_compute = {}
            self._action_loads = {}
            self._action_deferred_frees = {}
            self._action_native_receipts = OrderedDict()
            self._action_legacy_decider = False
            self.connector.set_operation_feedback_observer(self._connector_action_feedback)
        if self.connector.decoupled_store_enabled:
            self.connector.set_operation_decider(self._decide_store_operation)

    def _check_action_entry(self):
        feedback = getattr(self, "_action_feedback", None)
        if feedback is not None:
            feedback.check_entry()

    def set_prefill_action_policy(self, policy=None):
        """Trusted synchronous policy; immutable views, no native ownership handles.

        decide(step, candidate) returns OperationAction. Optional on_feedback
        observes native transitions and may only update policy-owned state.
        None restores the native fast path without constructing snapshots.
        """
        self._check_action_entry()
        if self._action_feedback is None:
            raise RuntimeError("lmcache.mp.action_feedback is disabled")
        if policy is not None and not callable(getattr(policy, "decide", None)):
            raise TypeError("action policy must implement decide")
        self._action_feedback.policy = policy
        self._action_legacy_decider = False

    def get_prefill_action_feedback(self):
        """Bounded scalar history and CPU samples; no polling or ownership handles."""
        if self._action_feedback is None:
            raise RuntimeError("lmcache.mp.action_feedback is disabled")
        return tuple(self._action_feedback.history)

    def _emit_action_feedback(self, feedback):
        from dataclasses import asdict
        fields = asdict(feedback)
        request_id = fields.pop("request_id")
        fields["result"] = feedback.result.value
        wait_emit("prefill_operation_feedback", request_id, **fields)

    def _action_attempt(self, request, kind, start=None, end=None, *, renew=False):
        feedback = self._action_feedback
        evidence = feedback.attempt(request.request_id, kind, start, end, renew=renew)
        self._action_attempts[(request.request_id, kind)] = evidence
        return evidence

    def _action_blocked(self, request, kind, reason):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        evidence = self._action_attempts.get((request.request_id, kind))
        if evidence is None:
            evidence = self._action_attempt(request, kind)
        self._action_feedback.report(evidence, ResultKind.BLOCKED, reason=reason)

    def _action_selected(self, request, p, quota, branch, grant=None):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        evidence = self._action_attempts.get((request.request_id, "compute"))
        if evidence is None:
            evidence = self._action_attempt(request, "compute", p, p + quota)
        identity = f"{evidence.operation_id}:compute:{self.current_step}"
        def commit(feedback):
            if grant is not None:
                self._compute_budget_commit(grant, branch)
        selected = self._action_feedback.report(
            evidence, ResultKind.SELECTED, commitment_id=identity,
            start_token=p, end_token=p + quota, granted_tokens=quota,
            native_phase=branch, effect=commit,
        )
        if selected is not None:
            self._action_compute[(self.current_step, request.request_id)] = selected

    def _action_revoke(self, request_id):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        evidence = self._action_compute.pop((self.current_step, request_id), None)
        if evidence is not None:
            def revoke(feedback):
                if getattr(self, "_compute_budget", None) is not None:
                    self._compute_budget_revoke(request_id)
            self._action_feedback.report(
                evidence, ResultKind.REVOKED, reason="native_same_step_rollback", effect=revoke)

    def _action_load_reserved(self, request, start, end):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        evidence = self._action_attempts.get((request.request_id, "load"))
        if evidence is None:
            evidence = self._action_attempt(request, "load", start, end)
        identity = f"{evidence.operation_id}:load"
        acquired = self._action_feedback.report(
            evidence, ResultKind.RESOURCES_ACQUIRED, commitment_id=identity,
            start_token=start, end_token=end, native_phase="target_reserved",
        )
        if acquired is not None:
            self._action_loads[request.request_id] = acquired

    def _action_load_transition(self, request_id, result, **fields):
        if self._action_feedback is None:
            return
        evidence = self._action_loads.get(request_id)
        if evidence is not None:
            updated = self._action_feedback.report(evidence, result, **fields)
            if updated is not None:
                self._action_loads[request_id] = updated

    def _connector_action_feedback(self, view, result, **fields):
        """Called only at native connector transitions, never polls or matches."""
        from vllm.v1.core.sched.prefill_operations import ResultKind
        feedback = self._action_feedback
        if feedback is None:
            return
        kind, rid, identity = view["kind"], view["request_id"], view["operation_id"]
        if kind == "load":
            fields.update(transfer_start_token=view.get("start_token"),
                          transfer_end_token=view.get("end_token"),
                          skip_first_n_tokens=view.get("skip_first_n_tokens"))
            self._action_load_transition(rid, ResultKind(result), **fields)
            return
        evidence = feedback.find(rid, kind, commitment_id=identity)
        if evidence is None:
            evidence = feedback.attempt(
                rid, kind, view.get("start_token"), view.get("end_token"), operation_id=identity)
        feedback.report(evidence, ResultKind(result), commitment_id=identity, **fields)

    def _action_request_pending(self, request, reason):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        if self._action_feedback is not None:
            evidence = self._action_feedback.find(request.request_id, "lifetime")
            if evidence is None:
                evidence = self._action_attempt(request, "lifetime", 0, request.num_prompt_tokens)
            self._action_feedback.report(
                evidence, ResultKind.CANCEL_PENDING,
                commitment_id=evidence.commitment_id or f"{evidence.operation_id}:release",
                reason=reason, native_phase="native_delayed_release",
            )

    def _action_request_released(self, request_id):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        feedback = self._action_feedback
        if feedback is None:
            return
        evidence = feedback.find(request_id, "lifetime")
        if evidence is None:
            evidence = feedback.attempt(request_id, "lifetime")
        feedback.report(evidence, ResultKind.ENDED,
                        commitment_id=evidence.commitment_id or f"{evidence.operation_id}:release",
                        native_phase="native_hbm_blocks_released",
                        resource_obligation="HBM released; lower_source_cleanup_is_separate")
        load = self._action_loads.pop(request_id, None)
        if load is not None:
            feedback.report(load, ResultKind.ENDED, native_phase="native_request_release")

    def _action_stores_finished(self, request_id):
        from vllm.v1.core.sched.prefill_operations import ResultKind
        feedback = self._action_feedback
        # Native finished_sending is the aggregate fence for immediate STORE.
        # Retained STOREs normally retired through their exact worker receipts.
        for identity in tuple(feedback.by_request.get((request_id, "store"), ())):
            evidence = feedback.find(request_id, "store", commitment_id=identity)
            for result in (ResultKind.COMPLETION, ResultKind.ADOPTED, ResultKind.ENDED):
                feedback.report(evidence, result, native_phase="native_finished_sending_aggregate")

    def _action_duplicate_native_receipt(self, direction, request_id):
        """Deduplicate known native receipts; unknown identities stay native errors.

        Native finished sets carry request IDs, not DMA epochs. This does not
        extend their protocol to arbitrary stale replies after ID reuse.
        """
        key = (direction, request_id)
        load = self._action_loads.get(request_id) if direction == "recv" else None
        request = self.requests.get(request_id)
        identity = load.commitment_id if load else (id(request) if request is not None else None)
        previous = self._action_native_receipts.get(key)
        if previous is not None and (identity is None or previous == identity):
            return True, identity
        return False, identity

    def _action_record_native_receipt(self, direction, request_id, identity):
        # Commit deduplication only after native adoption/release succeeds.
        # An exception must not make a later delivery silently skip cleanup.
        if identity is not None:
            self._action_native_receipts[(direction, request_id)] = identity
            while len(self._action_native_receipts) > self._operation_limit:
                self._action_native_receipts.popitem(last=False)

    def _register_operation_request(self, request: Request) -> None:
        if getattr(self, "_decoupled_control", False):
            self._operation_requests.setdefault(request.request_id, request)

    def _retire_operation_request(self, request: Request) -> None:
        if getattr(self, "_decoupled_control", False):
            self._operation_requests.pop(request.request_id, None)

    def get_prefill_operation_step_view(self):
        """Return observations only; no matching, allocation or ownership changes."""
        from vllm.v1.core.sched.prefill_operations import OperationView, StepOperationView

        if not getattr(self, "_prefill_operations_enabled", False):
            raise RuntimeError("decoupled control is disabled")
        connector_views = []
        for operation in self.connector.get_operation_snapshot():
            dependencies = operation.get("dependencies", ())
            if isinstance(dependencies, dict):
                dependencies = dependencies.items()
            connector_views.append(OperationView(
                operation["operation_id"], operation["request_id"],
                operation["kind"], operation.get("start_token"),
                operation.get("end_token"), operation["stage"],
                tuple(dependencies), operation.get("blocked_reason",
                                                  operation.get("block_reason")),
                owner=operation.get("owner", "native_connector"),
                native_phase=operation.get("native_phase"),
                source_ownership=operation.get("source_ownership"),
                physical_ready=operation.get("source_readable", "unknown"),
                resources=tuple(operation.get("resources", ())),
            ))
        return StepOperationView(
            self.current_step,
            self._operation_budget,
            self._operation_input_budget,
            self.kv_cache_manager.block_pool.get_num_free_blocks(),
            self._inflight_prefill_reserved_blocks(),
            tuple(self._step_operations[-self._operation_limit:])
            if self._action_feedback is not None and not self._action_legacy_decider
            else tuple(self._step_operations),
            tuple(connector_views),
        )

    def set_prefill_operation_decider(self, decider) -> None:
        self._check_action_entry()
        if not getattr(self, "_prefill_operations_enabled", False):
            raise RuntimeError("decoupled control is disabled")
        if not callable(getattr(decider, "decide", None)):
            raise TypeError("operation decider must provide decide(step, candidate)")
        self._operation_decider = decider
        if self._action_feedback is not None:
            self._action_legacy_decider = True
            self._action_feedback.policy = None

    def _decide_store_operation(self, candidate):
        from vllm.v1.core.sched.prefill_operations import OperationDecision, OperationView

        view = OperationView(
            candidate["operation_id"], candidate["request_id"], "store",
            candidate["start_token"], candidate["end_token"], candidate["stage"],
            tuple(candidate.get("dependencies", ())),
            candidate.get("blocked_reason", candidate.get("block_reason")),
            owner=candidate.get("owner", "native_connector"),
            native_phase=candidate.get("native_phase"),
            source_ownership=candidate.get("source_ownership"),
            physical_ready=candidate.get("source_readable", "unknown"),
            resources=tuple(candidate.get("resources", ())),
        )
        self._step_operations.append(view)
        if self._action_feedback is not None and not self._action_legacy_decider:
            del self._step_operations[:-self._operation_limit]
        if self._action_feedback is not None and not self._action_legacy_decider:
            from vllm.v1.core.sched.prefill_operations import ActionKind, ResultKind
            action = self._action_feedback.decide(view, self.get_prefill_operation_step_view)
            if action.kind == ActionKind.KEEP_RETAINED_STORE:
                evidence = self._action_feedback.find(view.request_id, "store", commitment_id=view.operation_id)
                if evidence is not None:
                    # A per-step choice, not another acquisition of source pages.
                    attempt = self._action_feedback.attempt(view.request_id, "store_choice",
                                                           view.start_token, view.end_token)
                    self._action_feedback.report(attempt, ResultKind.NOT_ATTEMPTED,
                                                 reason="keep_retained_store")
            return "defer" if action.kind in (ActionKind.RETAIN_STORE, ActionKind.KEEP_RETAINED_STORE) else "submit"
        decision = self._operation_decider.decide(
            self.get_prefill_operation_step_view(), view
        )
        if (not isinstance(decision, OperationDecision)
                or decision.action not in ("attempt", "defer")
                or decision.compute_tokens is not None):
            raise ValueError("STORE decision must be attempt/defer without quota")
        return "submit" if decision.action == "attempt" else "defer"

    def _progress_retained_store_operations(self):
        if not self.connector.decoupled_store_enabled:
            return
        candidates = self.connector.get_retained_store_candidates(self._operation_limit)
        for candidate in candidates:
            if self._decide_store_operation(candidate) == "submit":
                self.connector.request_store_submission([candidate["operation_id"]])

    def _observe_operation(self, request, kind, stage, start=0, end=None,
                           dependencies=(), reason=None, proposed=None, granted=None):
        from vllm.v1.core.sched.prefill_operations import OperationView

        if end is None:
            end = request.num_prompt_tokens
        identity = f"{request.request_id}:{kind}:{request.num_preemptions}"
        if kind == "compute":
            identity += f":{self.current_step}:{start}:{end}"
        elif kind == "lookup":
            state = self.connector.lookup_operation_state(request.request_id)
            identity = state["operation_id"] or identity
        view = OperationView(
            identity,
            request.request_id, kind, start, end, stage, tuple(dependencies),
            reason, proposed, granted,
            request_status=request.status.name,
            input_tokens=request.num_prompt_tokens,
            accounted_computed_tokens=request.num_computed_tokens,
            in_flight_tokens=request.num_in_flight_tokens,
        )
        self._step_operations.append(view)
        if self._action_feedback is not None and not self._action_legacy_decider:
            del self._step_operations[:-self._operation_limit]
        wait_emit(
            "prefill_operation", request.request_id, step=self.current_step,
            operation_id=view.operation_id, operation=kind, stage=stage,
            start_token=start, end_token=end, dependencies=dict(dependencies),
            reason=reason, proposed_tokens=proposed, granted_tokens=granted,
        )
        return view

    def _decide_operation(self, request, kind, start, end, token_budget,
                          input_budget, proposed=None):
        from vllm.v1.core.sched.prefill_operations import OperationDecision

        self._operation_budget = token_budget
        self._operation_input_budget = input_budget
        view = self._observe_operation(
            request, kind, "dependencies_checked", start, end,
            dependencies=(("scheduler_prerequisites", "confirmed"),
                          ("capacity_acquired", "unknown"),
                          ("physical_inputs_ready", "unknown")), proposed=proposed,
        )
        if self._action_feedback is not None and not self._action_legacy_decider:
            from vllm.v1.core.sched.prefill_operations import ActionKind, ResultKind
            evidence = self._action_attempt(request, kind, start, end, renew=True)
            action = self._action_feedback.decide(view, self.get_prefill_operation_step_view)
            if action.kind != ActionKind.ATTEMPT:
                self._action_feedback.report(evidence, ResultKind.NOT_ATTEMPTED,
                                             reason=action.kind.value)
            return action
        decision = self._operation_decider.decide(
            self.get_prefill_operation_step_view(), view
        )
        if not isinstance(decision, OperationDecision):
            raise TypeError("operation decider must return OperationDecision")
        if decision.action not in ("attempt", "defer"):
            raise ValueError("unknown operation decision")
        quota = decision.compute_tokens
        if quota is not None and (
            kind != "compute" or type(quota) is not int
            or not 0 < quota <= proposed
        ):
            raise ValueError("compute quota must be within the native proposal")
        if decision.action == "defer":
            self._observe_operation(request, kind, "discovered", start, end,
                                    reason="decision_deferred")
            return None
        if kind != "compute":
            return True
        return proposed if quota is None else quota

    def _begin_prefill_operation_step(self) -> None:
        if getattr(self, "_global_sjf_input_enabled", False) or getattr(
            self, "_conditional_end_enabled", False
        ):
            raise ValueError("decoupled control requires native request ordering")
        self._step_operations = []
        if self._action_feedback is not None:
            self._action_feedback.begin(self.current_step)
            self._action_attempts.clear()
        self._operation_budget = self.max_num_scheduled_tokens
        self._operation_input_budget = self.scheduler_config.max_num_batched_tokens
        if not self._decoupled_control:
            return
        self.connector.begin_prefill_control_step(self.current_step)
        candidates = []
        for _ in range(min(self._operation_limit, len(self._operation_requests))):
            rid, request = self._operation_requests.popitem(last=False)
            if self.requests.get(rid) is not request or request.is_finished():
                continue
            if request.status == RequestStatus.RUNNING:
                continue
            self._operation_requests[rid] = request
            candidates.append(request)
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                finished = rid in self.finished_recving_kv_req_ids
                self._observe_operation(
                    request, "load",
                    "completed_observed" if finished else "committed_metadata",
                    dependencies=(("worker_completion_received",
                                   "confirmed" if finished else "pending"),),
                )
                if finished and self._try_promote_blocked_waiting_request(request):
                    self._observe_operation(request, "load", "published")
            if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                lookup = self.connector.lookup_operation_state(rid)
                if request.num_computed_tokens or request.num_preemptions:
                    kind = "compute"
                elif lookup["result_known"]:
                    kind = "load" if lookup["result_tokens"] > 0 else "compute"
                else:
                    kind = "lookup"
                self._observe_operation(
                    request, kind, "discovered",
                    dependencies=(("lookup_result", "confirmed" if
                                   lookup["result_known"] else "unknown"),
                                  ("hbm_match", "unknown"),
                                  ("capacity_acquired", "unknown"),
                                  ("physical_inputs_ready", "unknown")),
                )
        self._operation_candidates = tuple(candidates)

    def _prepare_new_lookup_operations(self) -> None:
        if not self.connector.prepare_lookup_operations:
            return
        for request in sorted(self._operation_candidates,
                              key=lambda r: r.arrival_time):
            if (request.status != RequestStatus.WAITING
                    or request.num_computed_tokens != 0):
                continue
            state = self.connector.lookup_operation_state(request.request_id)
            if state["submitted"] or state["result_known"]:
                continue
            self._observe_operation(request, "lookup", "discovered")
            decision = self._decide_operation(
                request, "lookup", 0, request.num_prompt_tokens,
                self._operation_budget, self._operation_input_budget,
            )
            from vllm.v1.core.sched.prefill_operations import OperationAction, ActionKind
            if decision is None or (isinstance(decision, OperationAction)
                                    and decision.kind != ActionKind.ATTEMPT):
                continue
            self.connector.prepare_lookup_operation(request)
            state = self.connector.lookup_operation_state(request.request_id)
            self._observe_operation(
                request, "lookup", "submitted" if state["submitted"]
                else "discovered",
            )

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens

        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)

        end = start + num_new_tokens
        use_internal_checkpoint = (
            self.mamba_has_prefill_checkpoint_blocks and start % block_size == 0
        )
        if use_internal_checkpoint:
            last_cache_position = 0
        # Invariant: slot p holds the state after exactly (p + 1) * block_size
        # tokens. State is written at chunk ends, so chunk ends must be block
        # aligned. Exempt: the prompt's last chunk, whose slot decode advances
        # to the boundary. A block too wide for one chunk advances sub-block
        # and re-aligns at the next boundary.
        if end < prefill_end and not use_internal_checkpoint:
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
        stops = (
            # Same invariant: a chunk starting mid-block stops at the boundary
            # rather than running past it.
            next_block_boundary if start % block_size != 0 else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction, block-floored (a sub-block
            # junction's state is not separately cacheable): cache its state
            # so sibling requests sharing the prefix can reuse it.
            start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end
            else 0,
        )
        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    def _timed_local_prefix_cache_hit(self, request, phase, reconcile=False):
        started = time.monotonic_ns()
        result = (
            self.kv_cache_manager.get_computed_blocks(request) if reconcile
            else self._get_local_prefix_cache_hit(request)
        )
        elapsed = time.monotonic_ns() - started
        wait_emit(
            "prefill_local_prefix_match", request.request_id,
            phase=phase, elapsed_ns=elapsed, local_tokens=result[1],
            call_kind="divergent_reconciliation" if reconcile else "prefix",
            batch_id=f"{os.getpid()}:{self.current_step}",
            input_tokens=request.num_prompt_tokens,
            semantics="native local match call including its pre-existing nested observation costs; this timing event emission excluded; not uninstrumented pure lookup time",
        )
        return result

    def _match_waiting_prefix(self, request, phase="native", existing_lookup_only=False):
        # Extracted from the frozen WAITING path. Blocks are ephemeral and are
        # adopted only by allocate_slots in the same synchronous admission.
        num_external_computed_tokens = 0
        load_kv_async = False
        connector_prefix_cache_queries = 0
        (
            new_computed_blocks,
            num_new_local_computed_tokens,
            shared_prefix_boundary,
            hit_diverged,
        ) = self._timed_local_prefix_cache_hit(request, phase)
        if phase == "native":
            request.shared_prefix_boundary = shared_prefix_boundary
        wait_emit(
            "hbm_match",
            request.request_id,
            hit_tokens=num_new_local_computed_tokens, phase=phase,
        )
        # Get externally-cached tokens if using a KVConnector.
        if self.connector is not None:
            # Present a block-aligned local hit to the connector so
            # a strictly longer remote hit can supersede a local
            # sub-block tail without racing its copy-on-write.
            partial_tail = num_new_local_computed_tokens % self.block_size
            block_aligned_local = (
                num_new_local_computed_tokens - partial_tail
            )
            method = self.connector.get_num_new_matched_tokens
            if phase == "prepare":
                method = self.connector.observe_num_new_matched_tokens
            elif existing_lookup_only:
                method = self.connector.get_num_new_matched_tokens_from_existing_lookup
            lookup_started = time.monotonic_ns()
            ext_tokens, load_kv_async = method(request, block_aligned_local)
            lookup_elapsed = time.monotonic_ns() - lookup_started
            counters = getattr(self, "_prefix_match_attempt_counts", None)
            if counters is None:
                counters = self._prefix_match_attempt_counts = {}
            counter_key = (request.request_id, phase)
            counters[counter_key] = counters.get(counter_key, 0) + 1
            wait_emit(
                "prefill_lower_match_attempt", request.request_id,
                phase=phase, elapsed_ns=lookup_elapsed,
                batch_id=f"{os.getpid()}:{self.current_step}",
                existing_lookup_only=existing_lookup_only,
                attempt_number=counters[counter_key],
                external_tokens=ext_tokens, async_load=load_kv_async,
                semantics="connector call/result observation; not necessarily a new remote LOOKUP",
            )
            wait_emit(
                "lower_match_poll",
                request.request_id,
                external_tokens=ext_tokens, phase=phase,
                async_load=load_kv_async,
            )
            if ext_tokens is None:
                return None
            if partial_tail and ext_tokens > partial_tail:
                # Remote strictly exceeds the full local hit: drop the
                # sub-block tail so no CoW is needed, and let the load
                # cover it. Trim the partial block out of the local
                # computed blocks so it is not adopted from the cache.
                new_computed_blocks = (
                    self.kv_cache_manager.truncate_computed_blocks(
                        new_computed_blocks, block_aligned_local
                    )
                )
                num_new_local_computed_tokens = block_aligned_local
                num_external_computed_tokens = ext_tokens
            elif partial_tail:
                # Remote does not exceed the full local hit: keep the
                # local sub-block tail and load nothing external.
                num_external_computed_tokens = 0
                # Nothing to load remotely -> not an async-load step;
                # clearing avoids the `load_kv_async` assert below.
                load_kv_async = False
            else:
                num_external_computed_tokens = ext_tokens
            if hit_diverged and num_external_computed_tokens == 0:
                # No external tokens back the deeper local hit, so its
                # resume boundary would have no valid Mamba state.
                # Reconcile to the boundary every group agrees on.
                (
                    new_computed_blocks,
                    num_new_local_computed_tokens,
                    shared_prefix_boundary,
                ) = self._timed_local_prefix_cache_hit(request, phase, reconcile=True)
                if phase == "native":
                    request.shared_prefix_boundary = shared_prefix_boundary
            connector_prefix_cache_queries = (
                request.num_tokens - num_new_local_computed_tokens
            )
            connector_prefix_cache_hits = num_external_computed_tokens
        # Total computed tokens (local + external).
        num_computed_tokens = (
            num_new_local_computed_tokens + num_external_computed_tokens
        )
        assert num_computed_tokens <= request.num_tokens
        return PrefixMatch(
            new_computed_blocks, num_new_local_computed_tokens,
            num_external_computed_tokens, num_computed_tokens, load_kv_async,
            shared_prefix_boundary, connector_prefix_cache_queries,
        )

    def _get_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        connector = self.connector
        if connector is not None and connector.supports_divergent_local_hybrid_hits:
            return self.kv_cache_manager.get_computed_blocks_for_connector(request)

        blocks, num_local, shared_prefix_boundary = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
        return blocks, num_local, shared_prefix_boundary, False

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Never end a prefill chunk within num_prefill_lookahead of the
        prefill end.

        At a chunked-prefill boundary, the multi-module MTP drafter consumes
        the next num_prefill_lookahead known prefill tokens as draft inputs. A
        boundary closer to the end than that would make it fall back to
        sampled drafts, permanently polluting the trailing modules' KV caches.
        Either finish the prefill or leave at least num_prefill_lookahead for
        the next chunk. No-op for eagle-family drafters (lookahead 1).
        """
        remaining = request.num_tokens - num_computed_tokens - num_new_tokens
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        return max(num_new_tokens, 0)

    @compute_scheduled
    @retirement_call("schedule")
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._check_action_entry()
        self.current_step += 1
        wait_emit("schedule_begin", step=self.current_step)
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False
        # Whether any scheduled request has a synchronous connector KV load.
        has_sync_kv_loads = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()
        decoupled_control = getattr(self, "_decoupled_control", False)
        operations_enabled = getattr(self, "_prefill_operations_enabled", False)
        if operations_enabled:
            control_wall, control_cpu = time.perf_counter_ns(), time.thread_time_ns()
            self._begin_prefill_operation_step()
            wait_emit("operation_control_timing", step=self.current_step, phase="progress_adopt",
                      wall_ns=time.perf_counter_ns()-control_wall,
                      thread_cpu_ns=time.thread_time_ns()-control_cpu)
        conditional_end = getattr(self, "_conditional_end_enabled", False)
        conditional_stop = False
        if conditional_end:
            self._conditional_begin()
        compute_budget_enabled = getattr(self, "_compute_budget", None) is not None
        compute_batch_end = False
        if compute_budget_enabled:
            self._compute_budget_begin()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # First, schedule the RUNNING requests.
        global_order = getattr(self, "_global_sjf_input_enabled", False)
        step_skipped_waiting = create_request_queue(self.policy)
        action_feedback = getattr(self, "_action_feedback", None) is not None
        policy_skipped = set()
        policy_held_queues = {}
        scheduled_loras: set[int] = set()
        req_index = 0
        ledger = None
        if global_order:
            from vllm.v1.core.sched.global_sjf_rollback import StepAllocationLedger
            ledger = StepAllocationLedger(self, emit=wait_emit)

        def mark(request, reason, **fields):
            compute_mark(self, request, reason, **fields)
            if action_feedback and reason in (
                "lookup_pending", "blocked_waiting_dependency", "stale_output_pending",
                "waiting_allocation_failed", "running_allocation_failed", "native_zero_quota",
                "output_limit_guard", "operation_max_num_seqs",
            ):
                kind = "load" if active_branch == "restore_control" or request.status == RequestStatus.WAITING_FOR_REMOTE_KVS else "compute"
                if (request.request_id, "load") in self._action_attempts:
                    kind = "load"
                self._action_blocked(request, kind, reason)
            if decoupled_control and reason in (
                "lookup_pending", "blocked_waiting_dependency",
                "stale_output_pending", "waiting_allocation_failed",
                "running_allocation_failed", "native_zero_quota",
                "output_limit_guard", "operation_max_num_seqs",
            ):
                self._observe_operation(
                    request, "load" if active_branch == "restore_control"
                    else "compute", "discovered", reason=reason,
                )
            if global_order:
                self._global_sjf_observe_result(
                    request, outcome="native_check", branch=active_branch,
                    reason=reason, remaining_budget=token_budget, **fields,
                )

        def undo_preempted(victim):
            nonlocal token_budget, input_budget, encoder_compute_budget
            # The native rule already popped the primary victim from self.running.
            # The ledger invalidates only this step's unexecuted tail, and includes
            # consumers whose adopted prefix would otherwise lose its producer.
            rollback_requests = ledger.rollback_closure(victim)
            self._skip_zero_block_ids.difference_update(ledger.last_rollback['restore_zeroing_skip_ids'])
            rollback_buckets = {item['request_id']: item['bucket'] for item in ledger.last_rollback['entries']}
            for request in rollback_requests:
                rid = request.request_id
                bucket = rollback_buckets.get(rid, 'running')
                restored = num_scheduled_tokens.pop(rid, 0)
                if compute_budget_enabled:
                    self._compute_budget_revoke(rid)
                token_budget += restored
                if restored:
                    input_budget += restored + draft_slots
                for selected in (scheduled_new_reqs, scheduled_resumed_reqs, scheduled_running_reqs):
                    if request in selected:
                        selected.remove(request)
                req_to_new_blocks.pop(rid, None)
                scheduled_spec_decode_tokens.pop(rid, None)
                preempted_encoder_inputs = scheduled_encoder_inputs.pop(rid, None)
                if preempted_encoder_inputs:
                    encoder_compute_budget += sum(request.get_num_encoder_embeds(i) for i in preempted_encoder_inputs)
                if request in self.running:
                    self.running.remove(request)
                if bucket == "async_load":
                    # This LOAD has only been reserved; no connector metadata/DMA was
                    # dispatched. It loses a local-prefix dependency of this step.
                    step_skipped_waiting.remove_request(request)
                    self._free_request_blocks(request)
                    self._inflight_prefills.discard(request)
                    request.status = RequestStatus.PREEMPTED
                    request.num_computed_tokens = 0
                    request.num_preemptions += 1
                    self.waiting.prepend_request(request)
                    self.reset_preempted_req_ids.add(rid)
                else:
                    self._preempt_request(request, scheduled_timestamp, drop_stale_output=self.requires_kv_delivery)
                preempted_reqs.append(request)
                self._global_sjf_observe_result(
                    request, outcome="same_step_rollback", branch=bucket,
                    reason="native_victim" if request is victim else "producer_rolled_back",
                    primary_victim=victim.request_id, revoked_quota=restored,
                    remaining_budget=token_budget,
                )

        def process_running(request):
            nonlocal req_index, token_budget, input_budget, encoder_compute_budget, prefill_scheduled, conditional_stop, compute_batch_end
            mark(request, "running_visited")
            if input_budget <= draft_slots:
                return False

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                mark(request, "output_limit_guard")
                req_index += 1
                return True

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                mark(request, "decode_cadence")
                req_index += 1
                return True

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                mark(request, "prefill_throttle")
                req_index += 1
                return True

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )

            # Multi-module MTP: avoid ending a prefill chunk within
            # num_prefill_lookahead of the prefill end.
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # 5. Insufficient budget to keep a multi-module MTP prefill
                #    chunk out of the prefill-lookahead window.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                mark(request, "native_zero_quota")
                req_index += 1
                return True

            if conditional_end and self._conditional_should_stop(
                request, p=request.num_computed_tokens, quota=num_new_tokens,
                branch="running", selected_tokens=num_scheduled_tokens,
            ):
                conditional_stop = True
                return False

            action_end_after_commit = False
            if decoupled_control or action_feedback:
                quota = self._decide_operation(
                    request, "compute", request.num_computed_tokens,
                    request.num_computed_tokens + num_new_tokens,
                    token_budget, input_budget, num_new_tokens,
                )
                if quota is None:
                    req_index += 1
                    return True
                from vllm.v1.core.sched.prefill_operations import OperationAction, ActionKind
                if isinstance(quota, OperationAction):
                    if quota.kind == ActionKind.END_COMPUTE_BATCH:
                        compute_batch_end = True
                        return False
                    if quota.kind == ActionKind.SKIP_CANDIDATE:
                        policy_skipped.add(request.request_id)
                        req_index += 1
                        return True
                    action_end_after_commit = quota.end_after_commit
                    num_new_tokens = quota.compute_tokens or num_new_tokens
                else:
                    num_new_tokens = quota

            # Schedule newly needed KV blocks for the request.
            native_compute_tokens = num_new_tokens
            compute_grant = None
            allocation_retried = False
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    if action_feedback and allocation_retried:
                        self._action_attempt(request, "compute", request.num_computed_tokens,
                                             request.num_computed_tokens + native_compute_tokens, renew=True)
                    allocation_retried = True
                    if compute_budget_enabled:
                        # Re-propose after a same-step rollback; its successful
                        # compute terms may have changed. Never probe allocation.
                        compute_grant = self._compute_budget_propose(
                            request, request.num_computed_tokens,
                            native_compute_tokens, "running",
                        )
                        if compute_grant.tokens == 0:
                            compute_batch_end = True
                            return False
                        num_new_tokens = compute_grant.tokens
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    mark(request, 'running_allocation_failed')
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        # Record the index of the preemption victim to
                        # maintain accurate loop state.
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        # Decrement the loop cursor if the removed request
                        # preceded the current iteration, preventing the
                        # silent omission of the subsequent request.
                        if victim_index < req_index:
                            req_index -= 1

                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            if compute_budget_enabled and not action_feedback:
                                self._compute_budget_revoke(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                    else:
                        preempted_req = self.running.pop()

                    if global_order:
                        undo_preempted(preempted_req)
                        if request.status != RequestStatus.RUNNING:
                            new_blocks = None
                            break
                        continue

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    if action_feedback:
                        self._action_revoke(preempted_req.request_id)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                return False

            if global_order:
                ledger.record(request, 'running', request.num_computed_tokens, num_new_tokens)
            mark(request, "running_allocated", quota=num_new_tokens)
            if decoupled_control:
                self._observe_operation(
                    request, "compute", "selected", request.num_computed_tokens,
                    request.num_computed_tokens + num_new_tokens,
                    granted=num_new_tokens,
                )
            if conditional_end:
                self._conditional_selected(request, request.num_computed_tokens, num_new_tokens, "running")
            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            if action_feedback:
                self._action_selected(request, request.num_computed_tokens, num_new_tokens, "running", compute_grant)
            elif compute_budget_enabled:
                self._compute_budget_commit(compute_grant, "running")
            compute_batch_end = action_end_after_commit or bool(compute_grant and compute_grant.end_after_commit)
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
            return not compute_batch_end

        def process_waiting(request_queue, restore_only=False):
            nonlocal token_budget, input_budget, encoder_compute_budget, has_sync_kv_loads, conditional_stop, compute_batch_end
            if not restore_only and input_budget <= draft_slots:
                return False
            # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
            # in `running` but still hold a model-runner request slot.
            num_running = len(self.running) + self.num_waiting_for_streaming_input
            if num_running >= self.max_num_running_reqs:
                if decoupled_control or action_feedback:
                    mark(request_queue.peek_request(), "operation_max_num_seqs")
                return False
            request = request_queue.peek_request()
            request_id = request.request_id
            def skip_policy_candidate():
                policy_skipped.add(request_id)
                if not restore_only:
                    request_queue.pop_request()
                    entry = policy_held_queues.setdefault(id(request_queue), (request_queue, []))
                    entry[1].append(request)

            if action_feedback and (request_id in policy_skipped
                                    or request_id in self._action_feedback.skipped_requests):
                skip_policy_candidate()
                return True
            def defer_waiting_request():
                if not restore_only:
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)

            mark(request, "waiting_visited")
            # try to promote blocked statuses while traversing skipped queue.
            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                mark(request, 'blocked_waiting_dependency', status=request.status.name)
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request_id,
                    )
                defer_waiting_request()
                return True
            if (
                request.num_stale_output_tokens > 0
                and not request.drop_stale_output
            ):
                mark(request, 'stale_output_pending')
                # Deliverable stale output still in flight: resuming now
                # could resample a position that output later delivers.
                # It drains within the pipeline depth.
                defer_waiting_request()
                return True
            # Check that adding the request still respects the max_loras
            # constraint.
            if (
                self.lora_config
                and request.lora_request
                and (
                    len(scheduled_loras) == self.lora_config.max_loras
                    and request.lora_request.lora_int_id not in scheduled_loras
                )
            ):
                # Scheduling would exceed max_loras, skip.
                defer_waiting_request()
                return True
            num_external_computed_tokens = 0
            load_kv_async = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
            did_prefix_cache_lookup = False
            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                did_prefix_cache_lookup = True
                prepared = getattr(self, "_global_sjf_match_prepare", False)
                match = self._match_waiting_prefix(
                    request, phase="restore_control" if restore_only else (
                        "takeover" if prepared else "native"),
                    existing_lookup_only=restore_only or prepared or (conditional_end and self._conditional_existing_lookup(request)) or (
                        compute_budget_enabled
                        and request.request_id in self._compute_budget_retry_matches
                    ),
                )
                if match is None:
                    mark(request, "lookup_pending")
                    defer_waiting_request()
                    return True
                new_computed_blocks = match.blocks
                num_new_local_computed_tokens = match.local_tokens
                num_external_computed_tokens = match.external_tokens
                num_computed_tokens = match.matched_tokens
                load_kv_async = match.load_kv_async
                request.shared_prefix_boundary = match.shared_prefix_boundary
                connector_prefix_cache_queries = match.connector_queries
                connector_prefix_cache_hits = match.external_tokens
                if prepared:
                    self._observe_takeover_match(request, match)
                # Skip request with pending mm encoding prefetches
                if (
                    self.ec_connector is not None
                    and request.mm_features
                    and not self.ec_connector.ensure_cache_available(
                        request, num_computed_tokens
                    )
                ):
                    defer_waiting_request()
                    return True
                # Track first scheduled prefill, not post-preemption repeat prefills
                if request.prefill_stats and request.num_preemptions <= 0:
                    assert num_computed_tokens <= request.num_prompt_tokens
                    request.prefill_stats.set(
                        num_prompt_tokens=request.num_prompt_tokens,
                        num_local_cached_tokens=num_new_local_computed_tokens,
                        num_external_cached_tokens=num_external_computed_tokens,
                    )
            else:
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens
            mark(request, "matched", matched=num_computed_tokens, async_load=load_kv_async, external=num_external_computed_tokens)
            if restore_only and not load_kv_async:
                self._observe_operation(
                    request, "load", "dependencies_checked",
                    num_new_local_computed_tokens, num_computed_tokens,
                    reason="current_hbm_needs_no_restore",
                )
                return True
            encoder_inputs_to_schedule = None
            external_load_encoder_input = []
            new_encoder_compute_budget = encoder_compute_budget
            pad_spec_decode = False
            if load_kv_async:
                mark(request, "restore_pending", matched=num_computed_tokens)
                # KVTransfer: loading remote KV, do not allocate for new work.
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                # DP prefill balancing: defer this step's local prefill
                # compute to a cadence-aligned step.
                return False
            else:
                request_token_budget = min(token_budget, input_budget - draft_slots)
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                # Pad new decode requests to uniform spec decoding size to
                # preserve full cudagraph for this step.
                # Not for diffusion where draft tokens can't be padded.
                if (
                    (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                    and self.num_sampled_tokens_per_step > 0
                    and num_new_tokens == 1
                    and (scheduled_running_reqs and not prefill_scheduled)
                ):
                    padded_num_tokens = 1 + self.num_spec_tokens
                    # Pad only when there is room for the sampled token(s).
                    if (
                        num_computed_tokens
                        + padded_num_tokens
                        + self.num_sampled_tokens_per_step
                        <= self.max_model_len
                    ):
                        if padded_num_tokens > request_token_budget:
                            # Prefer to not schedule than schedule un-padded.
                            return False
                        num_new_tokens = padded_num_tokens
                        pad_spec_decode = True
                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold
                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > request_token_budget
                ):
                    # If chunked_prefill is disabled,
                    # we can stop the scheduling here.
                    return False
                num_new_tokens = min(num_new_tokens, request_token_budget)
                assert num_new_tokens > 0
                # Apply Mamba alignment before encoder caps.
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        return False
                    if (
                        pad_spec_decode
                        and num_new_tokens != 1 + self.num_spec_tokens
                    ):
                        # Alignment clipped the placeholder rows. The split
                        # aligns prefill chunks, but the padded tail rows are
                        # speculative positions, not prefill tokens. A padded
                        # request must keep all 1 + num_spec rows or the
                        # sampler's row count stops matching its query rows,
                        # so drop the padding instead of shortening it.
                        num_new_tokens = 1
                        pad_spec_decode = False
                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_compute_budget,
                        external_load_encoder_input,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        encoder_compute_budget,
                        shift_computed_tokens=self.num_prefill_lookahead,
                    )
                # Multi-module MTP: avoid ending a prefill chunk within
                # num_prefill_lookahead of the prefill end.
                num_new_tokens = self._reserve_prefill_lookahead(
                    request, num_computed_tokens, num_new_tokens
                )
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    return False
            # During async KV load, no forward pass is run yet.
            # Allocate speculative lookahead slots later to avoid
            # mismatching local and remote block counts.
            limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
            effective_lookahead_tokens = (
                0 if limit_lookahead_tokens else self.num_lookahead_tokens
            )
            # Determine if we need to allocate cross-attention blocks.
            num_encoder_tokens = 0
            if (
                self.is_encoder_decoder
                and request.has_encoder_inputs
                and encoder_inputs_to_schedule
            ):
                num_encoder_tokens = sum(
                    request.get_num_encoder_embeds(i)
                    for i in encoder_inputs_to_schedule
                )
            reserved_blocks = 0
            if load_kv_async:
                # An async load holds its blocks for the whole transfer with
                # no forward progress and isn't preemptible here. Admit it
                # only if it fits in (free - other in-flight reservations), to
                # avoid deadlock and predictable preemptions.
                reserved_blocks = self._inflight_prefill_reserved_blocks()
            if conditional_end and not load_kv_async and self._conditional_should_stop(
                request, p=num_computed_tokens, quota=num_new_tokens,
                branch="waiting", selected_tokens=num_scheduled_tokens,
                local_tokens=num_new_local_computed_tokens,
                external_tokens=num_external_computed_tokens, blocks=new_computed_blocks,
                full_sequence=self.scheduler_reserve_full_isl,
                reserved_blocks=reserved_blocks, has_scheduled_reqs=bool(self.running),
            ):
                conditional_stop = True
                return False
            action_end_after_commit = False
            if decoupled_control or action_feedback:
                decision = self._decide_operation(
                    request, "load" if load_kv_async else "compute",
                    num_new_local_computed_tokens if load_kv_async else
                    num_computed_tokens,
                    num_computed_tokens if load_kv_async else
                    num_computed_tokens + num_new_tokens,
                    token_budget, input_budget,
                    None if load_kv_async else num_new_tokens,
                )
                if decision is None:
                    return False
                from vllm.v1.core.sched.prefill_operations import OperationAction, ActionKind
                if isinstance(decision, OperationAction):
                    if decision.kind == ActionKind.END_COMPUTE_BATCH:
                        compute_batch_end = True
                        return False
                    if decision.kind == ActionKind.SKIP_CANDIDATE:
                        skip_policy_candidate()
                        return True
                    action_end_after_commit = decision.end_after_commit
                    if not load_kv_async:
                        num_new_tokens = decision.compute_tokens or num_new_tokens
                elif not load_kv_async:
                    num_new_tokens = decision
            compute_grant = None
            if compute_budget_enabled and not load_kv_async:
                compute_grant = self._compute_budget_propose(
                    request, num_computed_tokens, num_new_tokens, "waiting"
                )
                if compute_grant.tokens == 0:
                    self._compute_budget_retry_matches.add(request_id)
                    compute_batch_end = True
                    return False
                num_new_tokens = compute_grant.tokens
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=load_kv_async,
                num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=self.scheduler_reserve_full_isl,
                reserved_blocks=reserved_blocks,
                has_scheduled_reqs=bool(self.running),
            )
            if new_blocks is None:
                mark(request, 'waiting_allocation_failed')
                if (compute_budget_enabled and not load_kv_async
                        and self._compute_budget.mode != "off"):
                    self._compute_budget_retry_matches.add(request_id)
                # The request cannot be scheduled.
                # NOTE: we need to untouch the request from the encode cache
                # manager
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                return False
            wait_emit(
                "target_reserved",
                request.request_id,
                external_tokens=num_external_computed_tokens,
                async_load=load_kv_async,
                block_ids=self.kv_cache_manager.get_blocks(
                    request_id
                ).get_block_ids(),
            )
            # KVTransfer: the connector uses this info to determine
            # if a load is needed. Note that
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    self.kv_cache_manager.get_blocks(request_id),
                    num_external_computed_tokens,
                )
                if (
                    self.connector_prefix_cache_stats is not None
                    and connector_prefix_cache_queries != 0
                ):
                    self.connector_prefix_cache_stats.record(
                        num_tokens=connector_prefix_cache_queries,
                        num_hits=connector_prefix_cache_hits,
                        preempted=request.num_preemptions > 0,
                    )
            if conditional_end:
                self._conditional_adopted(request)
            # Record at admission so unscheduled lookups are not counted.
            if did_prefix_cache_lookup:
                self.kv_cache_manager.record_prefix_cache_stats(
                    request, num_new_local_computed_tokens
                )
            if not restore_only:
                request = request_queue.pop_request()
            if load_kv_async:
                if compute_budget_enabled:
                    self._compute_budget_retry_matches.discard(request_id)
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                mark(request, 'restore_reserved', matched=num_computed_tokens)
                if not restore_only:
                    step_skipped_waiting.prepend_request(request)
                # Set num_computed_tokens even though KVs are not yet loaded.
                # request.num_computed_tokens will not be used anywhere until
                # the request finished the KV transfer.
                #
                # If a transfer error is reported by the connector,
                # request.num_computed_tokens will be re-set accordingly in
                # _update_requests_with_invalid_blocks.
                #
                # When the transfer is finished, either successfully or not,
                # request.num_computed_tokens will correctly reflect the number
                # of computed tokens.
                # _update_waiting_for_remote_kv will then cache
                # only the successfully loaded tokens.
                request.num_computed_tokens = num_computed_tokens
                if decoupled_control:
                    self._observe_operation(
                        request, "load", "resources_acquired",
                        num_new_local_computed_tokens, num_computed_tokens,
                        dependencies=(("hbm_target_reserved", "confirmed"),
                                      ("physical_completion", "pending")),
                    )
                if global_order:
                    ledger.record(request, 'async_load', num_computed_tokens, 0)
                self._inflight_prefills.add(request)
                if self.needs_kv_cache_zeroing:
                    # Skip zeroing of the blocks the async load will
                    # overwrite; the zeroing could race the write.
                    self._skip_zero_block_ids.update(
                        self.kv_cache_manager.get_zeroing_block_ids_in_range(
                            request.request_id,
                            num_new_local_computed_tokens,
                            num_computed_tokens,
                        )
                    )
                if action_feedback:
                    self._action_load_reserved(request, num_new_local_computed_tokens, num_computed_tokens)
                return True
            if global_order:
                ledger.record(request, 'new' if request.status == RequestStatus.WAITING else 'resumed', num_computed_tokens, num_new_tokens)
            self.running.append(request)
            self._retire_operation_request(request)
            if num_external_computed_tokens > 0:
                # load_kv_async is False here
                has_sync_kv_loads = True
            if self.log_stats:
                request.record_event(
                    EngineCoreEventType.SCHEDULED, scheduled_timestamp
                )
            if request.status == RequestStatus.WAITING or (
                global_order and request.request_id not in self._global_sjf_worker_initialized
            ):
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")
            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                request_id
            )
            mark(request, "waiting_allocated", quota=num_new_tokens, matched=num_computed_tokens)
            if decoupled_control:
                self._observe_operation(
                    request, "compute", "selected", num_computed_tokens,
                    num_computed_tokens + num_new_tokens,
                    granted=num_new_tokens,
                )
            if conditional_end:
                self._conditional_selected(request, num_computed_tokens, num_new_tokens, "waiting")
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            if compute_budget_enabled and not action_feedback:
                self._compute_budget_commit(compute_grant, "waiting")
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            if action_feedback:
                self._action_selected(request, num_computed_tokens, num_new_tokens, "waiting", compute_grant)
            compute_batch_end = action_end_after_commit or bool(compute_grant and compute_grant.end_after_commit)
            if pad_spec_decode:
                assert num_new_tokens == 1 + self.num_spec_tokens
                scheduled_spec_decode_tokens[request_id] = [
                    -1
                ] * self.num_spec_tokens
            # Only track requests that will still be prefilling after this chunk.
            if num_computed_tokens + num_new_tokens < request.num_tokens:
                self._inflight_prefills.add(request)
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            # Allocate for external load encoder cache
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
            return not compute_batch_end

        if not global_order:
            active_branch = "running"
            while req_index < len(self.running) and token_budget > 0:
                if not process_running(self.running[req_index]):
                    break
            # Record the LoRAs in scheduled_running_reqs
            if self.lora_config:
                scheduled_loras = set(
                    req.lora_request.lora_int_id
                    for req in scheduled_running_reqs
                    if req.lora_request and req.lora_request.lora_int_id > 0
                )
                assert len(scheduled_loras) <= self.lora_config.max_loras

            if (decoupled_control and not preempted_reqs
                    and self._pause_state == PauseState.UNPAUSED):
                self._operation_budget = token_budget
                self._operation_input_budget = input_budget
                control_wall, control_cpu = time.perf_counter_ns(), time.thread_time_ns()
                self._prepare_new_lookup_operations()

                class RestoreCandidateReference:
                    def __init__(self, request):
                        self.request = request

                    def peek_request(self):
                        return self.request

                active_branch = "restore_control"
                # This switch gates only extra admission after RUNNING.
                # Native WAITING matching/LOAD and start-of-step adoption
                # keep their original paths. Both modes use the same RR view.
                for request in (self._operation_candidates
                                if self.connector.prepare_load_operations else ()):
                    if (request.status not in (
                        RequestStatus.WAITING, RequestStatus.PREEMPTED
                    ) or request.num_computed_tokens != 0):
                        continue
                    lookup = self.connector.lookup_operation_state(request.request_id)
                    self._observe_operation(
                        request, "load", "discovered",
                        dependencies=(("lookup_result", "confirmed" if
                                       lookup["result_known"] else "unknown"),),
                    )
                    if (lookup["result_known"]
                            and lookup["result_tokens"] > 0):
                        wait_emit("extra_load_prepare_attempt", request.request_id,
                                  step=self.current_step, remaining_budget=token_budget,
                                  lookup_result_tokens=lookup["result_tokens"],
                                  status=request.status.name)
                        process_waiting(RestoreCandidateReference(request),
                                        restore_only=True)
                wait_emit("operation_control_timing", step=self.current_step, phase="extra_prepare",
                          extra_load_enabled=self.connector.prepare_load_operations,
                          wall_ns=time.perf_counter_ns()-control_wall,
                          thread_cpu_ns=time.thread_time_ns()-control_cpu)

            if not conditional_stop and not compute_batch_end and not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
                active_branch = "waiting"
                while (self.waiting or self.skipped_waiting) and token_budget > 0:
                    queue = self._select_waiting_queue_for_scheduling()
                    assert queue is not None
                    if not process_waiting(queue):
                        break
                if step_skipped_waiting:
                    self.skipped_waiting.prepend_requests(step_skipped_waiting)
                if not defer_prefills:
                    self.prefill_capacity_bound = bool(self.waiting)
        else:
            # A view removes only the visited WAITING request if its native branch
            # consumes it. The owning queue and self.running keep their native order.
            class QueueView:
                def __init__(self, queue, request):
                    self.queue, self.request = queue, request
                def peek_request(self):
                    return self.request
                def pop_request(self):
                    self.queue.remove_request(self.request)
                    return self.request

            initial_running = {id(r) for r in self.running}
            owners = {id(r): q for q in (self.waiting, self.skipped_waiting) for r in q}
            visited = set()
            waiting_closed = self._pause_state != PauseState.UNPAUSED
            for request in self._global_sjf_candidates():
                if compute_batch_end or token_budget <= 0 or input_budget <= draft_slots:
                    break
                if id(request) in visited:
                    raise AssertionError("candidate visited twice")
                visited.add(id(request))
                if request in preempted_reqs:
                    continue
                active_branch = "running" if id(request) in initial_running else "waiting"
                if active_branch == "running":
                    if request.status != RequestStatus.RUNNING or request not in self.running:
                        self._global_sjf_observe_result(request, outcome="not_checked", branch=active_branch, reason="ownership_changed")
                        continue
                    req_index = self.running.index(request)
                    process_running(request)
                elif waiting_closed or preempted_reqs:
                    self._global_sjf_observe_result(request, outcome="not_checked", branch=active_branch, reason="waiting_admission_closed")
                else:
                    if len(self.running) + self.num_waiting_for_streaming_input >= self.max_num_running_reqs:
                        waiting_closed = True
                        self._global_sjf_observe_result(request, outcome="blocked", branch=active_branch, reason="max_num_seqs", remaining_budget=token_budget)
                        continue
                    queue = owners[id(request)]
                    if not process_waiting(QueueView(queue, request)):
                        waiting_closed = True
                if request.request_id in num_scheduled_tokens:
                    self._global_sjf_observe_result(request, outcome="allocated", branch=active_branch, quota=num_scheduled_tokens[request.request_id], remaining_budget=token_budget)
                elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    self._global_sjf_observe_result(request, outcome="restore_pending", branch=active_branch, reason="native_remote_dependency", remaining_budget=token_budget)
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)
            if self._pause_state == PauseState.UNPAUSED and not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)


        # Check if the scheduling constraints are satisfied.
        for queue, held in policy_held_queues.values():
            queue.prepend_requests(reversed(held))
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        if compute_budget_enabled:
            self._compute_budget.assert_output(num_scheduled_tokens)
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    uses_mrope=self.model_uses_mrope,
                    uses_xdrope=self.model_uses_xdrope,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    uses_mrope=self.model_uses_mrope,
                    uses_xdrope=self.model_uses_xdrope,
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # Mamba "align" boundary states must be handed off with exact block ids;
        # they cannot be reconstructed from a connector's append-only block
        # table. Drained every step so stale offers cannot accumulate.
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()

        kv_connector_block_state = None
        if self.connector is not None:
            snapshot_req_ids = {req.req_id for req in new_reqs_data}
            snapshot_req_ids.update(
                req_id
                for req_id, block_ids in zip(
                    cached_reqs_data.req_ids,
                    cached_reqs_data.new_block_ids,
                    strict=True,
                )
                if block_ids
            )
            snapshot_req_ids.update(
                req_id for req_id in boundary_state_offloads if req_id in self.requests
            )
            kv_connector_block_state = KVConnectorBlockState(
                block_ids={
                    req_id: self.kv_cache_manager.get_block_ids(req_id)
                    for req_id in snapshot_req_ids
                },
                boundary_state_offloads=boundary_state_offloads,
            )

        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        scheduled_encoder_input_stats = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs
            )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=has_sync_kv_loads,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_connector_block_state=kv_connector_block_state,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
        )
        if action_feedback:
            # Local correlation only; does not change the worker protocol.
            scheduler_output._action_step_id = self.current_step

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            if operations_enabled:
                self._operation_budget = token_budget
                self._operation_input_budget = input_budget
                self._progress_retained_store_operations()
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        if decoupled_control:
            self._operation_budget = token_budget
            self._operation_input_budget = input_budget

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Connector-only block state must not be dispatched to workers.
        scheduler_output.kv_connector_block_state = None

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        if global_order:
            self._global_sjf_observe_step_end(scheduler_output)

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        wait_emit(
            "schedule_output",
            step=self.current_step,
            tokens=dict(num_scheduled_tokens),
            waiting=len(self.waiting),
            running=len(self.running),
            free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            preempted=list(scheduler_output.preempted_req_ids or ()),
        )
        for wait_rid, wait_tokens in num_scheduled_tokens.items():
            wait_once("first_schedule", wait_rid, tokens=wait_tokens)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None

        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        return new_block_ids_to_zero or None

    @retirement_call("preempt")
    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self._register_operation_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        encoder_window_end = (
            num_computed_tokens + num_new_tokens + shift_computed_tokens
        )
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=encoder_window_end,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def _make_scheduled_encoder_input_stats(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> ScheduledEncoderInputStats | None:
        stats = ScheduledEncoderInputStats()

        for req_id, input_ids in scheduled_encoder_inputs.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            for input_id in input_ids:
                mm_feature = request.mm_features[input_id]
                stats.num_inputs += 1
                stats.output_tokens += mm_feature.mm_position.get_num_embeds()

        return stats if stats.num_inputs else None

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    @retirement_call("consume")
    @compute_adopted
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        self._check_action_entry()
        action_step = getattr(scheduler_output, "_action_step_id", None)
        action_results = {}
        if action_step is not None:
            from vllm.v1.core.sched.prefill_operations import ResultKind
            for rid in scheduler_output.num_scheduled_tokens:
                evidence = self._action_compute.get((action_step, rid))
                if evidence is not None:
                    action_results[rid] = evidence
                    self._action_feedback.report(evidence, ResultKind.COMPLETION,
                                                 native_phase="model_runner_output_received")
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        ec_connector_output = model_runner_output.ec_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale and request.drop_stale_output:
                notify_retirement(
                    "stale_drop",
                    self,
                    request_id=req_id,
                    scheduled_tokens=num_tokens_scheduled,
                    model_runner_output=model_runner_output,
                )
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )
                if request.spec_decode_metrics is not None:
                    # Exclude grammar-invalidated drafts from the proposed
                    # count, mirroring make_spec_decoding_stats; the accepted
                    # bucket (j) is unaffected.
                    adj_draft_tokens = num_draft_tokens
                    if scheduler_output.num_invalid_spec_tokens:
                        adj_draft_tokens -= (
                            scheduler_output.num_invalid_spec_tokens.get(req_id, 0)
                        )
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=adj_draft_tokens,
                        num_accepted=num_accepted,
                        detailed=self.spec_decode_metrics_level == "detailed",
                    )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_sampling_mask = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            prefill_stats = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)
            if new_token_ids and num_output_tokens_before == 0:
                wait_once("first_token_core", request.request_id)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_mm_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if req_id in action_results:
                self._action_feedback.report(action_results[req_id], ResultKind.ADOPTED,
                                             native_phase="native_output_processed")
            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if self.return_sampling_mask:
                sampling_masks = model_runner_output.sampling_masks
                if new_token_ids and sampling_masks is not None:
                    new_sampling_mask = sampling_masks.slice_request(
                        req_index, len(new_token_ids)
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_sampling_mask=new_sampling_mask,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        spec_decode_metrics=(
                            request.spec_decode_metrics
                            if finish_reason is not None
                            else None
                        ),
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        error_req_ids = set(self.grammar_compile_error_reqs)
        self.grammar_compile_error_reqs.clear()
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            error_req_ids.update(failed_kv_load_req_ids)

        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # EC Connector: update state from worker-side EC connector output.
        if self.ec_connector is not None and ec_connector_output:
            self.ec_connector.update_connector_output(ec_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        for rid, evidence in action_results.items():
            self._action_feedback.report(evidence, ResultKind.ENDED,
                                         native_phase="native_output_processed")
            self._action_compute.pop((action_step, rid), None)
        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)
        self._register_operation_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's read-ahead has also passed it,
        # mirroring the shift the encoder scheduling path applies.
        spec_lookahead = self.num_prefill_lookahead

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return self.kv_cache_manager.usage

    def add_request(self, request: Request) -> None:
        self._check_action_entry()
        if getattr(self, "_prefill_operations_enabled", False) and (
            request.max_tokens != 1 or request.resumable or request.has_encoder_inputs
        ):
            raise ValueError("decoupled control accepts text P-only max_tokens=1")
        wait_emit(
            "input_known",
            request.request_id,
            prompt_tokens=request.num_prompt_tokens,
            logical_id=(request.sampling_params.extra_args or {}).get(
                "native_token_replay_id"
            )
            if request.sampling_params
            else None,
            block_hashes=[str(h) for h in request.block_hashes],
        )
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.spec_decode_metrics_level != "none":
                request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                    self.num_spec_tokens
                )
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        self._check_action_entry()
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        self._retire_operation_request(request)
        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if getattr(self, "_action_feedback", None) is not None and delay_free_blocks:
            self._action_request_pending(request, "native_connector_or_load_obligation")
        from vllm.v1 import eviction_probe
        eviction_probe.request_hold(self, request, delay_free_blocks, connector_delay_free_blocks)
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]
        if getattr(self, "_action_feedback", None) is not None:
            if request.request_id not in self._action_deferred_frees.values():
                self._action_request_released(request.request_id)

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        logger.info("setting pause state to %s", pause_state.name)
        self._pause_state = pause_state

    @retirement_call("free_request")
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))
            if getattr(self, "_action_feedback", None) is not None and request.is_finished():
                self._action_deferred_frees[id(blocks)] = request.request_id
                self._action_request_pending(request, "native_gpu_fence_pending")

    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    @retirement_call("drain_frees")
    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
            if getattr(self, "_action_feedback", None) is not None:
                rid = self._action_deferred_frees.pop(id(blocks), None)
                if rid is not None:
                    self._action_request_released(rid)

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            if getattr(self, "_action_feedback", None) is not None:
                from vllm.v1.core.sched.prefill_operations import ResultKind
                self._action_load_transition(request.request_id, ResultKind.FAILURE,
                                             native_phase="native_invalid_blocks_received")
            wait_emit(
                "load_failure_recompute",
                request.request_id,
                recompute_enabled=self.recompute_kv_load_failures,
            )
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        wait_emit(
            "kv_published", request.request_id, tokens=request.num_computed_tokens
        )
        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            if getattr(self, "_action_feedback", None) is not None:
                from vllm.v1.core.sched.prefill_operations import ResultKind
                self._action_load_transition(request.request_id, ResultKind.ADOPTED,
                                             native_phase="native_kv_publication_and_status_adoption")
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not structured_output_req or structured_output_req.grammar is None:
                return False
            if isinstance(structured_output_req.grammar, Exception):
                self.grammar_compile_error_reqs.add(request.request_id)
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            if getattr(self, "_action_feedback", None) is not None:
                duplicate, receipt_identity = self._action_duplicate_native_receipt("recv", req_id)
                if duplicate:
                    continue
                from vllm.v1.core.sched.prefill_operations import ResultKind
                self._action_load_transition(req_id, ResultKind.COMPLETION,
                                             native_phase="native_finished_recving")
            wait_emit("scheduler_recv_notification", req_id)
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
            if getattr(self, "_action_feedback", None) is not None:
                self._action_record_native_receipt("recv", req_id, receipt_identity)
        for req_id in kv_connector_output.finished_sending or ():
            if getattr(self, "_action_feedback", None) is not None:
                duplicate, receipt_identity = self._action_duplicate_native_receipt("send", req_id)
                if duplicate:
                    continue
                self._action_stores_finished(req_id)
            wait_emit("scheduler_store_completion_observed", req_id)
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])
            if getattr(self, "_action_feedback", None) is not None:
                self._action_record_native_receipt("send", req_id, receipt_identity)

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
