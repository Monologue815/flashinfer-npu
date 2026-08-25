"""Shared lifecycle for FlashInfer-compatible batch Attention facades."""

from __future__ import annotations

from copy import copy
from typing import Optional, Sequence

from flashinfer_npu.runtime import DispatchError, SchemaError

from .frontend import (
    finalize_reference_result,
    require_reference_backend,
    require_reference_tensor,
    validate_framework_workspace_buffer,
    validate_workspace_buffer,
)
from .operator_resolver import AttentionOperatorRuntime
from .plan_selection import (
    AttentionPlanSelection,
    build_provider_plan_selection,
    build_reference_plan_selection,
)
from .planner import (
    AttentionFrameworkPlan,
    AttentionFrameworkSession,
    AttentionStateError,
)
from .reference import (
    ReferenceAttentionExecutor,
    ReferenceBuffer,
    ReferenceKVInput,
    ReferenceTensor,
    ScaleInput,
)
from .schema import AttentionMetadata, AttentionMode, AttentionPlanSpec
from .workspace import AttentionWorkspaceContract
from .tensor_contract import (
    reference_attention_launch_options,
    validate_reference_attention_views,
)
from .execution_identity import (
    AttentionCapturedExecution,
    AttentionGraphResourceContract,
    AttentionPersistentBufferSpec,
    build_reference_execution_identity,
)


_GRAPH_BUFFER_NAMES = {
    AttentionMode.BATCH_PREFILL_PAGED: (
        "qo_indptr",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "custom_mask",
        "mask_indptr",
    ),
    AttentionMode.BATCH_PREFILL_RAGGED: (
        "qo_indptr",
        "kv_indptr",
        "custom_mask",
        "mask_indptr",
    ),
    AttentionMode.BATCH_DECODE_PAGED: (
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
    ),
}

_GRAPH_BUFFER_COUNTS = {
    AttentionMode.BATCH_PREFILL_PAGED: (4, 6),
    AttentionMode.BATCH_PREFILL_RAGGED: (2, 4),
    AttentionMode.BATCH_DECODE_PAGED: (3,),
}


class HostBatchReferenceWrapper:
    """Reference lifecycle plus private provider routing for public facades."""

    def _init_host_wrapper(
        self,
        *,
        mode: AttentionMode,
        float_workspace_buffer,
        backend: str,
        graph_enabled: bool,
        fixed_batch_size: Optional[int],
        graph_buffers: Sequence[ReferenceTensor] = (),
        provider_runtime_enabled: bool = False,
    ) -> None:
        self._operator_runtime = None
        if backend != "reference":
            if isinstance(float_workspace_buffer, ReferenceTensor):
                require_reference_backend(backend)
            if backend != "auto":
                raise DispatchError(
                    "non-reference batch wrappers currently require backend='auto'"
                )
            if not provider_runtime_enabled:
                raise NotImplementedError(
                    "provider runtime is not connected to the %s public wrapper"
                    % mode.value
                )
            if graph_enabled:
                raise NotImplementedError(
                    "provider graph resources are not bound to public batch wrappers"
                )
            float_workspace_buffer = validate_framework_workspace_buffer(
                float_workspace_buffer, "float_workspace_buffer"
            )
            shape = tuple(int(dim) for dim in float_workspace_buffer.shape)
            device = str(float_workspace_buffer.device)
            if device.split(":", 1)[0] != "npu":
                raise DispatchError(
                    "backend='auto' batch wrappers require an npu[:index] workspace"
                )
            # Imported lazily to avoid making the public facade own registry state.
            from .holistic import attention_operator_runtime_registry_snapshot

            snapshot = attention_operator_runtime_registry_snapshot()
            self._operator_runtime_registry_snapshot = snapshot
            self._operator_runtime = AttentionOperatorRuntime(
                device,
                snapshot.registry,
                snapshot.operation_catalog,
                mode=mode,
            )
            self._float_workspace_buffer = float_workspace_buffer
            self._int_workspace_buffer = None
            self._session = None
            self._executor = None
            self._custom_mask_data = None
            self._capture_record = None
            self._capture_generation = 0
            self._graph_resources = AttentionGraphResourceContract.disabled(mode)
            self._workspace_contract = AttentionWorkspaceContract(
                backend="auto",
                device=device,
                float_capacity_bytes=shape[0],
                int_capacity_bytes=0,
            )
            return
        require_reference_backend(backend)
        self._float_workspace_buffer = validate_workspace_buffer(
            float_workspace_buffer, "float_workspace_buffer"
        )
        self._int_workspace_buffer = ReferenceTensor.zeros(
            (0,), dtype="uint8", device=self._float_workspace_buffer.device
        )
        self._session = AttentionFrameworkSession(
            mode,
            graph_enabled=graph_enabled,
            fixed_batch_size=fixed_batch_size,
        )
        self._executor = ReferenceAttentionExecutor()
        self._custom_mask_data = None
        self._capture_record = None
        self._capture_generation = 0
        if graph_enabled:
            names = _GRAPH_BUFFER_NAMES[mode]
            if len(graph_buffers) not in _GRAPH_BUFFER_COUNTS[mode]:
                raise SchemaError("unexpected persistent graph buffer set")
            persistent = tuple(
                AttentionPersistentBufferSpec(
                    name,
                    buffer.dtype,
                    buffer.shape[0],
                    buffer.device,
                )
                for name, buffer in zip(names, graph_buffers)
            )
            self._graph_resources = AttentionGraphResourceContract(
                mode,
                True,
                fixed_batch_size,
                persistent,
            )
        else:
            self._graph_resources = AttentionGraphResourceContract.disabled(mode)
        self._workspace_contract = AttentionWorkspaceContract.for_host_reference(
            device=self._float_workspace_buffer.device,
            float_capacity_bytes=self._float_workspace_buffer.shape[0],
            int_capacity_bytes=self._int_workspace_buffer.shape[0],
            graph_enabled=graph_enabled,
        )

    @property
    def plan_state(self) -> AttentionFrameworkPlan:
        if self._operator_runtime is not None:
            return self._operator_runtime.plan_state
        return self._session.plan_state

    @property
    def plan_selection(self) -> AttentionPlanSelection:
        """Describe the selected route without exposing an executable handle."""

        plan = self.plan_state
        if self._operator_runtime is not None:
            return build_provider_plan_selection(
                plan,
                self._operator_runtime.operator_session.active_plan,
                registry_generation=(
                    self._operator_runtime_registry_snapshot.generation
                ),
            )
        return build_reference_plan_selection(plan)

    def _provider_workspace_query_probe(self):
        """Fork wrapper state for a plan-equivalent, non-publishing size query."""

        if self._operator_runtime is None:
            raise AttentionStateError(
                "provider workspace query requires a provider runtime"
            )
        probe = copy(self)
        probe._operator_runtime = self._operator_runtime.fork_unplanned()
        probe._workspace_contract = AttentionWorkspaceContract(
            backend="auto",
            device=self._workspace_contract.device,
            float_capacity_bytes=(1 << 63) - 1,
            int_capacity_bytes=(1 << 63) - 1,
        )
        probe._capture_record = None
        return probe

    @property
    def is_graph_enabled(self) -> bool:
        if self._operator_runtime is not None:
            return False
        return self._session.graph_enabled

    @property
    def is_cuda_graph_enabled(self) -> bool:
        """Compatibility alias; internally this is a backend-neutral graph flag."""

        return self.is_graph_enabled

    @property
    def workspace_contract(self) -> AttentionWorkspaceContract:
        return self._workspace_contract

    @property
    def graph_resource_contract(self) -> AttentionGraphResourceContract:
        return self._graph_resources

    @property
    def capture_record(self) -> Optional[AttentionCapturedExecution]:
        """Host-only structural capture identity; never a device graph handle."""

        return self._capture_record

    def _commit_plan(
        self,
        spec: AttentionPlanSpec,
        metadata: AttentionMetadata,
        custom_mask_data,
    ) -> None:
        if self._operator_runtime is not None:
            if custom_mask_data is not None or spec.custom_mask is not None:
                raise NotImplementedError(
                    "provider custom-mask plan binding is not implemented"
                )
            self._operator_runtime.plan(
                spec,
                metadata,
                workspace_contract=self._workspace_contract,
            )
            self._workspace_contract = self._operator_runtime.workspace_contract
            self._capture_record = None
            return
        plan = self._session.plan(spec, metadata)
        self._workspace_contract = self._workspace_contract.bind_plan(
            plan.generation
        )
        self._custom_mask_data = custom_mask_data
        self._capture_record = None

    def reset_workspace_buffer(
        self, float_workspace_buffer, int_workspace_buffer
    ) -> None:
        if self._operator_runtime is not None:
            float_buffer = validate_framework_workspace_buffer(
                float_workspace_buffer, "float_workspace_buffer"
            )
            int_buffer = validate_framework_workspace_buffer(
                int_workspace_buffer,
                "int_workspace_buffer",
                device=str(float_buffer.device),
            )
            if float_buffer is int_buffer:
                raise SchemaError("float and int workspace buffers cannot alias")
            contract = self._workspace_contract.rebind(
                device=str(float_buffer.device),
                float_capacity_bytes=int(float_buffer.shape[0]),
                int_capacity_bytes=int(int_buffer.shape[0]),
                allow_device_change=False,
            )
            if self._operator_runtime.is_planned:
                self._operator_runtime.rebind_workspace_contract(contract)
                contract = self._operator_runtime.workspace_contract
            self._float_workspace_buffer = float_buffer
            self._int_workspace_buffer = int_buffer
            self._workspace_contract = contract
            return
        float_buffer = validate_workspace_buffer(
            float_workspace_buffer, "float_workspace_buffer"
        )
        int_buffer = validate_workspace_buffer(
            int_workspace_buffer,
            "int_workspace_buffer",
            device=float_buffer.device,
        )
        if float_buffer is int_buffer:
            raise SchemaError("float and int workspace buffers cannot alias")
        allow_device_change = not self._session.is_planned and not self.is_graph_enabled
        contract = self._workspace_contract.rebind(
            device=float_buffer.device,
            float_capacity_bytes=float_buffer.shape[0],
            int_capacity_bytes=int_buffer.shape[0],
            allow_device_change=allow_device_change,
        )
        self._float_workspace_buffer = float_buffer
        self._int_workspace_buffer = int_buffer
        self._workspace_contract = contract

    def _execute_reference(
        self,
        q,
        kv_data: ReferenceKVInput,
        *,
        args,
        q_scale: ScaleInput,
        k_scale: ScaleInput,
        v_scale: ScaleInput,
        out: Optional[ReferenceBuffer],
        lse: Optional[ReferenceBuffer],
        return_lse: bool,
        enable_pdl,
        window_left,
        sinks,
        kv_cache_sf,
        skip_softmax_threshold_scale_factor,
        use_fp16_softmax=None,
        uses_spcompress=None,
    ):
        if args:
            raise NotImplementedError(
                "custom JIT run arguments are not implemented by the Host facade"
            )
        if enable_pdl not in (None, False):
            raise NotImplementedError("enable_pdl is CUDA-specific and unsupported")
        if sinks is not None:
            raise NotImplementedError("attention sinks are not in the Host oracle yet")
        if kv_cache_sf is not None:
            raise NotImplementedError("NVFP4 kv_cache_sf is not implemented")
        if skip_softmax_threshold_scale_factor is not None:
            raise NotImplementedError("skip-softmax sparsity is not implemented")
        if use_fp16_softmax not in (None, False):
            raise NotImplementedError(
                "FP16 softmax is not implemented by the Host oracle"
            )
        if uses_spcompress not in (None, False):
            raise NotImplementedError("SP-compressed attention is not implemented")
        query = require_reference_tensor(q, "q")
        self._workspace_contract.validate_run(
            device=query.device,
            plan_generation=self.plan_state.generation,
        )
        if window_left is not None and window_left != self.plan_state.spec.window_left:
            raise SchemaError("run window_left must match the planned value")
        wants_lse = bool(return_lse) or lse is not None
        auxiliary, run_options = reference_attention_launch_options(
            self.plan_state,
            query.device,
            custom_mask_data=self._custom_mask_data,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        tensor_contract = validate_reference_attention_views(
            query,
            kv_data,
            out=out,
            lse=lse,
            workspace_float=self._float_workspace_buffer,
            workspace_int=self._int_workspace_buffer,
            auxiliary=auxiliary,
            run_options=run_options,
            plan=self.plan_state,
        )
        if self.is_graph_enabled:
            identity = build_reference_execution_identity(
                self.plan_state,
                self._workspace_contract,
                self._graph_resources,
                tensor_contract,
                return_lse=bool(return_lse),
            )
            if self._capture_record is None:
                self._capture_generation += 1
                self._capture_record = AttentionCapturedExecution(
                    identity,
                    "host_contract",
                    self._capture_generation,
                )
            else:
                self._capture_record.validate_reuse(identity)
        result = self._executor.execute(
            self.plan_state,
            query,
            kv_data,
            return_lse=wants_lse,
            custom_mask_data=self._custom_mask_data,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return finalize_reference_result(
            result,
            out=out,
            lse=lse,
            return_lse=bool(return_lse),
        )


def require_no_host_plan_extensions(**values) -> None:
    enabled = [
        name
        for name, value in values.items()
        if value is not None and value is not False and value != 0
    ]
    if enabled:
        raise NotImplementedError(
            "Host reference plan does not implement: %s" % ", ".join(enabled)
        )


def validate_graph_buffer(
    value,
    name: str,
    *,
    dtype: str,
    length: Optional[int] = None,
    minimum_length: Optional[int] = None,
) -> ReferenceTensor:
    tensor = require_reference_tensor(value, name)
    if len(tensor.shape) != 1 or tensor.dtype != dtype:
        raise SchemaError("%s must be a rank-1 %s ReferenceTensor" % (name, dtype))
    if length is not None and tensor.shape[0] != length:
        raise SchemaError("%s length must be %d" % (name, length))
    if minimum_length is not None and tensor.shape[0] < minimum_length:
        raise SchemaError("%s capacity must be at least %d" % (name, minimum_length))
    return tensor
