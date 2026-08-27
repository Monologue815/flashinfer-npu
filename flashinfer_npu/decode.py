"""FlashInfer-compatible decode facade backed by the Host reference oracle."""

from __future__ import annotations

import math
import warnings

from .runtime import SchemaError

from .attention.batch import HostBatchReferenceWrapper, validate_graph_buffer
from .attention.frontend import (
    adapt_framework_single_qkv,
    adapt_paged_kv_data,
    adapt_single_qkv,
    canonicalize_kv_dtype,
    finite_scalar,
    optional_finite_scalar,
    parse_kv_layout,
    parse_pos_encoding_mode,
    framework_index_values,
    reference_index_values,
)
from .attention.planner import AttentionFrameworkSession
from .attention.reference import ReferenceAttentionExecutor, ReferenceTensor
from .attention.jit_protocol import (
    freeze_jit_buffer,
    make_single_jit_buffers,
    record_single_jit_protocol_call,
    require_jit_run,
    upstream_kv_layout_code,
)
from .attention.schema import AttentionMode, AttentionPlanSpec, PagedKVMetadata
from .attention.operator_quantization import (
    AttentionOperatorQuantizedTensorInput,
    combine_attention_operator_quantized_kv_input,
)


def single_decode_with_kv_cache(
    q,
    k,
    v,
    kv_layout="NHD",
    pos_encoding_mode="NONE",
    use_tensor_cores=False,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    window_left=-1,
    logits_soft_cap=None,
    sm_scale=None,
    rope_scale=None,
    rope_theta=None,
    return_lse=False,
):
    """Execute single-request decode through reference or automatic provider."""

    if not isinstance(q, ReferenceTensor):
        adapted_provider = adapt_framework_single_qkv(
            q,
            k,
            v,
            mode=AttentionMode.SINGLE_DECODE,
            kv_layout=kv_layout,
        )
        if use_tensor_cores:
            raise NotImplementedError(
                "provider single-decode matrix-core preference is not bound"
            )
        if (
            q_scale is not None or k_scale is not None or v_scale is not None
        ) and adapted_provider.kv_quant_spec is None:
            raise NotImplementedError(
                "provider single-decode Q/K/V scale binding requires quantized K/V"
            )
        provider_q_scale = (
            None if q_scale is None else finite_scalar(q_scale, "q_scale")
        )
        provider_k_scale = (
            None if k_scale is None else finite_scalar(k_scale, "k_scale")
        )
        provider_v_scale = (
            None if v_scale is None else finite_scalar(v_scale, "v_scale")
        )
        provider_softmax_scale = (
            1.0 / math.sqrt(adapted_provider.head_dim_qk)
            if sm_scale is None
            else finite_scalar(sm_scale, "sm_scale")
        )
        if provider_q_scale is not None:
            provider_softmax_scale *= provider_q_scale
        if provider_k_scale is not None:
            provider_softmax_scale *= provider_k_scale
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_DECODE,
            num_qo_heads=adapted_provider.num_qo_heads,
            num_kv_heads=adapted_provider.num_kv_heads,
            head_dim_qk=adapted_provider.head_dim_qk,
            head_dim_vo=adapted_provider.head_dim_vo,
            kv_layout=adapted_provider.layout,
            pos_encoding_mode=parse_pos_encoding_mode(pos_encoding_mode),
            q_dtype=adapted_provider.q_dtype,
            kv_dtype=adapted_provider.kv_dtype,
            kv_quant_spec=adapted_provider.kv_quant_spec,
            o_dtype=adapted_provider.q_dtype,
            sm_scale=provider_softmax_scale,
            logits_soft_cap=(
                0.0
                if logits_soft_cap is None
                else finite_scalar(logits_soft_cap, "logits_soft_cap")
            ),
            window_left=window_left,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        from .attention.holistic import (
            attention_operator_runtime_registry_snapshot,
        )
        from .attention.operator_resolver import AttentionOperatorRuntime

        snapshot = attention_operator_runtime_registry_snapshot()
        runtime = AttentionOperatorRuntime(
            adapted_provider.device,
            snapshot.registry,
            snapshot.operation_catalog,
            mode=AttentionMode.SINGLE_DECODE,
        )
        runtime.plan(spec, adapted_provider.metadata)
        if adapted_provider.kv_quant_spec is not None:
            provider_kv_input = combine_attention_operator_quantized_kv_input(k, v)
        else:
            if isinstance(k, AttentionOperatorQuantizedTensorInput) or isinstance(
                v, AttentionOperatorQuantizedTensorInput
            ):
                raise SchemaError(
                    "dense provider plan cannot consume quantized tensor inputs"
                )
            provider_kv_input = (k, v)
        result = runtime.run(
            q,
            provider_kv_input,
            return_lse=bool(return_lse),
            v_scale=provider_v_scale,
            logits_soft_cap=spec.logits_soft_cap,
        )
        if return_lse:
            if not isinstance(result, tuple) or len(result) != 2:
                raise SchemaError(
                    "provider did not return the requested output and LSE pair"
                )
            return result
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]
        return result

    # ReferenceTensor inputs are themselves an explicit reference-backend opt-in.
    adapted = adapt_single_qkv(
        q, k, v, mode=AttentionMode.SINGLE_DECODE, kv_layout=kv_layout
    )
    del use_tensor_cores  # Algorithm preference has no effect on the scalar oracle.
    q_scale_value = optional_finite_scalar(q_scale, "q_scale")
    k_scale_value = optional_finite_scalar(k_scale, "k_scale")
    v_scale_value = optional_finite_scalar(v_scale, "v_scale")
    softmax_scale = (
        1.0 / math.sqrt(adapted.head_dim_qk)
        if sm_scale is None
        else finite_scalar(sm_scale, "sm_scale")
    )
    softmax_scale *= q_scale_value * k_scale_value
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_DECODE,
        num_qo_heads=adapted.num_qo_heads,
        num_kv_heads=adapted.num_kv_heads,
        head_dim_qk=adapted.head_dim_qk,
        head_dim_vo=adapted.head_dim_vo,
        kv_layout=adapted.layout,
        pos_encoding_mode=parse_pos_encoding_mode(pos_encoding_mode),
        q_dtype=adapted.q.dtype,
        kv_dtype=adapted.kv_data.spec.dtype,
        kv_quant_spec=adapted.kv_data.spec.quant_spec,
        o_dtype=adapted.q.dtype,
        sm_scale=softmax_scale,
        logits_soft_cap=(
            0.0
            if logits_soft_cap is None
            else finite_scalar(logits_soft_cap, "logits_soft_cap")
        ),
        window_left=window_left,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    plan = AttentionFrameworkSession(AttentionMode.SINGLE_DECODE).plan(
        spec, adapted.metadata
    )
    result = ReferenceAttentionExecutor().execute(
        plan,
        adapted.q,
        adapted.kv_data,
        return_lse=bool(return_lse),
        v_scale=v_scale_value,
    )
    return (result.output, result.lse) if return_lse else result.output


def single_decode_with_kv_cache_return_lse(*args, **kwargs):
    kwargs["return_lse"] = True
    return single_decode_with_kv_cache(*args, **kwargs)


def single_decode_with_kv_cache_with_jit_module(
    jit_module,
    q,
    k,
    v,
    *args,
    kv_layout="NHD",
    window_left=-1,
    return_lse=False,
):
    """Run an injected single-decode module through the frozen Host call ABI.

    FlashInfer's current compatibility behavior allocates LSE when requested
    but still returns only the output tensor; this facade preserves that API.
    """

    adapted = adapt_single_qkv(
        q, k, v, mode=AttentionMode.SINGLE_DECODE, kv_layout=kv_layout
    )
    if (
        not isinstance(window_left, int)
        or isinstance(window_left, bool)
        or window_left < -1
    ):
        raise SchemaError("window_left must be -1 or a non-negative integer")
    tmp, output, lse = make_single_jit_buffers(
        adapted.q,
        output_shape=adapted.q.shape,
        lse_shape=(adapted.num_qo_heads,),
        return_lse=bool(return_lse),
    )
    layout_code = upstream_kv_layout_code(adapted.layout)
    with record_single_jit_protocol_call(
        mode=AttentionMode.SINGLE_DECODE.value,
        jit_module=jit_module,
        q=adapted.q,
        k=k,
        v=v,
        tmp=tmp,
        output=output,
        lse=lse,
        layout_code=layout_code,
        mask_code=None,
        window_left=window_left,
        return_lse=bool(return_lse),
        extra_args=args,
    ):
        require_jit_run(jit_module)(
            adapted.q,
            k,
            v,
            tmp,
            output,
            lse,
            layout_code,
            window_left,
            *args,
        )
        frozen_output = freeze_jit_buffer(output, "output")
    return frozen_output


_BATCH_DECODE_PLAN_LEGACY_POS_ARGS = (
    "pos_encoding_mode",
    "window_left",
    "logits_soft_cap",
    "q_data_type",
    "kv_data_type",
    "o_data_type",
    "data_type",
    "sm_scale",
    "rope_scale",
    "rope_theta",
    "non_blocking",
    "block_tables",
    "seq_lens",
    "fixed_split_size",
    "disable_split_kv",
    "q_len_per_req",
)


def _optional_plan_scalar(value, name):
    return None if value is None else finite_scalar(value, name)


def _require_decode_forward_matches_plan(
    plan,
    *,
    pos_encoding_mode,
    window_left,
    logits_soft_cap,
    sm_scale,
    rope_scale,
    rope_theta,
):
    values = {
        "pos_encoding_mode": parse_pos_encoding_mode(pos_encoding_mode),
        "window_left": window_left,
        "logits_soft_cap": (
            0.0
            if logits_soft_cap is None
            else finite_scalar(logits_soft_cap, "logits_soft_cap")
        ),
        "sm_scale": (
            1.0 / math.sqrt(plan.spec.head_dim_qk)
            if sm_scale is None
            else finite_scalar(sm_scale, "sm_scale")
        ),
        "rope_scale": (
            1.0 if rope_scale is None else finite_scalar(rope_scale, "rope_scale")
        ),
        "rope_theta": (
            1e4 if rope_theta is None else finite_scalar(rope_theta, "rope_theta")
        ),
    }
    for name, value in values.items():
        if getattr(plan.spec, name) != value:
            raise SchemaError("deprecated forward %s must match the planned value" % name)


class BatchDecodeWithPagedKVCacheWrapper(HostBatchReferenceWrapper):
    """FlashInfer-compatible paged decode lifecycle on the Host oracle."""

    def __init__(
        self,
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=False,
        use_tensor_cores=False,
        paged_kv_indptr_buffer=None,
        paged_kv_indices_buffer=None,
        paged_kv_last_page_len_buffer=None,
        backend="auto",
        jit_args=None,
    ):
        layout = parse_kv_layout(kv_layout)
        if jit_args is not None:
            raise NotImplementedError("custom JIT modules are not implemented")
        provider_workspace = (
            backend == "auto"
            and str(getattr(float_workspace_buffer, "device", "")).split(":", 1)[0]
            == "npu"
        )
        if provider_workspace and use_cuda_graph:
            raise NotImplementedError(
                "provider graph resources are not bound to paged decode"
            )
        if provider_workspace and use_tensor_cores:
            raise NotImplementedError(
                "provider matrix-core preference is not bound to paged decode"
            )
        fixed_batch_size = None
        graph_buffers = ()
        if use_cuda_graph:
            if any(
                item is None
                for item in (
                    paged_kv_indptr_buffer,
                    paged_kv_indices_buffer,
                    paged_kv_last_page_len_buffer,
                )
            ):
                raise ValueError("graph mode requires all paged metadata buffers")
            last_buffer = validate_graph_buffer(
                paged_kv_last_page_len_buffer,
                "paged_kv_last_page_len_buffer",
                dtype="int32",
            )
            if last_buffer.shape[0] < 1:
                raise ValueError("graph mode requires a positive fixed batch size")
            fixed_batch_size = last_buffer.shape[0]
            graph_buffers = (
                validate_graph_buffer(
                    paged_kv_indptr_buffer,
                    "paged_kv_indptr_buffer",
                    dtype="int32",
                    length=fixed_batch_size + 1,
                ),
                validate_graph_buffer(
                    paged_kv_indices_buffer,
                    "paged_kv_indices_buffer",
                    dtype="int32",
                ),
                last_buffer,
            )
        self._init_host_wrapper(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            float_workspace_buffer=float_workspace_buffer,
            backend=backend,
            graph_enabled=bool(use_cuda_graph),
            fixed_batch_size=fixed_batch_size,
            graph_buffers=graph_buffers,
            provider_runtime_enabled=True,
        )
        if any(
            buffer.device != self._float_workspace_buffer.device
            for buffer in graph_buffers
        ):
            raise ValueError("graph buffers must be on the workspace device")
        self._kv_layout = layout
        self._use_tensor_cores = bool(use_tensor_cores)
        self._graph_indices_capacity = (
            graph_buffers[1].shape[0] if graph_buffers else None
        )

    @property
    def use_tensor_cores(self):
        return self._use_tensor_cores

    @property
    def use_matrix_cores(self):
        """Backend-neutral name for the public compatibility preference."""

        return self._use_tensor_cores

    def workspace_size(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        window_left=-1,
        logits_soft_cap=None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        data_type=None,
        sm_scale=None,
        rope_scale=None,
        rope_theta=None,
        block_tables=None,
        seq_lens=None,
        fixed_split_size=None,
        disable_split_kv=False,
        q_len_per_req=1,
    ):
        """Return caller-workspace bytes without mutating this wrapper's plan."""

        probe = (
            self._provider_workspace_query_probe()
            if self._operator_runtime is not None
            else BatchDecodeWithPagedKVCacheWrapper(
                self._float_workspace_buffer,
                kv_layout=self._kv_layout.value,
                use_tensor_cores=self.use_tensor_cores,
                backend="reference",
            )
        )
        probe.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=False,
            block_tables=block_tables,
            seq_lens=seq_lens,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            q_len_per_req=q_len_per_req,
        )
        return probe.workspace_contract.required_sizes

    def plan(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        *deprecated_positional_args,
        **kwargs,
    ):
        if len(deprecated_positional_args) > len(
            _BATCH_DECODE_PLAN_LEGACY_POS_ARGS
        ):
            raise TypeError("too many deprecated optional positional arguments")
        merged = dict(kwargs)
        for name, value in zip(
            _BATCH_DECODE_PLAN_LEGACY_POS_ARGS, deprecated_positional_args
        ):
            if name in merged:
                raise TypeError("multiple values for argument %r" % name)
            merged[name] = value
        if deprecated_positional_args:
            warnings.warn(
                "optional BatchDecode plan arguments should be passed by keyword",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._plan_impl(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            **merged,
        )

    def _plan_impl(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        *,
        pos_encoding_mode="NONE",
        window_left=-1,
        window_right=0,
        logits_soft_cap=None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        data_type=None,
        sm_scale=None,
        rope_scale=None,
        rope_theta=None,
        non_blocking=True,
        block_tables=None,
        seq_lens=None,
        fixed_split_size=None,
        disable_split_kv=False,
        q_len_per_req=1,
    ):
        del non_blocking
        if self._operator_runtime is not None and (
            fixed_split_size is not None or disable_split_kv
        ):
            raise NotImplementedError(
                "provider split-K planning controls are not bound"
            )
        del fixed_split_size, disable_split_kv
        if block_tables is not None or seq_lens is not None:
            raise NotImplementedError(
                "alternate block_tables/seq_lens metadata is not implemented"
            )
        if q_len_per_req > 1 and not self.use_tensor_cores:
            raise ValueError(
                "q_len_per_req > 1 requires use_tensor_cores=True in the public contract"
            )
        if data_type is not None:
            q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type
        q_dtype = str(q_data_type)
        kv_dtype, kv_quant_spec = canonicalize_kv_dtype(kv_data_type, q_dtype)
        o_dtype = q_dtype if o_data_type is None else str(o_data_type)
        index_reader = (
            framework_index_values
            if self._operator_runtime is not None
            else reference_index_values
        )
        metadata = PagedKVMetadata(
            index_reader(indptr, "indptr"),
            index_reader(indices, "indices"),
            index_reader(last_page_len, "last_page_len"),
            page_size,
        )
        if (
            self._graph_indices_capacity is not None
            and len(metadata.indices) > self._graph_indices_capacity
        ):
            raise ValueError("indices exceed graph buffer capacity")
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            kv_layout=self._kv_layout,
            pos_encoding_mode=parse_pos_encoding_mode(pos_encoding_mode),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            kv_quant_spec=kv_quant_spec,
            o_dtype=o_dtype,
            sm_scale=_optional_plan_scalar(sm_scale, "sm_scale"),
            logits_soft_cap=_optional_plan_scalar(
                logits_soft_cap, "logits_soft_cap"
            ),
            window_left=window_left,
            window_right=window_right,
            rope_scale=_optional_plan_scalar(rope_scale, "rope_scale"),
            rope_theta=_optional_plan_scalar(rope_theta, "rope_theta"),
            q_len_per_req=q_len_per_req,
        )
        self._commit_plan(spec, metadata, None)

    begin_forward = plan

    def forward(
        self,
        q,
        paged_kv_cache,
        pos_encoding_mode="NONE",
        q_scale=None,
        k_scale=None,
        v_scale=None,
        window_left=-1,
        logits_soft_cap=None,
        sm_scale=None,
        rope_scale=None,
        rope_theta=None,
    ):
        """Deprecated compatibility alias for :meth:`run`."""

        warnings.warn(
            "forward is deprecated; use run instead",
            DeprecationWarning,
            stacklevel=2,
        )
        plan = self.plan_state
        _require_decode_forward_matches_plan(
            plan,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return self.run(
            q,
            paged_kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            window_left=window_left,
        )

    def run(
        self,
        q,
        paged_kv_cache,
        *args,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        out=None,
        lse=None,
        return_lse=False,
        enable_pdl=None,
        window_left=None,
        sinks=None,
        q_len_per_req=None,
        skip_softmax_threshold_scale_factor=None,
        kv_cache_sf=None,
    ):
        plan = self.plan_state
        if q_len_per_req is not None:
            warnings.warn(
                "run q_len_per_req is deprecated; pass it to plan",
                DeprecationWarning,
                stacklevel=2,
            )
            if q_len_per_req != plan.spec.q_len_per_req:
                raise ValueError("run q_len_per_req must match the planned value")
        if self._operator_runtime is not None:
            if args:
                raise NotImplementedError(
                    "provider custom JIT run arguments are not implemented"
                )
            if q_scale is not None and plan.spec.kv_quant_spec is None:
                raise NotImplementedError(
                    "provider query-scale binding requires quantized K/V"
                )
            if enable_pdl not in (None, False):
                raise NotImplementedError(
                    "enable_pdl has no authorized Ascend provider binding"
                )
            if window_left is not None and window_left != plan.spec.window_left:
                raise SchemaError("run window_left must match the planned value")
            if sinks is not None:
                raise NotImplementedError(
                    "provider attention-sink run binding is not implemented"
                )
            if skip_softmax_threshold_scale_factor is not None:
                raise NotImplementedError(
                    "provider skip-softmax run binding is not implemented"
                )
            result = self._operator_runtime.run(
                q,
                paged_kv_cache,
                return_lse=bool(return_lse) or lse is not None,
                out=out,
                lse=lse,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                logits_soft_cap=plan.spec.logits_soft_cap,
                profiler_buffer=None,
                kv_cache_sf=kv_cache_sf,
            )
            if return_lse:
                if not isinstance(result, tuple) or len(result) != 2:
                    raise SchemaError(
                        "provider did not return the requested output and LSE pair"
                    )
                return result
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]
            return result
        kv_data = adapt_paged_kv_data(
            paged_kv_cache,
            page_size=plan.metadata.page_size,
            num_kv_heads=plan.spec.num_kv_heads,
            head_dim_qk=plan.spec.head_dim_qk,
            head_dim_vo=int(plan.spec.head_dim_vo),
            dtype=plan.spec.kv_dtype,
            layout=plan.spec.kv_layout,
            quant_spec=plan.spec.kv_quant_spec,
        )
        return self._execute_reference(
            q,
            kv_data,
            args=args,
            q_scale=optional_finite_scalar(q_scale, "q_scale"),
            k_scale=optional_finite_scalar(k_scale, "k_scale"),
            v_scale=optional_finite_scalar(v_scale, "v_scale"),
            out=out,
            lse=lse,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            window_left=window_left,
            sinks=sinks,
            kv_cache_sf=kv_cache_sf,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )

    def run_return_lse(self, *args, **kwargs):
        kwargs["return_lse"] = True
        return self.run(*args, **kwargs)

    def end_forward(self):
        warnings.warn("end_forward is deprecated and has no effect", DeprecationWarning)


__all__ = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "single_decode_with_kv_cache",
    "single_decode_with_kv_cache_return_lse",
    "single_decode_with_kv_cache_with_jit_module",
]
