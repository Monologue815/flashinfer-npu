"""FlashInfer-compatible prefill facades with reference/provider routing."""

from __future__ import annotations

import math
import warnings

from .runtime import DispatchError, SchemaError

from .attention.frontend import (
    adapt_batch_custom_mask,
    canonicalize_framework_paged_fp8_kv_input,
    canonicalize_flashinfer_paged_kv_dtype,
    canonicalize_kv_dtype,
    adapt_head_scale,
    adapt_paged_kv_data,
    adapt_ragged_kv_data,
    adapt_single_custom_mask,
    adapt_framework_single_qkv,
    adapt_single_qkv,
    canonicalize_dtype_name,
    finite_scalar,
    fp8_per_tensor_quant_spec,
    framework_index_values,
    multiply_scale,
    optional_finite_scalar,
    parse_kv_layout,
    parse_pos_encoding_mode,
    reference_index_values,
    require_reference_backend,
    single_fp8_per_head_quant_spec,
)
from .attention.nvfp4_scale_factor import (
    flashinfer_nvfp4_kv_quant_spec,
    is_flashinfer_nvfp4_kv_quant_spec,
)
from .attention.batch import (
    HostBatchReferenceWrapper,
    require_no_host_plan_extensions,
    validate_graph_buffer,
)
from .attention.planner import AttentionFrameworkSession
from .attention.reference import ReferenceAttentionExecutor, ReferenceTensor
from .attention.jit_protocol import (
    freeze_jit_buffer,
    make_single_jit_buffers,
    record_single_jit_protocol_call,
    require_jit_run,
    upstream_kv_layout_code,
    validate_upstream_mask_mode,
)
from .attention.schema import (
    AttentionMode,
    AttentionPlanSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVMetadata,
)
from .attention.operator_quantization import (
    AttentionOperatorImplicitUnitScale,
    AttentionOperatorQuantizedTensorInput,
    combine_attention_operator_quantized_kv_input,
)


def single_prefill_with_kv_cache(
    q,
    k,
    v,
    scale_q=None,
    scale_k=None,
    scale_v=None,
    o_dtype=None,
    custom_mask=None,
    packed_custom_mask=None,
    causal=False,
    kv_layout="NHD",
    pos_encoding_mode="NONE",
    use_fp16_qk_reduction=False,
    sm_scale=None,
    window_left=-1,
    logits_soft_cap=None,
    rope_scale=None,
    rope_theta=None,
    backend="auto",
    return_lse=False,
    kv_cache_sf=None,
    k_scale=None,
    v_scale=None,
):
    """Execute single-request prefill through reference or automatic provider."""

    if backend != "reference" and not isinstance(q, ReferenceTensor):
        if backend != "auto":
            raise DispatchError(
                "provider single prefill currently requires backend='auto'"
            )
        provider_key = k
        provider_value = v
        provider_q_head_scale = scale_q
        provider_k_head_scale = scale_k
        provider_v_head_scale = scale_v
        adapted_provider = adapt_framework_single_qkv(
            q,
            provider_key,
            provider_value,
            mode=AttentionMode.SINGLE_PREFILL,
            kv_layout=kv_layout,
            packed_kv_quant_spec=(
                flashinfer_nvfp4_kv_quant_spec()
                if kv_cache_sf is not None
                else None
            ),
            reject_unquantized_uint8=True,
        )
        head_scales = (scale_q, scale_k, scale_v)
        fp8_dtypes_match = (
            adapted_provider.q_dtype == adapted_provider.kv_dtype
            and adapted_provider.q_dtype
            in {"float8_e4m3fn", "float8_e5m2"}
        )
        if adapted_provider.kv_quant_spec is None and fp8_dtypes_match:
            fp8_quant_spec = single_fp8_per_head_quant_spec(
                adapted_provider.kv_dtype, kv_layout
            )
            provider_q_head_scale = (
                scale_q
                if scale_q is not None
                else AttentionOperatorImplicitUnitScale(
                    source="run.q_head_scale",
                    shape=(adapted_provider.num_qo_heads,),
                    dtype=fp8_quant_spec.scale_dtype,
                    device=adapted_provider.device,
                )
            )
            key_quant_scale = (
                scale_k
                if scale_k is not None
                else AttentionOperatorImplicitUnitScale(
                    source="kv.key.scale",
                    shape=(adapted_provider.num_kv_heads,),
                    dtype=fp8_quant_spec.scale_dtype,
                    device=adapted_provider.device,
                )
            )
            value_quant_scale = (
                scale_v
                if scale_v is not None
                else AttentionOperatorImplicitUnitScale(
                    source="kv.value.scale",
                    shape=(adapted_provider.num_kv_heads,),
                    dtype=fp8_quant_spec.scale_dtype,
                    device=adapted_provider.device,
                )
            )
            provider_key = AttentionOperatorQuantizedTensorInput(
                quant_spec=fp8_quant_spec,
                logical_shape=tuple(int(dim) for dim in k.shape),
                storage=k,
                scale=key_quant_scale,
            )
            provider_value = AttentionOperatorQuantizedTensorInput(
                quant_spec=fp8_quant_spec,
                logical_shape=tuple(int(dim) for dim in v.shape),
                storage=v,
                scale=value_quant_scale,
            )
            adapted_provider = adapt_framework_single_qkv(
                q,
                provider_key,
                provider_value,
                mode=AttentionMode.SINGLE_PREFILL,
                kv_layout=kv_layout,
            )
            provider_k_head_scale = None
            provider_v_head_scale = None
        elif adapted_provider.kv_quant_spec is None and any(
            value is not None for value in head_scales
        ):
            raise NotImplementedError(
                "provider query-scale/per-head K/V scale binding requires "
                "quantized K/V"
            )
        if (
            k_scale is not None or v_scale is not None
        ) and adapted_provider.kv_quant_spec is None:
            raise NotImplementedError(
                "provider single-prefill KV scale binding requires quantized K/V"
            )
        provider_k_scale = (
            None if k_scale is None else finite_scalar(k_scale, "k_scale")
        )
        provider_v_scale = (
            None if v_scale is None else finite_scalar(v_scale, "v_scale")
        )
        if custom_mask is not None or packed_custom_mask is not None:
            raise NotImplementedError(
                "provider single-prefill custom-mask binding is not implemented"
            )
        if use_fp16_qk_reduction:
            raise NotImplementedError(
                "provider single-prefill FP16 QK reduction is not bound"
            )
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_PREFILL,
            num_qo_heads=adapted_provider.num_qo_heads,
            num_kv_heads=adapted_provider.num_kv_heads,
            head_dim_qk=adapted_provider.head_dim_qk,
            head_dim_vo=adapted_provider.head_dim_vo,
            kv_layout=adapted_provider.layout,
            causal=bool(causal),
            pos_encoding_mode=parse_pos_encoding_mode(pos_encoding_mode),
            q_dtype=adapted_provider.q_dtype,
            kv_dtype=adapted_provider.kv_dtype,
            kv_quant_spec=adapted_provider.kv_quant_spec,
            o_dtype=(
                adapted_provider.q_dtype
                if o_dtype is None
                else canonicalize_dtype_name(o_dtype)
            ),
            sm_scale=(
                1.0 / math.sqrt(adapted_provider.head_dim_qk)
                if sm_scale is None
                else finite_scalar(sm_scale, "sm_scale")
            ),
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
            mode=AttentionMode.SINGLE_PREFILL,
            runtime_declaration_bindings=(
                snapshot.runtime_declaration_binding_tuples
            ),
            plan_scoring_manifest_binding=(
                snapshot.plan_scoring_manifest_binding
            ),
            provider_integration_bundle_binding=(
                snapshot.provider_integration_bundle_binding
            ),
        )
        runtime.plan(spec, adapted_provider.metadata)
        if is_flashinfer_nvfp4_kv_quant_spec(adapted_provider.kv_quant_spec):
            provider_kv_input = (provider_key, provider_value)
        elif adapted_provider.kv_quant_spec is not None:
            provider_kv_input = combine_attention_operator_quantized_kv_input(
                provider_key, provider_value
            )
        else:
            if isinstance(
                provider_key, AttentionOperatorQuantizedTensorInput
            ) or isinstance(
                provider_value, AttentionOperatorQuantizedTensorInput
            ):
                raise SchemaError(
                    "dense provider plan cannot consume quantized tensor inputs"
                )
            provider_kv_input = (provider_key, provider_value)
        result = runtime.run(
            q,
            provider_kv_input,
            return_lse=bool(return_lse),
            k_scale=provider_k_scale,
            v_scale=provider_v_scale,
            q_head_scale=provider_q_head_scale,
            k_head_scale=provider_k_head_scale,
            v_head_scale=provider_v_head_scale,
            logits_soft_cap=spec.logits_soft_cap,
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

    require_reference_backend(backend)
    if kv_cache_sf is not None:
        raise NotImplementedError(
            "kv_cache_sf/NVFP4 storage is not implemented by the Host facade"
        )
    adapted = adapt_single_qkv(
        q, k, v, mode=AttentionMode.SINGLE_PREFILL, kv_layout=kv_layout
    )
    mask_spec, mask_data = adapt_single_custom_mask(
        custom_mask,
        packed_custom_mask,
        qo_len=adapted.metadata.qo_len,
        kv_len=adapted.metadata.kv_len,
        device=adapted.q.device,
    )
    pos_mode = parse_pos_encoding_mode(pos_encoding_mode)
    softmax_scale = (
        1.0 / math.sqrt(adapted.head_dim_qk)
        if sm_scale is None
        else finite_scalar(sm_scale, "sm_scale")
    )
    softmax_scale *= optional_finite_scalar(k_scale, "k_scale")
    output_dtype = adapted.q.dtype if o_dtype is None else str(o_dtype)
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=adapted.num_qo_heads,
        num_kv_heads=adapted.num_kv_heads,
        head_dim_qk=adapted.head_dim_qk,
        head_dim_vo=adapted.head_dim_vo,
        kv_layout=adapted.layout,
        causal=bool(causal),
        pos_encoding_mode=pos_mode,
        q_dtype=adapted.q.dtype,
        kv_dtype=adapted.kv_data.spec.dtype,
        kv_quant_spec=adapted.kv_data.spec.quant_spec,
        o_dtype=output_dtype,
        sm_scale=softmax_scale,
        logits_soft_cap=(
            0.0
            if logits_soft_cap is None
            else finite_scalar(logits_soft_cap, "logits_soft_cap")
        ),
        window_left=window_left,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        use_fp16_qk_reduction=bool(use_fp16_qk_reduction),
        custom_mask=mask_spec,
    )
    plan = AttentionFrameworkSession(AttentionMode.SINGLE_PREFILL).plan(
        spec, adapted.metadata
    )
    q_scale_input = adapt_head_scale(
        scale_q,
        name="scale_q",
        num_heads=adapted.num_qo_heads,
        device=adapted.q.device,
    )
    k_scale_input = adapt_head_scale(
        scale_k,
        name="scale_k",
        num_heads=adapted.num_kv_heads,
        device=adapted.q.device,
    )
    v_scale_input = adapt_head_scale(
        scale_v,
        name="scale_v",
        num_heads=adapted.num_kv_heads,
        device=adapted.q.device,
    )
    v_scale_input = multiply_scale(
        v_scale_input, optional_finite_scalar(v_scale, "v_scale")
    )
    result = ReferenceAttentionExecutor().execute(
        plan,
        adapted.q,
        adapted.kv_data,
        return_lse=bool(return_lse),
        custom_mask_data=mask_data,
        q_scale=q_scale_input,
        k_scale=k_scale_input,
        v_scale=v_scale_input,
    )
    return (result.output, result.lse) if return_lse else result.output


def single_prefill_with_kv_cache_return_lse(*args, **kwargs):
    kwargs["return_lse"] = True
    return single_prefill_with_kv_cache(*args, **kwargs)


def single_prefill_with_kv_cache_with_jit_module(
    jit_module,
    q,
    k,
    v,
    *args,
    kv_layout="NHD",
    mask_mode=0,
    window_left=-1,
    return_lse=False,
):
    """Run an injected single-prefill module through the frozen Host call ABI.

    This compatibility path validates and forwards buffers exactly; it does not
    compile or authenticate a JIT artifact.  Production providers must use the
    artifact/launcher evidence path instead.
    """

    adapted = adapt_single_qkv(
        q, k, v, mode=AttentionMode.SINGLE_PREFILL, kv_layout=kv_layout
    )
    if (
        not isinstance(window_left, int)
        or isinstance(window_left, bool)
        or window_left < -1
    ):
        raise SchemaError("window_left must be -1 or a non-negative integer")
    mask_code = validate_upstream_mask_mode(mask_mode)
    tmp, output, lse = make_single_jit_buffers(
        adapted.q,
        output_shape=adapted.q.shape[:-1] + (adapted.head_dim_vo,),
        lse_shape=(adapted.metadata.qo_len, adapted.num_qo_heads),
        return_lse=bool(return_lse),
    )
    layout_code = upstream_kv_layout_code(adapted.layout)
    with record_single_jit_protocol_call(
        mode=AttentionMode.SINGLE_PREFILL.value,
        jit_module=jit_module,
        q=adapted.q,
        k=k,
        v=v,
        tmp=tmp,
        output=output,
        lse=lse,
        layout_code=layout_code,
        mask_code=mask_code,
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
            mask_code,
            layout_code,
            window_left,
            *args,
        )
        frozen_output = freeze_jit_buffer(output, "output")
        frozen_lse = None
        if return_lse:
            assert lse is not None
            frozen_lse = freeze_jit_buffer(lse, "LSE")
    return (frozen_output, frozen_lse) if return_lse else frozen_output


def _optional_plan_scalar(value, name):
    return None if value is None else finite_scalar(value, name)


def _canonical_dtype(value, default=None):
    return default if value is None else str(value)


def _require_prefill_forward_matches_plan(
    plan,
    *,
    causal,
    pos_encoding_mode,
    use_fp16_qk_reduction,
    window_left,
    logits_soft_cap,
    sm_scale,
    rope_scale,
    rope_theta,
):
    values = {
        "causal": bool(causal),
        "pos_encoding_mode": parse_pos_encoding_mode(pos_encoding_mode),
        "use_fp16_qk_reduction": bool(use_fp16_qk_reduction),
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


class BatchPrefillWithPagedKVCacheWrapper(HostBatchReferenceWrapper):
    """FlashInfer-compatible paged prefill lifecycle owned by one wrapper."""

    def __init__(
        self,
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=False,
        qo_indptr_buf=None,
        paged_kv_indptr_buf=None,
        paged_kv_indices_buf=None,
        paged_kv_last_page_len_buf=None,
        custom_mask_buf=None,
        mask_indptr_buf=None,
        backend="auto",
        jit_args=None,
        jit_kwargs=None,
    ):
        layout = parse_kv_layout(kv_layout)
        if jit_args is not None or jit_kwargs is not None:
            raise NotImplementedError("custom JIT modules are not implemented")
        if (
            backend == "auto"
            and str(getattr(float_workspace_buffer, "device", "")).split(":", 1)[0]
            == "npu"
            and use_cuda_graph
        ):
            raise NotImplementedError(
                "provider graph resources are not bound to paged prefill"
            )
        fixed_batch_size = None
        graph_buffers = ()
        if use_cuda_graph:
            if any(
                item is None
                for item in (
                    qo_indptr_buf,
                    paged_kv_indptr_buf,
                    paged_kv_indices_buf,
                    paged_kv_last_page_len_buf,
                )
            ):
                raise ValueError("graph mode requires all paged metadata buffers")
            qo_buffer = validate_graph_buffer(
                qo_indptr_buf, "qo_indptr_buf", dtype="int32"
            )
            if qo_buffer.shape[0] < 2:
                raise ValueError("qo_indptr_buf must define a positive batch size")
            fixed_batch_size = qo_buffer.shape[0] - 1
            graph_buffers = (
                qo_buffer,
                validate_graph_buffer(
                    paged_kv_indptr_buf,
                    "paged_kv_indptr_buf",
                    dtype="int32",
                    length=fixed_batch_size + 1,
                ),
                validate_graph_buffer(
                    paged_kv_indices_buf,
                    "paged_kv_indices_buf",
                    dtype="int32",
                ),
                validate_graph_buffer(
                    paged_kv_last_page_len_buf,
                    "paged_kv_last_page_len_buf",
                    dtype="int32",
                    length=fixed_batch_size,
                ),
            )
            if (custom_mask_buf is None) != (mask_indptr_buf is None):
                raise ValueError(
                    "custom_mask_buf and mask_indptr_buf must be provided together"
                )
            if custom_mask_buf is not None:
                graph_buffers += (
                    validate_graph_buffer(
                        custom_mask_buf, "custom_mask_buf", dtype="uint8"
                    ),
                    validate_graph_buffer(
                        mask_indptr_buf,
                        "mask_indptr_buf",
                        dtype="int32",
                        length=fixed_batch_size + 1,
                    ),
                )
        self._init_host_wrapper(
            mode=AttentionMode.BATCH_PREFILL_PAGED,
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
        self._graph_indices_capacity = (
            graph_buffers[2].shape[0] if graph_buffers else None
        )
        self._graph_mask_capacity = (
            graph_buffers[4].shape[0] if len(graph_buffers) > 4 else None
        )

    def workspace_size(
        self,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        head_dim_vo=None,
        custom_mask=None,
        packed_custom_mask=None,
        causal=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        sm_scale=None,
        window_left=-1,
        logits_soft_cap=None,
        rope_scale=None,
        rope_theta=None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        token_pos_in_items_len=0,
        max_item_len_ptr=None,
        seq_lens=None,
        seq_lens_q=None,
        block_tables=None,
        max_token_per_sequence=None,
        max_sequence_kv=None,
        fixed_split_size=None,
        disable_split_kv=False,
    ):
        """Return caller-workspace bytes without mutating this wrapper's plan."""

        probe = (
            self._provider_workspace_query_probe()
            if self._operator_runtime is not None
            else BatchPrefillWithPagedKVCacheWrapper(
                self._float_workspace_buffer,
                kv_layout=self._kv_layout.value,
                backend="reference",
            )
        )
        probe.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            head_dim_vo=head_dim_vo,
            custom_mask=custom_mask,
            packed_custom_mask=packed_custom_mask,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            non_blocking=False,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            seq_lens=seq_lens,
            seq_lens_q=seq_lens_q,
            block_tables=block_tables,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        return probe.workspace_contract.required_sizes

    def plan(
        self,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        head_dim_vo=None,
        custom_mask=None,
        packed_custom_mask=None,
        causal=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        sm_scale=None,
        window_left=-1,
        logits_soft_cap=None,
        rope_scale=None,
        rope_theta=None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        non_blocking=True,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        token_pos_in_items_len=0,
        max_item_len_ptr=None,
        seq_lens=None,
        seq_lens_q=None,
        block_tables=None,
        max_token_per_sequence=None,
        max_sequence_kv=None,
        fixed_split_size=None,
        disable_split_kv=False,
    ):
        del non_blocking
        if self._operator_runtime is not None:
            if fixed_split_size is not None or disable_split_kv:
                raise NotImplementedError(
                    "provider split-K planning controls are not bound"
                )
            if use_fp16_qk_reduction:
                raise NotImplementedError(
                    "provider FP16 QK reduction is not bound"
                )
        del fixed_split_size, disable_split_kv
        require_no_host_plan_extensions(
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            seq_lens=seq_lens,
            seq_lens_q=seq_lens_q,
            block_tables=block_tables,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
        )
        index_reader = (
            framework_index_values
            if self._operator_runtime is not None
            else reference_index_values
        )
        qo_values = index_reader(qo_indptr, "qo_indptr")
        page_indptr = index_reader(
            paged_kv_indptr, "paged_kv_indptr"
        )
        page_indices = index_reader(
            paged_kv_indices, "paged_kv_indices"
        )
        last_page = index_reader(
            paged_kv_last_page_len, "paged_kv_last_page_len"
        )
        paged = PagedKVMetadata(page_indptr, page_indices, last_page, page_size)
        metadata = PagedPrefillMetadata(qo_values, paged)
        if (
            self._graph_indices_capacity is not None
            and len(page_indices) > self._graph_indices_capacity
        ):
            raise ValueError("paged_kv_indices exceed graph buffer capacity")
        segment_sizes = tuple(
            q_len * kv_len
            for q_len, kv_len in zip(
                metadata.qo_lengths, metadata.paged_kv.sequence_lengths
            )
        )
        if self._operator_runtime is not None:
            if custom_mask is not None or packed_custom_mask is not None:
                raise NotImplementedError(
                    "provider custom-mask plan binding is not implemented"
                )
            mask_spec, mask_data = None, None
        else:
            mask_spec, mask_data = adapt_batch_custom_mask(
                custom_mask,
                packed_custom_mask,
                segment_sizes=segment_sizes,
                device=self._float_workspace_buffer.device,
            )
        if (
            mask_spec is not None
            and self.is_graph_enabled
            and self._graph_mask_capacity is None
        ):
            raise ValueError("graph custom mask requires constructor mask buffers")
        if (
            mask_spec is not None
            and self._graph_mask_capacity is not None
            and sum((size + 7) // 8 for size in segment_sizes)
            > self._graph_mask_capacity
        ):
            raise ValueError("custom mask exceeds graph buffer capacity")
        vo_dim = head_dim_qk if head_dim_vo is None else head_dim_vo
        q_dtype = _canonical_dtype(q_data_type)
        kv_dtype, kv_quant_spec = canonicalize_kv_dtype(kv_data_type, q_dtype)
        if self._operator_runtime is not None and kv_quant_spec is None:
            kv_dtype, kv_quant_spec = canonicalize_flashinfer_paged_kv_dtype(
                kv_dtype, q_dtype
            )
        if (
            self._operator_runtime is not None
            and kv_quant_spec is None
            and kv_dtype in {"float8_e4m3fn", "float8_e5m2"}
        ):
            kv_quant_spec = fp8_per_tensor_quant_spec(kv_dtype)
        o_dtype = _canonical_dtype(o_data_type, q_dtype)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_PAGED,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=vo_dim,
            kv_layout=self._kv_layout,
            causal=bool(causal),
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
            rope_scale=_optional_plan_scalar(rope_scale, "rope_scale"),
            rope_theta=_optional_plan_scalar(rope_theta, "rope_theta"),
            use_fp16_qk_reduction=bool(use_fp16_qk_reduction),
            custom_mask=mask_spec,
        )
        self._commit_plan(spec, metadata, mask_data)

    begin_forward = plan

    def forward(
        self,
        q,
        paged_kv_cache,
        causal=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
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
        _require_prefill_forward_matches_plan(
            plan,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return self.run(
            q,
            paged_kv_cache,
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
        kv_cache_sf=None,
        skip_softmax_threshold_scale_factor=None,
        use_fp16_softmax=None,
        uses_spcompress=None,
    ):
        plan = self.plan_state
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
            if use_fp16_softmax not in (None, False):
                raise NotImplementedError(
                    "provider FP16-softmax run binding is not implemented"
                )
            if uses_spcompress not in (None, False):
                raise NotImplementedError(
                    "provider SP-compression run binding is not implemented"
                )
            provider_kv_cache = paged_kv_cache
            quant_spec = plan.spec.kv_quant_spec
            if (
                quant_spec is not None
                and quant_spec.storage_dtype
                in {"float8_e4m3fn", "float8_e5m2"}
                and quant_spec.fingerprint
                == fp8_per_tensor_quant_spec(
                    quant_spec.storage_dtype
                ).fingerprint
            ):
                provider_kv_cache = canonicalize_framework_paged_fp8_kv_input(
                    paged_kv_cache,
                    quant_spec,
                    device=self._float_workspace_buffer.device,
                )
            result = self._operator_runtime.run(
                q,
                provider_kv_cache,
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
            page_size=plan.metadata.paged_kv.page_size,
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
            q_scale=adapt_head_scale(
                q_scale,
                name="q_scale",
                num_heads=plan.spec.num_qo_heads,
                device=self._float_workspace_buffer.device,
            ),
            k_scale=adapt_head_scale(
                k_scale,
                name="k_scale",
                num_heads=plan.spec.num_kv_heads,
                device=self._float_workspace_buffer.device,
            ),
            v_scale=adapt_head_scale(
                v_scale,
                name="v_scale",
                num_heads=plan.spec.num_kv_heads,
                device=self._float_workspace_buffer.device,
            ),
            out=out,
            lse=lse,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            window_left=window_left,
            sinks=sinks,
            kv_cache_sf=kv_cache_sf,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            use_fp16_softmax=use_fp16_softmax,
            uses_spcompress=uses_spcompress,
        )

    def run_return_lse(self, *args, **kwargs):
        kwargs["return_lse"] = True
        return self.run(*args, **kwargs)

    def end_forward(self):
        warnings.warn("end_forward is deprecated and has no effect", DeprecationWarning)


class BatchPrefillWithRaggedKVCacheWrapper(HostBatchReferenceWrapper):
    """FlashInfer-compatible ragged prefill lifecycle owned by one wrapper."""

    def __init__(
        self,
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=False,
        qo_indptr_buf=None,
        kv_indptr_buf=None,
        custom_mask_buf=None,
        mask_indptr_buf=None,
        backend="auto",
        jit_args=None,
        jit_kwargs=None,
    ):
        layout = parse_kv_layout(kv_layout)
        if jit_args is not None or jit_kwargs is not None:
            raise NotImplementedError("custom JIT modules are not implemented")
        if (
            backend == "auto"
            and str(getattr(float_workspace_buffer, "device", "")).split(":", 1)[0]
            == "npu"
            and use_cuda_graph
        ):
            raise NotImplementedError(
                "provider graph resources are not bound to ragged prefill"
            )
        fixed_batch_size = None
        graph_buffers = ()
        if use_cuda_graph:
            if qo_indptr_buf is None or kv_indptr_buf is None:
                raise ValueError("graph mode requires qo_indptr_buf and kv_indptr_buf")
            qo_buffer = validate_graph_buffer(
                qo_indptr_buf, "qo_indptr_buf", dtype="int32"
            )
            if qo_buffer.shape[0] < 2:
                raise ValueError("qo_indptr_buf must define a positive batch size")
            fixed_batch_size = qo_buffer.shape[0] - 1
            graph_buffers = (
                qo_buffer,
                validate_graph_buffer(
                    kv_indptr_buf,
                    "kv_indptr_buf",
                    dtype="int32",
                    length=fixed_batch_size + 1,
                ),
            )
            if (custom_mask_buf is None) != (mask_indptr_buf is None):
                raise ValueError(
                    "custom_mask_buf and mask_indptr_buf must be provided together"
                )
            if custom_mask_buf is not None:
                graph_buffers += (
                    validate_graph_buffer(
                        custom_mask_buf, "custom_mask_buf", dtype="uint8"
                    ),
                    validate_graph_buffer(
                        mask_indptr_buf,
                        "mask_indptr_buf",
                        dtype="int32",
                        length=fixed_batch_size + 1,
                    ),
                )
        self._init_host_wrapper(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
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
        self._graph_mask_capacity = (
            graph_buffers[2].shape[0] if len(graph_buffers) > 2 else None
        )

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=None,
        custom_mask=None,
        packed_custom_mask=None,
        causal=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        window_left=-1,
        logits_soft_cap=None,
        sm_scale=None,
        rope_scale=None,
        rope_theta=None,
        q_data_type="float16",
        kv_data_type=None,
        o_data_type=None,
        non_blocking=True,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        token_pos_in_items_len=0,
        max_item_len_ptr=None,
        fixed_split_size=None,
        disable_split_kv=False,
        seq_lens=None,
        seq_lens_q=None,
        max_token_per_sequence=None,
        max_sequence_kv=None,
        v_indptr=None,
        o_indptr=None,
    ):
        del non_blocking
        if self._operator_runtime is not None:
            if fixed_split_size is not None or disable_split_kv:
                raise NotImplementedError(
                    "provider split-K planning controls are not bound"
                )
            if use_fp16_qk_reduction:
                raise NotImplementedError(
                    "provider FP16 QK reduction is not bound"
                )
        del fixed_split_size, disable_split_kv
        require_no_host_plan_extensions(
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            seq_lens=seq_lens,
            seq_lens_q=seq_lens_q,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            v_indptr=v_indptr,
            o_indptr=o_indptr,
        )
        index_reader = (
            framework_index_values
            if self._operator_runtime is not None
            else reference_index_values
        )
        metadata = RaggedKVMetadata(
            index_reader(qo_indptr, "qo_indptr"),
            index_reader(kv_indptr, "kv_indptr"),
        )
        segment_sizes = tuple(
            q_len * kv_len
            for q_len, kv_len in zip(metadata.qo_lengths, metadata.kv_lengths)
        )
        if self._operator_runtime is not None:
            if custom_mask is not None or packed_custom_mask is not None:
                raise NotImplementedError(
                    "provider custom-mask plan binding is not implemented"
                )
            mask_spec, mask_data = None, None
        else:
            mask_spec, mask_data = adapt_batch_custom_mask(
                custom_mask,
                packed_custom_mask,
                segment_sizes=segment_sizes,
                device=self._float_workspace_buffer.device,
            )
        if (
            mask_spec is not None
            and self.is_graph_enabled
            and self._graph_mask_capacity is None
        ):
            raise ValueError("graph custom mask requires constructor mask buffers")
        if (
            mask_spec is not None
            and self._graph_mask_capacity is not None
            and sum((size + 7) // 8 for size in segment_sizes)
            > self._graph_mask_capacity
        ):
            raise ValueError("custom mask exceeds graph buffer capacity")
        vo_dim = head_dim_qk if head_dim_vo is None else head_dim_vo
        q_dtype = _canonical_dtype(q_data_type)
        kv_dtype, kv_quant_spec = canonicalize_kv_dtype(kv_data_type, q_dtype)
        o_dtype = _canonical_dtype(o_data_type, q_dtype)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=vo_dim,
            kv_layout=self._kv_layout,
            causal=bool(causal),
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
            rope_scale=_optional_plan_scalar(rope_scale, "rope_scale"),
            rope_theta=_optional_plan_scalar(rope_theta, "rope_theta"),
            use_fp16_qk_reduction=bool(use_fp16_qk_reduction),
            custom_mask=mask_spec,
        )
        self._commit_plan(spec, metadata, mask_data)

    begin_forward = plan

    def forward(
        self,
        q,
        k,
        v,
        causal=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
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
        _require_prefill_forward_matches_plan(
            plan,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return self.run(q, k, v)

    def run(
        self,
        q,
        k,
        v,
        *args,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        o_scale=None,
        out=None,
        lse=None,
        return_lse=False,
        enable_pdl=None,
        kv_cache_sf=None,
    ):
        plan = self.plan_state
        if self._operator_runtime is not None:
            if args:
                raise NotImplementedError(
                    "provider custom JIT run arguments are not implemented"
                )
            if q_scale is not None and plan.spec.kv_quant_spec is None:
                raise NotImplementedError(
                    "provider query-scale binding requires quantized K/V"
                )
            if o_scale is not None and plan.spec.kv_quant_spec is None:
                raise NotImplementedError(
                    "provider output-scale binding requires quantized K/V"
                )
            if enable_pdl not in (None, False):
                raise NotImplementedError(
                    "enable_pdl has no authorized Ascend provider binding"
                )
            if plan.spec.kv_quant_spec is not None:
                provider_kv_input = combine_attention_operator_quantized_kv_input(
                    k, v
                )
            else:
                if isinstance(
                    k, AttentionOperatorQuantizedTensorInput
                ) or isinstance(v, AttentionOperatorQuantizedTensorInput):
                    raise SchemaError(
                        "dense provider plan cannot consume quantized tensor inputs"
                    )
                provider_kv_input = (k, v)
            result = self._operator_runtime.run(
                q,
                provider_kv_input,
                return_lse=bool(return_lse) or lse is not None,
                out=out,
                lse=lse,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                o_scale=o_scale,
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
        if o_scale is not None:
            raise NotImplementedError(
                "ragged o_scale requires an authorized provider output-scale binding"
            )
        kv_data = adapt_ragged_kv_data(
            k,
            v,
            total_kv_tokens=plan.metadata.total_kv_tokens,
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
            q_scale=adapt_head_scale(
                q_scale,
                name="q_scale",
                num_heads=plan.spec.num_qo_heads,
                device=self._float_workspace_buffer.device,
            ),
            k_scale=adapt_head_scale(
                k_scale,
                name="k_scale",
                num_heads=plan.spec.num_kv_heads,
                device=self._float_workspace_buffer.device,
            ),
            v_scale=adapt_head_scale(
                v_scale,
                name="v_scale",
                num_heads=plan.spec.num_kv_heads,
                device=self._float_workspace_buffer.device,
            ),
            out=out,
            lse=lse,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            window_left=None,
            sinks=None,
            kv_cache_sf=kv_cache_sf,
            skip_softmax_threshold_scale_factor=None,
        )

    def run_return_lse(self, *args, **kwargs):
        kwargs["return_lse"] = True
        return self.run(*args, **kwargs)

    def end_forward(self):
        warnings.warn("end_forward is deprecated and has no effect", DeprecationWarning)


__all__ = [
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "single_prefill_with_kv_cache",
    "single_prefill_with_kv_cache_return_lse",
    "single_prefill_with_kv_cache_with_jit_module",
]
