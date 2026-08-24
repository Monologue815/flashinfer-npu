"""Deterministic micro-corpus covering the P0 Attention framework cells."""

from __future__ import annotations

import math

from flashinfer_npu.runtime import QuantSpec

from .corpus import AttentionTraceCase, AttentionTraceCorpus
from .reference import (
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
)
from .schema import (
    AttentionMode,
    AttentionPlanSpec,
    CustomMaskSpec,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    SingleAttentionMetadata,
)
from .trace import AttentionTrace


def _tensor(value, dtype="float32"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device="cpu")


def _flat(shape, values, dtype="float32"):
    return ReferenceTensor(tuple(shape), tuple(values), dtype=dtype, device="cpu")


def _quantized(
    shape,
    storage,
    scale,
    spec,
    *,
    zero_point=None,
    storage_shape=None,
):
    return ReferenceQuantizedTensor(
        logical_shape=tuple(shape),
        storage=_flat(
            storage_shape or shape,
            storage,
            "uint8"
            if spec.storage_dtype in {"int4_packed", "uint4_packed"}
            else spec.storage_dtype,
        ),
        scale=_tensor(scale, spec.scale_dtype),
        zero_point=(
            _tensor(zero_point, "int32") if zero_point is not None else None
        ),
        quant_spec=spec,
    )


def _single_decode_dense() -> AttentionTraceCase:
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_DECODE,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
    )
    cache = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(1, 2),
        q=_tensor([[1.0]]),
        kv_data=ReferenceKVData(
            cache,
            (_tensor([[[1.0]], [[0.0]]]), _tensor([[[2.0]], [[4.0]]])),
        ),
    )
    return AttentionTraceCase("single_decode_dense", trace)


def _single_prefill_int4_rope() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int4_packed",
        compute_dtype="float32",
        accumulator_dtype="float32",
        packing_order="low_nibble_first",
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=2,
        kv_layout=KVLayout.HND,
        causal=True,
        pos_encoding_mode=PosEncodingMode.ROPE_LLAMA,
        q_dtype="float32",
        kv_dtype="int4_packed",
        kv_quant_spec=quant,
    )
    cache = RaggedKVCacheSpec(
        2, 1, 2, 2, "int4_packed", KVLayout.HND, "cpu", quant
    )
    key = _quantized(
        (1, 2, 2), (0x01, 0x10), 1.0, quant, storage_shape=(1, 2, 1)
    )
    value = _quantized(
        (1, 2, 2), (0x42, 0x76), 1.0, quant, storage_shape=(1, 2, 1)
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(2, 2),
        q=_tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ),
        kv_data=ReferenceQuantizedKVData(cache, key, value),
    )
    return AttentionTraceCase("single_prefill_int4_hnd_rope", trace)


def _paged_prefill_int8_mask() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
    )
    paged = PagedKVMetadata((0, 1), (0,), (2,), 2)
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="int8",
        kv_quant_spec=quant,
        custom_mask=CustomMaskSpec(2),
    )
    cache = PagedKVCacheSpec(
        1, 2, 1, 1, 1, "int8", structure="separate", device="cpu", quant_spec=quant
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=PagedPrefillMetadata((0, 1), paged),
        q=_tensor([[[0.0]]]),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized((1, 2, 1, 1), (0, 0), 1.0, quant),
            _quantized((1, 2, 1, 1), (2, 6), 2.0, quant),
        ),
        custom_mask_data=(True, False),
    )
    return AttentionTraceCase("paged_prefill_int8_mask", trace)


def _ragged_prefill_uint8_group_packed() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="asymmetric",
        storage_dtype="uint8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        granularity="group",
        group_size=(1, 1, 1),
        axis=(0, 1, 2),
        has_zero_point=True,
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_RAGGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=1,
        kv_layout=KVLayout.HND,
        q_dtype="float32",
        kv_dtype="uint8",
        kv_quant_spec=quant,
        custom_mask=CustomMaskSpec(1, packed=True),
    )
    cache = RaggedKVCacheSpec(
        2, 1, 1, 1, "uint8", KVLayout.HND, "cpu", quant
    )
    scale = [[[0.5], [0.5]]]
    zero = [[[1], [1]]]
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=RaggedKVMetadata((0, 1), (0, 2)),
        q=_tensor([[[1.0], [2.0]]]),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized((1, 2, 1), (2, 0), scale, quant, zero_point=zero),
            _quantized((1, 2, 1), (3, 5), scale, quant, zero_point=zero),
        ),
        custom_mask_data=(0x03,),
    )
    return AttentionTraceCase("ragged_prefill_uint8_group_packed", trace)


def _paged_decode_int8_token_alibi_empty() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        granularity="token",
        axis=(2,),
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_DECODE_PAGED,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        kv_layout=KVLayout.HND,
        pos_encoding_mode=PosEncodingMode.ALIBI,
        q_dtype="float32",
        kv_dtype="int8",
        kv_quant_spec=quant,
    )
    cache = PagedKVCacheSpec(
        1, 1, 1, 1, 1, "int8", KVLayout.HND, "separate", "cpu", quant
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=PagedKVMetadata((0, 0, 1), (0,), (0, 1), 1),
        q=_tensor([[[0.0]], [[1.0]]]),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized((1, 1, 1, 1), (1,), [0.5], quant),
            _quantized((1, 1, 1, 1), (4,), [0.5], quant),
        ),
    )
    return AttentionTraceCase("paged_decode_int8_token_alibi_empty", trace)


def _paged_prefill_int4_multi_request_shared_page() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int4_packed",
        compute_dtype="float32",
        accumulator_dtype="float32",
        packing_order="low_nibble_first",
    )
    paged = PagedKVMetadata(
        indptr=(0, 1, 3),
        indices=(1, 0, 1),
        last_page_len=(2, 1),
        page_size=2,
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=3,
        head_dim_vo=3,
        causal=True,
        q_dtype="float32",
        kv_dtype="int4_packed",
        kv_quant_spec=quant,
    )
    cache = PagedKVCacheSpec(
        2,
        2,
        1,
        3,
        3,
        "int4_packed",
        structure="separate",
        device="cpu",
        quant_spec=quant,
    )
    # Every logical vector has width 3, so byte 2 contains a zero padding
    # nibble. Page 1 is referenced by both requests to exercise page sharing.
    key_storage = (0x01, 0x0F, 0x10, 0x00, 0x11, 0x01, 0x0F, 0x01)
    value_storage = (0x32, 0x04, 0x54, 0x06, 0x21, 0x03, 0x43, 0x05)
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=PagedPrefillMetadata((0, 1, 3), paged),
        q=_tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ]
        ),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized(
                (2, 2, 1, 3),
                key_storage,
                0.5,
                quant,
                storage_shape=(2, 2, 1, 2),
            ),
            _quantized(
                (2, 2, 1, 3),
                value_storage,
                0.5,
                quant,
                storage_shape=(2, 2, 1, 2),
            ),
        ),
    )
    return AttentionTraceCase(
        "paged_prefill_int4_multi_request_shared_page", trace
    )


def _paged_decode_int8_group_gqa_distinct_dims() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        granularity="group",
        axis=(0, 1, 2, 3),
        group_size=(1, 1, 1, 2),
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_DECODE_PAGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=3,
        head_dim_vo=2,
        q_dtype="float32",
        kv_dtype="int8",
        kv_quant_spec=quant,
    )
    cache = PagedKVCacheSpec(
        2,
        2,
        1,
        3,
        2,
        "int8",
        structure="separate",
        device="cpu",
        quant_spec=quant,
    )
    key_scale = [
        [[ [0.5, 0.25] ], [ [0.5, 0.25] ]],
        [[ [1.0, 0.5] ], [ [1.0, 0.5] ]],
    ]
    value_scale = [
        [[ [0.5] ], [ [0.5] ]],
        [[ [1.0] ], [ [1.0] ]],
    ]
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=PagedKVMetadata((0, 1, 2), (1, 0), (2, 1), 2),
        q=_tensor(
            [
                [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]],
            ]
        ),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized(
                (2, 2, 1, 3),
                (2, 0, 4, 0, 2, 2, 1, 1, 0, 2, -1, 2),
                key_scale,
                quant,
            ),
            _quantized(
                (2, 2, 1, 2),
                (2, 4, 6, 8, 1, 3, 5, 7),
                value_scale,
                quant,
            ),
        ),
    )
    return AttentionTraceCase(
        "paged_decode_int8_group_gqa_distinct_dims", trace
    )


def _mixed_dense_distinct_dims() -> AttentionTraceCase:
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=2,
        head_dim_vo=1,
        q_dtype="float32",
        kv_dtype="float32",
    )
    cache = PagedKVCacheSpec(
        1, 1, 1, 2, 1, "float32", structure="separate", device="cpu"
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=MixedPagedKVMetadata((0, 1), (0, 1), (0,), (1,), 1),
        q=_tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        kv_data=ReferenceKVData(
            cache,
            (_tensor([[[[1.0, 1.0]]]]), _tensor([[[[3.0]]]])),
        ),
    )
    return AttentionTraceCase("mixed_dense_distinct_dims", trace)


def _mixed_int4_window_softcap_per_head_scale() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int4_packed",
        compute_dtype="float32",
        accumulator_dtype="float32",
        packing_order="low_nibble_first",
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=3,
        head_dim_vo=3,
        causal=True,
        window_left=1,
        logits_soft_cap=2.0,
        q_dtype="float32",
        kv_dtype="int4_packed",
        kv_quant_spec=quant,
    )
    cache = PagedKVCacheSpec(
        2,
        2,
        2,
        3,
        3,
        "int4_packed",
        structure="separate",
        device="cpu",
        quant_spec=quant,
    )
    logical_shape = (2, 2, 2, 3)
    storage_shape = (2, 2, 2, 2)
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=MixedPagedKVMetadata(
            (0, 1, 2),
            (0, 2, 4),
            (0, 1, 1, 1),
            (3, 4),
            2,
        ),
        q=_flat((2, 4, 3), (0.0,) * 24),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized(
                logical_shape,
                (0x00, 0x00) * 8,
                0.5,
                quant,
                storage_shape=storage_shape,
            ),
            _quantized(
                logical_shape,
                (0x11, 0x01) * 8,
                1.25,
                quant,
                storage_shape=storage_shape,
            ),
        ),
        k_scale=(0.75, 1.0),
        v_scale=(1.5, 1.25),
        logits_soft_cap=1.0,
    )
    return AttentionTraceCase(
        "mixed_int4_window_softcap_per_head_scale", trace
    )


def _mixed_asymmetric_uint8_channel_per_head_scale() -> AttentionTraceCase:
    quant = QuantSpec(
        scheme="asymmetric",
        storage_dtype="uint8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        granularity="channel",
        axis=(1,),
        has_zero_point=True,
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=1,
        head_dim_vo=1,
        kv_layout=KVLayout.HND,
        causal=True,
        logits_soft_cap=2.0,
        q_dtype="float32",
        kv_dtype="uint8",
        kv_quant_spec=quant,
    )
    cache = PagedKVCacheSpec(
        2,
        2,
        2,
        1,
        1,
        "uint8",
        KVLayout.HND,
        "separate",
        "cpu",
        quant,
    )
    logical_shape = (2, 2, 2, 1)
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=MixedPagedKVMetadata(
            (0, 1, 2),
            (0, 2, 4),
            (0, 1, 1, 1),
            (3, 4),
            2,
        ),
        q=_flat((2, 4, 1), (0.0,) * 8),
        kv_data=ReferenceQuantizedKVData(
            cache,
            _quantized(
                logical_shape,
                (3, 3, 9, 9, 3, 3, 9, 9),
                (0.25, 0.75),
                quant,
                zero_point=(3, 9),
            ),
            _quantized(
                logical_shape,
                (3, 5, 4, 1, 3, 5, 4, 1),
                (0.5, 2.0),
                quant,
                zero_point=(1, 2),
            ),
        ),
        k_scale=(0.75, 1.25),
        v_scale=(2.0, 0.5),
        logits_soft_cap=1.0,
    )
    return AttentionTraceCase(
        "mixed_asymmetric_uint8_channel_per_head_scale", trace
    )


def _single_prefill_all_mask() -> AttentionTraceCase:
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
        custom_mask=CustomMaskSpec(2),
    )
    cache = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(1, 2),
        q=_tensor([[[1.0]]]),
        kv_data=ReferenceKVData(
            cache,
            (_tensor([[[1.0]], [[2.0]]]), _tensor([[[3.0]], [[4.0]]])),
        ),
        custom_mask_data=(False, False),
    )
    return AttentionTraceCase("single_prefill_all_mask", trace)


def _single_prefill_nan_logit() -> AttentionTraceCase:
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
    )
    cache = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(1, 2),
        q=_tensor([[[1.0]]]),
        kv_data=ReferenceKVData(
            cache,
            (
                _tensor([[[math.nan]], [[1.0]]]),
                _tensor([[[2.0]], [[6.0]]]),
            ),
        ),
    )
    return AttentionTraceCase("single_prefill_nan_logit", trace)


def _single_prefill_positive_infinity_logits() -> AttentionTraceCase:
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
        sm_scale=1.0,
    )
    cache = RaggedKVCacheSpec(3, 1, 1, 1, "float32", device="cpu")
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(1, 3),
        q=_tensor([[[1e308]]]),
        kv_data=ReferenceKVData(
            cache,
            (
                _tensor([[[1e308]], [[1.0]], [[1e308]]]),
                _tensor([[[2.0]], [[math.inf]], [[6.0]]]),
            ),
        ),
    )
    return AttentionTraceCase("single_prefill_positive_infinity_logits", trace)


def _single_prefill_negative_infinity_row() -> AttentionTraceCase:
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
        sm_scale=1.0,
    )
    cache = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(1, 2),
        q=_tensor([[[1e308]]]),
        kv_data=ReferenceKVData(
            cache,
            (
                _tensor([[[-1e308]], [[-1e308]]]),
                _tensor([[[2.0]], [[6.0]]]),
            ),
        ),
    )
    return AttentionTraceCase("single_prefill_negative_infinity_row", trace)


def build_framework_attention_corpus() -> AttentionTraceCorpus:
    """Return the deterministic P0 framework smoke corpus."""

    return AttentionTraceCorpus(
        name="attention-framework-smoke-v4",
        description="Deterministic micro-cases for every core Attention mode.",
        cases=(
            _single_decode_dense(),
            _single_prefill_int4_rope(),
            _paged_prefill_int8_mask(),
            _ragged_prefill_uint8_group_packed(),
            _paged_decode_int8_token_alibi_empty(),
            _paged_prefill_int4_multi_request_shared_page(),
            _paged_decode_int8_group_gqa_distinct_dims(),
            _mixed_dense_distinct_dims(),
            _mixed_int4_window_softcap_per_head_scale(),
            _mixed_asymmetric_uint8_channel_per_head_scale(),
            _single_prefill_all_mask(),
            _single_prefill_nan_logit(),
            _single_prefill_positive_infinity_logits(),
            _single_prefill_negative_infinity_row(),
        ),
    )


__all__ = ["build_framework_attention_corpus"]
