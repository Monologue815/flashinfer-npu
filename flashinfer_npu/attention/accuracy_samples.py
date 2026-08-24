"""Deterministic quantization-accuracy cases for the Attention framework."""

from __future__ import annotations

from flashinfer_npu.runtime import QuantSpec

from .accuracy import (
    AttentionAccuracyBudget,
    AttentionAccuracyCase,
    AttentionAccuracyCorpus,
    AttentionErrorTolerance,
)
from .reference import (
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
)
from .schema import (
    AttentionMode,
    AttentionPlanSpec,
    RaggedKVCacheSpec,
    SingleAttentionMetadata,
)
from .trace import AttentionTrace


def _tensor(value, dtype="float32"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device="cpu")


def _pair(
    quant: QuantSpec,
    dense_key: ReferenceTensor,
    dense_value: ReferenceTensor,
    quantized_key: ReferenceQuantizedTensor,
    quantized_value: ReferenceQuantizedTensor,
):
    qk_dim = dense_key.shape[-1]
    vo_dim = dense_value.shape[-1]
    common = dict(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=qk_dim,
        head_dim_vo=vo_dim,
        q_dtype="float32",
        o_dtype="float32",
        sm_scale=1.0,
    )
    dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
    quantized_spec = AttentionPlanSpec(
        kv_dtype=quant.storage_dtype, kv_quant_spec=quant, **common
    )
    metadata = SingleAttentionMetadata(1, 2)
    q = ReferenceTensor.zeros((1, 1, qk_dim), device="cpu")
    dense_cache = RaggedKVCacheSpec(
        2, 1, qk_dim, vo_dim, "float32", device="cpu"
    )
    quantized_cache = RaggedKVCacheSpec(
        2,
        1,
        qk_dim,
        vo_dim,
        quant.storage_dtype,
        device="cpu",
        quant_spec=quant,
    )
    dense_trace = AttentionTrace.capture(
        spec=dense_spec,
        metadata=metadata,
        q=q,
        kv_data=ReferenceKVData(dense_cache, (dense_key, dense_value)),
    )
    quantized_trace = AttentionTrace.capture(
        spec=quantized_spec,
        metadata=metadata,
        q=q,
        kv_data=ReferenceQuantizedKVData(
            quantized_cache, quantized_key, quantized_value
        ),
    )
    return dense_trace, quantized_trace


def _exact_int8() -> AttentionAccuracyCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
    )
    scale = _tensor(1.0)
    dense, quantized = _pair(
        quant,
        _tensor([[[0.0]], [[0.0]]]),
        _tensor([[[1.0]], [[3.0]]]),
        ReferenceQuantizedTensor(
            (2, 1, 1), _tensor([[[0]], [[0]]], "int8"), scale, quant
        ),
        ReferenceQuantizedTensor(
            (2, 1, 1), _tensor([[[1]], [[3]]], "int8"), scale, quant
        ),
    )
    return AttentionAccuracyCase(
        "exact_int8",
        dense,
        quantized,
        AttentionAccuracyBudget(),
        True,
        "Exact INT8 representation must have zero quantization drift.",
    )


def _lossy_asymmetric_uint8() -> AttentionAccuracyCase:
    quant = QuantSpec(
        scheme="asymmetric",
        storage_dtype="uint8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        has_zero_point=True,
    )
    scale = _tensor(0.25)
    zero = _tensor(128, "int32")
    dense, quantized = _pair(
        quant,
        _tensor([[[0.0]], [[0.0]]]),
        _tensor([[[0.3]], [[1.1]]]),
        ReferenceQuantizedTensor(
            (2, 1, 1),
            _tensor([[[128]], [[128]]], "uint8"),
            scale,
            quant,
            zero,
        ),
        ReferenceQuantizedTensor(
            (2, 1, 1),
            _tensor([[[129]], [[132]]], "uint8"),
            scale,
            quant,
            zero,
        ),
    )
    return AttentionAccuracyCase(
        "lossy_asymmetric_uint8",
        dense,
        quantized,
        AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(atol=0.08)
        ),
        True,
        "Asymmetric UINT8 rounding is accepted only by its declared budget.",
    )


def _lossy_packed_int4_odd_dimension() -> AttentionAccuracyCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int4_packed",
        compute_dtype="float32",
        accumulator_dtype="float32",
        packing_order="low_nibble_first",
    )
    scale = _tensor(0.5)
    dense, quantized = _pair(
        quant,
        _tensor([[[0.0]], [[0.0]]]),
        _tensor([[[0.2, -0.7, 1.4]], [[1.1, 0.3, -1.2]]]),
        ReferenceQuantizedTensor(
            (2, 1, 1), _tensor([[[0x00]], [[0x00]]], "uint8"), scale, quant
        ),
        ReferenceQuantizedTensor(
            (2, 1, 3),
            _tensor([[[0xF0, 0x03]], [[0x12, 0x0E]]], "uint8"),
            scale,
            quant,
        ),
    )
    return AttentionAccuracyCase(
        "lossy_packed_int4_odd_dimension",
        dense,
        quantized,
        AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(atol=0.21)
        ),
        True,
        "Packed INT4 odd-dimension padding and rounding share one explicit budget.",
    )


def _int8_scale_overflow_rejected() -> AttentionAccuracyCase:
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
    )
    scale = _tensor(1.5e306)
    dense, quantized = _pair(
        quant,
        _tensor([[[0.0]], [[0.0]]]),
        _tensor([[[1e308]], [[1e308]]]),
        ReferenceQuantizedTensor(
            (2, 1, 1), _tensor([[[0]], [[0]]], "int8"), scale, quant
        ),
        ReferenceQuantizedTensor(
            (2, 1, 1), _tensor([[[127]], [[127]]], "int8"), scale, quant
        ),
    )
    return AttentionAccuracyCase(
        "int8_scale_overflow_rejected",
        dense,
        quantized,
        AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(atol=1e308)
        ),
        False,
        "Finite scale that overflows dequantized V must fail despite a large budget.",
    )


def build_attention_accuracy_corpus() -> AttentionAccuracyCorpus:
    return AttentionAccuracyCorpus(
        name="attention-quantization-accuracy-v1",
        description=(
            "Host-only paired dense/quantized cases separating quantization drift "
            "from future backend execution drift."
        ),
        cases=(
            _exact_int8(),
            _lossy_asymmetric_uint8(),
            _lossy_packed_int4_odd_dimension(),
            _int8_scale_overflow_rejected(),
        ),
    )


__all__ = ["build_attention_accuracy_corpus"]
