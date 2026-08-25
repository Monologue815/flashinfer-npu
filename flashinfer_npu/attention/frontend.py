"""Host-only adapters shared by FlashInfer-compatible Attention facades."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import DispatchError, QuantSpec, SchemaError

from .reference import (
    ReferenceAttentionResult,
    ReferenceBuffer,
    ReferenceKVData,
    ReferenceKVInput,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    ScaleInput,
)
from .schema import (
    AttentionMode,
    CustomMaskSpec,
    KVLayout,
    PagedKVCacheSpec,
    PosEncodingMode,
    RaggedKVCacheSpec,
    SingleAttentionMetadata,
)


MaskInput = Optional[ReferenceTensor]
FrontendScaleInput = Optional[Union[float, Sequence[float], ReferenceTensor]]
PagedKVInput = Union[
    ReferenceTensor,
    Tuple[ReferenceTensor, ReferenceTensor],
    ReferenceQuantizedKVData,
]


@dataclass(frozen=True)
class SingleQKVAdapterResult:
    q: ReferenceTensor
    kv_data: ReferenceKVInput
    metadata: SingleAttentionMetadata
    layout: KVLayout
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int


def require_reference_backend(backend: str) -> None:
    if backend == "auto":
        raise DispatchError(
            "backend='auto' cannot select the Host reference executor; "
            "pass backend='reference' explicitly"
        )
    if backend != "reference":
        raise DispatchError(
            "backend %r is unavailable in the Host-only Attention facade" % backend
        )


def parse_pos_encoding_mode(value: str) -> PosEncodingMode:
    try:
        return PosEncodingMode(value)
    except ValueError as error:
        raise SchemaError(
            "pos_encoding_mode must be NONE, ROPE_LLAMA, or ALIBI"
        ) from error


def parse_kv_layout(value: str) -> KVLayout:
    try:
        return KVLayout(value)
    except ValueError as error:
        raise SchemaError("kv_layout must be NHD or HND") from error


def canonicalize_kv_dtype(value, q_dtype: str) -> Tuple[str, Optional[QuantSpec]]:
    if isinstance(value, QuantSpec):
        return value.storage_dtype, value
    return (q_dtype if value is None else canonicalize_dtype_name(value)), None


def canonicalize_dtype_name(value) -> str:
    """Normalize string and torch-style dtype objects without importing torch."""

    text = str(value)
    if text.startswith("torch."):
        text = text[len("torch.") :]
    aliases = {
        "half": "float16",
        "float": "float32",
        "double": "float64",
    }
    text = aliases.get(text, text)
    if not text:
        raise SchemaError("dtype must be non-empty")
    return text


def framework_index_values(value, name: str) -> Tuple[int, ...]:
    """Read rank-1 plan metadata from reference, sequence, or tensor-like input."""

    if isinstance(value, ReferenceTensor):
        return reference_index_values(value, name)
    dtype = getattr(value, "dtype", None)
    if dtype is not None and canonicalize_dtype_name(dtype) != "int32":
        raise SchemaError("%s dtype must be int32" % name)
    if isinstance(value, (tuple, list)):
        raw = value
    else:
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise TypeError(
                "%s must be an int32 tensor-like value or integer sequence" % name
            )
        raw = tolist()
    if not isinstance(raw, (tuple, list)) or any(
        isinstance(item, (tuple, list)) for item in raw
    ):
        raise SchemaError("%s must be rank 1" % name)
    result = []
    for item in raw:
        if isinstance(item, bool):
            raise SchemaError("%s values must be integers" % name)
        try:
            integer = int(item)
        except (TypeError, ValueError) as error:
            raise SchemaError("%s values must be integers" % name) from error
        if item != integer:
            raise SchemaError("%s values must be integers" % name)
        result.append(integer)
    return tuple(result)


def require_reference_tensor(value, name: str) -> ReferenceTensor:
    if not isinstance(value, ReferenceTensor):
        raise TypeError("%s must be ReferenceTensor in the Host-only facade" % name)
    return value


def reference_index_values(value, name: str) -> Tuple[int, ...]:
    tensor = require_reference_tensor(value, name)
    if len(tensor.shape) != 1:
        raise SchemaError("%s must be rank 1" % name)
    if tensor.dtype != "int32":
        raise SchemaError("%s dtype must be int32" % name)
    result = []
    for item in tensor.data:
        if not math.isfinite(item) or not item.is_integer():
            raise SchemaError("%s values must be integers" % name)
        result.append(int(item))
    return tuple(result)


def validate_workspace_buffer(value, name: str, device: Optional[str] = None):
    tensor = require_reference_tensor(value, name)
    if len(tensor.shape) != 1 or tensor.dtype != "uint8":
        raise SchemaError("%s must be a rank-1 uint8 ReferenceTensor" % name)
    if device is not None and tensor.device != device:
        raise SchemaError("workspace buffers must be on the same device")
    return tensor


def validate_framework_workspace_buffer(
    value, name: str, device: Optional[str] = None
):
    """Validate reference or tensor-like workspace without importing torch."""

    if isinstance(value, ReferenceTensor):
        return validate_workspace_buffer(value, name, device=device)
    shape = getattr(value, "shape", None)
    try:
        shape = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as error:
        raise SchemaError("%s must expose a rank-1 shape" % name) from error
    if len(shape) != 1 or shape[0] < 0:
        raise SchemaError("%s must be rank 1" % name)
    if canonicalize_dtype_name(getattr(value, "dtype", "")) != "uint8":
        raise SchemaError("%s dtype must be uint8" % name)
    actual_device = str(getattr(value, "device", ""))
    if not actual_device:
        raise SchemaError("%s must expose a device" % name)
    if device is not None and actual_device != str(device):
        raise SchemaError("workspace buffers must be on the same device")
    return value


def adapt_batch_custom_mask(
    custom_mask: MaskInput,
    packed_custom_mask: MaskInput,
    *,
    segment_sizes: Sequence[int],
    device: str,
) -> Tuple[Optional[CustomMaskSpec], Optional[Tuple[Union[bool, int], ...]]]:
    if packed_custom_mask is not None:
        tensor = require_reference_tensor(packed_custom_mask, "packed_custom_mask")
        expected = sum((int(size) + 7) // 8 for size in segment_sizes)
        if tensor.shape != (expected,):
            raise SchemaError("packed_custom_mask shape must be (%d,)" % expected)
        if tensor.dtype != "uint8":
            raise SchemaError("packed_custom_mask dtype must be uint8")
        if tensor.device != device:
            raise SchemaError("packed_custom_mask must be on the workspace device")
        values = []
        for item in tensor.data:
            if not item.is_integer() or not 0 <= item <= 255:
                raise SchemaError(
                    "packed_custom_mask values must be integers in [0, 255]"
                )
            values.append(int(item))
        return CustomMaskSpec(expected, packed=True), tuple(values)
    if custom_mask is not None:
        tensor = require_reference_tensor(custom_mask, "custom_mask")
        expected = sum(int(size) for size in segment_sizes)
        if tensor.shape != (expected,):
            raise SchemaError("custom_mask shape must be (%d,)" % expected)
        if tensor.dtype != "bool":
            raise SchemaError("custom_mask dtype must be bool")
        if tensor.device != device:
            raise SchemaError("custom_mask must be on the workspace device")
        if any(item not in (0.0, 1.0) for item in tensor.data):
            raise SchemaError("custom_mask values must be boolean")
        return CustomMaskSpec(expected), tuple(bool(item) for item in tensor.data)
    return None, None


def adapt_paged_kv_data(
    value: PagedKVInput,
    *,
    page_size: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    dtype: str,
    layout: KVLayout,
    quant_spec: Optional[QuantSpec] = None,
) -> ReferenceKVInput:
    if isinstance(value, ReferenceQuantizedKVData):
        spec = value.spec
        expected = (
            page_size,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo,
            dtype,
            layout,
        )
        actual = (
            spec.page_size if isinstance(spec, PagedKVCacheSpec) else None,
            spec.num_kv_heads,
            spec.head_dim_qk,
            spec.head_dim_vo,
            spec.dtype,
            spec.layout,
            spec.quant_spec,
        )
        expected = expected + (quant_spec,)
        if actual != expected:
            raise SchemaError("quantized paged KV data does not match the plan")
        return value
    if quant_spec is not None:
        raise SchemaError("quantized plan requires ReferenceQuantizedKVData")
    if isinstance(value, ReferenceTensor):
        tensors = (value,)
        structure = "packed"
    elif isinstance(value, tuple) and len(value) == 2:
        tensors = tuple(
            require_reference_tensor(item, "paged_kv_cache[%d]" % index)
            for index, item in enumerate(value)
        )
        structure = "separate"
    else:
        raise TypeError(
            "paged_kv_cache must be a ReferenceTensor or a (k_cache, v_cache) tuple"
        )
    num_pages = tensors[0].shape[0] if tensors[0].shape else 0
    spec = PagedKVCacheSpec(
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        dtype=dtype,
        layout=layout,
        structure=structure,
        device=tensors[0].device,
    )
    return ReferenceKVData(spec, tensors)


def adapt_ragged_kv_data(
    k,
    v,
    *,
    total_kv_tokens: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    dtype: str,
    layout: KVLayout,
    quant_spec: Optional[QuantSpec] = None,
) -> ReferenceKVInput:
    if isinstance(k, ReferenceQuantizedTensor) or isinstance(
        v, ReferenceQuantizedTensor
    ):
        if not isinstance(k, ReferenceQuantizedTensor) or not isinstance(
            v, ReferenceQuantizedTensor
        ):
            raise TypeError("quantized ragged K and V must be provided together")
        if k.quant_spec != v.quant_spec:
            raise SchemaError("quantized K and V must use the same QuantSpec")
        if k.quant_spec != quant_spec:
            raise SchemaError("quantized ragged KV data does not match the plan")
        spec = RaggedKVCacheSpec(
            total_kv_tokens=total_kv_tokens,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            dtype=dtype,
            layout=layout,
            device=k.device,
            quant_spec=k.quant_spec,
        )
        return ReferenceQuantizedKVData(spec, k, v)
    if quant_spec is not None:
        raise SchemaError("quantized plan requires quantized ragged K and V")
    key = require_reference_tensor(k, "k")
    value = require_reference_tensor(v, "v")
    spec = RaggedKVCacheSpec(
        total_kv_tokens=total_kv_tokens,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        dtype=dtype,
        layout=layout,
        device=key.device,
    )
    return ReferenceKVData(spec, (key, value))


def finalize_reference_result(
    result: ReferenceAttentionResult,
    *,
    out: Optional[ReferenceBuffer],
    lse: Optional[ReferenceBuffer],
    return_lse: bool,
):
    output_value = result.output
    if out is not None:
        if not isinstance(out, ReferenceBuffer):
            raise TypeError("out must be ReferenceBuffer in the Host-only facade")
        out.copy_from(result.output)
        output_value = out
    lse_value = result.lse
    if lse is not None:
        if not isinstance(lse, ReferenceBuffer):
            raise TypeError("lse must be ReferenceBuffer in the Host-only facade")
        if result.lse is None:
            raise SchemaError("internal reference result did not produce LSE")
        lse.copy_from(result.lse)
        lse_value = lse
    return (output_value, lse_value) if return_lse else output_value


def adapt_single_qkv(
    q: ReferenceTensor,
    k,
    v,
    *,
    mode: AttentionMode,
    kv_layout: str,
) -> SingleQKVAdapterResult:
    require_reference_tensor(q, "q")
    quantized_kv = isinstance(k, ReferenceQuantizedTensor) or isinstance(
        v, ReferenceQuantizedTensor
    )
    if quantized_kv:
        if not isinstance(k, ReferenceQuantizedTensor) or not isinstance(
            v, ReferenceQuantizedTensor
        ):
            raise TypeError("quantized K and V must be provided together")
        if k.quant_spec != v.quant_spec:
            raise SchemaError("quantized K and V must use the same QuantSpec")
        kv_quant_spec = k.quant_spec
    else:
        require_reference_tensor(k, "k")
        require_reference_tensor(v, "v")
        kv_quant_spec = None
    layout = parse_kv_layout(kv_layout)

    expected_q_rank = 2 if mode == AttentionMode.SINGLE_DECODE else 3
    if len(q.shape) != expected_q_rank:
        raise SchemaError(
            "%s q must be rank %d" % (mode.value, expected_q_rank)
        )
    if len(k.shape) != 3 or len(v.shape) != 3:
        raise SchemaError("single-request k and v must be rank 3")
    if q.device != k.device or q.device != v.device:
        raise SchemaError("q, k, and v must be on the same device")
    if k.dtype != v.dtype:
        raise SchemaError("k and v must have the same dtype")

    if mode == AttentionMode.SINGLE_DECODE:
        qo_len = 1
        num_qo_heads, head_dim_qk = q.shape
    else:
        qo_len, num_qo_heads, head_dim_qk = q.shape
    if layout == KVLayout.NHD:
        kv_len, num_kv_heads, k_head_dim = k.shape
        v_kv_len, v_num_heads, head_dim_vo = v.shape
    else:
        num_kv_heads, kv_len, k_head_dim = k.shape
        v_num_heads, v_kv_len, head_dim_vo = v.shape
    if k_head_dim != head_dim_qk:
        raise SchemaError("q and k head dimensions must match")
    if v_kv_len != kv_len:
        raise SchemaError("k and v sequence lengths must match")
    if v_num_heads != num_kv_heads:
        raise SchemaError("k and v head counts must match")
    if num_qo_heads % num_kv_heads != 0:
        raise SchemaError("num_kv_heads must divide num_qo_heads")
    if mode == AttentionMode.SINGLE_DECODE and head_dim_vo != head_dim_qk:
        raise SchemaError("single decode requires equal q/k and v head dimensions")

    cache_spec = RaggedKVCacheSpec(
        total_kv_tokens=kv_len,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        dtype=k.dtype,
        layout=layout,
        device=k.device,
        quant_spec=kv_quant_spec,
    )
    kv_data = (
        ReferenceQuantizedKVData(cache_spec, k, v)
        if quantized_kv
        else ReferenceKVData(cache_spec, (k, v))
    )
    return SingleQKVAdapterResult(
        q=q,
        kv_data=kv_data,
        metadata=SingleAttentionMetadata(qo_len=qo_len, kv_len=kv_len),
        layout=layout,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
    )


def adapt_single_custom_mask(
    custom_mask: MaskInput,
    packed_custom_mask: MaskInput,
    *,
    qo_len: int,
    kv_len: int,
    device: str,
) -> Tuple[Optional[CustomMaskSpec], Optional[Tuple[Union[bool, int], ...]]]:
    # FlashInfer gives packed_custom_mask precedence over custom_mask.
    if packed_custom_mask is not None:
        if not isinstance(packed_custom_mask, ReferenceTensor):
            raise TypeError("packed_custom_mask must be ReferenceTensor")
        expected_numel = (qo_len * kv_len + 7) // 8
        if packed_custom_mask.shape != (expected_numel,):
            raise SchemaError(
                "packed_custom_mask shape must be (%d,)" % expected_numel
            )
        if packed_custom_mask.dtype != "uint8":
            raise SchemaError("packed_custom_mask dtype must be uint8")
        if packed_custom_mask.device != device:
            raise SchemaError("packed_custom_mask must be on the q device")
        packed_values = []
        for value in packed_custom_mask.data:
            if (
                not math.isfinite(value)
                or not value.is_integer()
                or not 0 <= value <= 255
            ):
                raise SchemaError(
                    "packed_custom_mask values must be integers in [0, 255]"
                )
            packed_values.append(int(value))
        return (
            CustomMaskSpec(numel=expected_numel, packed=True),
            tuple(packed_values),
        )
    if custom_mask is not None:
        if not isinstance(custom_mask, ReferenceTensor):
            raise TypeError("custom_mask must be ReferenceTensor")
        if custom_mask.shape != (qo_len, kv_len):
            raise SchemaError(
                "custom_mask shape must be (%d, %d)" % (qo_len, kv_len)
            )
        if custom_mask.dtype != "bool":
            raise SchemaError("custom_mask dtype must be bool")
        if custom_mask.device != device:
            raise SchemaError("custom_mask must be on the q device")
        if any(value not in (0.0, 1.0) for value in custom_mask.data):
            raise SchemaError("custom_mask values must be boolean")
        return (
            CustomMaskSpec(numel=qo_len * kv_len),
            tuple(bool(value) for value in custom_mask.data),
        )
    return None, None


def adapt_head_scale(
    value: FrontendScaleInput,
    *,
    name: str,
    num_heads: int,
    device: str,
) -> ScaleInput:
    if value is None:
        return 1.0
    if isinstance(value, ReferenceTensor):
        if value.shape != (num_heads,):
            raise SchemaError(
                "%s shape must be (%d,)" % (name, num_heads)
            )
        if value.dtype != "float32":
            raise SchemaError("%s dtype must be float32" % name)
        if value.device != device:
            raise SchemaError("%s must be on the q device" % name)
        return value.data
    return value


def optional_finite_scalar(value: Optional[float], name: str) -> float:
    if value is None:
        return 1.0
    return finite_scalar(value, name)


def finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SchemaError("%s must be a scalar" % name) from error
    if not math.isfinite(result):
        raise SchemaError("%s must be finite" % name)
    return result


def multiply_scale(value: ScaleInput, factor: float) -> ScaleInput:
    if isinstance(value, (int, float)):
        return float(value) * factor
    return tuple(float(item) * factor for item in value)
