"""Zero-dependency numerical reference for the Attention framework.

The implementation is intentionally scalar and slow. It exists to define and
test semantics for small workloads before a torch frontend or device backend
is available. It must never be selected silently for production inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .numerics import normalize_attention_logits
from .planner import AttentionFrameworkPlan
from .quant_layout import (
    infer_quant_scale_shape,
    infer_quant_storage_shape,
    normalize_quant_axes,
)
from .schema import (
    AttentionMode,
    KVCacheSpec,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    SingleAttentionMetadata,
    TensorSpec,
)


ScaleInput = Union[float, Sequence[float]]


def _numel(shape: Sequence[int]) -> int:
    return reduce(mul, shape, 1)


@dataclass(frozen=True)
class ReferenceTensor:
    """Small immutable dense tensor backed by a flat Python tuple."""

    shape: Tuple[int, ...]
    data: Tuple[float, ...]
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "data", tuple(float(value) for value in self.data))
        if any(dim < 0 for dim in self.shape):
            raise SchemaError("reference tensor shape cannot contain negative values")
        expected = _numel(self.shape)
        if len(self.data) != expected:
            raise SchemaError(
                "reference tensor data length must be %d, got %d"
                % (expected, len(self.data))
            )
        if not self.dtype or not self.device:
            raise SchemaError("reference tensor dtype and device must be non-empty")

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(self.shape, self.dtype, self.device)

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        dtype: str = "float32",
        device: str = "cpu",
    ) -> "ReferenceTensor":
        shape_tuple = tuple(int(dim) for dim in shape)
        return cls(shape_tuple, (0.0,) * _numel(shape_tuple), dtype, device)

    @classmethod
    def from_nested(
        cls,
        value: object,
        dtype: str = "float32",
        device: str = "cpu",
    ) -> "ReferenceTensor":
        def visit(node: object) -> Tuple[Tuple[int, ...], List[float]]:
            if isinstance(node, (list, tuple)):
                if not node:
                    return (0,), []
                child_results = [visit(child) for child in node]
                child_shape = child_results[0][0]
                if any(shape != child_shape for shape, _ in child_results):
                    raise SchemaError("nested reference tensor must be rectangular")
                flat: List[float] = []
                for _, values in child_results:
                    flat.extend(values)
                return (len(node),) + child_shape, flat
            try:
                return (), [float(node)]
            except (TypeError, ValueError) as error:
                raise SchemaError("reference tensor values must be numeric") from error

        shape, data = visit(value)
        return cls(shape, tuple(data), dtype, device)

    def at(self, *indices: int) -> float:
        if len(indices) != len(self.shape):
            raise IndexError("expected %d indices" % len(self.shape))
        offset = 0
        for index, dim in zip(indices, self.shape):
            if index < 0 or index >= dim:
                raise IndexError("tensor index out of range")
            offset = offset * dim + index
        return self.data[offset]


@dataclass
class ReferenceBuffer:
    """Mutable Host output buffer used to validate ``out``/``lse`` semantics."""

    shape: Tuple[int, ...]
    dtype: str = "float32"
    device: str = "cpu"
    data: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in self.shape):
            raise SchemaError("reference buffer shape cannot contain negative values")
        expected = _numel(self.shape)
        if not self.data:
            self.data = (0.0,) * expected
        else:
            self.data = tuple(float(value) for value in self.data)
            if len(self.data) != expected:
                raise SchemaError(
                    "reference buffer data length must be %d, got %d"
                    % (expected, len(self.data))
                )
        if not self.dtype or not self.device:
            raise SchemaError("reference buffer dtype and device must be non-empty")

    @property
    def spec(self) -> TensorSpec:
        return TensorSpec(self.shape, self.dtype, self.device)

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        dtype: str = "float32",
        device: str = "cpu",
    ) -> "ReferenceBuffer":
        return cls(tuple(int(dim) for dim in shape), dtype, device)

    def copy_from(self, tensor: ReferenceTensor) -> None:
        if self.spec != tensor.spec:
            raise SchemaError("reference buffer shape/dtype/device mismatch")
        self.data = tensor.data

    def at(self, *indices: int) -> float:
        return ReferenceTensor(self.shape, self.data, self.dtype, self.device).at(
            *indices
        )


@dataclass(frozen=True)
class ReferenceQuantizedTensor:
    """Explicit quantized tensor with inline indexed dequantization semantics.

    ``axis`` in :class:`QuantSpec` indexes the logical tensor shape. For
    ``group``/``block`` granularity, ``group_size`` is aligned one-to-one with
    ``axis`` and the scale index is ``logical_index // group_size``. Other
    non-tensor granularities select the declared axes directly.
    """

    logical_shape: Tuple[int, ...]
    storage: ReferenceTensor
    scale: ReferenceTensor
    quant_spec: QuantSpec
    zero_point: Optional[ReferenceTensor] = None

    def __post_init__(self) -> None:
        logical_shape = tuple(int(dim) for dim in self.logical_shape)
        object.__setattr__(self, "logical_shape", logical_shape)
        if not logical_shape or any(dim < 0 for dim in logical_shape):
            raise SchemaError(
                "quantized logical_shape must be non-empty and non-negative"
            )
        if self.quant_spec.scheme == "mx":
            raise NotImplementedError("MX quantization is not in the Host oracle")
        supported_storage = {
            "int8": ("int8", -128, 127),
            "uint8": ("uint8", 0, 255),
            "int4_packed": ("uint8", -8, 7),
            "uint4_packed": ("uint8", 0, 15),
            "float8_e4m3fn": ("float8_e4m3fn", None, None),
            "float8_e5m2": ("float8_e5m2", None, None),
        }
        if self.quant_spec.storage_dtype not in supported_storage:
            raise NotImplementedError(
                "Host quantized tensor does not support storage dtype %r"
                % self.quant_spec.storage_dtype
            )
        if self.quant_spec.physical_layout != "logical":
            raise NotImplementedError(
                "Host oracle only decodes physical_layout='logical'"
            )
        storage_tensor_dtype, minimum, maximum = supported_storage[
            self.quant_spec.storage_dtype
        ]
        if minimum is None and self.quant_spec.has_zero_point:
            raise SchemaError("FP8 quantized tensor cannot carry zero_point")
        if self.storage.dtype != storage_tensor_dtype:
            raise SchemaError(
                "quantized storage tensor dtype must be %s" % storage_tensor_dtype
            )
        expected_storage_shape = self._expected_storage_shape()
        if self.storage.shape != expected_storage_shape:
            raise SchemaError(
                "quantized storage shape must be %r, got %r"
                % (expected_storage_shape, self.storage.shape)
            )
        if self.scale.shape != self.scale_shape:
            raise SchemaError(
                "quantized scale shape must be %r, got %r"
                % (self.scale_shape, self.scale.shape)
            )
        if self.scale.dtype != self.quant_spec.scale_dtype:
            raise SchemaError(
                "quantized scale dtype must be %s" % self.quant_spec.scale_dtype
            )
        if self.storage.device != self.scale.device:
            raise SchemaError("quantized storage and scale must share a device")
        if any(not math.isfinite(value) or value <= 0 for value in self.scale.data):
            raise SchemaError("quantized scales must be finite and positive")
        if self.quant_spec.has_zero_point:
            if self.zero_point is None:
                raise SchemaError("quantized tensor requires zero_point")
            if self.zero_point.shape != self.scale_shape:
                raise SchemaError("zero_point shape must match scale shape")
            if self.zero_point.dtype != "int32":
                raise SchemaError("zero_point dtype must be int32")
            if self.zero_point.device != self.storage.device:
                raise SchemaError("zero_point must be on the storage device")
            for value in self.zero_point.data:
                if not value.is_integer() or not minimum <= value <= maximum:
                    raise SchemaError(
                        "zero_point values must be integral and in storage range"
                    )
        elif self.zero_point is not None:
            raise SchemaError("zero_point provided for a quant spec without one")
        self._validate_storage_values(storage_tensor_dtype)
        self._validate_int4_padding()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.logical_shape

    @property
    def dtype(self) -> str:
        return self.quant_spec.storage_dtype

    @property
    def device(self) -> str:
        return self.storage.device

    @property
    def normalized_axis(self) -> Tuple[int, ...]:
        return normalize_quant_axes(self.logical_shape, self.quant_spec)

    @property
    def scale_shape(self) -> Tuple[int, ...]:
        return infer_quant_scale_shape(self.logical_shape, self.quant_spec)

    def _expected_storage_shape(self) -> Tuple[int, ...]:
        return infer_quant_storage_shape(self.logical_shape, self.quant_spec)

    def _validate_storage_values(self, tensor_dtype: str) -> None:
        if tensor_dtype in {"float8_e4m3fn", "float8_e5m2"}:
            if any(not math.isfinite(value) for value in self.storage.data):
                raise SchemaError("FP8 storage values must be finite")
            return
        if tensor_dtype == "int8":
            minimum, maximum = -128, 127
        else:
            minimum, maximum = 0, 255
        for value in self.storage.data:
            if not value.is_integer() or not minimum <= value <= maximum:
                raise SchemaError(
                    "quantized storage values must be integral and in dtype range"
                )

    def _validate_int4_padding(self) -> None:
        if self.quant_spec.storage_dtype not in {
            "int4_packed",
            "uint4_packed",
        }:
            return
        if self.quant_spec.packing_order not in {
            "low_nibble_first",
            "high_nibble_first",
        }:
            raise SchemaError(
                "packed INT4 requires low_nibble_first or high_nibble_first"
            )
        if self.logical_shape[-1] % 2 == 0:
            return
        row_width = self.storage.shape[-1]
        rows = _numel(self.storage.shape[:-1])
        for row in range(rows):
            byte = int(self.storage.data[row * row_width + row_width - 1])
            padding = (
                byte >> 4
                if self.quant_spec.packing_order == "low_nibble_first"
                else byte & 0x0F
            )
            if padding != 0:
                raise SchemaError("unused INT4 padding nibble must be zero")

    def _scale_indices(self, indices: Tuple[int, ...]) -> Tuple[int, ...]:
        axes = self.normalized_axis
        if self.quant_spec.granularity == "tensor":
            return ()
        if self.quant_spec.granularity in {"group", "block"}:
            return tuple(
                indices[axis] // size
                for axis, size in zip(axes, self.quant_spec.group_size or ())
            )
        return tuple(indices[axis] for axis in axes)

    def quantized_at(self, *indices: int) -> float:
        if len(indices) != len(self.logical_shape):
            raise IndexError("expected %d logical indices" % len(self.logical_shape))
        for index, dim in zip(indices, self.logical_shape):
            if index < 0 or index >= dim:
                raise IndexError("quantized tensor index out of range")
        if self.quant_spec.storage_dtype not in {
            "int4_packed",
            "uint4_packed",
        }:
            if self.quant_spec.storage_dtype in {
                "float8_e4m3fn",
                "float8_e5m2",
            }:
                return self.storage.at(*indices)
            return int(self.storage.at(*indices))
        packed_indices = indices[:-1] + (indices[-1] // 2,)
        byte = int(self.storage.at(*packed_indices))
        first = indices[-1] % 2 == 0
        if self.quant_spec.packing_order == "low_nibble_first":
            nibble = byte & 0x0F if first else byte >> 4
        else:
            nibble = byte >> 4 if first else byte & 0x0F
        if self.quant_spec.storage_dtype == "int4_packed" and nibble >= 8:
            return nibble - 16
        return nibble

    def dequantized_at(self, *indices: int) -> float:
        scale_indices = self._scale_indices(tuple(indices))
        scale = self.scale.at(*scale_indices)
        zero_point = (
            self.zero_point.at(*scale_indices)
            if self.zero_point is not None
            else 0.0
        )
        return (self.quantized_at(*indices) - zero_point) * scale

    def dequantize(self) -> ReferenceTensor:
        values = []
        total = _numel(self.logical_shape)
        for flat_index in range(total):
            remainder = flat_index
            reversed_indices = []
            for dim in reversed(self.logical_shape):
                reversed_indices.append(remainder % dim)
                remainder //= dim
            indices = tuple(reversed(reversed_indices))
            values.append(self.dequantized_at(*indices))
        return ReferenceTensor(
            self.logical_shape,
            tuple(values),
            self.quant_spec.compute_dtype,
            self.device,
        )


@dataclass(frozen=True)
class ReferenceKVData:
    """Concrete reference values paired with a logical KV cache spec."""

    spec: KVCacheSpec
    tensors: Tuple[ReferenceTensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        if self.spec.quant_spec is not None:
            raise SchemaError(
                "dense reference KV cannot carry quant_spec; "
                "use ReferenceQuantizedKVData"
            )
        expected_shapes = self.spec.expected_shapes
        if len(self.tensors) != len(expected_shapes):
            raise SchemaError(
                "KV data requires %d tensor(s), got %d"
                % (len(expected_shapes), len(self.tensors))
            )
        for tensor, expected_shape in zip(self.tensors, expected_shapes):
            if tensor.shape != expected_shape:
                raise SchemaError(
                    "KV tensor shape must be %r, got %r"
                    % (expected_shape, tensor.shape)
                )
            if tensor.dtype != self.spec.dtype:
                raise SchemaError("KV tensor dtype does not match KV cache spec")
            if tensor.device != self.spec.device:
                raise SchemaError("KV tensor device does not match KV cache spec")

    def key(self, token: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, RaggedKVCacheSpec):
            raise TypeError("key(token, ...) requires ragged KV data")
        key_tensor = self.tensors[0]
        if self.spec.layout == KVLayout.NHD:
            return key_tensor.at(token, head, dim)
        return key_tensor.at(head, token, dim)

    def value(self, token: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, RaggedKVCacheSpec):
            raise TypeError("value(token, ...) requires ragged KV data")
        value_tensor = self.tensors[1]
        if self.spec.layout == KVLayout.NHD:
            return value_tensor.at(token, head, dim)
        return value_tensor.at(head, token, dim)

    def paged_key(self, page: int, entry: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, PagedKVCacheSpec):
            raise TypeError("paged_key requires paged KV data")
        if self.spec.structure == "packed":
            tensor = self.tensors[0]
            if self.spec.layout == KVLayout.NHD:
                return tensor.at(page, 0, entry, head, dim)
            return tensor.at(page, 0, head, entry, dim)
        tensor = self.tensors[0]
        if self.spec.layout == KVLayout.NHD:
            return tensor.at(page, entry, head, dim)
        return tensor.at(page, head, entry, dim)

    def paged_value(self, page: int, entry: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, PagedKVCacheSpec):
            raise TypeError("paged_value requires paged KV data")
        if self.spec.structure == "packed":
            tensor = self.tensors[0]
            if self.spec.layout == KVLayout.NHD:
                return tensor.at(page, 1, entry, head, dim)
            return tensor.at(page, 1, head, entry, dim)
        tensor = self.tensors[1]
        if self.spec.layout == KVLayout.NHD:
            return tensor.at(page, entry, head, dim)
        return tensor.at(page, head, entry, dim)


@dataclass(frozen=True)
class ReferenceQuantizedKVData:
    """K/V storage that dequantizes each consumed element at Attention read time."""

    spec: KVCacheSpec
    key_data: ReferenceQuantizedTensor
    value_data: ReferenceQuantizedTensor

    def __post_init__(self) -> None:
        if self.spec.quant_spec is None:
            raise SchemaError("quantized KV cache spec requires quant_spec")
        if isinstance(self.spec, PagedKVCacheSpec) and self.spec.structure != "separate":
            raise SchemaError(
                "quantized reference KV uses separate K/V storage and scales"
            )
        expected_shapes = self.spec.expected_shapes
        if len(expected_shapes) != 2:
            raise SchemaError("quantized reference KV requires separate K/V shapes")
        if self.key_data.logical_shape != expected_shapes[0]:
            raise SchemaError("quantized key logical shape does not match cache spec")
        if self.value_data.logical_shape != expected_shapes[1]:
            raise SchemaError("quantized value logical shape does not match cache spec")
        for name, value in (("key", self.key_data), ("value", self.value_data)):
            if value.quant_spec != self.spec.quant_spec:
                raise SchemaError("quantized %s spec does not match cache spec" % name)
            if value.dtype != self.spec.dtype:
                raise SchemaError("quantized %s dtype does not match cache spec" % name)
            if value.device != self.spec.device:
                raise SchemaError("quantized %s device does not match cache spec" % name)

    def key(self, token: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, RaggedKVCacheSpec):
            raise TypeError("key(token, ...) requires ragged KV data")
        if self.spec.layout == KVLayout.NHD:
            return self.key_data.dequantized_at(token, head, dim)
        return self.key_data.dequantized_at(head, token, dim)

    def value(self, token: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, RaggedKVCacheSpec):
            raise TypeError("value(token, ...) requires ragged KV data")
        if self.spec.layout == KVLayout.NHD:
            return self.value_data.dequantized_at(token, head, dim)
        return self.value_data.dequantized_at(head, token, dim)

    def paged_key(self, page: int, entry: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, PagedKVCacheSpec):
            raise TypeError("paged_key requires paged KV data")
        if self.spec.layout == KVLayout.NHD:
            return self.key_data.dequantized_at(page, entry, head, dim)
        return self.key_data.dequantized_at(page, head, entry, dim)

    def paged_value(self, page: int, entry: int, head: int, dim: int) -> float:
        if not isinstance(self.spec, PagedKVCacheSpec):
            raise TypeError("paged_value requires paged KV data")
        if self.spec.layout == KVLayout.NHD:
            return self.value_data.dequantized_at(page, entry, head, dim)
        return self.value_data.dequantized_at(page, head, entry, dim)


ReferenceKVInput = Union[ReferenceKVData, ReferenceQuantizedKVData]


@dataclass(frozen=True)
class ReferenceAttentionResult:
    output: ReferenceTensor
    lse: Optional[ReferenceTensor]


def _alibi_slopes(num_heads: int) -> Tuple[float, ...]:
    """Reference ALiBi slopes following the original power-of-two recipe."""

    def power_of_two_slopes(count: int) -> Tuple[float, ...]:
        start = 2 ** (-(2 ** -(math.log2(count) - 3)))
        ratio = start
        return tuple(start * ratio**index for index in range(count))

    if num_heads > 0 and num_heads & (num_heads - 1) == 0:
        return power_of_two_slopes(num_heads)
    nearest_power = 2 ** math.floor(math.log2(num_heads))
    base = power_of_two_slopes(nearest_power)
    extra_source = power_of_two_slopes(2 * nearest_power)
    extras = extra_source[0::2][: num_heads - nearest_power]
    return base + extras


class ReferenceAttentionExecutor:
    """Execute an :class:`AttentionFrameworkPlan` using scalar Python math."""

    def execute(
        self,
        plan: AttentionFrameworkPlan,
        q: ReferenceTensor,
        kv_data: ReferenceKVInput,
        *,
        return_lse: bool = False,
        custom_mask_data: Optional[Sequence[Union[bool, int]]] = None,
        alibi_slopes: Optional[Sequence[float]] = None,
        q_scale: ScaleInput = 1.0,
        k_scale: ScaleInput = 1.0,
        v_scale: ScaleInput = 1.0,
        logits_soft_cap: float = 0.0,
    ) -> ReferenceAttentionResult:
        result_spec = plan.validate_run(
            q.spec,
            kv_data.spec,
            return_lse=return_lse,
            logits_soft_cap=logits_soft_cap,
        )
        q_scales = self._normalize_head_scales(
            "q_scale", q_scale, plan.spec.num_qo_heads
        )
        k_scales = self._normalize_head_scales(
            "k_scale", k_scale, plan.spec.num_kv_heads
        )
        v_scales = self._normalize_head_scales(
            "v_scale", v_scale, plan.spec.num_kv_heads
        )

        mask_spec = plan.spec.custom_mask
        mask_values: Optional[Tuple[bool, ...]] = None
        mask_bytes: Optional[Tuple[int, ...]] = None
        if mask_spec is not None:
            if mask_spec.packed:
                if custom_mask_data is None:
                    raise SchemaError("custom_mask_data is required by the plan")
                converted_bytes: List[int] = []
                for value in custom_mask_data:
                    try:
                        byte = int(value)
                    except (TypeError, ValueError) as error:
                        raise SchemaError(
                            "packed custom mask values must be uint8"
                        ) from error
                    if byte < 0 or byte > 255 or float(value) != float(byte):
                        raise SchemaError("packed custom mask values must be uint8")
                    converted_bytes.append(byte)
                mask_bytes = tuple(converted_bytes)
                if len(mask_bytes) != mask_spec.numel:
                    raise SchemaError("custom_mask_data length does not match the plan")
            else:
                if custom_mask_data is None:
                    raise SchemaError("custom_mask_data is required by the plan")
                mask_values = tuple(bool(value) for value in custom_mask_data)
                if len(mask_values) != mask_spec.numel:
                    raise SchemaError("custom_mask_data length does not match the plan")
        elif custom_mask_data is not None:
            raise SchemaError("custom_mask_data was provided without a custom mask plan")

        slopes: Optional[Tuple[float, ...]] = None
        if plan.spec.pos_encoding_mode == PosEncodingMode.ALIBI:
            slopes = (
                tuple(float(value) for value in alibi_slopes)
                if alibi_slopes is not None
                else _alibi_slopes(plan.spec.num_qo_heads)
            )
            if len(slopes) != plan.spec.num_qo_heads:
                raise SchemaError("ALiBi slopes must have one value per query head")
        elif alibi_slopes is not None:
            raise SchemaError("ALiBi slopes require pos_encoding_mode='ALIBI'")
        output_values: List[float] = []
        lse_values: List[float] = []
        mask_offset = 0
        mask_byte_offset = 0

        for request, q_start, qo_len, kv_len, key_fn, value_fn in self._requests(
            plan, kv_data
        ):
            mask_segment = None
            if mask_values is not None:
                segment_size = qo_len * kv_len
                mask_segment = mask_values[mask_offset : mask_offset + segment_size]
                mask_offset += segment_size
            elif mask_bytes is not None:
                segment_size = qo_len * kv_len
                segment_num_bytes = (segment_size + 7) // 8
                packed_segment = mask_bytes[
                    mask_byte_offset : mask_byte_offset + segment_num_bytes
                ]
                mask_byte_offset += segment_num_bytes
                mask_segment = tuple(
                    bool(packed_segment[index // 8] & (1 << (index % 8)))
                    for index in range(segment_size)
                )
            request_output, request_lse = self._execute_request(
                plan,
                q,
                q_start,
                qo_len,
                kv_len,
                key_fn,
                value_fn,
                mask_segment,
                slopes,
                q_scales,
                k_scales,
                v_scales,
                logits_soft_cap,
            )
            output_values.extend(request_output)
            lse_values.extend(request_lse)

        output = ReferenceTensor(
            result_spec.output.shape,
            tuple(output_values),
            result_spec.output.dtype,
            result_spec.output.device,
        )
        lse = None
        if result_spec.lse is not None:
            lse = ReferenceTensor(
                result_spec.lse.shape,
                tuple(lse_values),
                result_spec.lse.dtype,
                result_spec.lse.device,
            )
        return ReferenceAttentionResult(output=output, lse=lse)

    def _requests(self, plan: AttentionFrameworkPlan, kv_data: ReferenceKVInput):
        metadata = plan.metadata
        mode = plan.spec.mode
        if isinstance(metadata, SingleAttentionMetadata):
            qo_len = metadata.qo_len
            kv_len = metadata.kv_len
            yield (
                0,
                0,
                qo_len,
                kv_len,
                lambda token, head, dim: kv_data.key(token, head, dim),
                lambda token, head, dim: kv_data.value(token, head, dim),
            )
            return

        if isinstance(metadata, RaggedKVMetadata):
            for request, (q_start, q_end, kv_start, kv_end) in enumerate(
                zip(
                    metadata.qo_indptr,
                    metadata.qo_indptr[1:],
                    metadata.kv_indptr,
                    metadata.kv_indptr[1:],
                )
            ):
                yield (
                    request,
                    q_start,
                    q_end - q_start,
                    kv_end - kv_start,
                    lambda token, head, dim, base=kv_start: kv_data.key(
                        base + token, head, dim
                    ),
                    lambda token, head, dim, base=kv_start: kv_data.value(
                        base + token, head, dim
                    ),
                )
            return

        if isinstance(metadata, PagedPrefillMetadata):
            paged = metadata.paged_kv
            q_indptr = metadata.qo_indptr
            kv_lengths = paged.sequence_lengths
            page_indptr = paged.indptr
            page_indices = paged.indices
            page_size = paged.page_size
        elif isinstance(metadata, PagedKVMetadata):
            paged = metadata
            q_len = plan.spec.q_len_per_req
            q_indptr = tuple(index * q_len for index in range(paged.batch_size + 1))
            kv_lengths = paged.sequence_lengths
            page_indptr = paged.indptr
            page_indices = paged.indices
            page_size = paged.page_size
        elif isinstance(metadata, MixedPagedKVMetadata):
            q_indptr = metadata.qo_indptr
            kv_lengths = metadata.kv_len_arr
            page_indptr = metadata.kv_indptr
            page_indices = metadata.kv_indices
            page_size = metadata.page_size
        else:
            raise TypeError("unsupported Attention metadata")

        for request, (q_start, q_end, kv_len) in enumerate(
            zip(q_indptr, q_indptr[1:], kv_lengths)
        ):
            page_base = page_indptr[request]

            def key_fn(
                token: int,
                head: int,
                dim: int,
                base: int = page_base,
            ) -> float:
                page = page_indices[base + token // page_size]
                return kv_data.paged_key(page, token % page_size, head, dim)

            def value_fn(
                token: int,
                head: int,
                dim: int,
                base: int = page_base,
            ) -> float:
                page = page_indices[base + token // page_size]
                return kv_data.paged_value(page, token % page_size, head, dim)

            yield (
                request,
                q_start,
                q_end - q_start,
                kv_len,
                key_fn,
                value_fn,
            )

    def _execute_request(
        self,
        plan: AttentionFrameworkPlan,
        q: ReferenceTensor,
        q_start: int,
        qo_len: int,
        kv_len: int,
        key_fn,
        value_fn,
        custom_mask: Optional[Sequence[bool]],
        alibi_slopes: Optional[Tuple[float, ...]],
        q_scales: Tuple[float, ...],
        k_scales: Tuple[float, ...],
        v_scales: Tuple[float, ...],
        run_logits_soft_cap: float,
    ) -> Tuple[List[float], List[float]]:
        spec = plan.spec
        group_size = spec.num_qo_heads // spec.num_kv_heads
        output: List[float] = []
        lse: List[float] = []
        cap = (
            run_logits_soft_cap
            if run_logits_soft_cap > 0
            else float(spec.logits_soft_cap or 0.0)
        )
        q_position_offset = kv_len - qo_len

        for local_q in range(qo_len):
            global_q = q_start + local_q
            q_position = q_position_offset + local_q
            for qo_head in range(spec.num_qo_heads):
                kv_head = qo_head // group_size
                logits: List[float] = []
                visible_keys: List[int] = []
                for key_position in range(kv_len):
                    visible = True
                    if custom_mask is not None:
                        visible = bool(custom_mask[local_q * kv_len + key_position])
                    elif spec.effective_causal:
                        visible = key_position <= q_position
                    if spec.window_left >= 0:
                        visible = visible and key_position >= q_position - spec.window_left
                        if spec.window_right >= 0:
                            visible = visible and key_position <= q_position + spec.window_right
                    if not visible:
                        continue
                    dot = 0.0
                    for dim in range(spec.head_dim_qk):
                        if spec.pos_encoding_mode == PosEncodingMode.ROPE_LLAMA:
                            q_value = self._rope_component(
                                lambda component: self._q_at(
                                    q,
                                    spec.mode,
                                    global_q,
                                    qo_head,
                                    component,
                                ),
                                q_position,
                                dim,
                                spec.head_dim_qk,
                                float(spec.rope_scale),
                                float(spec.rope_theta),
                            )
                            k_value = self._rope_component(
                                lambda component: key_fn(
                                    key_position, kv_head, component
                                ),
                                key_position,
                                dim,
                                spec.head_dim_qk,
                                float(spec.rope_scale),
                                float(spec.rope_theta),
                            )
                        else:
                            q_value = self._q_at(
                                q, spec.mode, global_q, qo_head, dim
                            )
                            k_value = key_fn(key_position, kv_head, dim)
                        dot += (
                            q_value
                            * q_scales[qo_head]
                            * k_value
                            * k_scales[kv_head]
                        )
                    logit = dot * float(spec.sm_scale)
                    if alibi_slopes is not None:
                        logit += alibi_slopes[qo_head] * (key_position - q_position)
                    if cap > 0:
                        logit = cap * math.tanh(logit / cap)
                    logits.append(logit)
                    visible_keys.append(key_position)

                probabilities, row_lse = normalize_attention_logits(logits)
                if not probabilities:
                    output.extend((0.0,) * int(spec.head_dim_vo))
                    lse.append(row_lse)
                    continue
                for dim in range(int(spec.head_dim_vo)):
                    value = sum(
                        probability
                        * value_fn(key_position, kv_head, dim)
                        * v_scales[kv_head]
                        for probability, key_position in zip(
                            probabilities, visible_keys
                        )
                        if probability != 0.0
                    )
                    output.append(value)
                lse.append(row_lse)
        return output, lse

    @staticmethod
    def _normalize_head_scales(
        name: str,
        value: ScaleInput,
        num_heads: int,
    ) -> Tuple[float, ...]:
        if isinstance(value, (int, float)):
            scales = (float(value),) * num_heads
        else:
            try:
                scales = tuple(float(item) for item in value)
            except (TypeError, ValueError) as error:
                raise SchemaError("%s must contain numeric values" % name) from error
            if len(scales) != num_heads:
                raise SchemaError(
                    "%s must be scalar or have one value per head (%d)"
                    % (name, num_heads)
                )
        if any(not math.isfinite(scale) for scale in scales):
            raise SchemaError("%s must be finite" % name)
        return scales

    @staticmethod
    def _q_at(
        q: ReferenceTensor,
        mode: AttentionMode,
        token: int,
        head: int,
        dim: int,
    ) -> float:
        if mode == AttentionMode.SINGLE_DECODE:
            return q.at(head, dim)
        return q.at(token, head, dim)

    @staticmethod
    def _rope_component(
        value_at,
        position: int,
        dim: int,
        head_dim: int,
        rope_scale: float,
        rope_theta: float,
    ) -> float:
        """Apply FlashInfer's non-interleaved LLaMA RoPE to one component."""

        half_dim = head_dim // 2
        frequency_index = dim % half_dim
        frequency = (1.0 / rope_scale) * rope_theta ** (
            -2.0 * frequency_index / head_dim
        )
        angle = float(position) * frequency
        cosine = math.cos(angle)
        sine = math.sin(angle)
        if dim < half_dim:
            return value_at(dim) * cosine - value_at(dim + half_dim) * sine
        return value_at(dim) * cosine + value_at(dim - half_dim) * sine
