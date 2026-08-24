"""Framework-independent tensor, alias, and stream contracts for Attention."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .quant_layout import infer_quant_scale_shape, infer_quant_storage_shape
from .quant_physical_layout import (
    QuantPhysicalLayoutDescriptor,
    QuantPhysicalLayoutShapes,
)
from .planner import AttentionFrameworkPlan
from .launch_contract import ATTENTION_RUN_OPTIONS_C_ABI, AttentionAuxiliaryRole
from .reference import (
    ReferenceBuffer,
    ReferenceKVData,
    ReferenceKVInput,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
)
from .schema import KVCacheSpec, TensorSpec


ATTENTION_TENSOR_CONTRACT_SCHEMA_VERSION = 1
ATTENTION_AUXILIARY_CONTRACT_SCHEMA_VERSION = 1
ATTENTION_RUN_OPTIONS_SCHEMA_VERSION = 1

_DTYPE_ITEMSIZE = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
}


def dtype_itemsize(dtype: str) -> int:
    try:
        return _DTYPE_ITEMSIZE[str(dtype)]
    except KeyError as error:
        raise SchemaError("unknown tensor dtype itemsize: %r" % dtype) from error


def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    shape_tuple = tuple(int(dim) for dim in shape)
    stride = 1
    result = []
    for dim in reversed(shape_tuple):
        result.append(stride)
        stride *= max(dim, 1)
    return tuple(reversed(result))


def _numel(shape: Sequence[int]) -> int:
    return reduce(mul, shape, 1)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TensorView:
    """A storage-safe tensor view expressed without importing torch.

    Strides and ``storage_offset`` are in elements.  ``storage_nbytes`` and
    alignment are in bytes.  Negative/overlapping strides are intentionally
    rejected by v1; future adapters may define a separate materialization path.
    """

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str
    device: str
    storage_id: str
    storage_nbytes: int
    storage_offset: int = 0
    data_ptr_alignment: int = 1
    writable: bool = False
    schema_version: int = ATTENTION_TENSOR_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_TENSOR_CONTRACT_SCHEMA_VERSION:
            raise SchemaError("unsupported TensorView schema version")
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(
            self, "strides", tuple(int(stride) for stride in self.strides)
        )
        try:
            object.__setattr__(self, "storage_nbytes", int(self.storage_nbytes))
            object.__setattr__(self, "storage_offset", int(self.storage_offset))
            object.__setattr__(
                self, "data_ptr_alignment", int(self.data_ptr_alignment)
            )
        except (TypeError, ValueError) as error:
            raise SchemaError("TensorView storage fields must be integers") from error
        if len(self.shape) != len(self.strides):
            raise SchemaError("TensorView shape and strides must have the same rank")
        if any(dim < 0 for dim in self.shape):
            raise SchemaError("TensorView shape cannot contain negative values")
        if any(stride < 0 for stride in self.strides):
            raise SchemaError("TensorView v1 does not support negative strides")
        if not self.dtype or not self.device or not self.storage_id:
            raise SchemaError("TensorView dtype/device/storage_id must be non-empty")
        if not isinstance(self.writable, bool):
            raise SchemaError("TensorView writable must be boolean")
        if self.storage_nbytes < 0 or self.storage_offset < 0:
            raise SchemaError("TensorView storage size/offset cannot be negative")
        if (
            self.data_ptr_alignment <= 0
            or self.data_ptr_alignment & (self.data_ptr_alignment - 1)
        ):
            raise SchemaError("TensorView alignment must be a positive power of two")
        _ = self.itemsize
        if self.required_storage_nbytes > self.storage_nbytes:
            raise SchemaError(
                "TensorView exceeds storage: requires %d bytes, has %d"
                % (self.required_storage_nbytes, self.storage_nbytes)
            )
        if self.has_internal_overlap:
            raise SchemaError("TensorView v1 does not support internal overlap")

    @property
    def itemsize(self) -> int:
        return dtype_itemsize(self.dtype)

    @property
    def numel(self) -> int:
        return _numel(self.shape)

    @property
    def max_element_offset(self) -> int:
        if self.numel == 0:
            return self.storage_offset
        return self.storage_offset + sum(
            (dim - 1) * stride for dim, stride in zip(self.shape, self.strides)
        )

    @property
    def required_storage_nbytes(self) -> int:
        if self.numel == 0:
            return self.storage_offset * self.itemsize
        return (self.max_element_offset + 1) * self.itemsize

    @property
    def is_contiguous(self) -> bool:
        return self.numel == 0 or self.strides == contiguous_strides(self.shape)

    @property
    def has_internal_overlap(self) -> bool:
        if self.numel <= 1:
            return False
        # Sorted-stride criterion is exact for positive dense rectangular views.
        required_span = 1
        for dim, stride in sorted(
            ((dim, stride) for dim, stride in zip(self.shape, self.strides) if dim > 1),
            key=lambda item: item[1],
        ):
            if stride < required_span:
                return True
            required_span += stride * (dim - 1)
        return False

    @property
    def byte_interval(self) -> Tuple[int, int]:
        start = self.storage_offset * self.itemsize
        end = start if self.numel == 0 else (self.max_element_offset + 1) * self.itemsize
        return start, end

    def overlaps(self, other: "TensorView") -> bool:
        if self.storage_id != other.storage_id:
            return False
        left_start, left_end = self.byte_interval
        right_start, right_end = other.byte_interval
        return max(left_start, right_start) < min(left_end, right_end)

    def require_alignment(self, alignment: int, name: str) -> None:
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("required alignment must be a positive power of two")
        if self.data_ptr_alignment < alignment:
            raise SchemaError("%s must be %d-byte aligned" % (name, alignment))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "dtype": self.dtype,
            "device": self.device,
            "storage_id": self.storage_id,
            "storage_nbytes": self.storage_nbytes,
            "storage_offset": self.storage_offset,
            "data_ptr_alignment": self.data_ptr_alignment,
            "writable": self.writable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorView":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("TensorView fields do not match schema version 1")
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

@dataclass(frozen=True)
class QuantizedTensorView:
    logical_shape: Tuple[int, ...]
    storage: TensorView
    scale: TensorView
    quant_spec: QuantSpec
    zero_point: Optional[TensorView] = None
    physical_layout_descriptor: Optional[QuantPhysicalLayoutDescriptor] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_shape", tuple(int(dim) for dim in self.logical_shape)
        )
        descriptor = self.physical_layout_descriptor
        if self.quant_spec.physical_layout == "logical":
            if descriptor is not None:
                raise SchemaError(
                    "logical quantized view cannot carry a physical layout descriptor"
                )
            expected_shapes = QuantPhysicalLayoutShapes(
                infer_quant_storage_shape(self.logical_shape, self.quant_spec),
                infer_quant_scale_shape(self.logical_shape, self.quant_spec),
                (
                    infer_quant_scale_shape(self.logical_shape, self.quant_spec)
                    if self.quant_spec.has_zero_point
                    else None
                ),
            )
        else:
            if descriptor is None:
                raise SchemaError(
                    "non-logical quantized view requires a physical layout descriptor"
                )
            if not isinstance(descriptor, QuantPhysicalLayoutDescriptor):
                raise TypeError(
                    "physical_layout_descriptor has the wrong type"
                )
            expected_shapes = descriptor.physical_shapes(
                self.logical_shape, self.quant_spec
            )
            self.storage.require_alignment(
                descriptor.storage_transform.required_alignment,
                "quantized physical storage",
            )
            if descriptor.scale_transform is not None:
                self.scale.require_alignment(
                    descriptor.scale_transform.required_alignment,
                    "quantized physical scale",
                )
            if (
                self.zero_point is not None
                and descriptor.zero_point_transform is not None
            ):
                self.zero_point.require_alignment(
                    descriptor.zero_point_transform.required_alignment,
                    "quantized physical zero_point",
                )
        if self.storage.shape != expected_shapes.storage:
            raise SchemaError("quantized storage view shape does not match QuantSpec")
        expected_storage_dtype = (
            "uint8"
            if self.quant_spec.storage_dtype in {"int4_packed", "uint4_packed"}
            else self.quant_spec.storage_dtype
        )
        if self.storage.dtype != expected_storage_dtype:
            raise SchemaError("quantized storage view dtype does not match QuantSpec")
        expected_scale_shape = expected_shapes.scale
        if self.scale.shape != expected_scale_shape:
            raise SchemaError("quantized scale view shape does not match QuantSpec")
        if self.scale.dtype != self.quant_spec.scale_dtype:
            raise SchemaError("quantized scale view dtype does not match QuantSpec")
        if self.storage.device != self.scale.device:
            raise SchemaError("quantized storage and scale views must share device")
        if self.quant_spec.has_zero_point:
            if self.zero_point is None:
                raise SchemaError("quantized tensor view requires zero_point")
            if self.zero_point.shape != expected_shapes.zero_point:
                raise SchemaError("zero_point view shape must match scale shape")
            if self.zero_point.dtype != "int32":
                raise SchemaError("zero_point view dtype must be int32")
            if self.zero_point.device != self.storage.device:
                raise SchemaError("zero_point view must share storage device")
        elif self.zero_point is not None:
            raise SchemaError("zero_point view provided for spec without zero point")
        views = [self.storage, self.scale]
        if self.zero_point is not None:
            views.append(self.zero_point)
        _reject_aliases("quantized tensor components", views)

    @property
    def device(self) -> str:
        return self.storage.device


@dataclass(frozen=True)
class KVCacheView:
    spec: KVCacheSpec
    key: object
    value: object
    packed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.packed, bool):
            raise SchemaError("KVCacheView packed must be boolean")
        if type(self.key) is not type(self.value):
            raise SchemaError("K and V views must use the same representation")
        if not isinstance(self.key, (TensorView, QuantizedTensorView)):
            raise TypeError("KVCacheView requires tensor or quantized tensor views")
        if self.key.device != self.value.device:
            raise SchemaError("K and V views must share device")
        expected = self.spec.expected_shapes
        if self.packed:
            if not isinstance(self.key, TensorView) or self.key is not self.value:
                raise SchemaError(
                    "packed KV view must reference one combined dense storage view"
                )
            if len(expected) != 1 or self.key.shape != expected[0]:
                raise SchemaError("packed KV view shape does not match cache spec")
            if self.key.dtype != self.spec.dtype:
                raise SchemaError("packed KV view dtype does not match cache spec")
        else:
            if len(expected) != 2:
                raise SchemaError("separate KV view requires separate cache spec")
            key_shape = (
                self.key.logical_shape
                if isinstance(self.key, QuantizedTensorView)
                else self.key.shape
            )
            value_shape = (
                self.value.logical_shape
                if isinstance(self.value, QuantizedTensorView)
                else self.value.shape
            )
            if key_shape != expected[0] or value_shape != expected[1]:
                raise SchemaError("K/V view shapes do not match cache spec")
            if isinstance(self.key, QuantizedTensorView):
                if self.spec.quant_spec is None:
                    raise SchemaError("quantized KV view requires quantized cache spec")
                if (
                    self.key.quant_spec != self.spec.quant_spec
                    or self.value.quant_spec != self.spec.quant_spec
                ):
                    raise SchemaError("K/V QuantSpec does not match cache spec")
                if (
                    self.key.physical_layout_descriptor
                    != self.value.physical_layout_descriptor
                ):
                    raise SchemaError(
                        "K/V physical layout descriptors must match"
                    )
            else:
                if self.spec.quant_spec is not None:
                    raise SchemaError("dense KV view cannot use quantized cache spec")
                if self.key.dtype != self.spec.dtype or self.value.dtype != self.spec.dtype:
                    raise SchemaError("K/V view dtype does not match cache spec")
            _reject_aliases("K and V", self.component_views)
        if self.device != self.spec.device:
            raise SchemaError("KV view device does not match cache spec")

    @property
    def device(self) -> str:
        return self.key.device

    @property
    def quantized(self) -> bool:
        return isinstance(self.key, QuantizedTensorView)

    @property
    def component_views(self) -> Tuple[TensorView, ...]:
        if self.packed:
            return (self.key,)
        result = []
        for value in (self.key, self.value):
            if isinstance(value, TensorView):
                result.append(value)
            else:
                result.extend((value.storage, value.scale))
                if value.zero_point is not None:
                    result.append(value.zero_point)
        return tuple(result)

    @property
    def named_component_views(self) -> Tuple[Tuple[str, TensorView], ...]:
        if self.packed:
            return (("kv.packed_storage", self.key),)
        if isinstance(self.key, TensorView):
            return (
                ("kv.key_storage", self.key),
                ("kv.value_storage", self.value),
            )
        result = []
        for prefix, value in (("kv.key", self.key), ("kv.value", self.value)):
            assert isinstance(value, QuantizedTensorView)
            result.extend(
                (
                    (prefix + "_storage", value.storage),
                    (prefix + "_scale", value.scale),
                )
            )
            if value.zero_point is not None:
                result.append((prefix + "_zero_point", value.zero_point))
        return tuple(result)


@dataclass(frozen=True)
class StreamContext:
    device: str
    stream_id: str
    ordered: bool = True

    def __post_init__(self) -> None:
        if not self.device or not self.stream_id:
            raise SchemaError("stream device/id must be non-empty")
        if not isinstance(self.ordered, bool):
            raise SchemaError("stream ordered must be boolean")


@dataclass(frozen=True)
class AttentionTensorAccessPolicy:
    require_contiguous_q: bool = False
    require_contiguous_kv: bool = False
    require_contiguous_output: bool = False
    required_alignment: int = 1
    permit_output_input_alias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "require_contiguous_q",
            "require_contiguous_kv",
            "require_contiguous_output",
            "permit_output_input_alias",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SchemaError("%s must be boolean" % name)
        if self.required_alignment <= 0 or self.required_alignment & (
            self.required_alignment - 1
        ):
            raise SchemaError("access policy alignment must be a power of two")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "require_contiguous_q": self.require_contiguous_q,
            "require_contiguous_kv": self.require_contiguous_kv,
            "require_contiguous_output": self.require_contiguous_output,
            "required_alignment": self.required_alignment,
            "permit_output_input_alias": self.permit_output_input_alias,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionRunOptions:
    """Scalar values copied into ``FlashInferNpuAttentionRunOptionsV1``.

    A per-head auxiliary scale, when present, replaces the corresponding
    scalar.  This keeps the public FlashInfer scalar-or-tensor convention
    unambiguous at the kernel boundary.
    """

    q_scale: float = 1.0
    k_scale: float = 1.0
    v_scale: float = 1.0
    logits_soft_cap: float = 0.0
    flags: int = 0
    schema_version: int = ATTENTION_RUN_OPTIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_RUN_OPTIONS_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention run-options version")
        for name in ("q_scale", "k_scale", "v_scale", "logits_soft_cap"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise SchemaError("Attention run-option scalars must be numeric") from error
            if not math.isfinite(value):
                raise SchemaError("Attention run-option scalars must be finite")
            object.__setattr__(self, name, value)
        if self.logits_soft_cap < 0:
            raise SchemaError("run-time logits_soft_cap cannot be negative")
        if not isinstance(self.flags, int) or isinstance(self.flags, bool):
            raise SchemaError("Attention run-options flags must be an integer")
        if self.flags != 0:
            raise SchemaError("Attention run-options v1 flags must be zero")

    def validate_plan(self, plan: AttentionFrameworkPlan) -> None:
        if self.logits_soft_cap > 0 and float(plan.spec.logits_soft_cap or 0.0) <= 0:
            raise SchemaError("non-zero run-time logits_soft_cap requires a capped plan")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionRunOptions":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionRunOptions fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionRunOptions fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def pack(self) -> bytes:
        return ATTENTION_RUN_OPTIONS_C_ABI.pack(
            {
                "q_scale": self.q_scale,
                "k_scale": self.k_scale,
                "v_scale": self.v_scale,
                "logits_soft_cap": self.logits_soft_cap,
                "flags": self.flags,
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "AttentionRunOptions":
        values = ATTENTION_RUN_OPTIONS_C_ABI.unpack(payload)
        return cls(
            q_scale=values["q_scale"],
            k_scale=values["k_scale"],
            v_scale=values["v_scale"],
            logits_soft_cap=values["logits_soft_cap"],
            flags=values["flags"],
        )


@dataclass(frozen=True)
class AttentionAuxiliaryTensor:
    role: AttentionAuxiliaryRole
    view: TensorView
    schema_version: int = ATTENTION_AUXILIARY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_AUXILIARY_CONTRACT_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention auxiliary tensor version")
        try:
            object.__setattr__(self, "role", AttentionAuxiliaryRole(self.role))
        except ValueError as error:
            raise SchemaError("unknown Attention auxiliary tensor role") from error
        if not isinstance(self.view, TensorView):
            raise TypeError("Attention auxiliary tensor view must be TensorView")
        if self.role == AttentionAuxiliaryRole.PROFILER:
            if not self.view.writable:
                raise SchemaError("profiler auxiliary tensor must be writable")
        elif self.view.writable:
            raise SchemaError("Attention input auxiliary tensors must be read-only")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role.name,
            "view": self.view.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAuxiliaryTensor":
        data = dict(value)
        if set(data) != {"schema_version", "role", "view"}:
            raise SchemaError("AttentionAuxiliaryTensor fields are invalid")
        if not isinstance(data.get("view"), Mapping):
            raise SchemaError("Attention auxiliary view must be an object")
        try:
            data["role"] = AttentionAuxiliaryRole[str(data["role"])]
            data["view"] = TensorView.from_dict(data["view"])
            return cls(**data)
        except (KeyError, TypeError) as error:
            raise SchemaError("AttentionAuxiliaryTensor fields are invalid") from error


@dataclass(frozen=True)
class AttentionAuxiliaryContract:
    components: Tuple[AttentionAuxiliaryTensor, ...] = ()
    schema_version: int = ATTENTION_AUXILIARY_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_AUXILIARY_CONTRACT_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention auxiliary contract version")
        values = tuple(self.components)
        if any(not isinstance(item, AttentionAuxiliaryTensor) for item in values):
            raise TypeError("auxiliary components must be AttentionAuxiliaryTensor")
        if len({item.role for item in values}) != len(values):
            raise SchemaError("Attention auxiliary tensor roles must be unique")
        canonical = tuple(sorted(values, key=lambda item: int(item.role)))
        if values != canonical:
            raise SchemaError("Attention auxiliary tensor roles must use canonical order")
        object.__setattr__(self, "components", values)

    @property
    def by_role(self) -> Dict[AttentionAuxiliaryRole, TensorView]:
        return {item.role: item.view for item in self.components}

    @property
    def input_views(self) -> Tuple[TensorView, ...]:
        return tuple(
            item.view
            for item in self.components
            if item.role != AttentionAuxiliaryRole.PROFILER
        )

    @property
    def output_views(self) -> Tuple[TensorView, ...]:
        return tuple(
            item.view
            for item in self.components
            if item.role == AttentionAuxiliaryRole.PROFILER
        )

    def validate_plan(self, plan: AttentionFrameworkPlan, device: str) -> None:
        roles = self.by_role
        if any(view.device != device for view in roles.values()):
            raise SchemaError("Attention auxiliary tensors must share the run device")

        mask = roles.get(AttentionAuxiliaryRole.CUSTOM_MASK)
        mask_spec = plan.spec.custom_mask
        if (mask is None) != (mask_spec is None):
            raise SchemaError("custom-mask auxiliary presence must match the plan")
        if mask is not None and mask_spec is not None:
            expected_dtype = "uint8" if mask_spec.packed else "bool"
            if mask.shape != (mask_spec.numel,) or mask.dtype != expected_dtype:
                raise SchemaError("custom-mask auxiliary shape/dtype does not match the plan")

        scale_specs = (
            (AttentionAuxiliaryRole.Q_SCALE, plan.spec.num_qo_heads),
            (AttentionAuxiliaryRole.K_SCALE, plan.spec.num_kv_heads),
            (AttentionAuxiliaryRole.V_SCALE, plan.spec.num_kv_heads),
        )
        for role, heads in scale_specs:
            view = roles.get(role)
            if view is not None and (
                view.shape != (heads,) or view.dtype not in {"float32", "float64"}
            ):
                raise SchemaError("%s auxiliary must be float32/float64 with one value per head" % role.name.lower())

        profiler = roles.get(AttentionAuxiliaryRole.PROFILER)
        if (profiler is None) == bool(plan.spec.use_profiler):
            raise SchemaError("profiler auxiliary presence must match the plan")
        if profiler is not None and (
            profiler.dtype != "uint64" or len(profiler.shape) != 1 or profiler.numel < 1
        ):
            raise SchemaError("profiler auxiliary must be a non-empty rank-1 uint64 tensor")

        slopes = roles.get(AttentionAuxiliaryRole.ALIBI_SLOPES)
        if slopes is not None:
            if plan.spec.pos_encoding_mode.value != "ALIBI":
                raise SchemaError("ALiBi slopes auxiliary requires an ALIBI plan")
            if slopes.shape != (plan.spec.num_qo_heads,) or slopes.dtype not in {
                "float32",
                "float64",
            }:
                raise SchemaError("ALiBi slopes must be float32/float64 with one value per query head")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "components": [item.to_dict() for item in self.components],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAuxiliaryContract":
        data = dict(value)
        if set(data) != {"schema_version", "components"} or not isinstance(
            data.get("components"), (list, tuple)
        ):
            raise SchemaError("AttentionAuxiliaryContract fields are invalid")
        return cls(
            tuple(AttentionAuxiliaryTensor.from_dict(item) for item in data["components"]),
            schema_version=data["schema_version"],
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "components": [
                    {
                        "role": item.role.name,
                        "shape": list(item.view.shape),
                        "strides": list(item.view.strides),
                        "dtype": item.view.dtype,
                        "device": item.view.device,
                        "writable": item.view.writable,
                    }
                    for item in self.components
                ],
            }
        )


@dataclass(frozen=True)
class AttentionRunTensorContract:
    q: TensorView
    kv: KVCacheView
    stream: StreamContext
    out: Optional[TensorView] = None
    lse: Optional[TensorView] = None
    workspace_float: Optional[TensorView] = None
    workspace_int: Optional[TensorView] = None
    auxiliary: AttentionAuxiliaryContract = AttentionAuxiliaryContract()
    run_options: AttentionRunOptions = AttentionRunOptions()

    def validate(
        self,
        policy: AttentionTensorAccessPolicy,
        plan: Optional[AttentionFrameworkPlan] = None,
    ) -> None:
        if not isinstance(self.auxiliary, AttentionAuxiliaryContract):
            raise TypeError("auxiliary must be AttentionAuxiliaryContract")
        if not isinstance(self.run_options, AttentionRunOptions):
            raise TypeError("run_options must be AttentionRunOptions")
        views = self._named_views()
        if any(view.device != self.stream.device for _, view in views):
            raise SchemaError("all Attention views must be on the stream device")
        if not self.stream.ordered:
            raise SchemaError("Attention requires an ordered execution stream")
        for name, view in views:
            view.require_alignment(policy.required_alignment, name)
        if policy.require_contiguous_q and not self.q.is_contiguous:
            raise SchemaError("q must be contiguous for the selected backend")
        if policy.require_contiguous_kv and any(
            not view.is_contiguous for view in self.kv.component_views
        ):
            raise SchemaError("KV views must be contiguous for the selected backend")
        if self.out is not None:
            if not self.out.writable:
                raise SchemaError("out view must be writable")
            if policy.require_contiguous_output and not self.out.is_contiguous:
                raise SchemaError("out must be contiguous for the selected backend")
        if self.lse is not None and not self.lse.writable:
            raise SchemaError("lse view must be writable")
        if self.workspace_float is not None and not self.workspace_float.writable:
            raise SchemaError("float workspace view must be writable")
        if self.workspace_int is not None and not self.workspace_int.writable:
            raise SchemaError("int workspace view must be writable")
        if (self.workspace_float is None) != (self.workspace_int is None):
            raise SchemaError("float/int workspace views must be provided together")

        input_views = (self.q,) + self.kv.component_views + self.auxiliary.input_views
        outputs = tuple(view for view in (self.out, self.lse) if view is not None) + self.auxiliary.output_views
        if not policy.permit_output_input_alias:
            for output in outputs:
                for input_view in input_views:
                    if output.overlaps(input_view):
                        raise SchemaError("Attention output cannot alias an input")
        _reject_aliases("Attention outputs", outputs)
        workspaces = tuple(
            view
            for view in (self.workspace_float, self.workspace_int)
            if view is not None
        )
        _reject_aliases("Attention workspaces", workspaces)
        for workspace in workspaces:
            for name, view in views:
                if view is workspace:
                    continue
                if workspace.overlaps(view):
                    raise SchemaError("workspace cannot alias Attention tensors")
        if plan is not None:
            self.auxiliary.validate_plan(plan, self.stream.device)
            self.run_options.validate_plan(plan)
            profiler = self.auxiliary.by_role.get(AttentionAuxiliaryRole.PROFILER)
            plan.validate_run(
                TensorSpec(self.q.shape, self.q.dtype, self.q.device),
                self.kv.spec,
                out=(
                    TensorSpec(self.out.shape, self.out.dtype, self.out.device)
                    if self.out is not None
                    else None
                ),
                lse=(
                    TensorSpec(self.lse.shape, self.lse.dtype, self.lse.device)
                    if self.lse is not None
                    else None
                ),
                return_lse=self.lse is not None,
                logits_soft_cap=self.run_options.logits_soft_cap,
                profiler_buffer=(
                    TensorSpec(profiler.shape, profiler.dtype, profiler.device)
                    if profiler is not None
                    else None
                ),
            )

    def _named_views(self) -> Tuple[Tuple[str, TensorView], ...]:
        result = [("q", self.q)]
        result.extend(self.kv.named_component_views)
        result.extend(
            ("aux.%s" % item.role.name.lower(), item.view)
            for item in self.auxiliary.components
        )
        for name, view in (
            ("out", self.out),
            ("lse", self.lse),
            ("workspace_float", self.workspace_float),
            ("workspace_int", self.workspace_int),
        ):
            if view is not None:
                result.append((name, view))
        return tuple(result)

    @property
    def named_views(self) -> Tuple[Tuple[str, TensorView], ...]:
        return self._named_views()


def _reject_aliases(name: str, views: Iterable[TensorView]) -> None:
    values = tuple(views)
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if left.overlaps(right):
                raise SchemaError("%s cannot alias" % name)


def _reference_view(
    value,
    *,
    storage_id: str,
    writable: bool = False,
) -> TensorView:
    return TensorView(
        shape=value.shape,
        strides=contiguous_strides(value.shape),
        dtype=value.dtype,
        device=value.device,
        # Runtime identity is intentionally process-local; it models aliasing,
        # while semantic traces use their own deterministic fingerprints.
        storage_id="reference:%s"
        % hashlib.sha256(str(id(value)).encode("ascii")).hexdigest()[:16],
        storage_nbytes=_numel(value.shape) * dtype_itemsize(value.dtype),
        data_ptr_alignment=64,
        writable=writable,
    )


def reference_quantized_tensor_view(
    value: ReferenceQuantizedTensor, name: str
) -> QuantizedTensorView:
    return QuantizedTensorView(
        logical_shape=value.logical_shape,
        storage=_reference_view(value.storage, storage_id="%s.storage" % name),
        scale=_reference_view(value.scale, storage_id="%s.scale" % name),
        zero_point=(
            _reference_view(value.zero_point, storage_id="%s.zero" % name)
            if value.zero_point is not None
            else None
        ),
        quant_spec=value.quant_spec,
    )


def reference_kv_cache_view(value: ReferenceKVInput) -> KVCacheView:
    if isinstance(value, ReferenceQuantizedKVData):
        return KVCacheView(
            value.spec,
            reference_quantized_tensor_view(value.key_data, "kv.key"),
            reference_quantized_tensor_view(value.value_data, "kv.value"),
        )
    if not isinstance(value, ReferenceKVData):
        raise TypeError("unsupported reference KV data")
    if len(value.tensors) == 1:
        packed = _reference_view(value.tensors[0], storage_id="kv.packed")
        return KVCacheView(value.spec, packed, packed, packed=True)
    if len(value.tensors) != 2:
        raise SchemaError("Host tensor contract requires packed or separate K/V")
    return KVCacheView(
        value.spec,
        _reference_view(value.tensors[0], storage_id="kv.key"),
        _reference_view(value.tensors[1], storage_id="kv.value"),
    )


def validate_reference_attention_views(
    q: ReferenceTensor,
    kv_data: ReferenceKVInput,
    *,
    out: Optional[ReferenceBuffer] = None,
    lse: Optional[ReferenceBuffer] = None,
    workspace_float: Optional[ReferenceTensor] = None,
    workspace_int: Optional[ReferenceTensor] = None,
    auxiliary: AttentionAuxiliaryContract = AttentionAuxiliaryContract(),
    run_options: AttentionRunOptions = AttentionRunOptions(),
    plan: Optional[AttentionFrameworkPlan] = None,
) -> AttentionRunTensorContract:
    contract = AttentionRunTensorContract(
        q=_reference_view(q, storage_id="q"),
        kv=reference_kv_cache_view(kv_data),
        stream=StreamContext(q.device, "host-synchronous"),
        out=(
            _reference_view(out, storage_id="out", writable=True)
            if out is not None
            else None
        ),
        lse=(
            _reference_view(lse, storage_id="lse", writable=True)
            if lse is not None
            else None
        ),
        workspace_float=(
            _reference_view(
                workspace_float, storage_id="workspace.float", writable=True
            )
            if workspace_float is not None
            else None
        ),
        workspace_int=(
            _reference_view(
                workspace_int, storage_id="workspace.int", writable=True
            )
            if workspace_int is not None
            else None
        ),
        auxiliary=auxiliary,
        run_options=run_options,
    )
    contract.validate(AttentionTensorAccessPolicy(), plan=plan)
    return contract


def reference_attention_launch_options(
    plan: AttentionFrameworkPlan,
    device: str,
    *,
    custom_mask_data=None,
    q_scale=1.0,
    k_scale=1.0,
    v_scale=1.0,
    logits_soft_cap: float = 0.0,
    profiler_buffer: Optional[ReferenceBuffer] = None,
    alibi_slopes=None,
) -> Tuple[AttentionAuxiliaryContract, AttentionRunOptions]:
    """Adapt reference values to the same scalar-or-tensor launch boundary."""

    components = []

    def sequence_view(role, value, dtype="float32", writable=False):
        values = tuple(value)
        view = TensorView(
            shape=(len(values),),
            strides=(1,),
            dtype=dtype,
            device=device,
            storage_id="reference:aux:%s:%s"
            % (role.name.lower(), hashlib.sha256(str(id(value)).encode("ascii")).hexdigest()[:16]),
            storage_nbytes=len(values) * dtype_itemsize(dtype),
            data_ptr_alignment=64,
            writable=writable,
        )
        components.append(AttentionAuxiliaryTensor(role, view))

    if custom_mask_data is not None:
        if plan.spec.custom_mask is None:
            raise SchemaError("custom_mask_data was provided without a custom mask plan")
        sequence_view(
            AttentionAuxiliaryRole.CUSTOM_MASK,
            custom_mask_data,
            "uint8" if plan.spec.custom_mask.packed else "bool",
        )

    scalars = []
    for role, value in (
        (AttentionAuxiliaryRole.Q_SCALE, q_scale),
        (AttentionAuxiliaryRole.K_SCALE, k_scale),
        (AttentionAuxiliaryRole.V_SCALE, v_scale),
    ):
        if isinstance(value, (int, float)):
            scalars.append(float(value))
        else:
            sequence_view(role, value)
            scalars.append(1.0)

    if profiler_buffer is not None:
        components.append(
            AttentionAuxiliaryTensor(
                AttentionAuxiliaryRole.PROFILER,
                _reference_view(profiler_buffer, storage_id="aux.profiler", writable=True),
            )
        )
    if alibi_slopes is not None:
        sequence_view(AttentionAuxiliaryRole.ALIBI_SLOPES, alibi_slopes)
    components.sort(key=lambda item: int(item.role))
    auxiliary = AttentionAuxiliaryContract(tuple(components))
    options = AttentionRunOptions(
        q_scale=scalars[0],
        k_scale=scalars[1],
        v_scale=scalars[2],
        logits_soft_cap=logits_soft_cap,
    )
    auxiliary.validate_plan(plan, device)
    options.validate_plan(plan)
    return auxiliary, options


__all__ = [
    "ATTENTION_AUXILIARY_CONTRACT_SCHEMA_VERSION",
    "ATTENTION_RUN_OPTIONS_SCHEMA_VERSION",
    "ATTENTION_TENSOR_CONTRACT_SCHEMA_VERSION",
    "AttentionRunTensorContract",
    "AttentionRunOptions",
    "AttentionAuxiliaryContract",
    "AttentionAuxiliaryTensor",
    "AttentionTensorAccessPolicy",
    "KVCacheView",
    "QuantizedTensorView",
    "StreamContext",
    "TensorView",
    "contiguous_strides",
    "dtype_itemsize",
    "reference_kv_cache_view",
    "reference_attention_launch_options",
    "reference_quantized_tensor_view",
    "validate_reference_attention_views",
]
