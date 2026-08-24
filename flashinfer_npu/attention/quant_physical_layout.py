"""Explicit, non-executing plans for quantized physical layout conversion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .quant_layout import (
    infer_logical_quant_storage_shape,
    infer_quant_scale_shape,
)


ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION = 1
_LAYOUT_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_COMPONENTS = ("storage", "scale", "zero_point")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _axis_token(value: str, rank: int) -> Tuple[str, int]:
    if not isinstance(value, str) or len(value) < 2 or value[0] not in {"o", "i"}:
        raise SchemaError("physical axis tokens must use o<axis> or i<axis>")
    try:
        axis = int(value[1:])
    except ValueError as error:
        raise SchemaError("physical axis token index is invalid") from error
    if axis < 0 or axis >= rank or str(axis) != value[1:]:
        raise SchemaError("physical axis token is outside transform rank")
    return value[0], axis


@dataclass(frozen=True)
class QuantPhysicalAxisTransform:
    """Reversible permutation/blocking transform over an input storage shape.

    ``oN`` is the outer coordinate of input axis N and ``iN`` its block-local
    coordinate. Every outer axis appears once; an inner axis appears exactly
    when its block size is greater than one.
    """

    input_rank: int
    axis_blocks: Tuple[int, ...]
    physical_axes: Tuple[str, ...]
    required_alignment: int = 1
    padding_value: float = 0.0
    schema_version: int = ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION:
            raise SchemaError("unsupported quant physical axis transform version")
        if type(self.input_rank) is not int or self.input_rank < 0:
            raise SchemaError("quant physical transform input_rank cannot be negative")
        blocks = tuple(int(value) for value in self.axis_blocks)
        axes = tuple(str(value) for value in self.physical_axes)
        object.__setattr__(self, "axis_blocks", blocks)
        object.__setattr__(self, "physical_axes", axes)
        if len(blocks) != self.input_rank or any(value <= 0 for value in blocks):
            raise SchemaError("axis_blocks must contain one positive value per input axis")
        if (
            type(self.required_alignment) is not int
            or self.required_alignment <= 0
            or self.required_alignment & (self.required_alignment - 1)
        ):
            raise SchemaError("physical layout alignment must be a power of two")
        padding = float(self.padding_value)
        if not math.isfinite(padding):
            raise SchemaError("physical layout padding_value must be finite")
        object.__setattr__(self, "padding_value", padding)
        parsed = tuple(_axis_token(value, self.input_rank) for value in axes)
        if len(set(parsed)) != len(parsed):
            raise SchemaError("physical axis tokens cannot repeat")
        outer = {axis for kind, axis in parsed if kind == "o"}
        inner = {axis for kind, axis in parsed if kind == "i"}
        expected_outer = set(range(self.input_rank))
        expected_inner = {
            axis for axis, block in enumerate(blocks) if block > 1
        }
        if outer != expected_outer or inner != expected_inner:
            raise SchemaError(
                "physical axes must contain every outer axis and exactly blocked inner axes"
            )

    def physical_shape(self, input_shape: Sequence[int]) -> Tuple[int, ...]:
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != self.input_rank or any(value < 0 for value in shape):
            raise SchemaError("physical transform input shape does not match rank")
        result = []
        for token in self.physical_axes:
            kind, axis = _axis_token(token, self.input_rank)
            block = self.axis_blocks[axis]
            result.append(
                (shape[axis] + block - 1) // block
                if kind == "o"
                else block
            )
        return tuple(result)

    def logical_to_physical(
        self, input_shape: Sequence[int], coordinate: Sequence[int]
    ) -> Tuple[int, ...]:
        shape = tuple(int(value) for value in input_shape)
        point = tuple(int(value) for value in coordinate)
        self.physical_shape(shape)
        if len(point) != self.input_rank or any(
            value < 0 or value >= dim for value, dim in zip(point, shape)
        ):
            raise SchemaError("logical storage coordinate is out of bounds")
        result = []
        for token in self.physical_axes:
            kind, axis = _axis_token(token, self.input_rank)
            block = self.axis_blocks[axis]
            result.append(point[axis] // block if kind == "o" else point[axis] % block)
        return tuple(result)

    def physical_to_logical(
        self, input_shape: Sequence[int], coordinate: Sequence[int]
    ) -> Optional[Tuple[int, ...]]:
        shape = tuple(int(value) for value in input_shape)
        physical_shape = self.physical_shape(shape)
        point = tuple(int(value) for value in coordinate)
        if len(point) != len(physical_shape) or any(
            value < 0 or value >= dim
            for value, dim in zip(point, physical_shape)
        ):
            raise SchemaError("physical storage coordinate is out of bounds")
        outer = [0] * self.input_rank
        inner = [0] * self.input_rank
        for token, value in zip(self.physical_axes, point):
            kind, axis = _axis_token(token, self.input_rank)
            if kind == "o":
                outer[axis] = value
            else:
                inner[axis] = value
        logical = tuple(
            outer[axis] * block + inner[axis]
            for axis, block in enumerate(self.axis_blocks)
        )
        return None if any(value >= dim for value, dim in zip(logical, shape)) else logical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_rank": self.input_rank,
            "axis_blocks": list(self.axis_blocks),
            "physical_axes": list(self.physical_axes),
            "required_alignment": self.required_alignment,
            "padding_value": self.padding_value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantPhysicalAxisTransform":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("QuantPhysicalAxisTransform fields are invalid")
        if not isinstance(data.get("axis_blocks"), (list, tuple)) or not isinstance(
            data.get("physical_axes"), (list, tuple)
        ):
            raise SchemaError("quant physical transform axes/blocks must be arrays")
        data["axis_blocks"] = tuple(data["axis_blocks"])
        data["physical_axes"] = tuple(data["physical_axes"])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("QuantPhysicalAxisTransform fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class QuantPhysicalLayoutShapes:
    storage: Tuple[int, ...]
    scale: Tuple[int, ...]
    zero_point: Optional[Tuple[int, ...]]

    def __post_init__(self) -> None:
        for name in ("storage", "scale"):
            value = tuple(int(dim) for dim in getattr(self, name))
            if any(dim < 0 for dim in value):
                raise SchemaError("quant physical component shapes cannot be negative")
            object.__setattr__(self, name, value)
        if self.zero_point is not None:
            value = tuple(int(dim) for dim in self.zero_point)
            if any(dim < 0 for dim in value):
                raise SchemaError("quant zero-point physical shape cannot be negative")
            object.__setattr__(self, "zero_point", value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage": list(self.storage),
            "scale": list(self.scale),
            "zero_point": list(self.zero_point) if self.zero_point is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantPhysicalLayoutShapes":
        data = dict(value)
        if set(data) != {"storage", "scale", "zero_point"}:
            raise SchemaError("QuantPhysicalLayoutShapes fields are invalid")
        for name in ("storage", "scale"):
            if not isinstance(data[name], (list, tuple)):
                raise SchemaError("quant physical %s shape must be an array" % name)
            data[name] = tuple(data[name])
        if data["zero_point"] is not None:
            if not isinstance(data["zero_point"], (list, tuple)):
                raise SchemaError("quant physical zero_point shape must be an array")
            data["zero_point"] = tuple(data["zero_point"])
        return cls(**data)


@dataclass(frozen=True)
class QuantPhysicalLayoutDescriptor:
    layout_id: str
    storage_dtypes: Tuple[str, ...]
    storage_transform: QuantPhysicalAxisTransform
    storage_converter_id: str
    storage_inverse_converter_id: str
    scale_transform: Optional[QuantPhysicalAxisTransform] = None
    scale_converter_id: Optional[str] = None
    scale_inverse_converter_id: Optional[str] = None
    zero_point_transform: Optional[QuantPhysicalAxisTransform] = None
    zero_point_converter_id: Optional[str] = None
    zero_point_inverse_converter_id: Optional[str] = None
    required_features: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION:
            raise SchemaError("unsupported quant physical layout descriptor version")
        if not isinstance(self.layout_id, str) or not _LAYOUT_ID.fullmatch(self.layout_id):
            raise SchemaError("quant physical layout_id is invalid")
        if self.layout_id == "logical":
            raise SchemaError("logical layout does not require a descriptor")
        dtypes = tuple(str(value) for value in self.storage_dtypes)
        features = tuple(str(value) for value in self.required_features)
        object.__setattr__(self, "storage_dtypes", dtypes)
        object.__setattr__(self, "required_features", features)
        if not dtypes or any(not value for value in dtypes) or len(set(dtypes)) != len(dtypes):
            raise SchemaError("quant physical storage_dtypes must be non-empty and unique")
        if any(not value for value in features) or len(set(features)) != len(features):
            raise SchemaError("quant physical required_features must be unique")
        if not isinstance(self.storage_transform, QuantPhysicalAxisTransform):
            raise TypeError("storage_transform must be QuantPhysicalAxisTransform")
        for name in ("storage_converter_id", "storage_inverse_converter_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise SchemaError("quant physical %s must be non-empty" % name)
        for component in ("scale", "zero_point"):
            transform = getattr(self, "%s_transform" % component)
            converter = getattr(self, "%s_converter_id" % component)
            inverse = getattr(self, "%s_inverse_converter_id" % component)
            if transform is None:
                if converter is not None or inverse is not None:
                    raise SchemaError(
                        "%s converter ids require a physical transform" % component
                    )
            else:
                if not isinstance(transform, QuantPhysicalAxisTransform):
                    raise TypeError("%s_transform has the wrong type" % component)
                if not converter or not inverse:
                    raise SchemaError(
                        "%s physical transform requires forward/inverse converter ids"
                        % component
                    )

    def validate_quant_spec(
        self, logical_shape: Sequence[int], quant_spec: QuantSpec
    ) -> None:
        if quant_spec.physical_layout != self.layout_id:
            raise SchemaError("QuantSpec physical_layout does not match descriptor")
        if quant_spec.storage_dtype not in self.storage_dtypes:
            raise SchemaError("quant physical layout does not support storage dtype")
        storage_shape = infer_logical_quant_storage_shape(logical_shape, quant_spec)
        scale_shape = infer_quant_scale_shape(logical_shape, quant_spec)
        if self.storage_transform.input_rank != len(storage_shape):
            raise SchemaError("storage transform rank does not match logical storage")
        if self.scale_transform is not None and self.scale_transform.input_rank != len(
            scale_shape
        ):
            raise SchemaError("scale transform rank does not match quant scale")
        if (
            self.zero_point_transform is not None
            and self.zero_point_transform.input_rank != len(scale_shape)
        ):
            raise SchemaError("zero-point transform rank does not match quant scale")

    def physical_shapes(
        self, logical_shape: Sequence[int], quant_spec: QuantSpec
    ) -> QuantPhysicalLayoutShapes:
        self.validate_quant_spec(logical_shape, quant_spec)
        storage = infer_logical_quant_storage_shape(logical_shape, quant_spec)
        scale = infer_quant_scale_shape(logical_shape, quant_spec)
        return QuantPhysicalLayoutShapes(
            self.storage_transform.physical_shape(storage),
            (
                self.scale_transform.physical_shape(scale)
                if self.scale_transform is not None
                else scale
            ),
            (
                self.zero_point_transform.physical_shape(scale)
                if quant_spec.has_zero_point and self.zero_point_transform is not None
                else scale
                if quant_spec.has_zero_point
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layout_id": self.layout_id,
            "storage_dtypes": list(self.storage_dtypes),
            "storage_transform": self.storage_transform.to_dict(),
            "storage_converter_id": self.storage_converter_id,
            "storage_inverse_converter_id": self.storage_inverse_converter_id,
            "scale_transform": (
                self.scale_transform.to_dict() if self.scale_transform is not None else None
            ),
            "scale_converter_id": self.scale_converter_id,
            "scale_inverse_converter_id": self.scale_inverse_converter_id,
            "zero_point_transform": (
                self.zero_point_transform.to_dict()
                if self.zero_point_transform is not None
                else None
            ),
            "zero_point_converter_id": self.zero_point_converter_id,
            "zero_point_inverse_converter_id": self.zero_point_inverse_converter_id,
            "required_features": list(self.required_features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantPhysicalLayoutDescriptor":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("QuantPhysicalLayoutDescriptor fields are invalid")
        for name in ("storage_dtypes", "required_features"):
            if not isinstance(data.get(name), (list, tuple)):
                raise SchemaError("quant physical descriptor %s must be an array" % name)
            data[name] = tuple(data[name])
        for name in ("storage_transform", "scale_transform", "zero_point_transform"):
            transform = data.get(name)
            if transform is not None:
                if not isinstance(transform, Mapping):
                    raise SchemaError("quant physical %s must be an object" % name)
                data[name] = QuantPhysicalAxisTransform.from_dict(transform)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("QuantPhysicalLayoutDescriptor fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class QuantPhysicalLayoutCatalog:
    descriptors: Tuple[QuantPhysicalLayoutDescriptor, ...] = ()
    schema_version: int = ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION:
            raise SchemaError("unsupported quant physical layout catalog version")
        values = tuple(self.descriptors)
        if any(not isinstance(value, QuantPhysicalLayoutDescriptor) for value in values):
            raise TypeError("quant physical catalog contains wrong descriptor type")
        ids = tuple(value.layout_id for value in values)
        if len(set(ids)) != len(ids):
            raise SchemaError("quant physical layout ids must be unique")
        object.__setattr__(self, "descriptors", values)

    def resolve(self, quant_spec: QuantSpec) -> QuantPhysicalLayoutDescriptor:
        matches = tuple(
            item for item in self.descriptors if item.layout_id == quant_spec.physical_layout
        )
        if len(matches) != 1:
            raise SchemaError(
                "QuantSpec physical_layout requires exactly one registered descriptor"
            )
        return matches[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_quant_physical_layout_catalog",
            "descriptors": [value.to_dict() for value in self.descriptors],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantPhysicalLayoutCatalog":
        data = dict(value)
        if set(data) != {"schema_version", "kind", "descriptors"}:
            raise SchemaError("QuantPhysicalLayoutCatalog fields are invalid")
        if data["kind"] != "attention_quant_physical_layout_catalog":
            raise SchemaError("quant physical layout catalog kind is invalid")
        if not isinstance(data["descriptors"], (list, tuple)):
            raise SchemaError("quant physical layout descriptors must be an array")
        return cls(
            tuple(
                QuantPhysicalLayoutDescriptor.from_dict(item)
                for item in data["descriptors"]
            ),
            int(data["schema_version"]),
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG = QuantPhysicalLayoutCatalog()


@dataclass(frozen=True)
class QuantLayoutConversionStep:
    component: str
    source_layout: str
    destination_layout: str
    converter_id: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    required_alignment: int

    def __post_init__(self) -> None:
        if self.component not in _COMPONENTS:
            raise SchemaError("quant layout conversion component is invalid")
        for name in ("source_layout", "destination_layout", "converter_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise SchemaError("quant layout conversion %s must be non-empty" % name)
        for name in ("input_shape", "output_shape"):
            shape = tuple(int(dim) for dim in getattr(self, name))
            if any(dim < 0 for dim in shape):
                raise SchemaError("quant layout conversion shapes cannot be negative")
            object.__setattr__(self, name, shape)
        if (
            type(self.required_alignment) is not int
            or self.required_alignment <= 0
            or self.required_alignment & (self.required_alignment - 1)
        ):
            raise SchemaError("quant layout conversion alignment must be power of two")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "source_layout": self.source_layout,
            "destination_layout": self.destination_layout,
            "converter_id": self.converter_id,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "required_alignment": self.required_alignment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantLayoutConversionStep":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("QuantLayoutConversionStep fields are invalid")
        for name in ("input_shape", "output_shape"):
            if not isinstance(data[name], (list, tuple)):
                raise SchemaError("quant layout conversion %s must be an array" % name)
            data[name] = tuple(data[name])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("QuantLayoutConversionStep fields are invalid") from error


@dataclass(frozen=True)
class QuantLayoutConversionPlan:
    logical_shape: Tuple[int, ...]
    source_quant_spec_fingerprint: str
    destination_quant_spec_fingerprint: str
    source_layout: str
    destination_layout: str
    source_shapes: QuantPhysicalLayoutShapes
    destination_shapes: QuantPhysicalLayoutShapes
    steps: Tuple[QuantLayoutConversionStep, ...]
    schema_version: int = ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION:
            raise SchemaError("unsupported quant layout conversion plan version")
        shape = tuple(int(dim) for dim in self.logical_shape)
        if not shape or any(dim < 0 for dim in shape):
            raise SchemaError("quant layout conversion logical_shape is invalid")
        object.__setattr__(self, "logical_shape", shape)
        for name in ("source_quant_spec_fingerprint", "destination_quant_spec_fingerprint"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise SchemaError("quant layout conversion fingerprints must be SHA-256")
        if not self.source_layout or not self.destination_layout:
            raise SchemaError("quant layout conversion layouts must be non-empty")
        if not isinstance(self.source_shapes, QuantPhysicalLayoutShapes) or not isinstance(
            self.destination_shapes, QuantPhysicalLayoutShapes
        ):
            raise TypeError("quant layout conversion physical shapes have wrong type")
        steps = tuple(self.steps)
        if any(not isinstance(step, QuantLayoutConversionStep) for step in steps):
            raise TypeError("quant layout conversion steps have wrong type")
        object.__setattr__(self, "steps", steps)

    @property
    def requires_conversion(self) -> bool:
        return bool(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_shape": list(self.logical_shape),
            "source_quant_spec_fingerprint": self.source_quant_spec_fingerprint,
            "destination_quant_spec_fingerprint": self.destination_quant_spec_fingerprint,
            "source_layout": self.source_layout,
            "destination_layout": self.destination_layout,
            "source_shapes": self.source_shapes.to_dict(),
            "destination_shapes": self.destination_shapes.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantLayoutConversionPlan":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("QuantLayoutConversionPlan fields are invalid")
        if not isinstance(data.get("logical_shape"), (list, tuple)):
            raise SchemaError("quant layout conversion logical_shape must be an array")
        data["logical_shape"] = tuple(data["logical_shape"])
        for name in ("source_shapes", "destination_shapes"):
            if not isinstance(data.get(name), Mapping):
                raise SchemaError("quant layout conversion %s must be an object" % name)
            data[name] = QuantPhysicalLayoutShapes.from_dict(data[name])
        if not isinstance(data.get("steps"), (list, tuple)):
            raise SchemaError("quant layout conversion steps must be an array")
        data["steps"] = tuple(
            QuantLayoutConversionStep.from_dict(step) for step in data["steps"]
        )
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("QuantLayoutConversionPlan fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _logical_shapes(
    logical_shape: Sequence[int], quant_spec: QuantSpec
) -> QuantPhysicalLayoutShapes:
    scale = infer_quant_scale_shape(logical_shape, quant_spec)
    return QuantPhysicalLayoutShapes(
        infer_logical_quant_storage_shape(logical_shape, quant_spec),
        scale,
        scale if quant_spec.has_zero_point else None,
    )


def infer_quant_physical_shapes(
    logical_shape: Sequence[int],
    quant_spec: QuantSpec,
    catalog: QuantPhysicalLayoutCatalog = EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
) -> QuantPhysicalLayoutShapes:
    if quant_spec.physical_layout == "logical":
        return _logical_shapes(logical_shape, quant_spec)
    if not isinstance(catalog, QuantPhysicalLayoutCatalog):
        raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
    return catalog.resolve(quant_spec).physical_shapes(logical_shape, quant_spec)


def _semantic_quant_dict(quant_spec: QuantSpec) -> Dict[str, Any]:
    value = quant_spec.to_dict()
    value.pop("physical_layout")
    return value


def _component_transform(
    descriptor: Optional[QuantPhysicalLayoutDescriptor], component: str
):
    if descriptor is None:
        return None, None, None
    return (
        getattr(descriptor, "%s_transform" % component),
        getattr(descriptor, "%s_converter_id" % component),
        getattr(descriptor, "%s_inverse_converter_id" % component),
    )


def plan_quant_layout_conversion(
    logical_shape: Sequence[int],
    source: QuantSpec,
    destination: QuantSpec,
    catalog: QuantPhysicalLayoutCatalog = EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
) -> QuantLayoutConversionPlan:
    if _semantic_quant_dict(source) != _semantic_quant_dict(destination):
        raise SchemaError(
            "quant layout conversion cannot change quantization semantics"
        )
    if not isinstance(catalog, QuantPhysicalLayoutCatalog):
        raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
    logical = _logical_shapes(logical_shape, source)
    source_descriptor = (
        None if source.physical_layout == "logical" else catalog.resolve(source)
    )
    destination_descriptor = (
        None
        if destination.physical_layout == "logical"
        else catalog.resolve(destination)
    )
    source_shapes = infer_quant_physical_shapes(logical_shape, source, catalog)
    destination_shapes = infer_quant_physical_shapes(
        logical_shape, destination, catalog
    )
    if source_descriptor is not None:
        source_descriptor.validate_quant_spec(logical_shape, source)
    if destination_descriptor is not None:
        destination_descriptor.validate_quant_spec(logical_shape, destination)

    steps = []
    logical_components = {
        "storage": logical.storage,
        "scale": logical.scale,
        "zero_point": logical.zero_point,
    }
    source_components = {
        "storage": source_shapes.storage,
        "scale": source_shapes.scale,
        "zero_point": source_shapes.zero_point,
    }
    destination_components = {
        "storage": destination_shapes.storage,
        "scale": destination_shapes.scale,
        "zero_point": destination_shapes.zero_point,
    }
    for component in _COMPONENTS:
        logical_component = logical_components[component]
        if logical_component is None:
            continue
        source_transform, _source_forward, source_inverse = _component_transform(
            source_descriptor, component
        )
        destination_transform, destination_forward, _destination_inverse = (
            _component_transform(destination_descriptor, component)
        )
        if source.physical_layout == destination.physical_layout:
            continue
        if source_transform is not None:
            assert source_inverse is not None
            steps.append(
                QuantLayoutConversionStep(
                    component,
                    source.physical_layout,
                    "logical",
                    source_inverse,
                    source_components[component],
                    logical_component,
                    1,
                )
            )
        if destination_transform is not None:
            assert destination_forward is not None
            steps.append(
                QuantLayoutConversionStep(
                    component,
                    "logical",
                    destination.physical_layout,
                    destination_forward,
                    logical_component,
                    destination_components[component],
                    destination_transform.required_alignment,
                )
            )
    return QuantLayoutConversionPlan(
        tuple(int(dim) for dim in logical_shape),
        source.fingerprint,
        destination.fingerprint,
        source.physical_layout,
        destination.physical_layout,
        source_shapes,
        destination_shapes,
        tuple(steps),
    )


__all__ = [
    "ATTENTION_QUANT_PHYSICAL_LAYOUT_VERSION",
    "EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG",
    "QuantLayoutConversionPlan",
    "QuantLayoutConversionStep",
    "QuantPhysicalAxisTransform",
    "QuantPhysicalLayoutCatalog",
    "QuantPhysicalLayoutDescriptor",
    "QuantPhysicalLayoutShapes",
    "infer_quant_physical_shapes",
    "plan_quant_layout_conversion",
]
