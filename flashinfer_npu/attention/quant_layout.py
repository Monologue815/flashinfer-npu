"""Pure logical-to-physical helpers shared by quantized tensor adapters."""

from __future__ import annotations

from typing import Sequence, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError


def normalize_quant_axes(
    logical_shape: Sequence[int], quant_spec: QuantSpec
) -> Tuple[int, ...]:
    shape = tuple(int(dim) for dim in logical_shape)
    rank = len(shape)
    axes = tuple(quant_spec.axis or ())
    normalized = tuple(axis + rank if axis < 0 else axis for axis in axes)
    if any(axis < 0 or axis >= rank for axis in normalized):
        raise SchemaError("quantization axis is out of logical tensor rank")
    if len(set(normalized)) != len(normalized):
        raise SchemaError("quantization axis cannot contain normalized duplicates")
    return normalized


def infer_quant_scale_shape(
    logical_shape: Sequence[int], quant_spec: QuantSpec
) -> Tuple[int, ...]:
    shape = tuple(int(dim) for dim in logical_shape)
    if not shape or any(dim < 0 for dim in shape):
        raise SchemaError("quantized logical shape must be non-empty and non-negative")
    granularity = quant_spec.granularity
    axes = normalize_quant_axes(shape, quant_spec)
    if granularity == "tensor":
        if axes or quant_spec.group_size is not None:
            raise SchemaError("tensor quantization cannot define axis/group_size")
        return ()
    if not axes:
        raise SchemaError("non-tensor quantization requires axis")
    if granularity in {"group", "block"}:
        groups = tuple(quant_spec.group_size or ())
        if len(groups) != len(axes):
            raise SchemaError("group_size must have one value per axis")
        return tuple(
            (shape[axis] + size - 1) // size
            for axis, size in zip(axes, groups)
        )
    if quant_spec.group_size is not None:
        raise SchemaError("%s quantization does not use group_size" % granularity)
    return tuple(shape[axis] for axis in axes)


def infer_logical_quant_storage_shape(
    logical_shape: Sequence[int], quant_spec: QuantSpec
) -> Tuple[int, ...]:
    shape = tuple(int(dim) for dim in logical_shape)
    if not shape or any(dim < 0 for dim in shape):
        raise SchemaError("quantized logical shape must be non-empty and non-negative")
    if quant_spec.storage_dtype in {"int4_packed", "uint4_packed"}:
        return shape[:-1] + ((shape[-1] + 1) // 2,)
    return shape


def infer_quant_storage_shape(
    logical_shape: Sequence[int], quant_spec: QuantSpec, *, layout_catalog=None
) -> Tuple[int, ...]:
    logical_storage = infer_logical_quant_storage_shape(logical_shape, quant_spec)
    if quant_spec.physical_layout == "logical":
        return logical_storage
    if layout_catalog is None:
        raise NotImplementedError(
            "physical storage shape requires a registered non-logical layout"
        )
    from .quant_physical_layout import QuantPhysicalLayoutCatalog

    if not isinstance(layout_catalog, QuantPhysicalLayoutCatalog):
        raise TypeError("layout_catalog must be QuantPhysicalLayoutCatalog")
    return layout_catalog.resolve(quant_spec).physical_shapes(
        logical_shape, quant_spec
    ).storage


__all__ = [
    "infer_quant_scale_shape",
    "infer_logical_quant_storage_shape",
    "infer_quant_storage_shape",
    "normalize_quant_axes",
]
