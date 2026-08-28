"""Host-side NVFP4 KV scale-factor shape and tensor contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .operator_run import AttentionOperatorTensorMetadataInspector
from .planner import AttentionFrameworkPlan
from .schema import (
    AttentionMode,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    SingleAttentionMetadata,
)
from .tensor_contract import TensorView


ATTENTION_NVFP4_SCALE_FACTOR_VERSION = 1
NVFP4_SCALE_FACTOR_DTYPE = "float8_e4m3fn"


@dataclass(frozen=True)
class AttentionNvfp4ScaleFactorView:
    """Validated public ``kv_cache_sf`` structure for one framework plan."""

    structure: str
    key_shape: Tuple[int, ...]
    value_shape: Tuple[int, ...]
    key: Optional[TensorView] = None
    value: Optional[TensorView] = None
    combined: Optional[TensorView] = None
    layout_id: str = "linear"
    schema_version: int = ATTENTION_NVFP4_SCALE_FACTOR_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_NVFP4_SCALE_FACTOR_VERSION:
            raise SchemaError("unsupported Attention NVFP4 scale-factor version")
        if self.structure not in {"separate", "combined"}:
            raise SchemaError("NVFP4 scale-factor structure is invalid")
        if self.layout_id != "linear":
            raise SchemaError("NVFP4 scale-factor layout is not registered")
        for name in ("key_shape", "value_shape"):
            shape = tuple(int(item) for item in getattr(self, name))
            if not shape or any(item <= 0 for item in shape):
                raise SchemaError("NVFP4 scale-factor shape is invalid")
            object.__setattr__(self, name, shape)
        if self.structure == "separate":
            if self.key is None or self.value is None or self.combined is not None:
                raise SchemaError("separate NVFP4 scale factors require K and V")
        else:
            if self.combined is None or self.key is not None or self.value is not None:
                raise SchemaError("combined NVFP4 scale factors require one tensor")

    @property
    def named_views(self) -> Tuple[Tuple[str, TensorView], ...]:
        if self.structure == "combined":
            return (("kv_cache_sf", self.combined),)
        return (("kv_cache_sf.key", self.key), ("kv_cache_sf.value", self.value))


def _inspect(
    inspector: AttentionOperatorTensorMetadataInspector,
    value,
    *,
    name: str,
    expected_shape: Tuple[int, ...],
    expected_device: str,
) -> TensorView:
    view = inspector.to_view(value, name=name, writable=False)
    if not isinstance(view, TensorView):
        raise TypeError("tensor metadata inspector must return TensorView")
    if view.shape != expected_shape:
        raise SchemaError("%s shape must be %r" % (name, expected_shape))
    if view.dtype != NVFP4_SCALE_FACTOR_DTYPE:
        raise SchemaError("%s dtype must be %s" % (name, NVFP4_SCALE_FACTOR_DTYPE))
    if view.device != expected_device:
        raise SchemaError("%s device must match the provider query" % name)
    if not view.is_contiguous:
        raise SchemaError("%s must use the linear contiguous layout" % name)
    return view


def _scale_shapes(plan: AttentionFrameworkPlan, num_pages: Optional[int]):
    spec = plan.spec
    if spec.head_dim_qk % 16 or int(spec.head_dim_vo) % 16:
        raise SchemaError("NVFP4 head dimensions must be divisible by 16")
    key_block = spec.head_dim_qk // 16
    value_block = int(spec.head_dim_vo) // 16
    metadata = plan.metadata
    if isinstance(metadata, SingleAttentionMetadata):
        if spec.mode not in {AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE}:
            raise SchemaError("NVFP4 scale factors do not match the Attention mode")
        if spec.kv_layout == KVLayout.NHD:
            return (
                (metadata.kv_len, spec.num_kv_heads, key_block),
                (metadata.kv_len, spec.num_kv_heads, value_block),
            )
        return (
            (spec.num_kv_heads, metadata.kv_len, key_block),
            (spec.num_kv_heads, metadata.kv_len, value_block),
        )
    if isinstance(metadata, PagedPrefillMetadata):
        paged = metadata.paged_kv
    elif isinstance(metadata, (PagedKVMetadata, MixedPagedKVMetadata)):
        paged = metadata
    else:
        raise SchemaError("NVFP4 kv_cache_sf requires single or paged metadata")
    if num_pages is None or num_pages <= paged.max_page_index:
        raise SchemaError("NVFP4 scale factors omit a referenced KV page")
    if spec.kv_layout == KVLayout.NHD:
        return (
            (num_pages, paged.page_size, spec.num_kv_heads, key_block),
            (num_pages, paged.page_size, spec.num_kv_heads, value_block),
        )
    return (
        (num_pages, spec.num_kv_heads, paged.page_size, key_block),
        (num_pages, spec.num_kv_heads, paged.page_size, value_block),
    )


def inspect_attention_nvfp4_kv_scale_factors(
    plan: AttentionFrameworkPlan,
    value,
    inspector: AttentionOperatorTensorMetadataInspector,
    expected_device: str,
) -> AttentionNvfp4ScaleFactorView:
    """Validate FlashInfer-shaped scale factors without executing a provider."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
        raise TypeError("inspector must implement AttentionOperatorTensorMetadataInspector")
    device = str(expected_device)
    if not device:
        raise SchemaError("expected_device must be non-empty")
    separate = isinstance(value, (tuple, list))
    if separate and len(value) != 2:
        raise SchemaError("kv_cache_sf tuple must contain key and value scales")
    if separate:
        first_view = inspector.to_view(value[0], name="kv_cache_sf.key", writable=False)
        if not isinstance(first_view, TensorView):
            raise TypeError("tensor metadata inspector must return TensorView")
        num_pages = None if isinstance(plan.metadata, SingleAttentionMetadata) else (
            first_view.shape[0] if first_view.shape else None
        )
        key_shape, value_shape = _scale_shapes(plan, num_pages)
        key = _inspect(
            inspector, value[0], name="kv_cache_sf.key",
            expected_shape=key_shape, expected_device=device,
        )
        val = _inspect(
            inspector, value[1], name="kv_cache_sf.value",
            expected_shape=value_shape, expected_device=device,
        )
        return AttentionNvfp4ScaleFactorView(
            structure="separate", key_shape=key_shape, value_shape=value_shape,
            key=key, value=val,
        )
    combined_view = inspector.to_view(value, name="kv_cache_sf", writable=False)
    if not isinstance(combined_view, TensorView):
        raise TypeError("tensor metadata inspector must return TensorView")
    if isinstance(plan.metadata, SingleAttentionMetadata):
        raise SchemaError("single Attention kv_cache_sf must be a (K, V) tuple")
    num_pages = combined_view.shape[0] if combined_view.shape else None
    key_shape, value_shape = _scale_shapes(plan, num_pages)
    if key_shape != value_shape:
        raise SchemaError("combined kv_cache_sf requires equal K/V head dimensions")
    combined_shape = (key_shape[0], 2) + key_shape[1:]
    combined = _inspect(
        inspector, value, name="kv_cache_sf", expected_shape=combined_shape,
        expected_device=device,
    )
    return AttentionNvfp4ScaleFactorView(
        structure="combined", key_shape=key_shape, value_shape=value_shape,
        combined=combined,
    )


__all__ = [
    "ATTENTION_NVFP4_SCALE_FACTOR_VERSION",
    "NVFP4_SCALE_FACTOR_DTYPE",
    "AttentionNvfp4ScaleFactorView",
    "inspect_attention_nvfp4_kv_scale_factors",
]
