"""Joint metadata contract for packed NVFP4 KV storage and scale factors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .nvfp4_scale_factor import (
    ATTENTION_NVFP4_SCALE_FACTOR_VERSION,
    NVFP4_SCALE_FACTOR_DTYPE,
    AttentionNvfp4ScaleFactorView,
    AttentionOperatorNvfp4ScaleFactorBinding,
    infer_attention_nvfp4_packed_storage_shape,
    inspect_attention_nvfp4_kv_scale_factors,
    validate_attention_nvfp4_kv_quant_spec,
)
from .operation_catalog import AttentionOperatorOperationSpec
from .operator_plan import AttentionOperatorActivePlan
from .operator_run import (
    AttentionLoweredOperatorCall,
    AttentionOperatorRunAdapter,
    AttentionOperatorRunRequest,
    AttentionOperatorTensorMetadataInspector,
)
from .planner import AttentionFrameworkPlan
from .schema import (
    AttentionMode,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    SingleAttentionMetadata,
)
from .tensor_contract import AttentionTensorAccessPolicy, TensorView


ATTENTION_NVFP4_PACKED_KV_VERSION = 1
NVFP4_STORAGE_SHAPE_RULE = "linear_last_dim_e2m1_pair_v1"
NVFP4_SCALE_SHAPE_RULE = "logical_outer_dims_d16_v1"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _power_of_two(value: int, name: str) -> int:
    if type(value) is not int or value <= 0 or value & (value - 1):
        raise SchemaError("%s must be a positive power of two" % name)
    return value


@dataclass(frozen=True)
class AttentionNvfp4PackedLayoutDescriptor:
    """Reviewed shape/alignment meaning for one provider NVFP4 layout id.

    This descriptor describes already-materialized input tensors.  Converter
    implementation and device execution deliberately remain outside this
    framework-only contract.
    """

    physical_layout: str
    packing_order: str
    storage_required_alignment: int = 1
    scale_required_alignment: int = 1
    storage_shape_rule: str = NVFP4_STORAGE_SHAPE_RULE
    scale_shape_rule: str = NVFP4_SCALE_SHAPE_RULE
    schema_version: int = ATTENTION_NVFP4_PACKED_KV_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_NVFP4_PACKED_KV_VERSION:
            raise SchemaError("unsupported Attention NVFP4 packed layout version")
        if (
            not isinstance(self.physical_layout, str)
            or not self.physical_layout
            or self.physical_layout == "logical"
        ):
            raise SchemaError("NVFP4 packed layout must name a non-logical layout")
        if not isinstance(self.packing_order, str) or not self.packing_order:
            raise SchemaError("NVFP4 packed layout must name the packing order")
        if self.storage_shape_rule != NVFP4_STORAGE_SHAPE_RULE:
            raise SchemaError("NVFP4 storage shape rule is not registered")
        if self.scale_shape_rule != NVFP4_SCALE_SHAPE_RULE:
            raise SchemaError("NVFP4 scale shape rule is not registered")
        _power_of_two(self.storage_required_alignment, "storage alignment")
        _power_of_two(self.scale_required_alignment, "scale alignment")

    def validate_quant_spec(self, quant_spec: QuantSpec) -> None:
        validate_attention_nvfp4_kv_quant_spec(quant_spec)
        if quant_spec.physical_layout != self.physical_layout:
            raise SchemaError("NVFP4 layout descriptor differs from QuantSpec")
        if quant_spec.packing_order != self.packing_order:
            raise SchemaError("NVFP4 packing descriptor differs from QuantSpec")

    def storage_shape(self, logical_shape) -> Tuple[int, ...]:
        return infer_attention_nvfp4_packed_storage_shape(logical_shape)

    def scale_shape(self, logical_shape) -> Tuple[int, ...]:
        try:
            shape = tuple(int(item) for item in logical_shape)
        except (TypeError, ValueError) as error:
            raise SchemaError("NVFP4 logical shape must contain integers") from error
        if not shape or any(item <= 0 for item in shape):
            raise SchemaError("NVFP4 logical shape must be non-empty and positive")
        if shape[-1] % 16:
            raise SchemaError("NVFP4 logical head dimension must be divisible by 16")
        return shape[:-1] + (shape[-1] // 16,)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "physical_layout": self.physical_layout,
            "packing_order": self.packing_order,
            "storage_required_alignment": self.storage_required_alignment,
            "scale_required_alignment": self.scale_required_alignment,
            "storage_shape_rule": self.storage_shape_rule,
            "scale_shape_rule": self.scale_shape_rule,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionNvfp4PackedLayoutDescriptor":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("NVFP4 packed layout descriptor fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError(
                "NVFP4 packed layout descriptor fields are invalid"
            ) from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorNvfp4PackedKVBinding:
    """Bind one NVFP4 parameter mapping to one packed-layout contract."""

    scale_factor_binding: AttentionOperatorNvfp4ScaleFactorBinding
    layout_descriptor: AttentionNvfp4PackedLayoutDescriptor
    schema_version: int = ATTENTION_NVFP4_PACKED_KV_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_NVFP4_PACKED_KV_VERSION:
            raise SchemaError("unsupported Attention NVFP4 packed binding version")
        if not isinstance(
            self.scale_factor_binding, AttentionOperatorNvfp4ScaleFactorBinding
        ):
            raise TypeError("scale_factor_binding has the wrong type")
        if not isinstance(
            self.layout_descriptor, AttentionNvfp4PackedLayoutDescriptor
        ):
            raise TypeError("layout_descriptor has the wrong type")
        self.layout_descriptor.validate_quant_spec(
            self.scale_factor_binding.quant_spec
        )

    @property
    def provider_id(self) -> str:
        return self.scale_factor_binding.provider_id

    @property
    def operation_id(self) -> str:
        return self.scale_factor_binding.operation_id

    @property
    def quant_spec(self) -> QuantSpec:
        return self.scale_factor_binding.quant_spec

    @property
    def accepted_structures(self) -> Tuple[str, ...]:
        return self.scale_factor_binding.accepted_structures

    def validate_operation(self, operation: AttentionOperatorOperationSpec) -> None:
        self.scale_factor_binding.validate_operation(operation)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "scale_factor_binding": self.scale_factor_binding.to_dict(),
            "layout_descriptor": self.layout_descriptor.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _require_tensor(
    view: TensorView,
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: str,
    device: str,
    alignment: int,
) -> None:
    if not isinstance(view, TensorView):
        raise TypeError("%s must be TensorView" % name)
    if view.shape != shape:
        raise SchemaError("%s shape must be %r" % (name, shape))
    if view.dtype != dtype:
        raise SchemaError("%s dtype must be %s" % (name, dtype))
    if view.device != device:
        raise SchemaError("%s device must match the provider query" % name)
    if not view.is_contiguous:
        raise SchemaError("%s must be contiguous" % name)
    view.require_alignment(alignment, name)


@dataclass(frozen=True)
class AttentionNvfp4PackedKVView:
    """One plan-bound, fully validated NVFP4 KV storage/scale view."""

    plan_fingerprint: str
    structure: str
    key_logical_shape: Tuple[int, ...]
    value_logical_shape: Tuple[int, ...]
    quant_spec: QuantSpec
    layout_descriptor: AttentionNvfp4PackedLayoutDescriptor
    scale_factors: AttentionNvfp4ScaleFactorView
    key_storage: Optional[TensorView] = None
    value_storage: Optional[TensorView] = None
    combined_storage: Optional[TensorView] = None
    schema_version: int = ATTENTION_NVFP4_PACKED_KV_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_NVFP4_PACKED_KV_VERSION:
            raise SchemaError("unsupported Attention NVFP4 packed KV version")
        if (
            not isinstance(self.plan_fingerprint, str)
            or len(self.plan_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.plan_fingerprint)
        ):
            raise SchemaError("NVFP4 packed KV requires a plan SHA-256 fingerprint")
        if self.structure not in {"separate", "combined"}:
            raise SchemaError("NVFP4 packed KV structure is invalid")
        for name in ("key_logical_shape", "value_logical_shape"):
            shape = tuple(int(item) for item in getattr(self, name))
            if not shape or any(item <= 0 for item in shape):
                raise SchemaError("NVFP4 packed KV logical shape is invalid")
            object.__setattr__(self, name, shape)
        if not isinstance(
            self.layout_descriptor, AttentionNvfp4PackedLayoutDescriptor
        ):
            raise TypeError("NVFP4 packed KV layout descriptor has the wrong type")
        if not isinstance(self.scale_factors, AttentionNvfp4ScaleFactorView):
            raise TypeError("NVFP4 packed KV scale factors have the wrong type")
        self.layout_descriptor.validate_quant_spec(self.quant_spec)
        if self.scale_factors.structure != self.structure:
            raise SchemaError("NVFP4 storage and scale-factor structures differ")

        key_storage_shape = self.layout_descriptor.storage_shape(
            self.key_logical_shape
        )
        value_storage_shape = self.layout_descriptor.storage_shape(
            self.value_logical_shape
        )
        key_scale_shape = self.layout_descriptor.scale_shape(self.key_logical_shape)
        value_scale_shape = self.layout_descriptor.scale_shape(
            self.value_logical_shape
        )
        if (
            self.scale_factors.key_shape != key_scale_shape
            or self.scale_factors.value_shape != value_scale_shape
        ):
            raise SchemaError("NVFP4 scale-factor shape differs from logical KV")

        scale_alignment = self.layout_descriptor.scale_required_alignment
        storage_alignment = self.layout_descriptor.storage_required_alignment
        if self.structure == "separate":
            if (
                self.key_storage is None
                or self.value_storage is None
                or self.combined_storage is not None
            ):
                raise SchemaError("separate NVFP4 KV requires K and V storage")
            device = self.key_storage.device
            _require_tensor(
                self.key_storage,
                name="kv.key.storage",
                shape=key_storage_shape,
                dtype="uint8",
                device=device,
                alignment=storage_alignment,
            )
            _require_tensor(
                self.value_storage,
                name="kv.value.storage",
                shape=value_storage_shape,
                dtype="uint8",
                device=device,
                alignment=storage_alignment,
            )
            _require_tensor(
                self.scale_factors.key,
                name="kv_cache_sf.key",
                shape=key_scale_shape,
                dtype=NVFP4_SCALE_FACTOR_DTYPE,
                device=device,
                alignment=scale_alignment,
            )
            _require_tensor(
                self.scale_factors.value,
                name="kv_cache_sf.value",
                shape=value_scale_shape,
                dtype=NVFP4_SCALE_FACTOR_DTYPE,
                device=device,
                alignment=scale_alignment,
            )
        else:
            if (
                self.combined_storage is None
                or self.key_storage is not None
                or self.value_storage is not None
            ):
                raise SchemaError("combined NVFP4 KV requires one storage tensor")
            if len(self.key_logical_shape) != 4:
                raise SchemaError("combined NVFP4 KV is only defined for paged cache")
            if key_storage_shape != value_storage_shape:
                raise SchemaError("combined NVFP4 KV requires equal K/V dimensions")
            combined_storage_shape = (
                key_storage_shape[0],
                2,
            ) + key_storage_shape[1:]
            combined_scale_shape = (key_scale_shape[0], 2) + key_scale_shape[1:]
            device = self.combined_storage.device
            _require_tensor(
                self.combined_storage,
                name="kv.packed_storage",
                shape=combined_storage_shape,
                dtype="uint8",
                device=device,
                alignment=storage_alignment,
            )
            _require_tensor(
                self.scale_factors.combined,
                name="kv_cache_sf",
                shape=combined_scale_shape,
                dtype=NVFP4_SCALE_FACTOR_DTYPE,
                device=device,
                alignment=scale_alignment,
            )

        views = self.component_views
        for index, first in enumerate(views):
            for second in views[index + 1 :]:
                if first.overlaps(second):
                    raise SchemaError("NVFP4 packed KV components cannot alias")

    @property
    def device(self) -> str:
        if self.combined_storage is not None:
            return self.combined_storage.device
        return self.key_storage.device

    @property
    def component_views(self) -> Tuple[TensorView, ...]:
        if self.structure == "combined":
            return (self.combined_storage, self.scale_factors.combined)
        return (
            self.key_storage,
            self.value_storage,
            self.scale_factors.key,
            self.scale_factors.value,
        )

    @property
    def named_views(self) -> Tuple[Tuple[str, TensorView], ...]:
        if self.structure == "combined":
            return (
                ("kv.packed_storage", self.combined_storage),
                ("kv_cache_sf", self.scale_factors.combined),
            )
        return (
            ("kv.key.storage", self.key_storage),
            ("kv.value.storage", self.value_storage),
            ("kv_cache_sf.key", self.scale_factors.key),
            ("kv_cache_sf.value", self.scale_factors.value),
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "plan_fingerprint": self.plan_fingerprint,
            "structure": self.structure,
            "key_logical_shape": list(self.key_logical_shape),
            "value_logical_shape": list(self.value_logical_shape),
            "quant_spec": self.quant_spec.to_dict(),
            "layout_descriptor": self.layout_descriptor.to_dict(),
            "views": {
                name: view.to_dict() for name, view in self.named_views
            },
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _paged_metadata(plan: AttentionFrameworkPlan):
    metadata = plan.metadata
    if isinstance(metadata, PagedPrefillMetadata):
        return metadata.paged_kv
    if isinstance(metadata, (PagedKVMetadata, MixedPagedKVMetadata)):
        return metadata
    return None


def _logical_kv_shapes(
    plan: AttentionFrameworkPlan, num_pages: Optional[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    spec = plan.spec
    metadata = plan.metadata
    if isinstance(metadata, SingleAttentionMetadata):
        if spec.mode not in {AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE}:
            raise SchemaError("NVFP4 packed KV does not match the Attention mode")
        if spec.kv_layout == KVLayout.NHD:
            return (
                (metadata.kv_len, spec.num_kv_heads, spec.head_dim_qk),
                (metadata.kv_len, spec.num_kv_heads, int(spec.head_dim_vo)),
            )
        return (
            (spec.num_kv_heads, metadata.kv_len, spec.head_dim_qk),
            (spec.num_kv_heads, metadata.kv_len, int(spec.head_dim_vo)),
        )
    paged = _paged_metadata(plan)
    if paged is None:
        raise SchemaError("NVFP4 packed KV requires single or paged metadata")
    if num_pages is None or num_pages <= paged.max_page_index:
        raise SchemaError("NVFP4 packed KV omits a referenced KV page")
    if spec.kv_layout == KVLayout.NHD:
        return (
            (num_pages, paged.page_size, spec.num_kv_heads, spec.head_dim_qk),
            (num_pages, paged.page_size, spec.num_kv_heads, int(spec.head_dim_vo)),
        )
    return (
        (num_pages, spec.num_kv_heads, paged.page_size, spec.head_dim_qk),
        (num_pages, spec.num_kv_heads, paged.page_size, int(spec.head_dim_vo)),
    )


def _inspect_storage(
    inspector: AttentionOperatorTensorMetadataInspector,
    value,
    *,
    name: str,
) -> TensorView:
    view = inspector.to_view(value, name=name, writable=False)
    if not isinstance(view, TensorView):
        raise TypeError("tensor metadata inspector must return TensorView")
    return view


def inspect_attention_nvfp4_packed_kv_input(
    plan: AttentionFrameworkPlan,
    kv_cache,
    kv_cache_sf,
    inspector: AttentionOperatorTensorMetadataInspector,
    expected_device: str,
    layout_descriptor: AttentionNvfp4PackedLayoutDescriptor,
) -> AttentionNvfp4PackedKVView:
    """Validate public packed KV and ``kv_cache_sf`` without operator execution."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
        raise TypeError("inspector must implement AttentionOperatorTensorMetadataInspector")
    if not isinstance(layout_descriptor, AttentionNvfp4PackedLayoutDescriptor):
        raise TypeError("layout_descriptor has the wrong type")
    device = str(expected_device)
    if not device:
        raise SchemaError("expected_device must be non-empty")
    quant_spec = plan.spec.kv_quant_spec
    if quant_spec is None:
        raise SchemaError("NVFP4 packed KV requires a planned QuantSpec")
    layout_descriptor.validate_quant_spec(quant_spec)

    separate = isinstance(kv_cache, (tuple, list))
    if separate and len(kv_cache) != 2:
        raise SchemaError("NVFP4 KV tuple must contain key and value storage")
    if separate:
        key_storage = _inspect_storage(
            inspector, kv_cache[0], name="kv.key.storage"
        )
        value_storage = _inspect_storage(
            inspector, kv_cache[1], name="kv.value.storage"
        )
        combined_storage = None
        first_shape = key_storage.shape
        structure = "separate"
    else:
        if isinstance(plan.metadata, SingleAttentionMetadata):
            raise SchemaError("single Attention NVFP4 KV must be a (K, V) tuple")
        key_storage = None
        value_storage = None
        combined_storage = _inspect_storage(
            inspector, kv_cache, name="kv.packed_storage"
        )
        first_shape = combined_storage.shape
        structure = "combined"
    num_pages = None if isinstance(plan.metadata, SingleAttentionMetadata) else (
        first_shape[0] if first_shape else None
    )
    key_logical_shape, value_logical_shape = _logical_kv_shapes(plan, num_pages)
    scale_factors = inspect_attention_nvfp4_kv_scale_factors(
        plan, kv_cache_sf, inspector, device
    )
    if scale_factors.structure != structure:
        raise SchemaError("NVFP4 storage and scale-factor structures differ")
    result = AttentionNvfp4PackedKVView(
        plan_fingerprint=plan.fingerprint,
        structure=structure,
        key_logical_shape=key_logical_shape,
        value_logical_shape=value_logical_shape,
        quant_spec=quant_spec,
        layout_descriptor=layout_descriptor,
        scale_factors=scale_factors,
        key_storage=key_storage,
        value_storage=value_storage,
        combined_storage=combined_storage,
    )
    if result.device != device:
        raise SchemaError("NVFP4 packed KV device must match the provider query")
    return result


class AttentionOperatorNvfp4PackedKVRunAdapter:
    """Lower one exact public NVFP4 KV input without executing the operation."""

    def __init__(
        self,
        base_adapter: AttentionOperatorRunAdapter,
        operation: AttentionOperatorOperationSpec,
        binding: AttentionOperatorNvfp4PackedKVBinding,
        tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector,
        tensor_access_policy: AttentionTensorAccessPolicy,
        expected_device: str,
    ) -> None:
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(binding, AttentionOperatorNvfp4PackedKVBinding):
            raise TypeError("binding must be AttentionOperatorNvfp4PackedKVBinding")
        binding.validate_operation(operation)
        if base_adapter.provider_id != operation.provider_id:
            raise SchemaError("NVFP4 packed base adapter provider differs")
        if not isinstance(
            tensor_metadata_inspector, AttentionOperatorTensorMetadataInspector
        ):
            raise TypeError("tensor_metadata_inspector has the wrong type")
        if not isinstance(tensor_access_policy, AttentionTensorAccessPolicy):
            raise TypeError("tensor_access_policy has the wrong type")
        if not str(expected_device):
            raise SchemaError("NVFP4 packed expected_device must be non-empty")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._base_adapter = base_adapter
        self._binding = binding
        self._inspector = tensor_metadata_inspector
        self._access_policy = tensor_access_policy
        self._expected_device = str(expected_device)

    def lower(
        self,
        active_plan: AttentionOperatorActivePlan,
        request: AttentionOperatorRunRequest,
    ) -> AttentionLoweredOperatorCall:
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        if not isinstance(request, AttentionOperatorRunRequest):
            raise TypeError("request must be AttentionOperatorRunRequest")
        if active_plan.prepared_plan.implementation_id != self.operation_id:
            raise SchemaError("NVFP4 packed active operation differs from binding")
        quant_spec = active_plan.framework_plan.spec.kv_quant_spec
        matches = (
            quant_spec is not None
            and quant_spec.fingerprint == self._binding.quant_spec.fingerprint
        )
        if not matches:
            if request.kv_cache_sf is not None:
                raise SchemaError(
                    "kv_cache_sf has no exact NVFP4 packed QuantSpec binding"
                )
            return self._base_adapter.lower(active_plan, request)
        if request.kv_cache_sf is None:
            raise SchemaError("bound NVFP4 packed KV requires kv_cache_sf")

        packed_kv = inspect_attention_nvfp4_packed_kv_input(
            active_plan.framework_plan,
            request.kv_cache,
            request.kv_cache_sf,
            self._inspector,
            self._expected_device,
            self._binding.layout_descriptor,
        )
        if packed_kv.structure not in self._binding.accepted_structures:
            raise SchemaError("NVFP4 packed KV structure is not bound by operation")
        for name, view in packed_kv.named_views:
            view.require_alignment(self._access_policy.required_alignment, name)
            if self._access_policy.require_contiguous_kv and not view.is_contiguous:
                raise SchemaError("%s must be contiguous for provider lowering" % name)
        if not self._access_policy.permit_output_input_alias:
            for output_name, output in (("out", request.out), ("lse", request.lse)):
                if output is None:
                    continue
                output_view = self._inspector.to_view(
                    output, name=output_name, writable=True
                )
                if not isinstance(output_view, TensorView):
                    raise TypeError("tensor metadata inspector must return TensorView")
                for input_name, input_view in packed_kv.named_views:
                    if output_view.overlaps(input_view):
                        raise SchemaError(
                            "%s cannot alias %s" % (output_name, input_name)
                        )

        delegated_request = replace(request, kv_cache_sf=None)
        lowered = self._base_adapter.lower(active_plan, delegated_request)
        if not isinstance(lowered, AttentionLoweredOperatorCall):
            raise TypeError("base run adapter returned an invalid call description")
        scale_binding = self._binding.scale_factor_binding
        if packed_kv.structure == "combined":
            injected = ((scale_binding.combined_argument, request.kv_cache_sf),)
        else:
            injected = (
                (scale_binding.key_argument, request.kv_cache_sf[0]),
                (scale_binding.value_argument, request.kv_cache_sf[1]),
            )
        existing_arguments = {
            name
            for name, _ in lowered.positional_arguments + lowered.keyword_arguments
        }
        collision = existing_arguments.intersection(name for name, _ in injected)
        if collision:
            raise SchemaError(
                "NVFP4 packed argument collides with provider lowering: %s"
                % sorted(collision)[0]
            )
        existing_views = tuple(lowered.validated_input_views)
        existing_view_names = {name for name, _ in existing_views}
        if existing_view_names.intersection(name for name, _ in packed_kv.named_views):
            raise SchemaError("NVFP4 packed validated input view name collides")
        return replace(
            lowered,
            keyword_arguments=lowered.keyword_arguments + injected,
            validated_input_views=existing_views + packed_kv.named_views,
            consumed_request_fields=request.consumed_fields,
        )


class AttentionOperatorNvfp4PackedKVRunAdapterFactory:
    """Late-bind the reviewed joint NVFP4 contract to a provider device."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        binding: AttentionOperatorNvfp4PackedKVBinding,
        tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector,
        tensor_access_policy: AttentionTensorAccessPolicy,
    ) -> None:
        if not isinstance(binding, AttentionOperatorNvfp4PackedKVBinding):
            raise TypeError("binding must be AttentionOperatorNvfp4PackedKVBinding")
        binding.validate_operation(operation)
        if not isinstance(
            tensor_metadata_inspector, AttentionOperatorTensorMetadataInspector
        ):
            raise TypeError("tensor_metadata_inspector has the wrong type")
        if not isinstance(tensor_access_policy, AttentionTensorAccessPolicy):
            raise TypeError("tensor_access_policy has the wrong type")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation
        self._binding = binding
        self._inspector = tensor_metadata_inspector
        self._access_policy = tensor_access_policy

    def build(
        self, base_adapter: AttentionOperatorRunAdapter, device: str
    ) -> AttentionOperatorRunAdapter:
        return AttentionOperatorNvfp4PackedKVRunAdapter(
            base_adapter,
            self._operation,
            self._binding,
            self._inspector,
            self._access_policy,
            str(device),
        )


__all__ = [
    "ATTENTION_NVFP4_PACKED_KV_VERSION",
    "NVFP4_SCALE_SHAPE_RULE",
    "NVFP4_STORAGE_SHAPE_RULE",
    "AttentionNvfp4PackedKVView",
    "AttentionNvfp4PackedLayoutDescriptor",
    "AttentionOperatorNvfp4PackedKVBinding",
    "AttentionOperatorNvfp4PackedKVRunAdapter",
    "AttentionOperatorNvfp4PackedKVRunAdapterFactory",
    "inspect_attention_nvfp4_packed_kv_input",
]
