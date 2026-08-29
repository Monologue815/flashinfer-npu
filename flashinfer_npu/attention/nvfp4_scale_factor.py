"""Host-side NVFP4 KV scale-factor shape and tensor contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError

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


ATTENTION_NVFP4_SCALE_FACTOR_VERSION = 1
NVFP4_SCALE_FACTOR_DTYPE = "float8_e4m3fn"
FLASHINFER_NVFP4_PHYSICAL_LAYOUT = "flashinfer_nvfp4_linear_e2m1x2_v1"
FLASHINFER_NVFP4_PACKING_ORDER = "low_nibble_even_high_nibble_odd_v1"
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def attention_nvfp4_kv_quant_spec(
    *,
    physical_layout: str,
    packing_order: str,
    compute_dtype: str = "float32",
    accumulator_dtype: str = "float32",
) -> QuantSpec:
    """Build an NVFP4 contract for already-materialized packed input tensors."""

    if str(physical_layout) == "logical" or not str(physical_layout):
        raise SchemaError("NVFP4 requires a named non-logical physical layout")
    if not str(packing_order):
        raise SchemaError("NVFP4 requires an explicit provider packing order")
    return QuantSpec(
        scheme="symmetric",
        storage_dtype="uint8",
        compute_dtype=str(compute_dtype),
        accumulator_dtype=str(accumulator_dtype),
        scale_dtype=NVFP4_SCALE_FACTOR_DTYPE,
        granularity="block",
        group_size=(16,),
        axis=(-1,),
        has_zero_point=False,
        physical_layout=str(physical_layout),
        packing_order=str(packing_order),
    )


def flashinfer_nvfp4_kv_quant_spec() -> QuantSpec:
    """Return the provider-neutral NVFP4 KV contract of the public facade.

    FlashInfer materializes two E2M1 values per ``uint8`` byte, with the even
    logical element in the low nibble, and uses one linear E4M3 scale for each
    block of 16 logical values.  Providers may lower this canonical input to a
    private operator layout, but provider-private layouts never enter the
    public Attention plan.
    """

    return attention_nvfp4_kv_quant_spec(
        physical_layout=FLASHINFER_NVFP4_PHYSICAL_LAYOUT,
        packing_order=FLASHINFER_NVFP4_PACKING_ORDER,
    )


def is_flashinfer_nvfp4_kv_quant_spec(value: Optional[QuantSpec]) -> bool:
    """Whether ``value`` is the exact public FlashInfer NVFP4 KV contract."""

    return (
        isinstance(value, QuantSpec)
        and value.fingerprint == flashinfer_nvfp4_kv_quant_spec().fingerprint
    )


def validate_attention_nvfp4_kv_quant_spec(quant_spec: QuantSpec) -> QuantSpec:
    """Require the canonical public NVFP4 KV quantization semantics."""

    if not isinstance(quant_spec, QuantSpec):
        raise TypeError("NVFP4 quant_spec must be QuantSpec")
    if (
        quant_spec.scheme != "symmetric"
        or quant_spec.storage_dtype != "uint8"
        or quant_spec.scale_dtype != NVFP4_SCALE_FACTOR_DTYPE
        or quant_spec.granularity != "block"
        or quant_spec.group_size != (16,)
        or quant_spec.axis != (-1,)
        or quant_spec.has_zero_point
        or quant_spec.physical_layout == "logical"
        or not quant_spec.physical_layout
        or not quant_spec.packing_order
    ):
        raise SchemaError("NVFP4 QuantSpec is not canonical")
    return quant_spec


def infer_attention_nvfp4_packed_storage_shape(
    logical_shape,
) -> Tuple[int, ...]:
    """Return two-E2M1-values-per-byte storage shape for logical K or V."""

    try:
        shape = tuple(int(item) for item in logical_shape)
    except (TypeError, ValueError) as error:
        raise SchemaError("NVFP4 logical shape must contain integers") from error
    if not shape or any(item <= 0 for item in shape):
        raise SchemaError("NVFP4 logical shape must be non-empty and positive")
    if shape[-1] % 2:
        raise SchemaError("NVFP4 logical head dimension must be even")
    return shape[:-1] + (shape[-1] // 2,)


@dataclass(frozen=True)
class AttentionOperatorNvfp4ScaleFactorBinding:
    """Exact operation arguments authorized to consume public ``kv_cache_sf``."""

    provider_id: str
    operation_id: str
    quant_spec: QuantSpec
    combined_argument: Optional[str] = None
    key_argument: Optional[str] = None
    value_argument: Optional[str] = None
    layout_id: str = "linear"
    schema_version: int = ATTENTION_NVFP4_SCALE_FACTOR_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_NVFP4_SCALE_FACTOR_VERSION:
            raise SchemaError("unsupported Attention NVFP4 binding version")
        if not str(self.provider_id) or not str(self.operation_id):
            raise SchemaError("NVFP4 binding identities must be non-empty")
        try:
            validate_attention_nvfp4_kv_quant_spec(self.quant_spec)
        except SchemaError as error:
            raise SchemaError("NVFP4 binding QuantSpec is not canonical") from error
        if self.layout_id != "linear":
            raise SchemaError("NVFP4 binding layout is not registered")
        arguments = tuple(
            item
            for item in (
                self.combined_argument,
                self.key_argument,
                self.value_argument,
            )
            if item is not None
        )
        if (self.key_argument is None) != (self.value_argument is None):
            raise SchemaError("NVFP4 separate binding requires K and V arguments")
        if not arguments:
            raise SchemaError("NVFP4 binding requires an operation argument")
        if any(not _ARGUMENT_NAME.fullmatch(str(item)) for item in arguments):
            raise SchemaError("NVFP4 binding argument name is invalid")
        if len(set(arguments)) != len(arguments):
            raise SchemaError("NVFP4 binding arguments must be unique")
        for name in ("provider_id", "operation_id"):
            object.__setattr__(self, name, str(getattr(self, name)))

    @property
    def accepted_structures(self) -> Tuple[str, ...]:
        values = []
        if self.combined_argument is not None:
            values.append("combined")
        if self.key_argument is not None:
            values.append("separate")
        return tuple(values)

    def validate_operation(self, operation: AttentionOperatorOperationSpec) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if (
            operation.provider_id != self.provider_id
            or operation.operation_id != self.operation_id
        ):
            raise SchemaError("NVFP4 binding operation identity differs")
        arguments = {
            item
            for item in (
                self.combined_argument,
                self.key_argument,
                self.value_argument,
            )
            if item is not None
        }
        if not arguments.issubset(operation.quant_arguments):
            raise SchemaError("NVFP4 binding uses a non-quant operation argument")
        if not arguments.issubset(operation.keyword_arguments):
            raise SchemaError("NVFP4 binding arguments must be keyword arguments")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "quant_spec": self.quant_spec.to_dict(),
            "combined_argument": self.combined_argument,
            "key_argument": self.key_argument,
            "value_argument": self.value_argument,
            "layout_id": self.layout_id,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


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


class AttentionOperatorNvfp4ScaleFactorRunAdapter:
    """Validate and inject an exactly bound ``kv_cache_sf`` argument."""

    def __init__(
        self,
        base_adapter: AttentionOperatorRunAdapter,
        operation: AttentionOperatorOperationSpec,
        binding: AttentionOperatorNvfp4ScaleFactorBinding,
        tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector,
        tensor_access_policy: AttentionTensorAccessPolicy,
        expected_device: str,
    ) -> None:
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(binding, AttentionOperatorNvfp4ScaleFactorBinding):
            raise TypeError("binding must be AttentionOperatorNvfp4ScaleFactorBinding")
        binding.validate_operation(operation)
        if base_adapter.provider_id != operation.provider_id:
            raise SchemaError("NVFP4 base adapter provider differs")
        if not isinstance(
            tensor_metadata_inspector, AttentionOperatorTensorMetadataInspector
        ):
            raise TypeError("tensor_metadata_inspector has the wrong type")
        if not isinstance(tensor_access_policy, AttentionTensorAccessPolicy):
            raise TypeError("tensor_access_policy has the wrong type")
        if not str(expected_device):
            raise SchemaError("NVFP4 expected_device must be non-empty")
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
            raise SchemaError("NVFP4 active operation differs from binding")
        if request.kv_cache_sf is None:
            return self._base_adapter.lower(active_plan, request)
        quant_spec = active_plan.framework_plan.spec.kv_quant_spec
        if (
            quant_spec is None
            or quant_spec.fingerprint != self._binding.quant_spec.fingerprint
        ):
            raise SchemaError("NVFP4 run does not match the bound QuantSpec")
        scale_factors = inspect_attention_nvfp4_kv_scale_factors(
            active_plan.framework_plan,
            request.kv_cache_sf,
            self._inspector,
            self._expected_device,
        )
        if scale_factors.structure not in self._binding.accepted_structures:
            raise SchemaError(
                "NVFP4 scale-factor structure is not bound by the operation"
            )
        for name, view in scale_factors.named_views:
            view.require_alignment(self._access_policy.required_alignment, name)
        if not self._access_policy.permit_output_input_alias:
            for output_name, output in (("out", request.out), ("lse", request.lse)):
                if output is None:
                    continue
                output_view = self._inspector.to_view(
                    output, name=output_name, writable=True
                )
                if not isinstance(output_view, TensorView):
                    raise TypeError("tensor metadata inspector must return TensorView")
                for input_name, input_view in scale_factors.named_views:
                    if output_view.overlaps(input_view):
                        raise SchemaError(
                            "%s cannot alias %s" % (output_name, input_name)
                        )
        delegated_request = replace(request, kv_cache_sf=None)
        lowered = self._base_adapter.lower(active_plan, delegated_request)
        if not isinstance(lowered, AttentionLoweredOperatorCall):
            raise TypeError("base run adapter returned an invalid call description")
        if scale_factors.structure == "combined":
            injected = ((self._binding.combined_argument, request.kv_cache_sf),)
        else:
            injected = (
                (self._binding.key_argument, request.kv_cache_sf[0]),
                (self._binding.value_argument, request.kv_cache_sf[1]),
            )
        existing = {
            name
            for name, _ in lowered.positional_arguments + lowered.keyword_arguments
        }
        collision = existing.intersection(name for name, _ in injected)
        if collision:
            raise SchemaError(
                "NVFP4 argument collides with provider lowering: %s"
                % sorted(collision)[0]
            )
        existing_views = tuple(lowered.validated_input_views)
        existing_view_names = {name for name, _ in existing_views}
        if existing_view_names.intersection(name for name, _ in scale_factors.named_views):
            raise SchemaError("NVFP4 validated input view name collides")
        return replace(
            lowered,
            keyword_arguments=lowered.keyword_arguments + injected,
            validated_input_views=existing_views + scale_factors.named_views,
            consumed_request_fields=request.consumed_fields,
        )


class AttentionOperatorNvfp4ScaleFactorRunAdapterFactory:
    """Attach the reviewed NVFP4 binding after device resolution."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        binding: AttentionOperatorNvfp4ScaleFactorBinding,
        tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector,
        tensor_access_policy: AttentionTensorAccessPolicy,
    ) -> None:
        if not isinstance(binding, AttentionOperatorNvfp4ScaleFactorBinding):
            raise TypeError("binding must be AttentionOperatorNvfp4ScaleFactorBinding")
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
        return AttentionOperatorNvfp4ScaleFactorRunAdapter(
            base_adapter,
            self._operation,
            self._binding,
            self._inspector,
            self._access_policy,
            str(device),
        )


__all__ = [
    "ATTENTION_NVFP4_SCALE_FACTOR_VERSION",
    "FLASHINFER_NVFP4_PACKING_ORDER",
    "FLASHINFER_NVFP4_PHYSICAL_LAYOUT",
    "NVFP4_SCALE_FACTOR_DTYPE",
    "AttentionNvfp4ScaleFactorView",
    "AttentionOperatorNvfp4ScaleFactorBinding",
    "AttentionOperatorNvfp4ScaleFactorRunAdapter",
    "AttentionOperatorNvfp4ScaleFactorRunAdapterFactory",
    "attention_nvfp4_kv_quant_spec",
    "flashinfer_nvfp4_kv_quant_spec",
    "infer_attention_nvfp4_packed_storage_shape",
    "is_flashinfer_nvfp4_kv_quant_spec",
    "inspect_attention_nvfp4_kv_scale_factors",
    "validate_attention_nvfp4_kv_quant_spec",
]
