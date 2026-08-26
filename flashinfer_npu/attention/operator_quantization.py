"""Exact quantization bindings for external Attention operation arguments.

Catalog entries only prove that an API exposes parameters with quantization-like
names.  They do not prove how a FlashInfer KV ``QuantSpec`` maps to those
parameters.  This module closes that semantic gap without importing or calling
an operator package.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from flashinfer_npu.runtime import KernelDescriptor, QuantSpec, SchemaError

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionRuntimeEnvironment,
)
from .launch_binding import bind_attention_kv_physical_layout
from .operation_catalog import AttentionOperatorOperationSpec
from .operator_integration import AttentionOperatorPlanGate
from .operator_plan import AttentionOperatorActivePlan
from .operator_physical_evidence import AttentionOperatorPhysicalLayoutEvidence
from .operator_run import (
    AttentionLoweredOperatorCall,
    AttentionOperatorRunAdapter,
    AttentionOperatorRunRequest,
)
from .planner import AttentionFrameworkPlan
from .quant_physical_layout import (
    EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
    QuantPhysicalLayoutCatalog,
    QuantPhysicalLayoutDescriptor,
)
from .schema import (
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    SingleAttentionMetadata,
)
from .tensor_contract import KVCacheView, QuantizedTensorView, TensorView


ATTENTION_OPERATOR_QUANTIZATION_VERSION = 1

_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUANT_ARGUMENT_SOURCES = {
    "kv.key.scale",
    "kv.value.scale",
    "kv.key.zero_point",
    "kv.value.zero_point",
    "run.k_scale",
    "run.v_scale",
}
_RUNTIME_SCALE_POLICIES = {"reject", "argument"}


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionOperatorQuantArgumentBinding:
    """One logical quantization source mapped to one catalog argument."""

    source: str
    argument_name: str
    schema_version: int = ATTENTION_OPERATOR_QUANTIZATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_QUANTIZATION_VERSION:
            raise SchemaError("unsupported Attention quant argument binding version")
        source = str(self.source)
        argument_name = str(self.argument_name)
        if source not in _QUANT_ARGUMENT_SOURCES:
            raise SchemaError("unknown Attention quant argument source")
        if not _ARGUMENT_NAME.fullmatch(argument_name):
            raise SchemaError("invalid Attention quant argument name")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "argument_name", argument_name)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "argument_name": self.argument_name,
        }


@dataclass(frozen=True)
class AttentionOperatorQuantizationBinding:
    """Exact KV quant semantics authorized for one provider operation."""

    provider_id: str
    operation_id: str
    quant_spec: QuantSpec
    argument_bindings: Tuple[AttentionOperatorQuantArgumentBinding, ...]
    runtime_k_scale_policy: str = "reject"
    runtime_v_scale_policy: str = "reject"
    kv_input_contract: str = "separate_storage_scale_zero_point"
    schema_version: int = ATTENTION_OPERATOR_QUANTIZATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_QUANTIZATION_VERSION:
            raise SchemaError("unsupported Attention quantization binding version")
        if not str(self.provider_id) or not str(self.operation_id):
            raise SchemaError("quantization binding identities must be non-empty")
        if not isinstance(self.quant_spec, QuantSpec):
            raise TypeError("quant_spec must be QuantSpec")
        bindings = tuple(self.argument_bindings)
        if not bindings or any(
            not isinstance(item, AttentionOperatorQuantArgumentBinding)
            for item in bindings
        ):
            raise TypeError(
                "argument_bindings must contain quant argument bindings"
            )
        sources = tuple(item.source for item in bindings)
        arguments = tuple(item.argument_name for item in bindings)
        if len(set(sources)) != len(sources):
            raise SchemaError("quantization binding sources must be unique")
        if len(set(arguments)) != len(arguments):
            raise SchemaError("quantization binding arguments must be unique")
        required_sources = {"kv.key.scale", "kv.value.scale"}
        zero_sources = {"kv.key.zero_point", "kv.value.zero_point"}
        if self.quant_spec.has_zero_point:
            required_sources.update(zero_sources)
        elif zero_sources.intersection(sources):
            raise SchemaError(
                "symmetric quantization cannot bind KV zero-point arguments"
            )
        if not required_sources.issubset(sources):
            missing = sorted(required_sources.difference(sources))[0]
            raise SchemaError("quantization binding is missing source %s" % missing)
        for name, source in (
            ("runtime_k_scale_policy", "run.k_scale"),
            ("runtime_v_scale_policy", "run.v_scale"),
        ):
            policy = str(getattr(self, name))
            if policy not in _RUNTIME_SCALE_POLICIES:
                raise SchemaError("unknown Attention runtime scale policy")
            if (policy == "argument") != (source in sources):
                raise SchemaError(
                    "%s policy does not match its argument binding" % name
                )
            object.__setattr__(self, name, policy)
        if self.kv_input_contract != "separate_storage_scale_zero_point":
            raise SchemaError("unsupported Attention quantized KV input contract")
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(
            self,
            "argument_bindings",
            tuple(sorted(bindings, key=lambda item: item.source)),
        )

    @property
    def arguments_by_source(self):
        return {item.source: item.argument_name for item in self.argument_bindings}

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "quant_spec": self.quant_spec.to_dict(),
            "argument_bindings": [
                item.to_dict() for item in self.argument_bindings
            ],
            "runtime_k_scale_policy": self.runtime_k_scale_policy,
            "runtime_v_scale_policy": self.runtime_v_scale_policy,
            "kv_input_contract": self.kv_input_contract,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorQuantizedTensorInput:
    """One opaque quantized tensor for public APIs with separate K/V slots."""

    quant_spec: QuantSpec
    logical_shape: Tuple[int, ...]
    storage: Any
    scale: Any
    zero_point: Any = None
    schema_version: int = ATTENTION_OPERATOR_QUANTIZATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_QUANTIZATION_VERSION:
            raise SchemaError("unsupported Attention quantized tensor input version")
        if not isinstance(self.quant_spec, QuantSpec):
            raise TypeError("quant_spec must be QuantSpec")
        try:
            logical_shape = tuple(int(dim) for dim in self.logical_shape)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "quantized tensor input logical_shape must contain integers"
            ) from error
        if not logical_shape or any(dim < 0 for dim in logical_shape):
            raise SchemaError(
                "quantized tensor input logical_shape must be non-empty and "
                "non-negative"
            )
        object.__setattr__(self, "logical_shape", logical_shape)
        if self.storage is None or self.scale is None:
            raise SchemaError("quantized tensor input requires storage and scale")
        if self.quant_spec.has_zero_point and self.zero_point is None:
            raise SchemaError("asymmetric quantized tensor input requires zero_point")
        if not self.quant_spec.has_zero_point and self.zero_point is not None:
            raise SchemaError(
                "symmetric quantized tensor input cannot carry zero_point"
            )


@dataclass(frozen=True)
class AttentionOperatorQuantizedKVInput:
    """Opaque provider tensors carried through the unchanged public run slot."""

    quant_spec: QuantSpec
    key_storage: Any
    value_storage: Any
    key_scale: Any
    value_scale: Any
    key_zero_point: Any = None
    value_zero_point: Any = None
    key_logical_shape: Optional[Tuple[int, ...]] = None
    value_logical_shape: Optional[Tuple[int, ...]] = None
    schema_version: int = ATTENTION_OPERATOR_QUANTIZATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_QUANTIZATION_VERSION:
            raise SchemaError("unsupported Attention quantized KV input version")
        if not isinstance(self.quant_spec, QuantSpec):
            raise TypeError("quant_spec must be QuantSpec")
        for name in ("key_storage", "value_storage", "key_scale", "value_scale"):
            if getattr(self, name) is None:
                raise SchemaError("quantized KV input requires %s" % name)
        zero_points = (self.key_zero_point, self.value_zero_point)
        if self.quant_spec.has_zero_point and any(
            item is None for item in zero_points
        ):
            raise SchemaError(
                "asymmetric quantized KV input requires independent zero points"
            )
        if not self.quant_spec.has_zero_point and any(
            item is not None for item in zero_points
        ):
            raise SchemaError(
                "symmetric quantized KV input cannot carry zero points"
            )
        logical_shapes = (self.key_logical_shape, self.value_logical_shape)
        if (logical_shapes[0] is None) != (logical_shapes[1] is None):
            raise SchemaError(
                "quantized KV input must declare both logical shapes or neither"
            )
        if logical_shapes[0] is not None:
            normalized = []
            for value in logical_shapes:
                try:
                    shape = tuple(int(dim) for dim in value)
                except (TypeError, ValueError) as error:
                    raise SchemaError(
                        "quantized KV logical shapes must contain integers"
                    ) from error
                if not shape or any(dim < 0 for dim in shape):
                    raise SchemaError(
                        "quantized KV logical shapes must be non-empty and "
                        "non-negative"
                    )
                normalized.append(shape)
            object.__setattr__(self, "key_logical_shape", normalized[0])
            object.__setattr__(self, "value_logical_shape", normalized[1])


def combine_attention_operator_quantized_kv_input(
    key: AttentionOperatorQuantizedTensorInput,
    value: AttentionOperatorQuantizedTensorInput,
) -> AttentionOperatorQuantizedKVInput:
    """Combine separate public K/V arguments without exposing provider state."""

    if not isinstance(key, AttentionOperatorQuantizedTensorInput):
        raise SchemaError(
            "quantized provider plan requires AttentionOperatorQuantizedTensorInput "
            "for key"
        )
    if not isinstance(value, AttentionOperatorQuantizedTensorInput):
        raise SchemaError(
            "quantized provider plan requires AttentionOperatorQuantizedTensorInput "
            "for value"
        )
    if key.quant_spec.fingerprint != value.quant_spec.fingerprint:
        raise SchemaError("quantized key/value inputs use different QuantSpec values")
    return AttentionOperatorQuantizedKVInput(
        quant_spec=key.quant_spec,
        key_storage=key.storage,
        value_storage=value.storage,
        key_scale=key.scale,
        value_scale=value.scale,
        key_zero_point=key.zero_point,
        value_zero_point=value.zero_point,
        key_logical_shape=key.logical_shape,
        value_logical_shape=value.logical_shape,
    )


@runtime_checkable
class AttentionOperatorTensorMetadataInspector(Protocol):
    """Read an opaque provider tensor as a framework ``TensorView``.

    Implementations may inspect shape/stride/dtype/device/storage metadata only;
    importing an operator package, allocating a tensor, or touching device data
    is outside this protocol.
    """

    def to_view(
        self, tensor: Any, *, name: str, writable: bool = False
    ) -> TensorView:
        """Return a non-copying metadata view for ``tensor``."""


def _inspect_quant_component(
    inspector: AttentionOperatorTensorMetadataInspector,
    value: Any,
    name: str,
) -> TensorView:
    view = inspector.to_view(value, name=name, writable=False)
    if not isinstance(view, TensorView):
        raise TypeError("tensor metadata inspector must return TensorView")
    if not view.is_contiguous:
        raise SchemaError("%s must be contiguous for provider lowering" % name)
    return view


def _quantized_kv_cache_spec(
    plan: AttentionFrameworkPlan,
    key_storage: TensorView,
    expected_device: str,
    physical_layout_descriptor: Optional[QuantPhysicalLayoutDescriptor] = None,
):
    spec = plan.spec
    metadata = plan.metadata
    quant_spec = spec.kv_quant_spec
    if quant_spec is None:
        raise SchemaError("quantized KV metadata validation requires a quantized plan")
    common = {
        "num_kv_heads": spec.num_kv_heads,
        "head_dim_qk": spec.head_dim_qk,
        "head_dim_vo": int(spec.head_dim_vo),
        "dtype": str(spec.kv_dtype),
        "layout": spec.kv_layout,
        "device": str(expected_device),
        "quant_spec": quant_spec,
    }
    if isinstance(
        metadata,
        (PagedKVMetadata, PagedPrefillMetadata, MixedPagedKVMetadata),
    ):
        if not key_storage.shape:
            raise SchemaError("kv.key.storage must expose a page dimension")
        paged = (
            metadata.paged_kv
            if isinstance(metadata, PagedPrefillMetadata)
            else metadata
        )
        if quant_spec.physical_layout == "logical":
            num_pages = key_storage.shape[0]
        else:
            descriptor = physical_layout_descriptor
            if descriptor is None:
                raise SchemaError(
                    "provider tensor metadata validation requires an explicit "
                    "non-logical physical layout descriptor"
                )
            transform = descriptor.storage_transform
            if transform.axis_blocks[0] != 1 or "o0" not in transform.physical_axes:
                raise SchemaError(
                    "non-logical paged KV layout must preserve an exact page axis"
                )
            page_axis = transform.physical_axes.index("o0")
            if page_axis >= len(key_storage.shape):
                raise SchemaError("physical KV storage rank omits the page axis")
            num_pages = key_storage.shape[page_axis]
        if num_pages <= paged.max_page_index:
            raise SchemaError(
                "quantized KV storage does not contain every referenced page"
            )
        return PagedKVCacheSpec(
            num_pages=num_pages,
            page_size=paged.page_size,
            structure="separate",
            **common,
        )
    if isinstance(metadata, RaggedKVMetadata):
        total_kv_tokens = metadata.total_kv_tokens
    elif isinstance(metadata, SingleAttentionMetadata):
        total_kv_tokens = metadata.kv_len
    else:  # guarded by the framework plan schema; retained for future modes
        raise SchemaError("unsupported metadata for quantized provider KV validation")
    return RaggedKVCacheSpec(total_kv_tokens=total_kv_tokens, **common)


def inspect_attention_operator_quantized_kv_input(
    plan: AttentionFrameworkPlan,
    value: AttentionOperatorQuantizedKVInput,
    inspector: AttentionOperatorTensorMetadataInspector,
    expected_device: str,
    physical_layout_descriptor: Optional[QuantPhysicalLayoutDescriptor] = None,
) -> KVCacheView:
    """Validate opaque quantized K/V tensors against one active logical plan."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(value, AttentionOperatorQuantizedKVInput):
        raise TypeError("value must be AttentionOperatorQuantizedKVInput")
    if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
        raise TypeError(
            "inspector must implement AttentionOperatorTensorMetadataInspector"
        )
    if not str(expected_device):
        raise SchemaError("expected provider tensor device must be non-empty")
    quant_spec = plan.spec.kv_quant_spec
    if quant_spec is None or value.quant_spec.fingerprint != quant_spec.fingerprint:
        raise SchemaError("quantized KV input does not match the active QuantSpec")
    if quant_spec.physical_layout == "logical":
        if physical_layout_descriptor is not None:
            raise SchemaError("logical quantized KV cannot use a physical descriptor")
    elif not isinstance(
        physical_layout_descriptor, QuantPhysicalLayoutDescriptor
    ):
        raise SchemaError(
            "non-logical quantized KV requires an exact physical descriptor"
        )

    components = {
        "key_storage": _inspect_quant_component(
            inspector, value.key_storage, "kv.key.storage"
        ),
        "value_storage": _inspect_quant_component(
            inspector, value.value_storage, "kv.value.storage"
        ),
        "key_scale": _inspect_quant_component(
            inspector, value.key_scale, "kv.key.scale"
        ),
        "value_scale": _inspect_quant_component(
            inspector, value.value_scale, "kv.value.scale"
        ),
    }
    if value.key_zero_point is not None:
        components["key_zero_point"] = _inspect_quant_component(
            inspector, value.key_zero_point, "kv.key.zero_point"
        )
        components["value_zero_point"] = _inspect_quant_component(
            inspector, value.value_zero_point, "kv.value.zero_point"
        )
    cache_spec = _quantized_kv_cache_spec(
        plan,
        components["key_storage"],
        str(expected_device),
        physical_layout_descriptor,
    )
    key_shape, value_shape = cache_spec.expected_shapes
    if value.key_logical_shape is not None:
        if value.key_logical_shape != key_shape:
            raise SchemaError(
                "quantized key logical_shape does not match the active plan"
            )
        if value.value_logical_shape != value_shape:
            raise SchemaError(
                "quantized value logical_shape does not match the active plan"
            )
    key = QuantizedTensorView(
        logical_shape=key_shape,
        storage=components["key_storage"],
        scale=components["key_scale"],
        zero_point=components.get("key_zero_point"),
        quant_spec=quant_spec,
        physical_layout_descriptor=physical_layout_descriptor,
    )
    val = QuantizedTensorView(
        logical_shape=value_shape,
        storage=components["value_storage"],
        scale=components["value_scale"],
        zero_point=components.get("value_zero_point"),
        quant_spec=quant_spec,
        physical_layout_descriptor=physical_layout_descriptor,
    )
    return KVCacheView(cache_spec, key, val)


def validate_attention_operator_quantization_bindings(
    operation: AttentionOperatorOperationSpec,
    profiles: Sequence[AttentionBackendCapabilityProfile],
    bindings: Sequence[AttentionOperatorQuantizationBinding],
) -> Tuple[AttentionOperatorQuantizationBinding, ...]:
    """Require exact agreement between capability rules and API arguments."""

    if not isinstance(operation, AttentionOperatorOperationSpec):
        raise TypeError("operation must be AttentionOperatorOperationSpec")
    profile_values = tuple(profiles)
    if any(
        not isinstance(item, AttentionBackendCapabilityProfile)
        for item in profile_values
    ):
        raise TypeError("profiles must contain capability profiles")
    binding_values = tuple(bindings)
    if any(
        not isinstance(item, AttentionOperatorQuantizationBinding)
        for item in binding_values
    ):
        raise TypeError("bindings must contain quantization bindings")
    binding_fingerprints = tuple(
        item.quant_spec.fingerprint for item in binding_values
    )
    if len(set(binding_fingerprints)) != len(binding_fingerprints):
        raise SchemaError("duplicate operation QuantSpec binding")
    catalog_quant_arguments = set(operation.quant_arguments)
    for binding in binding_values:
        if (
            binding.provider_id != operation.provider_id
            or binding.operation_id != operation.operation_id
        ):
            raise SchemaError("quantization binding operation identity differs")
        undeclared = set(binding.arguments_by_source.values()).difference(
            catalog_quant_arguments
        )
        if undeclared:
            raise SchemaError(
                "quantization binding uses non-catalog argument %s"
                % sorted(undeclared)[0]
            )
    profile_quant_specs = {
        quant_spec.fingerprint: quant_spec
        for profile in profile_values
        for rule in profile.rules
        if set(rule.modes).intersection(operation.candidate_modes)
        for quant_spec in rule.quant_specs
    }
    profile_fingerprints = set(profile_quant_specs)
    declared_fingerprints = set(binding_fingerprints)
    if profile_fingerprints != declared_fingerprints:
        missing = profile_fingerprints.difference(declared_fingerprints)
        if missing:
            raise SchemaError(
                "operator capability QuantSpec has no API argument binding"
            )
        raise SchemaError(
            "operator quantization binding has no capability QuantSpec"
        )
    return tuple(
        sorted(binding_values, key=lambda item: item.quant_spec.fingerprint)
    )


def validate_attention_operator_quant_physical_layouts(
    bindings: Sequence[AttentionOperatorQuantizationBinding],
    catalog: QuantPhysicalLayoutCatalog = EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
) -> QuantPhysicalLayoutCatalog:
    """Require an exact provider-owned descriptor set for non-logical bindings."""

    values = tuple(bindings)
    if any(
        not isinstance(item, AttentionOperatorQuantizationBinding)
        for item in values
    ):
        raise TypeError("bindings must contain quantization bindings")
    if not isinstance(catalog, QuantPhysicalLayoutCatalog):
        raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
    required_layouts = {
        item.quant_spec.physical_layout
        for item in values
        if item.quant_spec.physical_layout != "logical"
    }
    catalog_layouts = {item.layout_id for item in catalog.descriptors}
    if required_layouts != catalog_layouts:
        if required_layouts.difference(catalog_layouts):
            raise SchemaError(
                "non-logical quantization binding has no physical descriptor"
            )
        raise SchemaError(
            "quant physical layout catalog contains an unbound descriptor"
        )
    for binding in values:
        quant_spec = binding.quant_spec
        if quant_spec.physical_layout == "logical":
            continue
        descriptor = catalog.resolve(quant_spec)
        if quant_spec.storage_dtype not in descriptor.storage_dtypes:
            raise SchemaError(
                "quant physical layout does not support bound storage dtype"
            )
    return catalog


class AttentionOperatorQuantizationPlanGate:
    """Compose a provider gate with mandatory exact-QuantSpec admission."""

    def __init__(
        self,
        base_gate: AttentionOperatorPlanGate,
        operation: AttentionOperatorOperationSpec,
        bindings: Sequence[AttentionOperatorQuantizationBinding],
    ) -> None:
        if not isinstance(base_gate, AttentionOperatorPlanGate):
            raise TypeError("base_gate must implement AttentionOperatorPlanGate")
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if (
            base_gate.provider_id != operation.provider_id
            or base_gate.operation_id != operation.operation_id
        ):
            raise SchemaError("quantization gate component identities differ")
        values = tuple(bindings)
        if any(
            not isinstance(item, AttentionOperatorQuantizationBinding)
            for item in values
        ):
            raise TypeError("bindings must contain quantization bindings")
        if any(
            item.provider_id != operation.provider_id
            or item.operation_id != operation.operation_id
            for item in values
        ):
            raise SchemaError("quantization gate binding identity differs")
        fingerprints = tuple(item.quant_spec.fingerprint for item in values)
        if len(set(fingerprints)) != len(fingerprints):
            raise SchemaError("quantization gate contains duplicate QuantSpec")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._base_gate = base_gate
        self._bindings = {
            item.quant_spec.fingerprint: item for item in values
        }

    def rejection_reasons(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> Tuple[str, ...]:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        reasons = tuple(
            str(item) for item in self._base_gate.rejection_reasons(plan, str(device))
        )
        quant_spec = plan.spec.kv_quant_spec
        if (
            quant_spec is not None
            and quant_spec.fingerprint not in self._bindings
        ):
            reasons += (
                "provider operation has no exact KV QuantSpec argument binding",
            )
        return tuple(dict.fromkeys(reasons))


class AttentionOperatorQuantizationRunAdapter:
    """Inject bound quant arguments after pure provider run lowering."""

    def __init__(
        self,
        base_adapter: AttentionOperatorRunAdapter,
        operation: AttentionOperatorOperationSpec,
        bindings: Sequence[AttentionOperatorQuantizationBinding],
        tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector,
        expected_device: str,
        physical_layout_catalog: QuantPhysicalLayoutCatalog,
        profiles: Sequence[AttentionBackendCapabilityProfile],
        descriptors: Sequence[KernelDescriptor],
        observed_environment: AttentionRuntimeEnvironment,
        physical_layout_evidence: Sequence[
            AttentionOperatorPhysicalLayoutEvidence
        ],
    ) -> None:
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if base_adapter.provider_id != operation.provider_id:
            raise SchemaError("quantization run adapter providers differ")
        if not isinstance(
            tensor_metadata_inspector, AttentionOperatorTensorMetadataInspector
        ):
            raise TypeError(
                "tensor_metadata_inspector must implement "
                "AttentionOperatorTensorMetadataInspector"
            )
        if not str(expected_device):
            raise SchemaError("quantization run expected_device must be non-empty")
        if not isinstance(physical_layout_catalog, QuantPhysicalLayoutCatalog):
            raise TypeError("physical_layout_catalog must be QuantPhysicalLayoutCatalog")
        profile_values = tuple(profiles)
        descriptor_values = tuple(descriptors)
        if any(
            not isinstance(item, AttentionBackendCapabilityProfile)
            for item in profile_values
        ):
            raise TypeError("profiles must contain capability profiles")
        if any(not isinstance(item, KernelDescriptor) for item in descriptor_values):
            raise TypeError("descriptors must contain KernelDescriptor values")
        if not isinstance(observed_environment, AttentionRuntimeEnvironment):
            raise TypeError("observed_environment must be AttentionRuntimeEnvironment")
        physical_evidence_values = tuple(physical_layout_evidence)
        if any(
            not isinstance(item, AttentionOperatorPhysicalLayoutEvidence)
            for item in physical_evidence_values
        ):
            raise TypeError(
                "physical_layout_evidence must contain physical evidence records"
            )
        values = tuple(bindings)
        if any(
            not isinstance(item, AttentionOperatorQuantizationBinding)
            for item in values
        ):
            raise TypeError("bindings must contain quantization bindings")
        if any(
            item.provider_id != operation.provider_id
            or item.operation_id != operation.operation_id
            for item in values
        ):
            raise SchemaError("quantization run binding identity differs")
        fingerprints = tuple(item.quant_spec.fingerprint for item in values)
        if len(set(fingerprints)) != len(fingerprints):
            raise SchemaError("quantization run adapter contains duplicate QuantSpec")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._base_adapter = base_adapter
        self._bindings = {
            item.quant_spec.fingerprint: item for item in values
        }
        self._tensor_metadata_inspector = tensor_metadata_inspector
        self._expected_device = str(expected_device)
        self._physical_layout_catalog = physical_layout_catalog
        self._profiles = profile_values
        self._descriptors = descriptor_values
        self._observed_environment = observed_environment
        self._physical_layout_evidence = physical_evidence_values

    def lower(
        self,
        active_plan: AttentionOperatorActivePlan,
        request: AttentionOperatorRunRequest,
    ) -> AttentionLoweredOperatorCall:
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        if not isinstance(request, AttentionOperatorRunRequest):
            raise TypeError("request must be AttentionOperatorRunRequest")
        quant_spec = active_plan.framework_plan.spec.kv_quant_spec
        if quant_spec is None:
            return self._base_adapter.lower(active_plan, request)
        binding = self._bindings.get(quant_spec.fingerprint)
        if binding is None:
            raise SchemaError("active quantized plan has no exact API binding")
        kv_input = request.kv_cache
        if not isinstance(kv_input, AttentionOperatorQuantizedKVInput):
            raise SchemaError(
                "quantized provider plan requires AttentionOperatorQuantizedKVInput"
            )
        if kv_input.quant_spec.fingerprint != quant_spec.fingerprint:
            raise SchemaError("quantized KV input does not match the active QuantSpec")
        descriptor = (
            None
            if quant_spec.physical_layout == "logical"
            else self._physical_layout_catalog.resolve(quant_spec)
        )
        kv_view = inspect_attention_operator_quantized_kv_input(
            active_plan.framework_plan,
            kv_input,
            self._tensor_metadata_inspector,
            self._expected_device,
            descriptor,
        )
        if descriptor is not None:
            receipt = active_plan.dispatch_receipt
            profile = next(
                (
                    item
                    for item in self._profiles
                    if item.profile_id == receipt.profile_id
                    and item.fingerprint == receipt.profile_fingerprint
                ),
                None,
            )
            kernel = next(
                (
                    item
                    for item in self._descriptors
                    if item.fingerprint == receipt.kernel_fingerprint
                ),
                None,
            )
            if profile is None or kernel is None:
                raise SchemaError(
                    "physical KV layout authority is absent from package runtime"
                )
            physical_evidence = next(
                (
                    item
                    for item in self._physical_layout_evidence
                    if item.evidence_id == receipt.evidence_id
                    and item.result_digest == receipt.evidence_result_digest
                ),
                None,
            )
            if physical_evidence is None:
                raise SchemaError(
                    "active physical KV layout evidence is absent from package runtime"
                )
            bind_attention_kv_physical_layout(
                kv_view,
                self._physical_layout_catalog,
                profile,
                kernel,
                self._observed_environment,
                receipt,
                physical_evidence,
            )
        for field_name, policy in (
            ("k_scale", binding.runtime_k_scale_policy),
            ("v_scale", binding.runtime_v_scale_policy),
        ):
            if policy == "reject" and getattr(request, field_name) is not None:
                raise SchemaError(
                    "quantization binding rejects run-time %s" % field_name
                )
        delegated_request = replace(
            request,
            kv_cache=(kv_input.key_storage, kv_input.value_storage),
            k_scale=None,
            v_scale=None,
        )
        lowered = self._base_adapter.lower(active_plan, delegated_request)
        if not isinstance(lowered, AttentionLoweredOperatorCall):
            raise TypeError("base run adapter returned an invalid call description")
        values_by_source = {
            "kv.key.scale": kv_input.key_scale,
            "kv.value.scale": kv_input.value_scale,
            "kv.key.zero_point": kv_input.key_zero_point,
            "kv.value.zero_point": kv_input.value_zero_point,
            "run.k_scale": request.k_scale,
            "run.v_scale": request.v_scale,
        }
        injected = tuple(
            (item.argument_name, values_by_source[item.source])
            for item in binding.argument_bindings
            if values_by_source[item.source] is not None
        )
        existing_names = {name for name, _ in lowered.keyword_arguments}
        collision = existing_names.intersection(name for name, _ in injected)
        if collision:
            raise SchemaError(
                "quantization argument collides with provider lowering: %s"
                % sorted(collision)[0]
            )
        return replace(
            lowered,
            keyword_arguments=lowered.keyword_arguments + injected,
            consumed_request_fields=request.consumed_fields,
        )


class AttentionOperatorQuantizationRunAdapterFactory:
    """Attach quantized tensor validation after device resolution."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        bindings: Sequence[AttentionOperatorQuantizationBinding],
        tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector,
        physical_layout_catalog: QuantPhysicalLayoutCatalog,
        profiles: Sequence[AttentionBackendCapabilityProfile],
        descriptors: Sequence[KernelDescriptor],
        observed_environment: AttentionRuntimeEnvironment,
        physical_layout_evidence: Sequence[
            AttentionOperatorPhysicalLayoutEvidence
        ],
    ) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        values = tuple(bindings)
        if not values or any(
            not isinstance(item, AttentionOperatorQuantizationBinding)
            for item in values
        ):
            raise TypeError("bindings must contain quantization bindings")
        if any(
            item.provider_id != operation.provider_id
            or item.operation_id != operation.operation_id
            for item in values
        ):
            raise SchemaError("quantization run factory binding identity differs")
        if not isinstance(
            tensor_metadata_inspector, AttentionOperatorTensorMetadataInspector
        ):
            raise TypeError(
                "tensor_metadata_inspector must implement "
                "AttentionOperatorTensorMetadataInspector"
            )
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation
        self._bindings = values
        self._tensor_metadata_inspector = tensor_metadata_inspector
        self._physical_layout_catalog = validate_attention_operator_quant_physical_layouts(
            values, physical_layout_catalog
        )
        self._profiles = tuple(profiles)
        self._descriptors = tuple(descriptors)
        self._observed_environment = observed_environment
        self._physical_layout_evidence = tuple(physical_layout_evidence)

    def build(
        self, base_adapter: AttentionOperatorRunAdapter, device: str
    ) -> AttentionOperatorRunAdapter:
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        return AttentionOperatorQuantizationRunAdapter(
            base_adapter,
            self._operation,
            self._bindings,
            self._tensor_metadata_inspector,
            str(device),
            self._physical_layout_catalog,
            self._profiles,
            self._descriptors,
            self._observed_environment,
            self._physical_layout_evidence,
        )


__all__ = [
    "ATTENTION_OPERATOR_QUANTIZATION_VERSION",
    "AttentionOperatorQuantArgumentBinding",
    "AttentionOperatorQuantizationBinding",
    "AttentionOperatorQuantizationPlanGate",
    "AttentionOperatorQuantizationRunAdapter",
    "AttentionOperatorQuantizationRunAdapterFactory",
    "AttentionOperatorQuantizedTensorInput",
    "AttentionOperatorQuantizedKVInput",
    "AttentionOperatorTensorMetadataInspector",
    "inspect_attention_operator_quantized_kv_input",
    "combine_attention_operator_quantized_kv_input",
    "validate_attention_operator_quantization_bindings",
    "validate_attention_operator_quant_physical_layouts",
]
