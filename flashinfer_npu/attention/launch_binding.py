"""Materialize validated Attention tensor contracts into the stable C POD ABI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from flashinfer_npu.runtime import KernelDescriptor, SchemaError

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionRuntimeEnvironment,
    validate_attention_kernel_bindings,
)
from .dispatch import AttentionDispatchReceipt

from .launch_contract import (
    ATTENTION_AUXILIARY_VIEW_C_ABI,
    ATTENTION_KV_CACHE_VIEW_C_ABI,
    ATTENTION_KV_CACHE_VIEW_V2_C_ABI,
    ATTENTION_MAX_TENSOR_RANK,
    ATTENTION_TENSOR_VIEW_C_ABI,
    AttentionAuxiliaryRole,
    AttentionDTypeCode,
    AttentionKVFlags,
    AttentionKVLayoutCode,
    AttentionKVPhysicalLayoutAccessCode,
    AttentionTensorFlags,
    AttentionTensorRole,
    attention_dtype_code,
    attention_kernel_binary_abi_v2,
)
from .quant_physical_layout import QuantPhysicalLayoutCatalog
from .schema import KVLayout
from .storage_lease import AttentionAddressBinding, AttentionHostBufferLease
from .tensor_contract import (
    AttentionAuxiliaryContract,
    KVCacheView,
    QuantizedTensorView,
    TensorView,
    contiguous_strides,
    dtype_itemsize,
)


ATTENTION_LAUNCH_BINDING_VERSION = 1
ATTENTION_KV_PHYSICAL_LAYOUT_BINDING_VERSION = 1
ATTENTION_KV_CACHE_VIEW_POD_V2_VERSION = 2
_KNOWN_TENSOR_FLAGS = int(
    AttentionTensorFlags.CONTIGUOUS
    | AttentionTensorFlags.WRITABLE
    | AttentionTensorFlags.EMPTY
)
_KNOWN_KV_FLAGS_V1 = int(AttentionKVFlags.PACKED | AttentionKVFlags.QUANTIZED)
_KNOWN_KV_FLAGS_V2 = int(
    AttentionKVFlags.PACKED
    | AttentionKVFlags.QUANTIZED
    | AttentionKVFlags.PHYSICAL_LAYOUT
)
_ZERO_SHA256 = "0" * 64


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str, *, allow_zero: bool = True) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)
    if not allow_zero and value == _ZERO_SHA256:
        raise SchemaError("%s cannot be the zero fingerprint" % name)


def _numel(shape: Sequence[int]) -> int:
    return reduce(mul, shape, 1)


@dataclass(frozen=True)
class AttentionTensorViewPOD:
    data_ptr: int
    storage_nbytes: int
    storage_offset_elements: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype_code: AttentionDTypeCode
    role_code: int
    flags: AttentionTensorFlags
    device_index: int
    schema_version: int = ATTENTION_LAUNCH_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_BINDING_VERSION:
            raise SchemaError("unsupported Attention tensor POD version")
        object.__setattr__(self, "shape", tuple(int(item) for item in self.shape))
        object.__setattr__(self, "strides", tuple(int(item) for item in self.strides))
        try:
            object.__setattr__(self, "dtype_code", AttentionDTypeCode(self.dtype_code))
            object.__setattr__(self, "flags", AttentionTensorFlags(self.flags))
        except ValueError as error:
            raise SchemaError("Attention tensor POD contains an unknown enum") from error
        for name in (
            "data_ptr",
            "storage_nbytes",
            "storage_offset_elements",
            "role_code",
            "device_index",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SchemaError("Attention tensor POD integer fields must be integers")
        if self.data_ptr < 0 or self.storage_nbytes < 0 or self.storage_offset_elements < 0:
            raise SchemaError("Attention tensor POD storage fields cannot be negative")
        if not 1 <= self.role_code <= 0xFFFF:
            raise SchemaError("Attention tensor POD role code is out of range")
        if self.device_index < 0:
            raise SchemaError("Attention tensor POD device index cannot be negative")
        if len(self.shape) != len(self.strides) or len(self.shape) > ATTENTION_MAX_TENSOR_RANK:
            raise SchemaError("Attention tensor POD rank is invalid")
        if any(dim < 0 for dim in self.shape) or any(stride < 0 for stride in self.strides):
            raise SchemaError("Attention tensor POD shape/strides cannot be negative")
        if int(self.flags) & ~_KNOWN_TENSOR_FLAGS:
            raise SchemaError("Attention tensor POD flags contain unknown bits")
        empty = _numel(self.shape) == 0
        contiguous = empty or self.strides == contiguous_strides(self.shape)
        if bool(self.flags & AttentionTensorFlags.EMPTY) != empty:
            raise SchemaError("Attention tensor POD EMPTY flag is not canonical")
        if bool(self.flags & AttentionTensorFlags.CONTIGUOUS) != contiguous:
            raise SchemaError("Attention tensor POD CONTIGUOUS flag is not canonical")
        if not empty and self.data_ptr == 0:
            raise SchemaError("non-empty Attention tensor POD requires a non-zero data pointer")
        dtype_name = _DTYPE_NAMES[self.dtype_code]
        itemsize = (
            1
            if dtype_name in {"int4", "int4_packed", "uint4_packed"}
            else dtype_itemsize(dtype_name)
        )
        if self.storage_offset_elements * itemsize > self.storage_nbytes:
            raise SchemaError("Attention tensor POD storage offset exceeds storage")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def pack(self) -> bytes:
        shape = self.shape + (0,) * (ATTENTION_MAX_TENSOR_RANK - self.ndim)
        strides = self.strides + (0,) * (ATTENTION_MAX_TENSOR_RANK - self.ndim)
        return ATTENTION_TENSOR_VIEW_C_ABI.pack(
            {
                "data_ptr": self.data_ptr,
                "storage_nbytes": self.storage_nbytes,
                "storage_offset_elements": self.storage_offset_elements,
                "shape": shape,
                "strides": strides,
                "ndim": self.ndim,
                "dtype_code": int(self.dtype_code),
                "role_code": self.role_code,
                "flags": int(self.flags),
                "device_index": self.device_index,
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "AttentionTensorViewPOD":
        values = ATTENTION_TENSOR_VIEW_C_ABI.unpack(payload)
        ndim = values["ndim"]
        if ndim > ATTENTION_MAX_TENSOR_RANK:
            raise SchemaError("Attention tensor POD ndim exceeds ABI maximum")
        if any(values["shape"][ndim:]) or any(values["strides"][ndim:]):
            raise SchemaError("Attention tensor POD trailing shape/stride entries must be zero")
        return cls(
            data_ptr=values["data_ptr"],
            storage_nbytes=values["storage_nbytes"],
            storage_offset_elements=values["storage_offset_elements"],
            shape=tuple(values["shape"][:ndim]),
            strides=tuple(values["strides"][:ndim]),
            dtype_code=values["dtype_code"],
            role_code=values["role_code"],
            flags=values["flags"],
            device_index=values["device_index"],
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.pack()).hexdigest()


_DTYPE_NAMES = {
    attention_dtype_code(name): name
    for name in (
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "float8_e4m3fn",
        "float8_e5m2",
        "int4",
        "int4_packed",
        "uint4_packed",
    )
}


def materialize_attention_tensor_view(
    binding: AttentionAddressBinding,
    role_code: int,
    device_index: int,
) -> AttentionTensorViewPOD:
    if not isinstance(binding, AttentionAddressBinding):
        raise TypeError("binding must be AttentionAddressBinding")
    view = binding.view
    flags = AttentionTensorFlags(0)
    if view.is_contiguous:
        flags |= AttentionTensorFlags.CONTIGUOUS
    if view.writable:
        flags |= AttentionTensorFlags.WRITABLE
    if view.numel == 0:
        flags |= AttentionTensorFlags.EMPTY
    return AttentionTensorViewPOD(
        data_ptr=binding.data_address,
        storage_nbytes=view.storage_nbytes,
        storage_offset_elements=view.storage_offset,
        shape=view.shape,
        strides=view.strides,
        dtype_code=attention_dtype_code(view.dtype),
        role_code=int(role_code),
        flags=flags,
        device_index=device_index,
    )


def _binding_for(
    bindings: Mapping[str, AttentionAddressBinding], role: str, expected: TensorView
) -> AttentionAddressBinding:
    try:
        binding = bindings[role]
    except KeyError as error:
        raise SchemaError("missing Attention address binding for %s" % role) from error
    if binding.role != role or binding.view != expected:
        raise SchemaError("Attention address binding does not match %s view" % role)
    return binding


def _binding_map(
    bindings: Sequence[AttentionAddressBinding],
) -> Dict[str, AttentionAddressBinding]:
    result = {item.role: item for item in bindings}
    if len(result) != len(tuple(bindings)):
        raise SchemaError("Attention address binding roles must be unique")
    return result


def _kv_components(kv: KVCacheView):
    if kv.packed:
        return (("kv.packed_storage", AttentionTensorRole.KV_PACKED_STORAGE, kv.key),)
    if isinstance(kv.key, TensorView):
        return (
            ("kv.key_storage", AttentionTensorRole.KV_KEY_STORAGE, kv.key),
            ("kv.value_storage", AttentionTensorRole.KV_VALUE_STORAGE, kv.value),
        )
    result = []
    for prefix, value, roles in (
        (
            "kv.key",
            kv.key,
            (
                AttentionTensorRole.KV_KEY_STORAGE,
                AttentionTensorRole.KV_KEY_SCALE,
                AttentionTensorRole.KV_KEY_ZERO_POINT,
            ),
        ),
        (
            "kv.value",
            kv.value,
            (
                AttentionTensorRole.KV_VALUE_STORAGE,
                AttentionTensorRole.KV_VALUE_SCALE,
                AttentionTensorRole.KV_VALUE_ZERO_POINT,
            ),
        ),
    ):
        assert isinstance(value, QuantizedTensorView)
        result.extend(
            (
                (prefix + "_storage", roles[0], value.storage),
                (prefix + "_scale", roles[1], value.scale),
            )
        )
        if value.zero_point is not None:
            result.append((prefix + "_zero_point", roles[2], value.zero_point))
    return tuple(result)


def _validate_component_table_lease(
    lease: AttentionHostBufferLease, component_blob: bytes
) -> None:
    if not isinstance(lease, AttentionHostBufferLease):
        raise TypeError("component table lease must be AttentionHostBufferLease")
    if lease.capacity_bytes < len(component_blob):
        raise SchemaError("component table lease capacity is incompatible")
    if component_blob and (lease.base_address == 0 or lease.alignment < 8):
        raise SchemaError("component table lease address/alignment is incompatible")
    if component_blob and not lease.writable:
        raise SchemaError("component table lease must be writable for materialization")


@dataclass(frozen=True)
class AttentionKVPhysicalLayoutBinding:
    """Host evidence binding for direct kernel consumption of a physical KV layout.

    This object does not execute a converter or a kernel. It joins an exact
    physical-layout descriptor/catalog to the capability, artifact and dispatch
    identities that are required before a v2 KV POD may be materialized.
    """

    quant_spec_fingerprint: str
    descriptor_fingerprint: str
    catalog_fingerprint: str
    dispatch_receipt_fingerprint: str
    profile_fingerprint: str
    rule_id: str
    environment_fingerprint: str
    evidence_result_digest: str
    kernel_fingerprint: str
    required_features: Tuple[str, ...]
    access_code: AttentionKVPhysicalLayoutAccessCode = (
        AttentionKVPhysicalLayoutAccessCode.KERNEL_NATIVE
    )
    schema_version: int = ATTENTION_KV_PHYSICAL_LAYOUT_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_KV_PHYSICAL_LAYOUT_BINDING_VERSION:
            raise SchemaError("unsupported Attention KV physical layout binding version")
        try:
            object.__setattr__(
                self, "access_code", AttentionKVPhysicalLayoutAccessCode(self.access_code)
            )
        except ValueError as error:
            raise SchemaError("unknown Attention KV physical layout access code") from error
        if self.access_code != AttentionKVPhysicalLayoutAccessCode.KERNEL_NATIVE:
            raise SchemaError("physical layout binding v1 only supports kernel-native access")
        for name in (
            "quant_spec_fingerprint",
            "descriptor_fingerprint",
            "catalog_fingerprint",
            "dispatch_receipt_fingerprint",
            "profile_fingerprint",
            "environment_fingerprint",
            "evidence_result_digest",
            "kernel_fingerprint",
        ):
            _require_sha256(name, getattr(self, name), allow_zero=False)
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise SchemaError("Attention KV physical layout rule_id must be non-empty")
        features = tuple(sorted(str(item) for item in self.required_features))
        if any(not item for item in features) or len(set(features)) != len(features):
            raise SchemaError("physical layout required_features must be unique and non-empty")
        object.__setattr__(self, "required_features", features)

    def validate(
        self,
        kv: KVCacheView,
        catalog: QuantPhysicalLayoutCatalog,
        receipt: AttentionDispatchReceipt,
    ) -> None:
        if not isinstance(kv, KVCacheView) or not kv.quantized:
            raise SchemaError("physical layout binding requires quantized KV")
        quant_spec = kv.spec.quant_spec
        assert quant_spec is not None
        if quant_spec.physical_layout == "logical":
            raise SchemaError("physical layout binding cannot authorize logical KV")
        descriptor = catalog.resolve(quant_spec)
        if not isinstance(kv.key, QuantizedTensorView):
            raise SchemaError("physical layout binding requires separate quantized KV")
        if (
            kv.key.physical_layout_descriptor != descriptor
            or kv.value.physical_layout_descriptor != descriptor
            or self.quant_spec_fingerprint != quant_spec.fingerprint
            or self.descriptor_fingerprint != descriptor.fingerprint
            or self.catalog_fingerprint != catalog.fingerprint
            or self.dispatch_receipt_fingerprint != receipt.fingerprint
            or self.profile_fingerprint != receipt.profile_fingerprint
            or self.rule_id != receipt.rule_id
            or self.environment_fingerprint != receipt.environment_fingerprint
            or self.evidence_result_digest != receipt.evidence_result_digest
            or self.kernel_fingerprint != receipt.kernel_fingerprint
            or self.required_features != descriptor.required_features
        ):
            raise SchemaError("Attention KV physical layout binding is stale")

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["required_features"] = list(self.required_features)
        result["access_code"] = int(self.access_code)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionKVPhysicalLayoutBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionKVPhysicalLayoutBinding fields are invalid")
        if not isinstance(data.get("required_features"), (list, tuple)):
            raise SchemaError("physical layout required_features must be an array")
        data["required_features"] = tuple(data["required_features"])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionKVPhysicalLayoutBinding fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@runtime_checkable
class AttentionKVPhysicalLayoutEvidence(Protocol):
    """External evidence capable of authorizing one physical KV binding."""

    evidence_id: str
    result_digest: str

    def validate_authority(
        self,
        kv: KVCacheView,
        catalog: QuantPhysicalLayoutCatalog,
        profile: AttentionBackendCapabilityProfile,
        kernel: KernelDescriptor,
        observed_environment: AttentionRuntimeEnvironment,
        receipt: AttentionDispatchReceipt,
    ) -> None:
        """Raise unless every physical-layout authority identity is exact."""


def bind_attention_kv_physical_layout(
    kv: KVCacheView,
    catalog: QuantPhysicalLayoutCatalog,
    profile: AttentionBackendCapabilityProfile,
    kernel: KernelDescriptor,
    observed_environment: AttentionRuntimeEnvironment,
    receipt: AttentionDispatchReceipt,
    physical_layout_evidence: Optional[AttentionKVPhysicalLayoutEvidence] = None,
) -> AttentionKVPhysicalLayoutBinding:
    """Verify and bind one kernel-native physical KV layout candidate."""

    if not isinstance(catalog, QuantPhysicalLayoutCatalog):
        raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
    if not isinstance(profile, AttentionBackendCapabilityProfile):
        raise TypeError("profile must be AttentionBackendCapabilityProfile")
    if not isinstance(kernel, KernelDescriptor):
        raise TypeError("kernel must be KernelDescriptor")
    if not isinstance(observed_environment, AttentionRuntimeEnvironment):
        raise TypeError("observed_environment must be AttentionRuntimeEnvironment")
    if not isinstance(receipt, AttentionDispatchReceipt):
        raise TypeError("receipt must be AttentionDispatchReceipt")
    if not isinstance(kv, KVCacheView) or not kv.quantized:
        raise SchemaError("physical layout binding requires quantized KV")
    quant_spec = kv.spec.quant_spec
    assert quant_spec is not None
    if quant_spec.physical_layout == "logical":
        raise SchemaError("physical layout binding requires a non-logical QuantSpec")
    if not isinstance(kv.key, QuantizedTensorView):
        raise SchemaError("physical layout binding requires separate quantized KV")
    descriptor = catalog.resolve(quant_spec)
    if (
        kv.key.physical_layout_descriptor != descriptor
        or kv.value.physical_layout_descriptor != descriptor
    ):
        raise SchemaError("KV physical layout descriptor is not catalog-owned")

    validate_attention_kernel_bindings((profile,), (kernel,))
    rule = next((item for item in profile.rules if item.rule_id == receipt.rule_id), None)
    if physical_layout_evidence is None:
        evidence = next(
            (
                item
                for item in profile.evidence
                if item.evidence_id == receipt.evidence_id
                and item.result_digest == receipt.evidence_result_digest
            ),
            None,
        )
        evidence_digest = evidence.result_digest if evidence is not None else None
    else:
        if not isinstance(
            physical_layout_evidence, AttentionKVPhysicalLayoutEvidence
        ):
            raise TypeError(
                "physical_layout_evidence must implement "
                "AttentionKVPhysicalLayoutEvidence"
            )
        physical_layout_evidence.validate_authority(
            kv,
            catalog,
            profile,
            kernel,
            observed_environment,
            receipt,
        )
        evidence = physical_layout_evidence
        evidence_digest = physical_layout_evidence.result_digest
    binding = kernel.capability_binding
    if (
        observed_environment != profile.environment
        or receipt.profile_id != profile.profile_id
        or receipt.profile_fingerprint != profile.fingerprint
        or receipt.environment_fingerprint != observed_environment.fingerprint
        or receipt.kernel_id != kernel.kernel_id
        or receipt.kernel_fingerprint != kernel.fingerprint
        or kernel.artifact is None
        or receipt.artifact_fingerprint != kernel.artifact.fingerprint
        or kernel.launch_abi is None
        or receipt.launch_abi_fingerprint != kernel.launch_abi.fingerprint
        or kernel.binary_abi is None
        or receipt.binary_abi_fingerprint != kernel.binary_abi.fingerprint
        or kernel.binary_abi.fingerprint != attention_kernel_binary_abi_v2().fingerprint
        or binding is None
        or binding.rule_id != receipt.rule_id
        or rule is None
        or evidence is None
        or quant_spec not in rule.quant_specs
    ):
        raise SchemaError("physical layout capability/dispatch evidence is incompatible")
    features = set(descriptor.required_features)
    if (
        not features
        or not features <= set(rule.required_features)
        or not features <= set(kernel.constraints.required_features)
        or not features <= set(observed_environment.features)
    ):
        raise SchemaError("physical layout required features are not fully evidenced")
    value = AttentionKVPhysicalLayoutBinding(
        quant_spec.fingerprint,
        descriptor.fingerprint,
        catalog.fingerprint,
        receipt.fingerprint,
        profile.fingerprint,
        rule.rule_id,
        observed_environment.fingerprint,
        evidence_digest,
        kernel.fingerprint,
        descriptor.required_features,
    )
    value.validate(kv, catalog, receipt)
    return value


@dataclass(frozen=True)
class AttentionKVCacheViewPOD:
    components_ptr: int
    components: Tuple[AttentionTensorViewPOD, ...]
    layout_code: AttentionKVLayoutCode
    flags: AttentionKVFlags
    quant_spec_fingerprint: str
    schema_version: int = ATTENTION_LAUNCH_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_BINDING_VERSION:
            raise SchemaError("unsupported Attention KV POD version")
        object.__setattr__(self, "components", tuple(self.components))
        try:
            object.__setattr__(self, "layout_code", AttentionKVLayoutCode(self.layout_code))
            object.__setattr__(self, "flags", AttentionKVFlags(self.flags))
        except ValueError as error:
            raise SchemaError("Attention KV POD contains an unknown enum") from error
        if not self.components or self.components_ptr <= 0:
            raise SchemaError("Attention KV POD requires a component table")
        if int(self.flags) & ~_KNOWN_KV_FLAGS_V1:
            raise SchemaError("Attention KV POD flags contain unknown bits")
        if len(self.quant_spec_fingerprint) != 64 or any(
            item not in "0123456789abcdef" for item in self.quant_spec_fingerprint
        ):
            raise SchemaError("Attention KV POD quant fingerprint must be SHA-256")
        roles = tuple(item.role_code for item in self.components)
        dense_roles = (
            int(AttentionTensorRole.KV_KEY_STORAGE),
            int(AttentionTensorRole.KV_VALUE_STORAGE),
        )
        quantized_roles = {
            (
                int(AttentionTensorRole.KV_KEY_STORAGE),
                int(AttentionTensorRole.KV_KEY_SCALE),
                int(AttentionTensorRole.KV_VALUE_STORAGE),
                int(AttentionTensorRole.KV_VALUE_SCALE),
            ),
            (
                int(AttentionTensorRole.KV_KEY_STORAGE),
                int(AttentionTensorRole.KV_KEY_SCALE),
                int(AttentionTensorRole.KV_KEY_ZERO_POINT),
                int(AttentionTensorRole.KV_VALUE_STORAGE),
                int(AttentionTensorRole.KV_VALUE_SCALE),
                int(AttentionTensorRole.KV_VALUE_ZERO_POINT),
            ),
        }
        if roles not in {
            (int(AttentionTensorRole.KV_PACKED_STORAGE),),
            dense_roles,
            *quantized_roles,
        }:
            raise SchemaError("Attention KV POD component role sequence is not canonical")
        if bool(self.flags & AttentionKVFlags.PACKED) != (
            roles == (int(AttentionTensorRole.KV_PACKED_STORAGE),)
        ):
            raise SchemaError("Attention KV POD packed flag/components disagree")
        quantized = any(
            role
            in {
                int(AttentionTensorRole.KV_KEY_SCALE),
                int(AttentionTensorRole.KV_VALUE_SCALE),
            }
            for role in roles
        )
        if bool(self.flags & AttentionKVFlags.QUANTIZED) != quantized:
            raise SchemaError("Attention KV POD quantized flag/components disagree")
        if quantized != (self.quant_spec_fingerprint != "0" * 64):
            raise SchemaError("Attention KV POD quant fingerprint presence is invalid")

    @property
    def component_blob(self) -> bytes:
        return b"".join(item.pack() for item in self.components)

    def pack(self) -> bytes:
        return ATTENTION_KV_CACHE_VIEW_C_ABI.pack(
            {
                "components_ptr": self.components_ptr,
                "component_count": len(self.components),
                "layout_code": int(self.layout_code),
                "flags": int(self.flags),
                "quant_spec_fingerprint": tuple(bytes.fromhex(self.quant_spec_fingerprint)),
            }
        )

    @classmethod
    def from_bytes(cls, descriptor: bytes, component_blob: bytes) -> "AttentionKVCacheViewPOD":
        values = ATTENTION_KV_CACHE_VIEW_C_ABI.unpack(descriptor)
        expected = values["component_count"] * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes
        if len(component_blob) != expected:
            raise SchemaError("Attention KV component blob size is invalid")
        components = tuple(
            AttentionTensorViewPOD.from_bytes(
                component_blob[offset : offset + ATTENTION_TENSOR_VIEW_C_ABI.size_bytes]
            )
            for offset in range(0, expected, ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        )
        return cls(
            values["components_ptr"],
            components,
            values["layout_code"],
            values["flags"],
            bytes(values["quant_spec_fingerprint"]).hex(),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.pack() + self.component_blob).hexdigest()


def materialize_attention_kv_cache_view(
    kv: KVCacheView,
    bindings: Sequence[AttentionAddressBinding],
    component_table_lease: AttentionHostBufferLease,
    device_index: int,
) -> AttentionKVCacheViewPOD:
    if (
        kv.quantized
        and kv.spec.quant_spec is not None
        and kv.spec.quant_spec.physical_layout != "logical"
    ):
        raise SchemaError(
            "Attention KV POD v1 cannot bind a non-logical physical layout descriptor"
        )
    mapping = _binding_map(bindings)
    components = tuple(
        materialize_attention_tensor_view(
            _binding_for(mapping, name, view), int(role), device_index
        )
        for name, role, view in _kv_components(kv)
    )
    blob = b"".join(item.pack() for item in components)
    _validate_component_table_lease(component_table_lease, blob)
    flags = AttentionKVFlags(0)
    if kv.packed:
        flags |= AttentionKVFlags.PACKED
    if kv.quantized:
        flags |= AttentionKVFlags.QUANTIZED
    return AttentionKVCacheViewPOD(
        component_table_lease.base_address,
        components,
        AttentionKVLayoutCode.NHD if kv.spec.layout == KVLayout.NHD else AttentionKVLayoutCode.HND,
        flags,
        kv.spec.quant_spec.fingerprint if kv.spec.quant_spec is not None else "0" * 64,
    )


@dataclass(frozen=True)
class AttentionKVCacheViewPODV2:
    components_ptr: int
    components: Tuple[AttentionTensorViewPOD, ...]
    layout_code: AttentionKVLayoutCode
    flags: AttentionKVFlags
    quant_spec_fingerprint: str
    physical_layout_access_code: AttentionKVPhysicalLayoutAccessCode
    physical_layout_descriptor_fingerprint: str = _ZERO_SHA256
    physical_layout_catalog_fingerprint: str = _ZERO_SHA256
    physical_layout_binding_fingerprint: str = _ZERO_SHA256
    dispatch_receipt_fingerprint: str = _ZERO_SHA256
    schema_version: int = ATTENTION_KV_CACHE_VIEW_POD_V2_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_KV_CACHE_VIEW_POD_V2_VERSION:
            raise SchemaError("unsupported Attention KV POD v2 version")
        object.__setattr__(self, "components", tuple(self.components))
        try:
            object.__setattr__(self, "layout_code", AttentionKVLayoutCode(self.layout_code))
            object.__setattr__(self, "flags", AttentionKVFlags(self.flags))
            object.__setattr__(
                self,
                "physical_layout_access_code",
                AttentionKVPhysicalLayoutAccessCode(
                    self.physical_layout_access_code
                ),
            )
        except ValueError as error:
            raise SchemaError("Attention KV POD v2 contains an unknown enum") from error
        if not self.components or self.components_ptr <= 0:
            raise SchemaError("Attention KV POD v2 requires a component table")
        if int(self.flags) & ~_KNOWN_KV_FLAGS_V2:
            raise SchemaError("Attention KV POD v2 flags contain unknown bits")
        for name in (
            "quant_spec_fingerprint",
            "physical_layout_descriptor_fingerprint",
            "physical_layout_catalog_fingerprint",
            "physical_layout_binding_fingerprint",
            "dispatch_receipt_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))

        roles = tuple(item.role_code for item in self.components)
        dense_roles = (
            int(AttentionTensorRole.KV_KEY_STORAGE),
            int(AttentionTensorRole.KV_VALUE_STORAGE),
        )
        quantized_roles = {
            (
                int(AttentionTensorRole.KV_KEY_STORAGE),
                int(AttentionTensorRole.KV_KEY_SCALE),
                int(AttentionTensorRole.KV_VALUE_STORAGE),
                int(AttentionTensorRole.KV_VALUE_SCALE),
            ),
            (
                int(AttentionTensorRole.KV_KEY_STORAGE),
                int(AttentionTensorRole.KV_KEY_SCALE),
                int(AttentionTensorRole.KV_KEY_ZERO_POINT),
                int(AttentionTensorRole.KV_VALUE_STORAGE),
                int(AttentionTensorRole.KV_VALUE_SCALE),
                int(AttentionTensorRole.KV_VALUE_ZERO_POINT),
            ),
        }
        if roles not in {
            (int(AttentionTensorRole.KV_PACKED_STORAGE),),
            dense_roles,
            *quantized_roles,
        }:
            raise SchemaError("Attention KV POD v2 component role sequence is not canonical")
        packed = roles == (int(AttentionTensorRole.KV_PACKED_STORAGE),)
        quantized = any(
            role
            in {
                int(AttentionTensorRole.KV_KEY_SCALE),
                int(AttentionTensorRole.KV_VALUE_SCALE),
            }
            for role in roles
        )
        if bool(self.flags & AttentionKVFlags.PACKED) != packed:
            raise SchemaError("Attention KV POD v2 packed flag/components disagree")
        if bool(self.flags & AttentionKVFlags.QUANTIZED) != quantized:
            raise SchemaError("Attention KV POD v2 quantized flag/components disagree")
        if quantized != (self.quant_spec_fingerprint != _ZERO_SHA256):
            raise SchemaError("Attention KV POD v2 quant fingerprint presence is invalid")

        physical = bool(self.flags & AttentionKVFlags.PHYSICAL_LAYOUT)
        physical_hashes = (
            self.physical_layout_descriptor_fingerprint,
            self.physical_layout_catalog_fingerprint,
            self.physical_layout_binding_fingerprint,
            self.dispatch_receipt_fingerprint,
        )
        if physical:
            if (
                not quantized
                or self.physical_layout_access_code
                != AttentionKVPhysicalLayoutAccessCode.KERNEL_NATIVE
                or any(value == _ZERO_SHA256 for value in physical_hashes)
            ):
                raise SchemaError("physical Attention KV POD v2 evidence is incomplete")
        elif (
            self.physical_layout_access_code
            != AttentionKVPhysicalLayoutAccessCode.LOGICAL
            or any(value != _ZERO_SHA256 for value in physical_hashes)
        ):
            raise SchemaError("logical Attention KV POD v2 must use canonical zero layout evidence")

    @property
    def component_blob(self) -> bytes:
        return b"".join(item.pack() for item in self.components)

    def pack(self) -> bytes:
        return ATTENTION_KV_CACHE_VIEW_V2_C_ABI.pack(
            {
                "components_ptr": self.components_ptr,
                "component_count": len(self.components),
                "layout_code": int(self.layout_code),
                "flags": int(self.flags),
                "physical_layout_access_code": int(
                    self.physical_layout_access_code
                ),
                "quant_spec_fingerprint": tuple(
                    bytes.fromhex(self.quant_spec_fingerprint)
                ),
                "physical_layout_descriptor_fingerprint": tuple(
                    bytes.fromhex(self.physical_layout_descriptor_fingerprint)
                ),
                "physical_layout_catalog_fingerprint": tuple(
                    bytes.fromhex(self.physical_layout_catalog_fingerprint)
                ),
                "physical_layout_binding_fingerprint": tuple(
                    bytes.fromhex(self.physical_layout_binding_fingerprint)
                ),
                "dispatch_receipt_fingerprint": tuple(
                    bytes.fromhex(self.dispatch_receipt_fingerprint)
                ),
            }
        )

    @classmethod
    def from_bytes(
        cls, descriptor: bytes, component_blob: bytes
    ) -> "AttentionKVCacheViewPODV2":
        values = ATTENTION_KV_CACHE_VIEW_V2_C_ABI.unpack(descriptor)
        expected = values["component_count"] * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes
        if len(component_blob) != expected:
            raise SchemaError("Attention KV POD v2 component blob size is invalid")
        components = tuple(
            AttentionTensorViewPOD.from_bytes(
                component_blob[offset : offset + ATTENTION_TENSOR_VIEW_C_ABI.size_bytes]
            )
            for offset in range(0, expected, ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        )
        return cls(
            components_ptr=values["components_ptr"],
            components=components,
            layout_code=values["layout_code"],
            flags=values["flags"],
            quant_spec_fingerprint=bytes(values["quant_spec_fingerprint"]).hex(),
            physical_layout_access_code=values["physical_layout_access_code"],
            physical_layout_descriptor_fingerprint=bytes(
                values["physical_layout_descriptor_fingerprint"]
            ).hex(),
            physical_layout_catalog_fingerprint=bytes(
                values["physical_layout_catalog_fingerprint"]
            ).hex(),
            physical_layout_binding_fingerprint=bytes(
                values["physical_layout_binding_fingerprint"]
            ).hex(),
            dispatch_receipt_fingerprint=bytes(
                values["dispatch_receipt_fingerprint"]
            ).hex(),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.pack() + self.component_blob).hexdigest()


def materialize_attention_kv_cache_view_v2(
    kv: KVCacheView,
    bindings: Sequence[AttentionAddressBinding],
    component_table_lease: AttentionHostBufferLease,
    device_index: int,
    *,
    catalog: Optional[QuantPhysicalLayoutCatalog] = None,
    physical_layout_binding: Optional[AttentionKVPhysicalLayoutBinding] = None,
    dispatch_receipt: Optional[AttentionDispatchReceipt] = None,
) -> AttentionKVCacheViewPODV2:
    """Materialize KV POD v2 without executing a converter or kernel."""

    mapping = _binding_map(bindings)
    components = tuple(
        materialize_attention_tensor_view(
            _binding_for(mapping, name, view), int(role), device_index
        )
        for name, role, view in _kv_components(kv)
    )
    blob = b"".join(item.pack() for item in components)
    _validate_component_table_lease(component_table_lease, blob)
    flags = AttentionKVFlags(0)
    if kv.packed:
        flags |= AttentionKVFlags.PACKED
    if kv.quantized:
        flags |= AttentionKVFlags.QUANTIZED

    quant_spec = kv.spec.quant_spec
    nonlogical = bool(
        quant_spec is not None and quant_spec.physical_layout != "logical"
    )
    if nonlogical:
        if (
            catalog is None
            or physical_layout_binding is None
            or dispatch_receipt is None
        ):
            raise SchemaError("non-logical KV POD v2 requires catalog, binding and receipt")
        if not isinstance(physical_layout_binding, AttentionKVPhysicalLayoutBinding):
            raise TypeError(
                "physical_layout_binding must be AttentionKVPhysicalLayoutBinding"
            )
        if not isinstance(dispatch_receipt, AttentionDispatchReceipt):
            raise TypeError("dispatch_receipt must be AttentionDispatchReceipt")
        if not isinstance(catalog, QuantPhysicalLayoutCatalog):
            raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
        physical_layout_binding.validate(kv, catalog, dispatch_receipt)
        flags |= AttentionKVFlags.PHYSICAL_LAYOUT
        descriptor = catalog.resolve(quant_spec)
        access = AttentionKVPhysicalLayoutAccessCode.KERNEL_NATIVE
        descriptor_fingerprint = descriptor.fingerprint
        catalog_fingerprint = catalog.fingerprint
        binding_fingerprint = physical_layout_binding.fingerprint
        receipt_fingerprint = dispatch_receipt.fingerprint
    else:
        if any(
            value is not None
            for value in (catalog, physical_layout_binding, dispatch_receipt)
        ):
            raise SchemaError("logical KV POD v2 cannot carry physical layout evidence")
        access = AttentionKVPhysicalLayoutAccessCode.LOGICAL
        descriptor_fingerprint = _ZERO_SHA256
        catalog_fingerprint = _ZERO_SHA256
        binding_fingerprint = _ZERO_SHA256
        receipt_fingerprint = _ZERO_SHA256
    return AttentionKVCacheViewPODV2(
        component_table_lease.base_address,
        components,
        AttentionKVLayoutCode.NHD
        if kv.spec.layout == KVLayout.NHD
        else AttentionKVLayoutCode.HND,
        flags,
        quant_spec.fingerprint if quant_spec is not None else _ZERO_SHA256,
        access,
        descriptor_fingerprint,
        catalog_fingerprint,
        binding_fingerprint,
        receipt_fingerprint,
    )


@dataclass(frozen=True)
class AttentionAuxiliaryViewPOD:
    components_ptr: int
    components: Tuple[AttentionTensorViewPOD, ...]
    flags: int = 0
    schema_version: int = ATTENTION_LAUNCH_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_BINDING_VERSION:
            raise SchemaError("unsupported Attention auxiliary POD version")
        object.__setattr__(self, "components", tuple(self.components))
        if self.flags != 0:
            raise SchemaError("Attention auxiliary POD v1 flags must be zero")
        if self.components and self.components_ptr <= 0:
            raise SchemaError("non-empty Attention auxiliary POD requires a component table")
        if not self.components and self.components_ptr != 0:
            raise SchemaError("empty Attention auxiliary POD must use a null component table")
        roles = tuple(item.role_code for item in self.components)
        if roles != tuple(sorted(roles)) or len(set(roles)) != len(roles):
            raise SchemaError("Attention auxiliary POD roles must be unique and canonical")
        for role in roles:
            try:
                AttentionAuxiliaryRole(role)
            except ValueError as error:
                raise SchemaError("Attention auxiliary POD contains an unknown role") from error

    @property
    def component_blob(self) -> bytes:
        return b"".join(item.pack() for item in self.components)

    def pack(self) -> bytes:
        return ATTENTION_AUXILIARY_VIEW_C_ABI.pack(
            {
                "components_ptr": self.components_ptr,
                "component_count": len(self.components),
                "flags": self.flags,
            }
        )

    @classmethod
    def from_bytes(
        cls, descriptor: bytes, component_blob: bytes
    ) -> "AttentionAuxiliaryViewPOD":
        values = ATTENTION_AUXILIARY_VIEW_C_ABI.unpack(descriptor)
        expected = values["component_count"] * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes
        if len(component_blob) != expected:
            raise SchemaError("Attention auxiliary component blob size is invalid")
        components = tuple(
            AttentionTensorViewPOD.from_bytes(
                component_blob[offset : offset + ATTENTION_TENSOR_VIEW_C_ABI.size_bytes]
            )
            for offset in range(0, expected, ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        )
        return cls(values["components_ptr"], components, values["flags"])

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.pack() + self.component_blob).hexdigest()


def materialize_attention_auxiliary_view(
    auxiliary: AttentionAuxiliaryContract,
    bindings: Sequence[AttentionAddressBinding],
    component_table_lease: AttentionHostBufferLease,
    device_index: int,
) -> AttentionAuxiliaryViewPOD:
    if not isinstance(auxiliary, AttentionAuxiliaryContract):
        raise TypeError("auxiliary must be AttentionAuxiliaryContract")
    mapping = _binding_map(bindings)
    components = tuple(
        materialize_attention_tensor_view(
            _binding_for(mapping, "aux.%s" % item.role.name.lower(), item.view),
            int(item.role),
            device_index,
        )
        for item in auxiliary.components
    )
    blob = b"".join(item.pack() for item in components)
    if components:
        _validate_component_table_lease(component_table_lease, blob)
        address = component_table_lease.base_address
    else:
        if component_table_lease.capacity_bytes != 0 or component_table_lease.base_address != 0:
            raise SchemaError("empty auxiliary component table must use an empty lease")
        address = 0
    return AttentionAuxiliaryViewPOD(address, components)


__all__ = [
    "ATTENTION_KV_CACHE_VIEW_POD_V2_VERSION",
    "ATTENTION_KV_PHYSICAL_LAYOUT_BINDING_VERSION",
    "ATTENTION_LAUNCH_BINDING_VERSION",
    "AttentionAuxiliaryViewPOD",
    "AttentionKVCacheViewPOD",
    "AttentionKVCacheViewPODV2",
    "AttentionKVPhysicalLayoutBinding",
    "AttentionKVPhysicalLayoutEvidence",
    "AttentionTensorViewPOD",
    "bind_attention_kv_physical_layout",
    "materialize_attention_auxiliary_view",
    "materialize_attention_kv_cache_view",
    "materialize_attention_kv_cache_view_v2",
    "materialize_attention_tensor_view",
]
