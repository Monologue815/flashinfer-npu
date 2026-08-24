"""Canonical, ownership-complete Host packet for a future Attention launch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple

from flashinfer_npu.runtime import SchemaError

from .dispatch import AttentionDispatchReceipt
from .execution_identity import (
    AttentionExecutionIdentity,
    attention_run_tensor_signature,
    attention_stream_context_fingerprint,
)
from .launch_binding import (
    AttentionAuxiliaryViewPOD,
    AttentionKVCacheViewPOD,
    AttentionTensorViewPOD,
    materialize_attention_auxiliary_view,
    materialize_attention_kv_cache_view,
    materialize_attention_tensor_view,
)
from .launch_contract import (
    ATTENTION_AUXILIARY_VIEW_C_ABI,
    ATTENTION_KV_CACHE_VIEW_C_ABI,
    ATTENTION_RUN_OPTIONS_C_ABI,
    ATTENTION_TENSOR_VIEW_C_ABI,
    AttentionAuxiliaryRole,
    AttentionTensorRole,
    attention_kernel_binary_abi,
)
from .metadata_wire import (
    AttentionPlanMetadataWire,
    materialize_attention_plan_metadata,
)
from .planner import AttentionFrameworkPlan
from .storage_lease import (
    AttentionAddressBinding,
    AttentionHostBufferLease,
    AttentionLaunchLeaseContract,
    AttentionStorageLifetime,
)
from .tensor_contract import (
    AttentionAuxiliaryContract,
    AttentionAuxiliaryTensor,
    AttentionRunOptions,
    AttentionRunTensorContract,
    StreamContext,
)


ATTENTION_LAUNCH_PACKET_VERSION = 1
ATTENTION_LAUNCH_PACKET_MAX_HOST_BYTES = 128 * 1024 * 1024


class AttentionHostBufferRole(str, Enum):
    Q_DESCRIPTOR = "q_descriptor"
    KV_DESCRIPTOR = "kv_descriptor"
    KV_COMPONENTS = "kv_components"
    AUX_DESCRIPTOR = "aux_descriptor"
    AUX_COMPONENTS = "aux_components"
    RUN_OPTIONS = "run_options"
    OUT_DESCRIPTOR = "out_descriptor"
    LSE_DESCRIPTOR = "lse_descriptor"
    PLAN_METADATA = "plan_metadata"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionStreamBinding:
    device: str
    device_index: int
    stream_id: str
    stream_handle: int
    runtime_id: str
    runtime_generation: int
    schema_version: int = ATTENTION_LAUNCH_PACKET_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_PACKET_VERSION:
            raise SchemaError("unsupported Attention stream binding version")
        for name in ("device", "stream_id", "runtime_id"):
            if not str(getattr(self, name)):
                raise SchemaError("Attention stream binding %s must be non-empty" % name)
        for name in ("device_index", "stream_handle", "runtime_generation"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SchemaError("Attention stream binding integer fields must be integers")
        if self.device_index < 0 or self.stream_handle <= 0:
            raise SchemaError("Attention stream binding device/handle is invalid")
        if self.runtime_generation < 1:
            raise SchemaError("Attention stream runtime_generation must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionStreamBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionStreamBinding fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionStreamBinding fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionHostBufferBinding:
    role: AttentionHostBufferRole
    lease: AttentionHostBufferLease
    content: bytes
    schema_version: int = ATTENTION_LAUNCH_PACKET_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_PACKET_VERSION:
            raise SchemaError("unsupported Attention host buffer binding version")
        try:
            object.__setattr__(self, "role", AttentionHostBufferRole(self.role))
        except ValueError as error:
            raise SchemaError("unknown Attention host buffer role") from error
        if not isinstance(self.lease, AttentionHostBufferLease):
            raise TypeError("host buffer binding lease must be AttentionHostBufferLease")
        if not isinstance(self.content, bytes):
            raise TypeError("host buffer binding content must be bytes")
        if len(self.content) > ATTENTION_LAUNCH_PACKET_MAX_HOST_BYTES:
            raise SchemaError("Attention host buffer content exceeds packet limit")
        if len(self.content) > self.lease.capacity_bytes:
            raise SchemaError("Attention host buffer content exceeds lease capacity")
        if self.content and (self.lease.base_address == 0 or self.lease.alignment < 8):
            raise SchemaError("Attention host buffer address/alignment is incompatible")
        if self.content and not self.lease.writable:
            raise SchemaError("Attention host buffer must be writable for materialization")
        if not self.content and (
            self.lease.base_address != 0 or self.lease.capacity_bytes != 0
        ):
            raise SchemaError("empty Attention host buffer must use a null/zero lease")

    @property
    def address(self) -> int:
        return self.lease.base_address

    @property
    def content_fingerprint(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role.value,
            "lease": self.lease.to_dict(),
            "content_hex": self.content.hex(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionHostBufferBinding":
        data = dict(value)
        if set(data) != {"schema_version", "role", "lease", "content_hex"}:
            raise SchemaError("AttentionHostBufferBinding fields are invalid")
        if not isinstance(data.get("lease"), Mapping) or not isinstance(
            data.get("content_hex"), str
        ):
            raise SchemaError("AttentionHostBufferBinding nested fields are invalid")
        if len(data["content_hex"]) > ATTENTION_LAUNCH_PACKET_MAX_HOST_BYTES * 2:
            raise SchemaError("Attention host buffer content exceeds packet limit")
        try:
            content = bytes.fromhex(data.pop("content_hex"))
        except ValueError as error:
            raise SchemaError("Attention host buffer content_hex is invalid") from error
        data["lease"] = AttentionHostBufferLease.from_dict(data["lease"])
        data["content"] = content
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionHostBufferBinding fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionLaunchArguments:
    q: int
    kv: int
    aux: int
    run_options: int
    out: int
    lse: int
    plan_metadata: int
    plan_metadata_nbytes: int
    float_workspace: int
    float_workspace_nbytes: int
    int_workspace: int
    int_workspace_nbytes: int
    stream: int
    schema_version: int = ATTENTION_LAUNCH_PACKET_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_PACKET_VERSION:
            raise SchemaError("unsupported Attention launch arguments version")
        for name in self.__dataclass_fields__:
            if name == "schema_version":
                continue
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SchemaError("Attention launch arguments must be non-negative integers")
        for name in ("q", "kv", "aux", "run_options", "out", "plan_metadata", "stream"):
            if getattr(self, name) == 0:
                raise SchemaError("Attention launch argument %s cannot be null" % name)
        if self.plan_metadata_nbytes <= 0:
            raise SchemaError("Attention plan metadata byte count must be positive")
        for pointer_name, size_name in (
            ("float_workspace", "float_workspace_nbytes"),
            ("int_workspace", "int_workspace_nbytes"),
        ):
            if (getattr(self, pointer_name) == 0) != (getattr(self, size_name) == 0):
                raise SchemaError("zero workspace size must use null and vice versa")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionLaunchArguments":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionLaunchArguments fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionLaunchArguments fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


_KV_ROLE_NAMES = {
    int(AttentionTensorRole.KV_KEY_STORAGE): "kv.key_storage",
    int(AttentionTensorRole.KV_KEY_SCALE): "kv.key_scale",
    int(AttentionTensorRole.KV_KEY_ZERO_POINT): "kv.key_zero_point",
    int(AttentionTensorRole.KV_VALUE_STORAGE): "kv.value_storage",
    int(AttentionTensorRole.KV_VALUE_SCALE): "kv.value_scale",
    int(AttentionTensorRole.KV_VALUE_ZERO_POINT): "kv.value_zero_point",
    int(AttentionTensorRole.KV_PACKED_STORAGE): "kv.packed_storage",
}


@dataclass(frozen=True)
class AttentionLaunchPacket:
    execution_identity: AttentionExecutionIdentity
    dispatch_receipt: AttentionDispatchReceipt
    launch_lease: AttentionLaunchLeaseContract
    stream_binding: AttentionStreamBinding
    host_buffers: Tuple[AttentionHostBufferBinding, ...]
    arguments: AttentionLaunchArguments
    graph_enabled: bool = False
    schema_version: int = ATTENTION_LAUNCH_PACKET_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCH_PACKET_VERSION:
            raise SchemaError("unsupported Attention launch packet version")
        for value, expected, name in (
            (self.execution_identity, AttentionExecutionIdentity, "execution_identity"),
            (self.dispatch_receipt, AttentionDispatchReceipt, "dispatch_receipt"),
            (self.launch_lease, AttentionLaunchLeaseContract, "launch_lease"),
            (self.stream_binding, AttentionStreamBinding, "stream_binding"),
            (self.arguments, AttentionLaunchArguments, "arguments"),
        ):
            if not isinstance(value, expected):
                raise TypeError("%s has the wrong type" % name)
        if self.execution_identity.binding_kind != "kernel":
            raise SchemaError("Attention launch packet requires a kernel identity")
        receipt = self.dispatch_receipt
        identity = self.execution_identity
        if (
            identity.plan_fingerprint != receipt.plan_fingerprint
            or identity.admission_fingerprint != receipt.admission_fingerprint
            or identity.backend != receipt.backend.value
            or identity.kernel_id != receipt.kernel_id
            or identity.kernel_fingerprint != receipt.kernel_fingerprint
            or identity.capability_profile_id != receipt.profile_id
            or identity.capability_profile_fingerprint != receipt.profile_fingerprint
            or identity.capability_rule_id != receipt.rule_id
            or identity.capability_evidence_id != receipt.evidence_id
        ):
            raise SchemaError("Attention launch identity and dispatch receipt disagree")
        if receipt.binary_abi_fingerprint != attention_kernel_binary_abi().fingerprint:
            raise SchemaError("Attention launch packet requires canonical binary ABI")
        if (
            self.launch_lease.execution_identity_fingerprint != identity.fingerprint
            or self.launch_lease.dispatch_receipt_fingerprint != receipt.fingerprint
        ):
            raise SchemaError("Attention launch lease fingerprints are stale")
        if (
            self.stream_binding.device != self.launch_lease.stream_device
            or self.stream_binding.stream_id != self.launch_lease.stream_id
            or self.arguments.stream != self.stream_binding.stream_handle
        ):
            raise SchemaError("Attention launch stream bindings disagree")
        stream_context_fingerprint = attention_stream_context_fingerprint(
            StreamContext(
                self.stream_binding.device,
                self.stream_binding.stream_id,
                ordered=True,
            )
        )
        if stream_context_fingerprint != identity.stream_context_fingerprint:
            raise SchemaError(
                "Attention launch stream context fingerprint is stale"
            )
        if not isinstance(self.graph_enabled, bool) or self.graph_enabled != identity.graph_enabled:
            raise SchemaError("Attention launch packet graph mode disagrees with identity")
        if self.graph_enabled != self.launch_lease.graph_enabled:
            raise SchemaError("Attention launch packet graph mode disagrees with lease")
        if identity.return_lse and self.arguments.lse == 0:
            raise SchemaError("return_lse launch identity requires a non-null LSE argument")

        buffers = tuple(self.host_buffers)
        if any(not isinstance(item, AttentionHostBufferBinding) for item in buffers):
            raise TypeError("host_buffers must contain AttentionHostBufferBinding")
        roles = tuple(item.role for item in buffers)
        if len(set(roles)) != len(roles):
            raise SchemaError("Attention launch host buffer roles must be unique")
        mapping = {item.role: item for item in buffers}
        required = {
            AttentionHostBufferRole.Q_DESCRIPTOR,
            AttentionHostBufferRole.KV_DESCRIPTOR,
            AttentionHostBufferRole.KV_COMPONENTS,
            AttentionHostBufferRole.AUX_DESCRIPTOR,
            AttentionHostBufferRole.AUX_COMPONENTS,
            AttentionHostBufferRole.RUN_OPTIONS,
            AttentionHostBufferRole.OUT_DESCRIPTOR,
            AttentionHostBufferRole.PLAN_METADATA,
        }
        if self.arguments.lse:
            required.add(AttentionHostBufferRole.LSE_DESCRIPTOR)
        if set(mapping) != required:
            raise SchemaError("Attention launch host buffer role set is not canonical")
        if sum(len(item.content) for item in buffers) > ATTENTION_LAUNCH_PACKET_MAX_HOST_BYTES:
            raise SchemaError("Attention launch packet host bytes exceed limit")
        for index, left in enumerate(buffers):
            for right in buffers[index + 1 :]:
                if left.lease.overlaps(right.lease):
                    raise SchemaError("Attention launch host buffers cannot overlap")
        if self.graph_enabled and any(
            item.lease.lifetime
            not in {AttentionStorageLifetime.PERSISTENT, AttentionStorageLifetime.CAPTURE}
            for item in buffers
        ):
            raise SchemaError("graph packet requires persistent/capture host buffers")
        object.__setattr__(self, "host_buffers", buffers)
        self._validate_buffer_contents(mapping)

    def _validate_buffer_contents(
        self, mapping: Mapping[AttentionHostBufferRole, AttentionHostBufferBinding]
    ) -> None:
        args = self.arguments
        address_fields = {
            AttentionHostBufferRole.Q_DESCRIPTOR: args.q,
            AttentionHostBufferRole.KV_DESCRIPTOR: args.kv,
            AttentionHostBufferRole.AUX_DESCRIPTOR: args.aux,
            AttentionHostBufferRole.RUN_OPTIONS: args.run_options,
            AttentionHostBufferRole.OUT_DESCRIPTOR: args.out,
            AttentionHostBufferRole.PLAN_METADATA: args.plan_metadata,
        }
        if args.lse:
            address_fields[AttentionHostBufferRole.LSE_DESCRIPTOR] = args.lse
        if any(mapping[role].address != address for role, address in address_fields.items()):
            raise SchemaError("Attention launch argument does not match host buffer address")

        q = AttentionTensorViewPOD.from_bytes(
            mapping[AttentionHostBufferRole.Q_DESCRIPTOR].content
        )
        out = AttentionTensorViewPOD.from_bytes(
            mapping[AttentionHostBufferRole.OUT_DESCRIPTOR].content
        )
        if q.role_code != int(AttentionTensorRole.Q) or out.role_code != int(
            AttentionTensorRole.OUT
        ):
            raise SchemaError("Attention launch q/out descriptor roles are invalid")
        lse = None
        if args.lse:
            lse = AttentionTensorViewPOD.from_bytes(
                mapping[AttentionHostBufferRole.LSE_DESCRIPTOR].content
            )
            if lse.role_code != int(AttentionTensorRole.LSE):
                raise SchemaError("Attention launch LSE descriptor role is invalid")
        kv = AttentionKVCacheViewPOD.from_bytes(
            mapping[AttentionHostBufferRole.KV_DESCRIPTOR].content,
            mapping[AttentionHostBufferRole.KV_COMPONENTS].content,
        )
        aux = AttentionAuxiliaryViewPOD.from_bytes(
            mapping[AttentionHostBufferRole.AUX_DESCRIPTOR].content,
            mapping[AttentionHostBufferRole.AUX_COMPONENTS].content,
        )
        options = AttentionRunOptions.from_bytes(
            mapping[AttentionHostBufferRole.RUN_OPTIONS].content
        )
        metadata_bytes = mapping[AttentionHostBufferRole.PLAN_METADATA].content
        metadata = AttentionPlanMetadataWire.from_bytes(metadata_bytes)
        if (
            kv.components_ptr
            != mapping[AttentionHostBufferRole.KV_COMPONENTS].address
            or aux.components_ptr
            != mapping[AttentionHostBufferRole.AUX_COMPONENTS].address
        ):
            raise SchemaError("Attention component table pointer is stale")
        if args.plan_metadata_nbytes != len(metadata_bytes):
            raise SchemaError("Attention launch plan metadata byte count is stale")
        receipt = self.dispatch_receipt
        identity = self.execution_identity
        if (
            metadata.plan_fingerprint != identity.plan_fingerprint
            or metadata.admission_fingerprint != identity.admission_fingerprint
            or metadata.dispatch_fingerprint != receipt.fingerprint
            or metadata.binary_abi_fingerprint != receipt.binary_abi_fingerprint
        ):
            raise SchemaError("Attention launch plan metadata fingerprints are stale")
        if options.fingerprint != identity.run_options_fingerprint:
            raise SchemaError("Attention launch run-options fingerprint is stale")

        bindings = {item.role: item for item in self.launch_lease.bindings}
        expected_roles = {"q", "out", "workspace_float", "workspace_int"}
        expected_roles.update(_KV_ROLE_NAMES[item.role_code] for item in kv.components)
        expected_roles.update(
            "aux.%s" % AttentionAuxiliaryRole(item.role_code).name.lower()
            for item in aux.components
        )
        if lse is not None:
            expected_roles.add("lse")
        if set(bindings) != expected_roles:
            raise SchemaError("Attention launch device binding role set is not canonical")
        auxiliary_contract = AttentionAuxiliaryContract(
            tuple(
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole(item.role_code),
                    bindings[
                        "aux.%s"
                        % AttentionAuxiliaryRole(item.role_code).name.lower()
                    ].view,
                )
                for item in aux.components
            )
        )
        if auxiliary_contract.fingerprint != identity.auxiliary_fingerprint:
            raise SchemaError("Attention launch auxiliary fingerprint is stale")

        descriptor_bindings = (("q", q), ("out", out))
        if lse is not None:
            descriptor_bindings += (("lse", lse),)
        descriptor_bindings += tuple(
            (_KV_ROLE_NAMES[item.role_code], item) for item in kv.components
        )
        descriptor_bindings += tuple(
            (
                "aux.%s" % AttentionAuxiliaryRole(item.role_code).name.lower(),
                item,
            )
            for item in aux.components
        )
        for role, pod in descriptor_bindings:
            expected = materialize_attention_tensor_view(
                bindings[role], pod.role_code, self.stream_binding.device_index
            )
            if expected != pod:
                raise SchemaError("Attention launch descriptor does not match %s lease" % role)

        for role, pointer, nbytes, required, alignment in (
            (
                "workspace_float",
                args.float_workspace,
                args.float_workspace_nbytes,
                receipt.float_workspace_bytes,
                receipt.float_workspace_alignment,
            ),
            (
                "workspace_int",
                args.int_workspace,
                args.int_workspace_nbytes,
                receipt.int_workspace_bytes,
                receipt.int_workspace_alignment,
            ),
        ):
            binding = bindings[role]
            view = binding.view
            if (
                view.dtype != "uint8"
                or len(view.shape) != 1
                or not view.is_contiguous
                or not view.writable
                or nbytes != view.numel
                or nbytes < required
                or pointer != (binding.data_address if nbytes else 0)
            ):
                raise SchemaError("Attention launch %s binding is incompatible" % role)
            view.require_alignment(alignment, role)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_identity": self.execution_identity.to_dict(),
            "dispatch_receipt": self.dispatch_receipt.to_dict(),
            "launch_lease": self.launch_lease.to_dict(),
            "stream_binding": self.stream_binding.to_dict(),
            "host_buffers": [item.to_dict() for item in self.host_buffers],
            "arguments": self.arguments.to_dict(),
            "graph_enabled": self.graph_enabled,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionLaunchPacket":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionLaunchPacket fields are invalid")
        for name in (
            "execution_identity",
            "dispatch_receipt",
            "launch_lease",
            "stream_binding",
            "arguments",
        ):
            if not isinstance(data.get(name), Mapping):
                raise SchemaError("AttentionLaunchPacket %s must be an object" % name)
        buffers = data.get("host_buffers")
        if not isinstance(buffers, (list, tuple)) or len(buffers) > len(
            AttentionHostBufferRole
        ):
            raise SchemaError("AttentionLaunchPacket host_buffers are invalid")
        data["execution_identity"] = AttentionExecutionIdentity.from_dict(
            data["execution_identity"]
        )
        data["dispatch_receipt"] = AttentionDispatchReceipt.from_dict(
            data["dispatch_receipt"]
        )
        data["launch_lease"] = AttentionLaunchLeaseContract.from_dict(
            data["launch_lease"]
        )
        data["stream_binding"] = AttentionStreamBinding.from_dict(
            data["stream_binding"]
        )
        data["host_buffers"] = tuple(
            AttentionHostBufferBinding.from_dict(item) for item in buffers
        )
        data["arguments"] = AttentionLaunchArguments.from_dict(data["arguments"])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionLaunchPacket fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _host_lease_map(
    leases: Mapping[AttentionHostBufferRole, AttentionHostBufferLease],
    required: Sequence[AttentionHostBufferRole],
) -> Dict[AttentionHostBufferRole, AttentionHostBufferLease]:
    result: Dict[AttentionHostBufferRole, AttentionHostBufferLease] = {}
    for key, lease in leases.items():
        try:
            role = AttentionHostBufferRole(key)
        except ValueError as error:
            raise SchemaError("unknown Attention host buffer lease role") from error
        if role in result:
            raise SchemaError("duplicate Attention host buffer lease role")
        if not isinstance(lease, AttentionHostBufferLease):
            raise TypeError("host buffer leases must be AttentionHostBufferLease")
        result[role] = lease
    if set(result) != set(required):
        raise SchemaError("Attention host buffer lease role set is not canonical")
    return result


def materialize_attention_launch_packet(
    plan: AttentionFrameworkPlan,
    tensors: AttentionRunTensorContract,
    execution_identity: AttentionExecutionIdentity,
    dispatch_receipt: AttentionDispatchReceipt,
    launch_lease: AttentionLaunchLeaseContract,
    stream_binding: AttentionStreamBinding,
    host_buffer_leases: Mapping[AttentionHostBufferRole, AttentionHostBufferLease],
) -> AttentionLaunchPacket:
    """Build a complete Host packet without loading or launching an artifact."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(tensors, AttentionRunTensorContract):
        raise TypeError("tensors must be AttentionRunTensorContract")
    if not isinstance(execution_identity, AttentionExecutionIdentity):
        raise TypeError("execution_identity must be AttentionExecutionIdentity")
    if not isinstance(dispatch_receipt, AttentionDispatchReceipt):
        raise TypeError("dispatch_receipt must be AttentionDispatchReceipt")
    if not isinstance(launch_lease, AttentionLaunchLeaseContract):
        raise TypeError("launch_lease must be AttentionLaunchLeaseContract")
    if not isinstance(stream_binding, AttentionStreamBinding):
        raise TypeError("stream_binding must be AttentionStreamBinding")
    if tensors.out is None:
        raise SchemaError("kernel launch packet requires caller-owned out TensorView")
    if execution_identity.return_lse and tensors.lse is None:
        raise SchemaError("return_lse kernel packet requires caller-owned LSE TensorView")
    if tensors.workspace_float is None or tensors.workspace_int is None:
        raise SchemaError("kernel launch packet requires both workspace TensorViews")
    if (
        execution_identity.plan_fingerprint != plan.fingerprint
        or execution_identity.admission_fingerprint != plan.admission_fingerprint
        or execution_identity.tensor_signature_fingerprint
        != attention_run_tensor_signature(tensors)
        or execution_identity.auxiliary_fingerprint != tensors.auxiliary.fingerprint
        or execution_identity.run_options_fingerprint != tensors.run_options.fingerprint
    ):
        raise SchemaError("execution identity is stale for Attention plan/tensors")
    if (
        dispatch_receipt.plan_fingerprint != plan.fingerprint
        or dispatch_receipt.admission_fingerprint != plan.admission_fingerprint
    ):
        raise SchemaError("dispatch receipt is stale for Attention plan")
    launch_lease.validate_tensor_contract(
        tensors, execution_identity_fingerprint=execution_identity.fingerprint
    )
    if launch_lease.dispatch_receipt_fingerprint != dispatch_receipt.fingerprint:
        raise SchemaError("launch lease is stale for dispatch receipt")
    if (
        stream_binding.device != tensors.stream.device
        or stream_binding.stream_id != tensors.stream.stream_id
    ):
        raise SchemaError("stream binding is stale for tensor contract")

    required = [
        AttentionHostBufferRole.Q_DESCRIPTOR,
        AttentionHostBufferRole.KV_DESCRIPTOR,
        AttentionHostBufferRole.KV_COMPONENTS,
        AttentionHostBufferRole.AUX_DESCRIPTOR,
        AttentionHostBufferRole.AUX_COMPONENTS,
        AttentionHostBufferRole.RUN_OPTIONS,
        AttentionHostBufferRole.OUT_DESCRIPTOR,
        AttentionHostBufferRole.PLAN_METADATA,
    ]
    if tensors.lse is not None:
        required.append(AttentionHostBufferRole.LSE_DESCRIPTOR)
    host_leases = _host_lease_map(host_buffer_leases, required)
    device_bindings = {item.role: item for item in launch_lease.bindings}
    q = materialize_attention_tensor_view(
        device_bindings["q"], AttentionTensorRole.Q, stream_binding.device_index
    )
    out = materialize_attention_tensor_view(
        device_bindings["out"], AttentionTensorRole.OUT, stream_binding.device_index
    )
    lse = (
        materialize_attention_tensor_view(
            device_bindings["lse"], AttentionTensorRole.LSE, stream_binding.device_index
        )
        if tensors.lse is not None
        else None
    )
    kv = materialize_attention_kv_cache_view(
        tensors.kv,
        launch_lease.bindings,
        host_leases[AttentionHostBufferRole.KV_COMPONENTS],
        stream_binding.device_index,
    )
    aux = materialize_attention_auxiliary_view(
        tensors.auxiliary,
        launch_lease.bindings,
        host_leases[AttentionHostBufferRole.AUX_COMPONENTS],
        stream_binding.device_index,
    )
    metadata = materialize_attention_plan_metadata(
        plan,
        dispatch_fingerprint=dispatch_receipt.fingerprint,
        binary_abi_fingerprint=dispatch_receipt.binary_abi_fingerprint,
    )
    contents = {
        AttentionHostBufferRole.Q_DESCRIPTOR: q.pack(),
        AttentionHostBufferRole.KV_DESCRIPTOR: kv.pack(),
        AttentionHostBufferRole.KV_COMPONENTS: kv.component_blob,
        AttentionHostBufferRole.AUX_DESCRIPTOR: aux.pack(),
        AttentionHostBufferRole.AUX_COMPONENTS: aux.component_blob,
        AttentionHostBufferRole.RUN_OPTIONS: tensors.run_options.pack(),
        AttentionHostBufferRole.OUT_DESCRIPTOR: out.pack(),
        AttentionHostBufferRole.PLAN_METADATA: metadata.to_bytes(),
    }
    if lse is not None:
        contents[AttentionHostBufferRole.LSE_DESCRIPTOR] = lse.pack()
    buffers = tuple(
        AttentionHostBufferBinding(role, host_leases[role], contents[role])
        for role in sorted(required, key=lambda item: item.value)
    )
    float_binding = device_bindings["workspace_float"]
    int_binding = device_bindings["workspace_int"]
    float_nbytes = tensors.workspace_float.numel
    int_nbytes = tensors.workspace_int.numel
    arguments = AttentionLaunchArguments(
        q=host_leases[AttentionHostBufferRole.Q_DESCRIPTOR].base_address,
        kv=host_leases[AttentionHostBufferRole.KV_DESCRIPTOR].base_address,
        aux=host_leases[AttentionHostBufferRole.AUX_DESCRIPTOR].base_address,
        run_options=host_leases[AttentionHostBufferRole.RUN_OPTIONS].base_address,
        out=host_leases[AttentionHostBufferRole.OUT_DESCRIPTOR].base_address,
        lse=(
            host_leases[AttentionHostBufferRole.LSE_DESCRIPTOR].base_address
            if lse is not None
            else 0
        ),
        plan_metadata=host_leases[AttentionHostBufferRole.PLAN_METADATA].base_address,
        plan_metadata_nbytes=len(contents[AttentionHostBufferRole.PLAN_METADATA]),
        float_workspace=(float_binding.data_address if float_nbytes else 0),
        float_workspace_nbytes=float_nbytes,
        int_workspace=(int_binding.data_address if int_nbytes else 0),
        int_workspace_nbytes=int_nbytes,
        stream=stream_binding.stream_handle,
    )
    return AttentionLaunchPacket(
        execution_identity,
        dispatch_receipt,
        launch_lease,
        stream_binding,
        buffers,
        arguments,
        graph_enabled=execution_identity.graph_enabled,
    )


__all__ = [
    "ATTENTION_LAUNCH_PACKET_MAX_HOST_BYTES",
    "ATTENTION_LAUNCH_PACKET_VERSION",
    "AttentionHostBufferBinding",
    "AttentionHostBufferRole",
    "AttentionLaunchArguments",
    "AttentionLaunchPacket",
    "AttentionStreamBinding",
    "materialize_attention_launch_packet",
]
