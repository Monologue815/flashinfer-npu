"""Execution and graph-capture identity contracts for Attention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import KernelDescriptor, SchemaError

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionRuntimeEnvironment,
)
from .dispatch import AttentionDispatchReceipt

from .numerics import (
    DEFAULT_ATTENTION_NUMERICS_POLICY,
    AttentionNumericsPolicy,
)
from .planner import AttentionFrameworkPlan
from .schema import AttentionMode
from .tensor_contract import (
    AttentionRunTensorContract,
    AttentionTensorAccessPolicy,
    StreamContext,
    TensorView,
)
from .workspace import AttentionWorkspaceContract


ATTENTION_EXECUTION_IDENTITY_VERSION = 3
_HASH_FIELDS = (
    "plan_fingerprint",
    "admission_fingerprint",
    "numerics_policy_fingerprint",
    "workspace_fingerprint",
    "graph_resources_fingerprint",
    "tensor_signature_fingerprint",
    "stream_context_fingerprint",
    "auxiliary_fingerprint",
    "run_options_fingerprint",
    "access_policy_fingerprint",
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionPersistentBufferSpec:
    name: str
    dtype: str
    capacity: int
    device: str

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.device:
            raise SchemaError("persistent buffer name/dtype/device must be non-empty")
        if self.capacity < 0:
            raise SchemaError("persistent buffer capacity cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "capacity": self.capacity,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionPersistentBufferSpec":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionPersistentBufferSpec fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionPersistentBufferSpec fields are invalid") from error


@dataclass(frozen=True)
class AttentionGraphResourceContract:
    mode: AttentionMode
    graph_enabled: bool
    fixed_batch_size: Optional[int]
    persistent_buffers: Tuple[AttentionPersistentBufferSpec, ...] = ()
    schema_version: int = ATTENTION_EXECUTION_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_EXECUTION_IDENTITY_VERSION:
            raise SchemaError("unsupported Attention graph resource version")
        object.__setattr__(self, "mode", AttentionMode(self.mode))
        if not isinstance(self.graph_enabled, bool):
            raise SchemaError("graph_enabled must be boolean")
        buffers = tuple(self.persistent_buffers)
        if len({item.name for item in buffers}) != len(buffers):
            raise SchemaError("persistent buffer names must be unique")
        object.__setattr__(self, "persistent_buffers", buffers)
        if self.graph_enabled:
            if self.fixed_batch_size is None or self.fixed_batch_size <= 0:
                raise SchemaError("graph resources require a positive fixed batch size")
            if not buffers:
                raise SchemaError("graph resources require persistent metadata buffers")
            if len({item.device for item in buffers}) != 1:
                raise SchemaError("persistent graph buffers must share one device")
        elif self.fixed_batch_size is not None or buffers:
            raise SchemaError("non-graph resources cannot freeze buffers or batch size")

    @classmethod
    def disabled(cls, mode: AttentionMode) -> "AttentionGraphResourceContract":
        return cls(mode, False, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "graph_enabled": self.graph_enabled,
            "fixed_batch_size": self.fixed_batch_size,
            "persistent_buffers": [item.to_dict() for item in self.persistent_buffers],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionGraphResourceContract":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionGraphResourceContract fields are invalid")
        buffers = data.get("persistent_buffers")
        if not isinstance(buffers, (list, tuple)):
            raise SchemaError("persistent_buffers must be an array")
        data["persistent_buffers"] = tuple(
            AttentionPersistentBufferSpec.from_dict(item) for item in buffers
        )
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionGraphResourceContract fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _view_signature(value: Optional[TensorView]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "strides": list(value.strides),
        "dtype": value.dtype,
        "device": value.device,
        "writable": value.writable,
    }


def attention_run_tensor_signature(contract: AttentionRunTensorContract) -> str:
    """Hash structural tensor ABI while excluding addresses/storage identity."""

    layout_descriptor = (
        contract.kv.key.physical_layout_descriptor
        if contract.kv.quantized
        else None
    )
    payload = {
        "q": _view_signature(contract.q),
        "kv_spec": contract.kv.spec.to_dict(),
        "kv_packed": contract.kv.packed,
        "kv_quantized": contract.kv.quantized,
        "kv_physical_layout_descriptor_fingerprint": (
            layout_descriptor.fingerprint if layout_descriptor is not None else None
        ),
        "kv_components": [
            _view_signature(item) for item in contract.kv.component_views
        ],
        "out": _view_signature(contract.out),
        "lse": _view_signature(contract.lse),
        "workspace_float": _view_signature(contract.workspace_float),
        "workspace_int": _view_signature(contract.workspace_int),
        "auxiliary": [
            {
                "role": item.role.name,
                "view": _view_signature(item.view),
            }
            for item in contract.auxiliary.components
        ],
        "run_options": contract.run_options.to_dict(),
        "stream": {
            "device": contract.stream.device,
            "stream_id": contract.stream.stream_id,
            "ordered": contract.stream.ordered,
        },
    }
    return _canonical_hash(payload)


def attention_stream_context_fingerprint(stream: StreamContext) -> str:
    """Hash the independently revalidatable execution-stream claim."""

    if not isinstance(stream, StreamContext):
        raise TypeError("stream must be StreamContext")
    return _canonical_hash(
        {
            "device": stream.device,
            "stream_id": stream.stream_id,
            "ordered": stream.ordered,
        }
    )


def _validate_workspace_tensor_binding(
    workspace: AttentionWorkspaceContract,
    tensors: AttentionRunTensorContract,
) -> None:
    views = (
        ("float", tensors.workspace_float, workspace.float_capacity_bytes),
        ("int", tensors.workspace_int, workspace.int_capacity_bytes),
    )
    for name, view, capacity in views:
        if view is None:
            raise SchemaError("execution identity requires both workspace tensor views")
        if (
            view.device != workspace.device
            or view.dtype != "uint8"
            or len(view.shape) != 1
            or not view.is_contiguous
            or view.numel != capacity
        ):
            raise SchemaError(
                "%s workspace tensor does not match its capacity/device contract"
                % name
            )


def _validate_execution_resources(
    plan: AttentionFrameworkPlan,
    workspace: AttentionWorkspaceContract,
    graph_resources: AttentionGraphResourceContract,
    tensors: AttentionRunTensorContract,
    access_policy: AttentionTensorAccessPolicy,
) -> None:
    if workspace.plan_generation != plan.generation:
        raise SchemaError("workspace is not bound to the identity plan generation")
    if workspace.graph_enabled != graph_resources.graph_enabled:
        raise SchemaError("workspace and graph resource modes do not match")
    if graph_resources.mode != plan.spec.mode:
        raise SchemaError("graph resources and plan mode do not match")
    if graph_resources.graph_enabled and graph_resources.fixed_batch_size != plan.batch_size:
        raise SchemaError("graph fixed batch size does not match the plan")
    if graph_resources.graph_enabled and any(
        item.device != workspace.device
        for item in graph_resources.persistent_buffers
    ):
        raise SchemaError("persistent graph buffers must share the workspace device")
    _validate_workspace_tensor_binding(workspace, tensors)
    tensors.validate(access_policy, plan=plan)


@dataclass(frozen=True)
class AttentionExecutionIdentity:
    plan_fingerprint: str
    admission_fingerprint: str
    numerics_policy_fingerprint: str
    workspace_fingerprint: str
    graph_resources_fingerprint: str
    tensor_signature_fingerprint: str
    stream_context_fingerprint: str
    auxiliary_fingerprint: str
    run_options_fingerprint: str
    access_policy_fingerprint: str
    backend: str
    graph_enabled: bool
    fixed_batch_size: Optional[int]
    q_len_per_req: int
    return_lse: bool
    capability_profile_id: Optional[str] = None
    capability_profile_fingerprint: Optional[str] = None
    capability_rule_id: Optional[str] = None
    capability_evidence_id: Optional[str] = None
    kernel_id: Optional[str] = None
    kernel_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_EXECUTION_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_EXECUTION_IDENTITY_VERSION:
            raise SchemaError("unsupported Attention execution identity version")
        for name in _HASH_FIELDS:
            _require_hash(name, getattr(self, name))
        if not self.backend:
            raise SchemaError("execution identity backend must be non-empty")
        if not isinstance(self.graph_enabled, bool):
            raise SchemaError("execution identity graph_enabled must be boolean")
        if self.graph_enabled:
            if self.fixed_batch_size is None or self.fixed_batch_size <= 0:
                raise SchemaError("graph execution requires fixed_batch_size")
        elif self.fixed_batch_size is not None:
            raise SchemaError("non-graph execution cannot freeze batch size")
        if self.q_len_per_req < 1:
            raise SchemaError("q_len_per_req must be positive")
        if not isinstance(self.return_lse, bool):
            raise SchemaError("return_lse must be boolean")
        binding = (
            self.capability_profile_id,
            self.capability_profile_fingerprint,
            self.capability_rule_id,
            self.capability_evidence_id,
            self.kernel_id,
            self.kernel_fingerprint,
        )
        if any(item is not None for item in binding) and not all(
            item is not None and item != "" for item in binding
        ):
            raise SchemaError("kernel execution binding fields become present together")
        if all(item is None for item in binding):
            if self.backend != "reference":
                raise SchemaError("non-reference identity requires kernel binding fields")
        else:
            _require_hash(
                "capability_profile_fingerprint",
                str(self.capability_profile_fingerprint),
            )
            _require_hash("kernel_fingerprint", str(self.kernel_fingerprint))
            if self.backend == "reference":
                raise SchemaError("reference identity cannot claim a kernel binding")

    @property
    def binding_kind(self) -> str:
        return "reference_contract" if self.kernel_id is None else "kernel"

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionExecutionIdentity":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionExecutionIdentity fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionExecutionIdentity fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def build_reference_execution_identity(
    plan: AttentionFrameworkPlan,
    workspace: AttentionWorkspaceContract,
    graph_resources: AttentionGraphResourceContract,
    tensors: AttentionRunTensorContract,
    *,
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
    access_policy: AttentionTensorAccessPolicy = AttentionTensorAccessPolicy(),
    return_lse: bool = False,
) -> AttentionExecutionIdentity:
    if workspace.backend != "reference":
        raise SchemaError("reference execution identity requires reference workspace")
    if not isinstance(numerics_policy, AttentionNumericsPolicy):
        raise TypeError("numerics_policy must be AttentionNumericsPolicy")
    if not isinstance(access_policy, AttentionTensorAccessPolicy):
        raise TypeError("access_policy must be AttentionTensorAccessPolicy")
    _validate_execution_resources(
        plan, workspace, graph_resources, tensors, access_policy
    )
    return AttentionExecutionIdentity(
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        numerics_policy_fingerprint=numerics_policy.fingerprint,
        workspace_fingerprint=workspace.fingerprint,
        graph_resources_fingerprint=graph_resources.fingerprint,
        tensor_signature_fingerprint=attention_run_tensor_signature(tensors),
        stream_context_fingerprint=attention_stream_context_fingerprint(
            tensors.stream
        ),
        auxiliary_fingerprint=tensors.auxiliary.fingerprint,
        run_options_fingerprint=tensors.run_options.fingerprint,
        access_policy_fingerprint=access_policy.fingerprint,
        backend="reference",
        graph_enabled=graph_resources.graph_enabled,
        fixed_batch_size=graph_resources.fixed_batch_size,
        q_len_per_req=plan.spec.q_len_per_req,
        return_lse=bool(return_lse),
    )


def build_kernel_execution_identity(
    plan: AttentionFrameworkPlan,
    workspace: AttentionWorkspaceContract,
    graph_resources: AttentionGraphResourceContract,
    tensors: AttentionRunTensorContract,
    dispatch: AttentionDispatchReceipt,
    profile: AttentionBackendCapabilityProfile,
    descriptor: KernelDescriptor,
    observed_environment: AttentionRuntimeEnvironment,
    *,
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
    access_policy: AttentionTensorAccessPolicy = AttentionTensorAccessPolicy(),
    return_lse: bool = False,
) -> AttentionExecutionIdentity:
    """Build a non-reference identity from a revalidated dispatch receipt.

    This is a framework contract only.  It does not load an artifact, launch a
    kernel, or claim that a device graph has been created.
    """

    if not isinstance(dispatch, AttentionDispatchReceipt):
        raise TypeError("dispatch must be AttentionDispatchReceipt")
    if not isinstance(numerics_policy, AttentionNumericsPolicy):
        raise TypeError("numerics_policy must be AttentionNumericsPolicy")
    if not isinstance(access_policy, AttentionTensorAccessPolicy):
        raise TypeError("access_policy must be AttentionTensorAccessPolicy")
    dispatch.validate(
        plan,
        profile,
        descriptor,
        observed_environment,
        numerics_policy=numerics_policy,
    )
    if workspace.backend != dispatch.backend.value:
        raise SchemaError("workspace backend does not match dispatch backend")
    if not workspace.requirements_known:
        raise SchemaError("kernel execution identity requires known workspace sizes")
    if workspace.required_sizes != dispatch.workspace_bytes:
        raise SchemaError("workspace requirement does not match dispatch formula")
    _validate_execution_resources(
        plan, workspace, graph_resources, tensors, access_policy
    )
    for view, alignment in zip(
        (tensors.workspace_float, tensors.workspace_int),
        dispatch.workspace_alignments,
    ):
        assert view is not None
        view.require_alignment(alignment, "kernel workspace")
    return AttentionExecutionIdentity(
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        numerics_policy_fingerprint=numerics_policy.fingerprint,
        workspace_fingerprint=workspace.fingerprint,
        graph_resources_fingerprint=graph_resources.fingerprint,
        tensor_signature_fingerprint=attention_run_tensor_signature(tensors),
        stream_context_fingerprint=attention_stream_context_fingerprint(
            tensors.stream
        ),
        auxiliary_fingerprint=tensors.auxiliary.fingerprint,
        run_options_fingerprint=tensors.run_options.fingerprint,
        access_policy_fingerprint=access_policy.fingerprint,
        backend=dispatch.backend.value,
        graph_enabled=graph_resources.graph_enabled,
        fixed_batch_size=graph_resources.fixed_batch_size,
        q_len_per_req=plan.spec.q_len_per_req,
        return_lse=bool(return_lse),
        capability_profile_id=dispatch.profile_id,
        capability_profile_fingerprint=dispatch.profile_fingerprint,
        capability_rule_id=dispatch.rule_id,
        capability_evidence_id=dispatch.evidence_id,
        kernel_id=dispatch.kernel_id,
        kernel_fingerprint=dispatch.kernel_fingerprint,
    )


class AttentionCaptureCompatibilityError(RuntimeError):
    """Raised when an execution attempts to reuse an incompatible capture."""


@dataclass(frozen=True)
class AttentionCapturedExecution:
    identity: AttentionExecutionIdentity
    capture_kind: str
    capture_generation: int
    schema_version: int = ATTENTION_EXECUTION_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_EXECUTION_IDENTITY_VERSION:
            raise SchemaError("unsupported Attention captured execution version")
        if self.capture_kind not in {"host_contract", "device_graph"}:
            raise SchemaError("capture_kind must be host_contract or device_graph")
        if self.capture_generation < 1:
            raise SchemaError("capture_generation must be positive")
        if not self.identity.graph_enabled:
            raise SchemaError("captured execution requires graph-enabled identity")
        if self.capture_kind == "host_contract" and self.identity.binding_kind != "reference_contract":
            raise SchemaError("host_contract capture requires reference identity")
        if self.capture_kind == "device_graph" and self.identity.binding_kind != "kernel":
            raise SchemaError("device_graph capture requires kernel identity")

    def validate_reuse(self, candidate: AttentionExecutionIdentity) -> None:
        mismatches = tuple(
            name
            for name in self.identity.__dataclass_fields__
            if getattr(self.identity, name) != getattr(candidate, name)
        )
        if mismatches:
            raise AttentionCaptureCompatibilityError(
                "Attention capture identity mismatch: %s" % ", ".join(mismatches)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "capture_kind": self.capture_kind,
            "capture_generation": self.capture_generation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionCapturedExecution":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionCapturedExecution fields are invalid")
        if not isinstance(data.get("identity"), Mapping):
            raise SchemaError("captured execution identity must be an object")
        data["identity"] = AttentionExecutionIdentity.from_dict(data["identity"])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionCapturedExecution fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "ATTENTION_EXECUTION_IDENTITY_VERSION",
    "AttentionCaptureCompatibilityError",
    "AttentionCapturedExecution",
    "AttentionExecutionIdentity",
    "AttentionGraphResourceContract",
    "AttentionPersistentBufferSpec",
    "attention_run_tensor_signature",
    "attention_stream_context_fingerprint",
    "build_kernel_execution_identity",
    "build_reference_execution_identity",
]
