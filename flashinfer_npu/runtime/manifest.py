"""Versioned kernel-manifest loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

from .schema import (
    ArtifactRef,
    Backend,
    KernelConstraints,
    KernelCapabilityBinding,
    KernelDescriptor,
    KernelBinaryABI,
    KernelLaunchABI,
    SchemaError,
    WorkspaceFormula,
)


KERNEL_MANIFEST_SCHEMA_VERSION = 3


def _tuple_signatures(values: Iterable[Iterable[str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(str(item) for item in signature) for signature in values)


def descriptor_from_dict(value: Mapping[str, Any]) -> KernelDescriptor:
    allowed_fields = {
        "kernel_id",
        "op",
        "backend",
        "constraints",
        "workspace",
        "int_workspace",
        "artifact",
        "launch_abi",
        "binary_abi",
        "priority",
        "op_schema_version",
        "tiling_schema_version",
        "capability_binding",
    }
    if not {"kernel_id", "op", "backend"} <= set(value):
        raise SchemaError("kernel descriptor is missing required fields")
    if not set(value) <= allowed_fields:
        raise SchemaError("kernel descriptor contains unknown fields")
    constraints_data = dict(value.get("constraints", {}))
    constraints = KernelConstraints(
        supported_socs=tuple(constraints_data.get("supported_socs", ())),
        dtype_signatures=_tuple_signatures(
            constraints_data.get("dtype_signatures", ())
        ),
        layout_signatures=_tuple_signatures(
            constraints_data.get("layout_signatures", ())
        ),
        required_features=tuple(constraints_data.get("required_features", ())),
        quant_storage_dtypes=tuple(
            constraints_data.get("quant_storage_dtypes", ())
        ),
        deterministic=constraints_data.get("deterministic"),
    )
    workspace_data = dict(value.get("workspace", {}))
    workspace = WorkspaceFormula(
        constant_bytes=int(workspace_data.get("constant_bytes", 0)),
        dynamic_coefficients=tuple(
            int(item) for item in workspace_data.get("dynamic_coefficients", ())
        ),
        alignment=int(workspace_data.get("alignment", 32)),
    )
    int_workspace_data = dict(value.get("int_workspace", {}))
    int_workspace = WorkspaceFormula(
        constant_bytes=int(int_workspace_data.get("constant_bytes", 0)),
        dynamic_coefficients=tuple(
            int(item)
            for item in int_workspace_data.get("dynamic_coefficients", ())
        ),
        alignment=int(int_workspace_data.get("alignment", 32)),
    )
    binding_value = value.get("capability_binding")
    if binding_value is not None and not isinstance(binding_value, Mapping):
        raise SchemaError("capability_binding must be an object or null")
    artifact_value = value.get("artifact")
    if artifact_value is not None and not isinstance(artifact_value, Mapping):
        raise SchemaError("artifact must be an object or null")
    launch_abi_value = value.get("launch_abi")
    if launch_abi_value is not None and not isinstance(launch_abi_value, Mapping):
        raise SchemaError("launch_abi must be an object or null")
    binary_abi_value = value.get("binary_abi")
    if binary_abi_value is not None and not isinstance(binary_abi_value, Mapping):
        raise SchemaError("binary_abi must be an object or null")
    return KernelDescriptor(
        kernel_id=str(value["kernel_id"]),
        op=str(value["op"]),
        backend=Backend(value["backend"]),
        constraints=constraints,
        workspace=workspace,
        int_workspace=int_workspace,
        artifact=(
            ArtifactRef.from_dict(artifact_value)
            if artifact_value is not None
            else None
        ),
        launch_abi=(
            KernelLaunchABI.from_dict(launch_abi_value)
            if launch_abi_value is not None
            else None
        ),
        binary_abi=(
            KernelBinaryABI.from_dict(binary_abi_value)
            if binary_abi_value is not None
            else None
        ),
        priority=int(value.get("priority", 0)),
        op_schema_version=int(value.get("op_schema_version", 1)),
        tiling_schema_version=int(value.get("tiling_schema_version", 1)),
        capability_binding=(
            KernelCapabilityBinding.from_dict(binding_value)
            if binding_value is not None
            else None
        ),
    )


def load_kernel_manifest(
    path: Union[str, Path]
) -> Tuple[KernelDescriptor, ...]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value: Dict[str, Any] = json.load(handle)
    if set(value) != {"schema_version", "generated_at", "kernels"}:
        raise SchemaError("kernel manifest fields do not match schema")
    version = value.get("schema_version")
    if version != KERNEL_MANIFEST_SCHEMA_VERSION:
        raise SchemaError("unsupported kernel manifest schema_version %r" % version)
    if not isinstance(value.get("generated_at"), str) or not value["generated_at"]:
        raise SchemaError("kernel manifest generated_at must be non-empty")
    if not isinstance(value.get("kernels"), list):
        raise SchemaError("kernel manifest kernels must be an array")
    descriptors = tuple(descriptor_from_dict(item) for item in value["kernels"])
    kernel_ids = tuple(descriptor.kernel_id for descriptor in descriptors)
    if len(kernel_ids) != len(set(kernel_ids)):
        raise SchemaError("kernel manifest contains duplicate kernel_id values")
    return descriptors
