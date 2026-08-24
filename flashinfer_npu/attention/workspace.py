"""Backend-explicit Attention workspace capacity and lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from flashinfer_npu.runtime import SchemaError


ATTENTION_WORKSPACE_SCHEMA_VERSION = 1


class WorkspaceRequirementUnknownError(RuntimeError):
    """Raised when a backend has not supplied a workspace size query yet."""


@dataclass(frozen=True)
class AttentionWorkspaceContract:
    """A bound pair of caller-owned byte buffers and backend requirements.

    Capacity and requirement are intentionally separate.  ``None`` means the
    backend requirement is unknown; it never means zero bytes.
    """

    backend: str
    device: str
    float_capacity_bytes: int
    int_capacity_bytes: int
    required_float_bytes: Optional[int] = None
    required_int_bytes: Optional[int] = None
    binding_generation: int = 1
    plan_generation: Optional[int] = None
    graph_enabled: bool = False
    schema_version: int = ATTENTION_WORKSPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_WORKSPACE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention workspace schema version")
        if not self.backend or not self.device:
            raise SchemaError("workspace backend/device must be non-empty")
        for name in (
            "float_capacity_bytes",
            "int_capacity_bytes",
            "binding_generation",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaError("%s cannot be negative" % name)
        if self.binding_generation < 1:
            raise SchemaError("binding_generation must be positive")
        if self.plan_generation is not None and self.plan_generation < 1:
            raise SchemaError("plan_generation must be positive")
        for name in ("required_float_bytes", "required_int_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise SchemaError("%s cannot be negative" % name)
        if (self.required_float_bytes is None) != (
            self.required_int_bytes is None
        ):
            raise SchemaError("float/int workspace requirements become known together")
        self.validate_capacity()

    @classmethod
    def for_host_reference(
        cls,
        *,
        device: str,
        float_capacity_bytes: int,
        int_capacity_bytes: int,
        graph_enabled: bool = False,
    ) -> "AttentionWorkspaceContract":
        return cls(
            backend="reference",
            device=device,
            float_capacity_bytes=float_capacity_bytes,
            int_capacity_bytes=int_capacity_bytes,
            required_float_bytes=0,
            required_int_bytes=0,
            graph_enabled=graph_enabled,
        )

    @property
    def requirements_known(self) -> bool:
        return self.required_float_bytes is not None

    @property
    def required_sizes(self) -> Tuple[int, int]:
        if not self.requirements_known:
            raise WorkspaceRequirementUnknownError(
                "workspace sizes are unknown for backend %r" % self.backend
            )
        return int(self.required_float_bytes), int(self.required_int_bytes)

    def validate_capacity(self) -> None:
        if not self.requirements_known:
            return
        required_float, required_int = self.required_sizes
        if self.float_capacity_bytes < required_float:
            raise SchemaError(
                "float workspace capacity is %d bytes, requires %d"
                % (self.float_capacity_bytes, required_float)
            )
        if self.int_capacity_bytes < required_int:
            raise SchemaError(
                "int workspace capacity is %d bytes, requires %d"
                % (self.int_capacity_bytes, required_int)
            )

    def rebind(
        self,
        *,
        device: str,
        float_capacity_bytes: int,
        int_capacity_bytes: int,
        allow_device_change: bool,
    ) -> "AttentionWorkspaceContract":
        if device != self.device and not allow_device_change:
            raise SchemaError(
                "workspace device cannot change after plan/graph resources are bound"
            )
        return replace(
            self,
            device=device,
            float_capacity_bytes=int(float_capacity_bytes),
            int_capacity_bytes=int(int_capacity_bytes),
            binding_generation=self.binding_generation + 1,
        )

    def bind_plan(self, generation: int) -> "AttentionWorkspaceContract":
        if generation < 1:
            raise SchemaError("plan generation must be positive")
        return replace(self, plan_generation=int(generation))

    def validate_run(self, *, device: str, plan_generation: int) -> None:
        self.validate_capacity()
        if device != self.device:
            raise SchemaError("q and workspace must be on the same device")
        if self.plan_generation != plan_generation:
            raise SchemaError("workspace binding is not associated with the active plan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "device": self.device,
            "float_capacity_bytes": self.float_capacity_bytes,
            "int_capacity_bytes": self.int_capacity_bytes,
            "required_float_bytes": self.required_float_bytes,
            "required_int_bytes": self.required_int_bytes,
            "binding_generation": self.binding_generation,
            "plan_generation": self.plan_generation,
            "graph_enabled": self.graph_enabled,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ATTENTION_WORKSPACE_SCHEMA_VERSION",
    "AttentionWorkspaceContract",
    "WorkspaceRequirementUnknownError",
]
