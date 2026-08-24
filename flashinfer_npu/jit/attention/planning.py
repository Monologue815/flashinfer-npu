"""Bind an exact loaded Attention JIT module to a provider plan factory.

The integration-supplied binder owns the interpretation of the module's
``plan`` entry point.  This framework layer records identity only and never
invokes a resolved symbol or an NPU operator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from flashinfer_npu.attention.operator_plan import AttentionOperatorPlanFactory
from flashinfer_npu.runtime import SchemaError

from .loading import AttentionJitLoadedModuleBinding


ATTENTION_JIT_PLANNER_BINDING_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("Attention JIT planner %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionJitPlannerBinding:
    """Receipt and opaque plan factory produced from one loaded module."""

    jit_module_binding_fingerprint: str
    provider_id: str
    operation_id: str
    binder_id: str
    binder_version: str
    planner_token: str
    factory: Any = field(repr=False, compare=False)
    schema_version: int = ATTENTION_JIT_PLANNER_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_PLANNER_BINDING_VERSION:
            raise SchemaError("unsupported Attention JIT planner binding version")
        _require_hash(
            "jit_module_binding_fingerprint",
            self.jit_module_binding_fingerprint,
        )
        for name in ("provider_id", "binder_id"):
            value = str(getattr(self, name))
            if not _IDENTIFIER.fullmatch(value):
                raise SchemaError("invalid Attention JIT planner %s" % name)
            object.__setattr__(self, name, value)
        for name in ("operation_id", "binder_version", "planner_token"):
            value = str(getattr(self, name))
            if not value or any(item in value for item in ("\x00", "\n", "\r")):
                raise SchemaError(
                    "Attention JIT planner %s must be safe and non-empty" % name
                )
            object.__setattr__(self, name, value)
        if not isinstance(self.factory, AttentionOperatorPlanFactory):
            raise TypeError("factory must implement AttentionOperatorPlanFactory")
        if (
            self.factory.provider_id != self.provider_id
            or self.factory.operation_id != self.operation_id
        ):
            raise SchemaError("Attention JIT planner factory identity differs")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "jit_module_binding_fingerprint": self.jit_module_binding_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "binder_id": self.binder_id,
            "binder_version": self.binder_version,
            "planner_token": self.planner_token,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate(self, module_binding: AttentionJitLoadedModuleBinding) -> None:
        if not isinstance(module_binding, AttentionJitLoadedModuleBinding):
            raise TypeError("module_binding must be AttentionJitLoadedModuleBinding")
        if self.jit_module_binding_fingerprint != module_binding.fingerprint:
            raise SchemaError("Attention JIT planner binding is stale")
        if not isinstance(self.factory, AttentionOperatorPlanFactory):
            raise TypeError("factory must implement AttentionOperatorPlanFactory")
        if (
            self.factory.provider_id != self.provider_id
            or self.factory.operation_id != self.operation_id
        ):
            raise SchemaError("Attention JIT planner factory identity differs")


@runtime_checkable
class AttentionJitPlannerBinder(Protocol):
    """Private adapter from a loaded module's plan symbol to a plan factory."""

    provider_id: str
    operation_id: str
    binder_id: str
    binder_version: str

    def bind(
        self,
        module_binding: AttentionJitLoadedModuleBinding,
        factory: AttentionOperatorPlanFactory,
    ) -> AttentionJitPlannerBinding:
        """Return a plan factory tied to the exact loaded module."""


__all__ = [
    "ATTENTION_JIT_PLANNER_BINDING_VERSION",
    "AttentionJitPlannerBinder",
    "AttentionJitPlannerBinding",
]
