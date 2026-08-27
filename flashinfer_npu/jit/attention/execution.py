"""Bind an exact loaded Attention JIT module to a runtime executor.

The binder is supplied by an authorized integration.  This module records and
validates identity only; it never calls a resolved symbol or an NPU operator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from flashinfer_npu.attention.operator_callable import (
    AttentionOperatorCallableBinding,
    AttentionOperatorRuntimeBinding,
)
from flashinfer_npu.runtime import SchemaError

from .loading import AttentionJitLoadedModuleBinding


ATTENTION_JIT_EXECUTOR_BINDING_VERSION = 1
ATTENTION_JIT_RUNTIME_EXECUTOR_BINDING_VERSION = 1
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
        raise SchemaError("Attention JIT executor %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionJitExecutorBinding:
    """Receipt and opaque executor produced from one loaded module."""

    jit_module_binding_fingerprint: str
    callable_binding_fingerprint: str
    provider_id: str
    operation_id: str
    binder_id: str
    binder_version: str
    executor_token: str
    executor: Any = field(repr=False, compare=False)
    schema_version: int = ATTENTION_JIT_EXECUTOR_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_EXECUTOR_BINDING_VERSION:
            raise SchemaError("unsupported Attention JIT executor binding version")
        _require_hash(
            "jit_module_binding_fingerprint",
            self.jit_module_binding_fingerprint,
        )
        _require_hash(
            "callable_binding_fingerprint",
            self.callable_binding_fingerprint,
        )
        for name in ("provider_id", "binder_id"):
            value = str(getattr(self, name))
            if not _IDENTIFIER.fullmatch(value):
                raise SchemaError("invalid Attention JIT executor %s" % name)
            object.__setattr__(self, name, value)
        for name in ("operation_id", "binder_version", "executor_token"):
            value = str(getattr(self, name))
            if not value or any(item in value for item in ("\x00", "\n", "\r")):
                raise SchemaError(
                    "Attention JIT executor %s must be safe and non-empty" % name
                )
            object.__setattr__(self, name, value)
        if self.executor is None:
            raise SchemaError("Attention JIT bound executor must be present")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "jit_module_binding_fingerprint": self.jit_module_binding_fingerprint,
            "callable_binding_fingerprint": self.callable_binding_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "binder_id": self.binder_id,
            "binder_version": self.binder_version,
            "executor_token": self.executor_token,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate(
        self,
        module_binding: AttentionJitLoadedModuleBinding,
        callable_binding: AttentionOperatorCallableBinding,
    ) -> None:
        if not isinstance(module_binding, AttentionJitLoadedModuleBinding):
            raise TypeError("module_binding must be AttentionJitLoadedModuleBinding")
        if not isinstance(callable_binding, AttentionOperatorCallableBinding):
            raise TypeError("callable_binding must be AttentionOperatorCallableBinding")
        if (
            self.jit_module_binding_fingerprint != module_binding.fingerprint
            or self.callable_binding_fingerprint != callable_binding.fingerprint
            or self.provider_id != callable_binding.provider_id
            or self.operation_id != callable_binding.operation_id
        ):
            raise SchemaError("Attention JIT executor binding is stale")


@dataclass(frozen=True)
class AttentionJitRuntimeExecutorBinding:
    """Bind the final runtime-authorized executor to one JIT executor receipt."""

    active_plan_fingerprint: str
    jit_executor_binding_fingerprint: str
    runtime_binding_fingerprint: str
    provider_id: str
    operation_id: str
    executor: Any = field(repr=False, compare=False)
    schema_version: int = ATTENTION_JIT_RUNTIME_EXECUTOR_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_RUNTIME_EXECUTOR_BINDING_VERSION:
            raise SchemaError("unsupported Attention JIT runtime executor version")
        for name in (
            "active_plan_fingerprint",
            "jit_executor_binding_fingerprint",
            "runtime_binding_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        for name in ("provider_id", "operation_id"):
            value = str(getattr(self, name))
            if not value or any(item in value for item in ("\x00", "\n", "\r")):
                raise SchemaError(
                    "Attention JIT runtime executor %s must be safe and non-empty"
                    % name
                )
            object.__setattr__(self, name, value)
        if self.executor is None:
            raise SchemaError("Attention JIT runtime executor must be present")
        if (
            getattr(self.executor, "provider_id", None) != self.provider_id
            or getattr(self.executor, "operation_id", None) != self.operation_id
        ):
            raise SchemaError("Attention JIT runtime executor identity differs")

    @classmethod
    def bind(
        cls,
        jit_binding: AttentionJitExecutorBinding,
        runtime_binding: AttentionOperatorRuntimeBinding,
        executor: Any,
    ) -> "AttentionJitRuntimeExecutorBinding":
        if not isinstance(jit_binding, AttentionJitExecutorBinding):
            raise TypeError("jit_binding must be AttentionJitExecutorBinding")
        if not isinstance(runtime_binding, AttentionOperatorRuntimeBinding):
            raise TypeError("runtime_binding must be AttentionOperatorRuntimeBinding")
        value = cls(
            active_plan_fingerprint=runtime_binding.active_plan_fingerprint,
            jit_executor_binding_fingerprint=jit_binding.fingerprint,
            runtime_binding_fingerprint=runtime_binding.fingerprint,
            provider_id=runtime_binding.provider_id,
            operation_id=runtime_binding.operation_id,
            executor=executor,
        )
        value.validate(jit_binding, runtime_binding, executor)
        return value

    def validate(
        self,
        jit_binding: AttentionJitExecutorBinding,
        runtime_binding: AttentionOperatorRuntimeBinding,
        executor: Any,
    ) -> None:
        if not isinstance(jit_binding, AttentionJitExecutorBinding):
            raise TypeError("jit_binding must be AttentionJitExecutorBinding")
        if not isinstance(runtime_binding, AttentionOperatorRuntimeBinding):
            raise TypeError("runtime_binding must be AttentionOperatorRuntimeBinding")
        if (
            self.active_plan_fingerprint != runtime_binding.active_plan_fingerprint
            or self.jit_executor_binding_fingerprint != jit_binding.fingerprint
            or self.runtime_binding_fingerprint != runtime_binding.fingerprint
            or self.provider_id != jit_binding.provider_id
            or self.provider_id != runtime_binding.provider_id
            or self.operation_id != jit_binding.operation_id
            or self.operation_id != runtime_binding.operation_id
            or jit_binding.callable_binding_fingerprint
            != runtime_binding.callable_binding_fingerprint
            or self.executor is not executor
        ):
            raise SchemaError("Attention JIT runtime executor binding is stale")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "jit_executor_binding_fingerprint": self.jit_executor_binding_fingerprint,
            "runtime_binding_fingerprint": self.runtime_binding_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@runtime_checkable
class AttentionJitExecutorBinder(Protocol):
    """Private adapter from a loaded module and package API to an executor."""

    provider_id: str
    operation_id: str
    binder_id: str
    binder_version: str

    def bind(
        self,
        module_binding: AttentionJitLoadedModuleBinding,
        callable_binding: AttentionOperatorCallableBinding,
        executor: Any,
    ) -> AttentionJitExecutorBinding:
        """Return an executor tied to the exact loaded module and callable."""


__all__ = [
    "ATTENTION_JIT_EXECUTOR_BINDING_VERSION",
    "ATTENTION_JIT_RUNTIME_EXECUTOR_BINDING_VERSION",
    "AttentionJitExecutorBinder",
    "AttentionJitExecutorBinding",
    "AttentionJitRuntimeExecutorBinding",
]
