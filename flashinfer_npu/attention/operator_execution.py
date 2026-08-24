"""Guarded execution of an injected, identity-bound Attention callable.

No package resolver or importer lives in this module.  A provider integration
must inject the callable object after separately proving package/signature
compatibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import AttentionOperatorOperationSpec
from .operator_callable import (
    AttentionOperatorCallableBinding,
    AttentionOperatorRuntimeBinding,
    observe_python_callable_signature,
)
from .operator_run import AttentionLoweredOperatorCall


ATTENTION_OPERATOR_EXECUTION_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class AttentionRuntimeBindingAwareExecutor(Protocol):
    """Executor candidate that requires the wrapper's final runtime binding."""

    provider_id: str
    operation_id: str

    def bind_runtime(
        self, runtime_binding: AttentionOperatorRuntimeBinding
    ) -> "AttentionRuntimeBindingAwareExecutor":
        """Return an executor authorized for one active plan generation."""


@dataclass(frozen=True)
class AttentionOperatorExecutionReceipt:
    runtime_binding_fingerprint: str
    active_plan_fingerprint: str
    provider_id: str
    operation_id: str
    return_names: Tuple[str, ...]
    schema_version: int = ATTENTION_OPERATOR_EXECUTION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_EXECUTION_VERSION:
            raise SchemaError("unsupported Attention execution receipt version")
        for name in ("runtime_binding_fingerprint", "active_plan_fingerprint"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError("%s must be lowercase SHA-256" % name)
        if not self.provider_id or not self.operation_id:
            raise SchemaError("execution receipt identities must be non-empty")
        returns = tuple(str(item) for item in self.return_names)
        if not returns or len(set(returns)) != len(returns):
            raise SchemaError("execution receipt return names must be unique")
        object.__setattr__(self, "return_names", returns)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "runtime_binding_fingerprint": self.runtime_binding_fingerprint,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "return_names": list(self.return_names),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionInjectedCallableExecutor:
    """Exact Python invocation for one observed and catalog-bound callable."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        callable_binding: AttentionOperatorCallableBinding,
        callable_object: Any,
        *,
        runtime_binding: AttentionOperatorRuntimeBinding = None,
    ) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(callable_binding, AttentionOperatorCallableBinding):
            raise TypeError("callable_binding must be AttentionOperatorCallableBinding")
        if not callable(callable_object):
            raise TypeError("callable_object must be callable")
        if (
            callable_binding.provider_id != operation.provider_id
            or callable_binding.operation_id != operation.operation_id
            or callable_binding.operation_fingerprint != operation.fingerprint
        ):
            raise SchemaError("injected callable binding does not match operation")
        observed_signature = observe_python_callable_signature(callable_object)
        if observed_signature.fingerprint != callable_binding.signature_fingerprint:
            raise SchemaError("injected callable signature differs from its binding")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation
        self._callable_binding = callable_binding
        self._callable_object = callable_object
        self._runtime_binding = None
        self._last_execution_receipt = None
        if runtime_binding is not None:
            self._validate_runtime_binding(runtime_binding)
            self._runtime_binding = runtime_binding

    @property
    def is_runtime_bound(self) -> bool:
        return self._runtime_binding is not None

    @property
    def last_execution_receipt(self) -> AttentionOperatorExecutionReceipt:
        if self._last_execution_receipt is None:
            raise RuntimeError("injected Attention callable has not executed successfully")
        return self._last_execution_receipt

    def _validate_runtime_binding(
        self, runtime_binding: AttentionOperatorRuntimeBinding
    ) -> None:
        if not isinstance(runtime_binding, AttentionOperatorRuntimeBinding):
            raise TypeError("runtime_binding must be AttentionOperatorRuntimeBinding")
        if (
            runtime_binding.provider_id != self.provider_id
            or runtime_binding.operation_id != self.operation_id
            or runtime_binding.operation_fingerprint != self._operation.fingerprint
            or runtime_binding.callable_binding_fingerprint
            != self._callable_binding.fingerprint
            or runtime_binding.observation_fingerprint
            != self._callable_binding.observation_fingerprint
        ):
            raise SchemaError("runtime binding does not authorize injected callable")

    def bind_runtime(
        self, runtime_binding: AttentionOperatorRuntimeBinding
    ) -> "AttentionInjectedCallableExecutor":
        self._validate_runtime_binding(runtime_binding)
        return AttentionInjectedCallableExecutor(
            self._operation,
            self._callable_binding,
            self._callable_object,
            runtime_binding=runtime_binding,
        )

    def execute(self, call: AttentionLoweredOperatorCall):
        if self._runtime_binding is None:
            raise RuntimeError("injected Attention callable executor is not runtime-bound")
        if not isinstance(call, AttentionLoweredOperatorCall):
            raise TypeError("call must be AttentionLoweredOperatorCall")
        runtime = self._runtime_binding
        if (
            call.provider_id != self.provider_id
            or call.operation_id != self.operation_id
            or call.active_plan_fingerprint != runtime.active_plan_fingerprint
        ):
            raise SchemaError("lowered call is not authorized by executor runtime")
        positional_names = tuple(name for name, _ in call.positional_arguments)
        if positional_names != self._operation.positional_arguments:
            raise SchemaError("executor positional arguments differ from operation")
        keyword_names = tuple(name for name, _ in call.keyword_arguments)
        if not set(keyword_names).issubset(self._operation.keyword_arguments):
            raise SchemaError("executor keyword arguments differ from operation")
        if not set(call.return_names).issubset(self._operation.return_names):
            raise SchemaError("executor return names differ from operation")

        positional_values = tuple(value for _, value in call.positional_arguments)
        keyword_values = dict(call.keyword_arguments)
        raw_result = self._callable_object(*positional_values, **keyword_values)
        if len(call.return_names) == 1:
            public_result = raw_result
        else:
            if not isinstance(raw_result, (tuple, list)):
                raise SchemaError("Attention callable must return multiple values")
            if len(raw_result) != len(call.return_names):
                raise SchemaError("Attention callable return arity differs from lowering")
            public_result = tuple(raw_result)
        self._last_execution_receipt = AttentionOperatorExecutionReceipt(
            runtime_binding_fingerprint=runtime.fingerprint,
            active_plan_fingerprint=call.active_plan_fingerprint,
            provider_id=self.provider_id,
            operation_id=self.operation_id,
            return_names=call.return_names,
        )
        return public_result


__all__ = [
    "ATTENTION_OPERATOR_EXECUTION_VERSION",
    "AttentionInjectedCallableExecutor",
    "AttentionOperatorExecutionReceipt",
    "AttentionRuntimeBindingAwareExecutor",
]
