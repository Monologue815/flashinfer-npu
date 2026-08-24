"""Provider-supplied callable observations bound to documented operations.

The framework never imports an operator package here.  A provider adapter may
submit an observation produced by its own resolver; tests use injected Python
callables so no torch_npu, CANN, or device runtime is involved.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import (
    AttentionOperatorOperationBinding,
    AttentionOperatorOperationSpec,
)
from .operator_plan import AttentionOperatorActivePlan
from .operator_provider import AttentionOperatorProviderProbe


ATTENTION_OPERATOR_CALLABLE_VERSION = 1

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AttentionOperatorCallableBindingError(RuntimeError):
    """Raised when an observed callable does not match its catalog operation."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _names(name: str, values) -> Tuple[str, ...]:
    try:
        result = tuple(str(item) for item in values)
    except TypeError as error:
        raise SchemaError("%s must be a sequence" % name) from error
    if any(not _ARGUMENT_NAME.fullmatch(item) for item in result):
        raise SchemaError("%s contains an invalid argument name" % name)
    if len(set(result)) != len(result):
        raise SchemaError("%s must contain unique names" % name)
    return result


@dataclass(frozen=True)
class AttentionObservedCallableSignature:
    """Normalized inspectable signature independent of a live callable object."""

    positional_arguments: Tuple[str, ...]
    keyword_arguments: Tuple[str, ...]
    has_var_positional: bool = False
    has_var_keyword: bool = False
    observation_kind: str = "adapter"
    schema_version: int = ATTENTION_OPERATOR_CALLABLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_CALLABLE_VERSION:
            raise SchemaError("unsupported observed callable signature version")
        positional = _names("positional_arguments", self.positional_arguments)
        keyword = _names("keyword_arguments", self.keyword_arguments)
        if set(positional).intersection(keyword):
            raise SchemaError("observed positional and keyword arguments overlap")
        if not str(self.observation_kind):
            raise SchemaError("signature observation_kind must be non-empty")
        object.__setattr__(self, "positional_arguments", positional)
        object.__setattr__(self, "keyword_arguments", keyword)
        object.__setattr__(self, "has_var_positional", bool(self.has_var_positional))
        object.__setattr__(self, "has_var_keyword", bool(self.has_var_keyword))
        object.__setattr__(self, "observation_kind", str(self.observation_kind))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "positional_arguments": list(self.positional_arguments),
            "keyword_arguments": list(self.keyword_arguments),
            "has_var_positional": self.has_var_positional,
            "has_var_keyword": self.has_var_keyword,
            "observation_kind": self.observation_kind,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionObservedCallableSignature":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("observed callable signature fields are invalid")
        try:
            data["positional_arguments"] = tuple(data["positional_arguments"])
            data["keyword_arguments"] = tuple(data["keyword_arguments"])
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("observed callable signature fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def observe_python_callable_signature(
    value: Any,
) -> AttentionObservedCallableSignature:
    """Normalize an injected Python callable without retaining or invoking it."""

    if not callable(value):
        raise TypeError("value must be callable")
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        raise SchemaError("Python callable signature is not inspectable") from error
    positional = []
    keyword = []
    has_var_positional = False
    has_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        elif (
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
        ):
            positional.append(parameter.name)
        else:
            keyword.append(parameter.name)
    return AttentionObservedCallableSignature(
        positional_arguments=tuple(positional),
        keyword_arguments=tuple(keyword),
        has_var_positional=has_var_positional,
        has_var_keyword=has_var_keyword,
        observation_kind="python_inspect",
    )


@dataclass(frozen=True)
class AttentionObservedOperatorCallable:
    """Serializable observation made by one provider adapter resolver."""

    provider_id: str
    package_name: str
    package_version: str
    callable_path: str
    api_version: str
    available: bool
    signature: Optional[AttentionObservedCallableSignature] = None
    unavailable_reasons: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_OPERATOR_CALLABLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_CALLABLE_VERSION:
            raise SchemaError("unsupported observed operator callable version")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid observed callable provider_id")
        for name in ("package_name", "package_version", "callable_path", "api_version"):
            if not str(getattr(self, name)):
                raise SchemaError("observed callable %s must be non-empty" % name)
        reasons = tuple(str(item) for item in self.unavailable_reasons)
        if any(not item for item in reasons) or len(set(reasons)) != len(reasons):
            raise SchemaError("observed callable rejection reasons must be unique")
        if self.available:
            if reasons or not isinstance(
                self.signature, AttentionObservedCallableSignature
            ):
                raise SchemaError(
                    "available callable requires a signature and no rejection reasons"
                )
        elif self.signature is not None or not reasons:
            raise SchemaError(
                "unavailable callable requires reasons and no observed signature"
            )
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "package_name", str(self.package_name))
        object.__setattr__(self, "package_version", str(self.package_version))
        object.__setattr__(self, "callable_path", str(self.callable_path))
        object.__setattr__(self, "api_version", str(self.api_version))
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "unavailable_reasons", reasons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "callable_path": self.callable_path,
            "api_version": self.api_version,
            "available": self.available,
            "signature": None if self.signature is None else self.signature.to_dict(),
            "unavailable_reasons": list(self.unavailable_reasons),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@runtime_checkable
class AttentionOperatorCallableInspector(Protocol):
    """Provider-owned resolver; no default real-package implementation exists yet."""

    provider_id: str

    def inspect(
        self, operation: AttentionOperatorOperationSpec
    ) -> AttentionObservedOperatorCallable:
        """Return metadata and signature without executing the callable."""


@dataclass(frozen=True)
class AttentionOperatorCallableReport:
    """Explain exact provider/package/API/signature compatibility."""

    provider_probe_fingerprint: str
    operation_fingerprint: str
    observation: AttentionObservedOperatorCallable
    accepted: bool
    reasons: Tuple[str, ...]
    schema_version: int = ATTENTION_OPERATOR_CALLABLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_CALLABLE_VERSION:
            raise SchemaError("unsupported callable compatibility report version")
        _require_hash("provider_probe_fingerprint", self.provider_probe_fingerprint)
        _require_hash("operation_fingerprint", self.operation_fingerprint)
        if not isinstance(self.observation, AttentionObservedOperatorCallable):
            raise TypeError("observation must be AttentionObservedOperatorCallable")
        reasons = tuple(str(item) for item in self.reasons)
        if any(not item for item in reasons) or len(set(reasons)) != len(reasons):
            raise SchemaError("callable compatibility reasons must be unique")
        if self.accepted == bool(reasons):
            raise SchemaError("accepted callable report must have no reasons")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class AttentionOperatorCallableBinding:
    """Stable binding from a provider probe and operation to an observation."""

    provider_id: str
    provider_probe_fingerprint: str
    operation_id: str
    operation_fingerprint: str
    observation_fingerprint: str
    signature_fingerprint: str
    package_name: str
    package_version: str
    callable_path: str
    api_version: str
    schema_version: int = ATTENTION_OPERATOR_CALLABLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_CALLABLE_VERSION:
            raise SchemaError("unsupported callable binding version")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid callable binding provider_id")
        for name in (
            "provider_probe_fingerprint",
            "operation_fingerprint",
            "observation_fingerprint",
            "signature_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        for name in (
            "operation_id",
            "package_name",
            "package_version",
            "callable_path",
            "api_version",
        ):
            if not str(getattr(self, name)):
                raise SchemaError("callable binding %s must be non-empty" % name)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                name: (
                    getattr(self, name)
                    if name != "schema_version"
                    else self.schema_version
                )
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class AttentionOperatorRuntimeBinding:
    """Closed active-plan -> operation -> callable authority chain."""

    active_plan_fingerprint: str
    provider_id: str
    provider_probe_fingerprint: str
    operation_binding_fingerprint: str
    callable_binding_fingerprint: str
    operation_id: str
    operation_fingerprint: str
    observation_fingerprint: str
    schema_version: int = ATTENTION_OPERATOR_CALLABLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_CALLABLE_VERSION:
            raise SchemaError("unsupported operator runtime binding version")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid operator runtime binding provider_id")
        for name in (
            "active_plan_fingerprint",
            "provider_probe_fingerprint",
            "operation_binding_fingerprint",
            "callable_binding_fingerprint",
            "operation_fingerprint",
            "observation_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if not str(self.operation_id):
            raise SchemaError("operator runtime operation_id must be non-empty")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            }
        )


def explain_attention_operator_callable(
    provider_probe: AttentionOperatorProviderProbe,
    operation: AttentionOperatorOperationSpec,
    observation: AttentionObservedOperatorCallable,
) -> AttentionOperatorCallableReport:
    """Pure comparison; does not import, resolve, or invoke any package object."""

    if not isinstance(provider_probe, AttentionOperatorProviderProbe):
        raise TypeError("provider_probe must be AttentionOperatorProviderProbe")
    if not isinstance(operation, AttentionOperatorOperationSpec):
        raise TypeError("operation must be AttentionOperatorOperationSpec")
    if not isinstance(observation, AttentionObservedOperatorCallable):
        raise TypeError("observation must be AttentionObservedOperatorCallable")
    reasons = []
    if provider_probe.provider_id != operation.provider_id:
        reasons.append("provider probe does not match catalog operation")
    if observation.provider_id != operation.provider_id:
        reasons.append("observed callable does not match catalog provider")
    if not provider_probe.available:
        reasons.extend(
            "provider unavailable: %s" % item
            for item in provider_probe.unavailable_reasons
        )
    package_versions = dict(provider_probe.package_versions)
    if operation.package_name not in package_versions:
        reasons.append(
            "provider probe does not report package %s" % operation.package_name
        )
    elif package_versions[operation.package_name] != observation.package_version:
        reasons.append("observed package version differs from provider probe")
    if observation.package_name != operation.package_name:
        reasons.append("observed package name does not match catalog operation")
    if observation.callable_path != operation.callable_path:
        reasons.append("observed callable path does not match catalog operation")
    if observation.api_version != operation.api_version:
        reasons.append("observed API version does not match catalog operation")
    if not observation.available:
        reasons.extend(
            "callable unavailable: %s" % item
            for item in observation.unavailable_reasons
        )
    else:
        signature = observation.signature
        if signature is None:  # guarded by observation schema; defensive
            reasons.append("available callable has no signature")
        else:
            if signature.positional_arguments != operation.positional_arguments:
                reasons.append("observed positional signature differs from catalog")
            if signature.keyword_arguments != operation.keyword_arguments:
                reasons.append("observed keyword signature differs from catalog")
            if signature.has_var_positional or signature.has_var_keyword:
                reasons.append("variadic callable cannot prove an exact signature")
    return AttentionOperatorCallableReport(
        provider_probe_fingerprint=provider_probe.fingerprint,
        operation_fingerprint=operation.fingerprint,
        observation=observation,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def bind_attention_operator_callable(
    provider_probe: AttentionOperatorProviderProbe,
    operation: AttentionOperatorOperationSpec,
    observation: AttentionObservedOperatorCallable,
) -> AttentionOperatorCallableBinding:
    report = explain_attention_operator_callable(
        provider_probe, operation, observation
    )
    if not report.accepted:
        raise AttentionOperatorCallableBindingError(
            "Attention operator callable is incompatible: %s"
            % "; ".join(report.reasons)
        )
    signature = observation.signature
    if signature is None:  # accepted reports always have a signature
        raise AssertionError("accepted callable observation has no signature")
    return AttentionOperatorCallableBinding(
        provider_id=operation.provider_id,
        provider_probe_fingerprint=provider_probe.fingerprint,
        operation_id=operation.operation_id,
        operation_fingerprint=operation.fingerprint,
        observation_fingerprint=observation.fingerprint,
        signature_fingerprint=signature.fingerprint,
        package_name=observation.package_name,
        package_version=observation.package_version,
        callable_path=observation.callable_path,
        api_version=observation.api_version,
    )


def bind_attention_operator_runtime(
    active_plan: AttentionOperatorActivePlan,
    operation_binding: AttentionOperatorOperationBinding,
    callable_binding: AttentionOperatorCallableBinding,
) -> AttentionOperatorRuntimeBinding:
    """Close identities before a wrapper publishes a prepared runtime."""

    if not isinstance(active_plan, AttentionOperatorActivePlan):
        raise TypeError("active_plan must be AttentionOperatorActivePlan")
    if not isinstance(operation_binding, AttentionOperatorOperationBinding):
        raise TypeError("operation_binding must be AttentionOperatorOperationBinding")
    if not isinstance(callable_binding, AttentionOperatorCallableBinding):
        raise TypeError("callable_binding must be AttentionOperatorCallableBinding")
    selection = active_plan.provider_selection
    if (
        operation_binding.active_plan_fingerprint != active_plan.fingerprint
        or operation_binding.provider_id != selection.provider_id
        or operation_binding.operation_id
        != active_plan.prepared_plan.implementation_id
    ):
        raise SchemaError("operation binding does not bind the active plan")
    if (
        callable_binding.provider_id != selection.provider_id
        or callable_binding.provider_probe_fingerprint
        != selection.provider_probe_fingerprint
        or callable_binding.operation_id != operation_binding.operation_id
        or callable_binding.operation_fingerprint
        != operation_binding.operation_fingerprint
    ):
        raise SchemaError("callable binding does not bind the active operation")
    return AttentionOperatorRuntimeBinding(
        active_plan_fingerprint=active_plan.fingerprint,
        provider_id=selection.provider_id,
        provider_probe_fingerprint=selection.provider_probe_fingerprint,
        operation_binding_fingerprint=operation_binding.fingerprint,
        callable_binding_fingerprint=callable_binding.fingerprint,
        operation_id=operation_binding.operation_id,
        operation_fingerprint=operation_binding.operation_fingerprint,
        observation_fingerprint=callable_binding.observation_fingerprint,
    )


__all__ = [
    "ATTENTION_OPERATOR_CALLABLE_VERSION",
    "AttentionObservedCallableSignature",
    "AttentionObservedOperatorCallable",
    "AttentionOperatorCallableBinding",
    "AttentionOperatorCallableBindingError",
    "AttentionOperatorCallableInspector",
    "AttentionOperatorCallableReport",
    "AttentionOperatorRuntimeBinding",
    "bind_attention_operator_runtime",
    "bind_attention_operator_callable",
    "explain_attention_operator_callable",
    "observe_python_callable_signature",
]
