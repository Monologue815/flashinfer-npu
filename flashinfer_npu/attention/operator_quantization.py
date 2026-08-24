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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .capability import AttentionBackendCapabilityProfile
from .operation_catalog import AttentionOperatorOperationSpec
from .operator_integration import AttentionOperatorPlanGate
from .planner import AttentionFrameworkPlan


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


__all__ = [
    "ATTENTION_OPERATOR_QUANTIZATION_VERSION",
    "AttentionOperatorQuantArgumentBinding",
    "AttentionOperatorQuantizationBinding",
    "AttentionOperatorQuantizationPlanGate",
    "validate_attention_operator_quantization_bindings",
]
