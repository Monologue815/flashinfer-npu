"""Audit the packaged model-facing Attention interface against its parity contract."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from flashinfer_npu.parity import load_packaged_manifest
from flashinfer_npu.runtime import SchemaError


ATTENTION_PUBLIC_INTERFACE_AUDIT_VERSION = 1

_INTERNAL_CONTROL_PARAMETER_NAMES = frozenset(
    {
        "adapter",
        "artifact",
        "callable",
        "executor",
        "jit_handle",
        "jit_module",
        "kernel_id",
        "module",
        "operation",
        "plan",
        "plan_handle",
        "provider",
        "provider_handle",
        "registration",
        "registry",
        "runtime",
        "runtime_declaration",
        "runtime_handle",
    }
)
_INTERNAL_CONTROL_PARAMETER_PREFIXES = (
    "adapter_",
    "artifact_",
    "callable_",
    "executor_",
    "jit_handle_",
    "jit_module_",
    "kernel_",
    "module_",
    "operation_",
    "provider_",
    "registration_",
    "registry_",
    "runtime_",
)
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


class AttentionPublicInterfaceAuditError(SchemaError):
    """The live Python interface violates the packaged Attention contract."""


@dataclass(frozen=True)
class AttentionParameterInterface:
    name: str
    kind: str
    required: bool
    has_default: bool
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "has_default": self.has_default,
        }
        if self.has_default:
            value["default"] = self.default
        return value


@dataclass(frozen=True)
class AttentionCallableInterface:
    local: str
    role: str
    parameters: Tuple[AttentionParameterInterface, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local": self.local,
            "role": self.role,
            "parameters": [item.to_dict() for item in self.parameters],
        }


@dataclass(frozen=True)
class AttentionPublicInterfaceAuditReport:
    upstream_ref: str
    model_facing_dispatch: str
    run_accepts_plan_handle: bool
    caller_selects_provider: bool
    model_facing: Tuple[AttentionCallableInterface, ...]
    advanced_injected_module: Tuple[AttentionCallableInterface, ...]
    schema_version: int = ATTENTION_PUBLIC_INTERFACE_AUDIT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_public_interface_audit",
            "upstream_ref": self.upstream_ref,
            "model_facing_dispatch": self.model_facing_dispatch,
            "run_accepts_plan_handle": self.run_accepts_plan_handle,
            "caller_selects_provider": self.caller_selects_provider,
            "model_facing": [item.to_dict() for item in self.model_facing],
            "advanced_injected_module": [
                item.to_dict() for item in self.advanced_injected_module
            ],
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def format(self) -> str:
        return "\n".join(
            (
                "Attention public interface audit",
                "  upstream-ref: %s" % self.upstream_ref,
                "  model-facing-dispatch: %s" % self.model_facing_dispatch,
                "  run-accepts-plan-handle: %s"
                % ("yes" if self.run_accepts_plan_handle else "no"),
                "  caller-selects-provider: %s"
                % ("yes" if self.caller_selects_provider else "no"),
                "  model-facing-callables: %d" % len(self.model_facing),
                "  advanced-injected-module-callables: %d"
                % len(self.advanced_injected_module),
                "  fingerprint: %s" % self.fingerprint,
            )
        )


def _resolve_public_symbol(local: str) -> Any:
    module_name, attribute = local.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (AttributeError, ImportError) as error:
        raise AttentionPublicInterfaceAuditError(
            "cannot resolve Attention public symbol %s" % local
        ) from error


def _parameter_interface(
    parameter: inspect.Parameter,
) -> AttentionParameterInterface:
    variadic = parameter.kind in {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    has_default = parameter.default is not inspect.Parameter.empty
    if has_default and not isinstance(parameter.default, _JSON_SCALAR_TYPES):
        raise AttentionPublicInterfaceAuditError(
            "Attention public parameter %s has a non-canonical default"
            % parameter.name
        )
    return AttentionParameterInterface(
        name=parameter.name,
        kind=parameter.kind.name.lower(),
        required=not has_default and not variadic,
        has_default=has_default,
        default=parameter.default if has_default else None,
    )


def _is_internal_control_parameter(name: str) -> bool:
    return name in _INTERNAL_CONTROL_PARAMETER_NAMES or name.startswith(
        _INTERNAL_CONTROL_PARAMETER_PREFIXES
    )


def _audit_callable(
    local: str,
    role: str,
    value: Any,
    *,
    model_facing: bool,
) -> AttentionCallableInterface:
    if not callable(value):
        raise AttentionPublicInterfaceAuditError(
            "Attention public symbol %s is not callable" % local
        )
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        raise AttentionPublicInterfaceAuditError(
            "Attention public symbol %s has no inspectable signature" % local
        ) from error
    parameters = tuple(
        _parameter_interface(parameter)
        for parameter in signature.parameters.values()
    )
    if model_facing:
        leaked = tuple(
            item.name
            for item in parameters
            if item.name != "self" and _is_internal_control_parameter(item.name)
        )
        if leaked:
            raise AttentionPublicInterfaceAuditError(
                "model-facing Attention callable %s leaks internal control parameters: %s"
                % (local, ", ".join(leaked))
            )
    else:
        if not parameters or parameters[0].name != "jit_module":
            raise AttentionPublicInterfaceAuditError(
                "advanced injected-module callable %s must take jit_module first"
                % local
            )
    return AttentionCallableInterface(local, role, parameters)


def audit_attention_public_interface() -> AttentionPublicInterfaceAuditReport:
    """Resolve and audit every callable bound by the packaged Attention contract."""

    manifest = load_packaged_manifest("attention")
    contract = manifest.attention_interface_contract
    if contract is None:
        raise AttentionPublicInterfaceAuditError(
            "packaged Attention parity has no interface contract"
        )

    model_facing = []
    for surface in manifest.attention_surfaces:
        value = _resolve_public_symbol(surface.local)
        if surface.public_lifecycle == "one_shot":
            if isinstance(value, type):
                raise AttentionPublicInterfaceAuditError(
                    "one-shot Attention surface cannot be a class"
                )
            model_facing.append(
                _audit_callable(
                    surface.local, "one_shot", value, model_facing=True
                )
            )
            continue
        if not isinstance(value, type):
            raise AttentionPublicInterfaceAuditError(
                "plan/run Attention surface must be a class"
            )
        model_facing.append(
            _audit_callable(
                surface.local, "constructor", value, model_facing=True
            )
        )
        for role in ("plan", "run"):
            method = getattr(value, role, None)
            model_facing.append(
                _audit_callable(
                    "%s.%s" % (surface.local, role),
                    role,
                    method,
                    model_facing=True,
                )
            )

    advanced = tuple(
        _audit_callable(
            local,
            "advanced_injected_module",
            _resolve_public_symbol(local),
            model_facing=False,
        )
        for local in contract.advanced_injected_module_symbols
    )
    records = tuple(model_facing) + advanced
    paths = tuple(item.local for item in records)
    if len(paths) != len(set(paths)):
        raise AttentionPublicInterfaceAuditError(
            "Attention public interface audit contains duplicate callables"
        )
    return AttentionPublicInterfaceAuditReport(
        upstream_ref=manifest.upstream_ref,
        model_facing_dispatch=contract.model_facing_dispatch,
        run_accepts_plan_handle=contract.run_accepts_plan_handle,
        caller_selects_provider=contract.caller_selects_provider,
        model_facing=tuple(model_facing),
        advanced_injected_module=advanced,
    )


__all__ = [
    "ATTENTION_PUBLIC_INTERFACE_AUDIT_VERSION",
    "AttentionCallableInterface",
    "AttentionParameterInterface",
    "AttentionPublicInterfaceAuditError",
    "AttentionPublicInterfaceAuditReport",
    "audit_attention_public_interface",
]
