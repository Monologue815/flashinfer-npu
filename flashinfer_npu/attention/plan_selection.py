"""Read-only description of the implementation selected by Attention plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Optional

from flashinfer_npu.runtime import Backend, SchemaError

from .operator_plan import AttentionOperatorActivePlan
from .planner import AttentionFrameworkPlan
from .schema import AttentionMode


ATTENTION_PLAN_SELECTION_VERSION = 2


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: Optional[str]) -> None:
    if value is None or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionPlanSelection:
    """Stable diagnostics for a wrapper-owned plan, never an execution handle."""

    mode: AttentionMode
    route: str
    backend: str
    plan_generation: int
    framework_plan_fingerprint: str
    provider_id: Optional[str] = None
    operation_id: Optional[str] = None
    active_plan_fingerprint: Optional[str] = None
    registry_generation: Optional[int] = None
    runtime_declaration_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_PLAN_SELECTION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PLAN_SELECTION_VERSION:
            raise SchemaError("unsupported Attention plan selection version")
        if not isinstance(self.mode, AttentionMode):
            raise TypeError("mode must be AttentionMode")
        if self.route not in {"reference", "provider"}:
            raise SchemaError("invalid Attention plan selection route")
        if not isinstance(self.plan_generation, int) or isinstance(
            self.plan_generation, bool
        ) or self.plan_generation <= 0:
            raise SchemaError("Attention plan selection generation must be positive")
        _require_hash(
            "framework_plan_fingerprint", self.framework_plan_fingerprint
        )
        provider_fields = (
            self.provider_id,
            self.operation_id,
            self.active_plan_fingerprint,
            self.registry_generation,
            self.runtime_declaration_fingerprint,
        )
        if self.route == "reference":
            if self.backend != "reference" or any(
                item is not None for item in provider_fields
            ):
                raise SchemaError(
                    "reference plan selection cannot contain provider identity"
                )
            return
        if self.backend not in {item.value for item in Backend if item.value != "reference"}:
            raise SchemaError("provider plan selection backend is invalid")
        if not self.provider_id or not self.operation_id:
            raise SchemaError("provider plan selection identity is incomplete")
        _require_hash("active_plan_fingerprint", self.active_plan_fingerprint)
        if not isinstance(self.registry_generation, int) or isinstance(
            self.registry_generation, bool
        ) or self.registry_generation < 0:
            raise SchemaError(
                "provider plan selection registry generation must be non-negative"
            )
        if self.runtime_declaration_fingerprint is not None:
            _require_hash(
                "runtime_declaration_fingerprint",
                self.runtime_declaration_fingerprint,
            )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "route": self.route,
            "backend": self.backend,
            "plan_generation": self.plan_generation,
            "framework_plan_fingerprint": self.framework_plan_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "registry_generation": self.registry_generation,
            "runtime_declaration_fingerprint": (
                self.runtime_declaration_fingerprint
            ),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def build_reference_plan_selection(
    plan: AttentionFrameworkPlan,
) -> AttentionPlanSelection:
    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    return AttentionPlanSelection(
        mode=plan.spec.mode,
        route="reference",
        backend="reference",
        plan_generation=plan.generation,
        framework_plan_fingerprint=plan.fingerprint,
    )


def build_provider_plan_selection(
    plan: AttentionFrameworkPlan,
    active_plan: AttentionOperatorActivePlan,
    *,
    registry_generation: int,
    runtime_declaration_fingerprint: Optional[str] = None,
) -> AttentionPlanSelection:
    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(active_plan, AttentionOperatorActivePlan):
        raise TypeError("active_plan must be AttentionOperatorActivePlan")
    if (
        active_plan.framework_plan.fingerprint != plan.fingerprint
        or active_plan.framework_plan.generation != plan.generation
    ):
        raise SchemaError("provider selection does not bind the wrapper plan")
    return AttentionPlanSelection(
        mode=plan.spec.mode,
        route="provider",
        backend=active_plan.provider_selection.backend.value,
        plan_generation=plan.generation,
        framework_plan_fingerprint=plan.fingerprint,
        provider_id=active_plan.provider_selection.provider_id,
        operation_id=active_plan.prepared_plan.implementation_id,
        active_plan_fingerprint=active_plan.fingerprint,
        registry_generation=registry_generation,
        runtime_declaration_fingerprint=runtime_declaration_fingerprint,
    )


__all__ = [
    "ATTENTION_PLAN_SELECTION_VERSION",
    "AttentionPlanSelection",
    "build_provider_plan_selection",
    "build_reference_plan_selection",
]
