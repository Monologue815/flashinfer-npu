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


ATTENTION_PLAN_SELECTION_VERSION = 5


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
    provider_integration_bundle_id: Optional[str] = None
    provider_integration_bundle_fingerprint: Optional[str] = None
    plan_scoring_manifest_id: Optional[str] = None
    plan_scoring_manifest_fingerprint: Optional[str] = None
    plan_scoring_policy_id: Optional[str] = None
    plan_scoring_policy_fingerprint: Optional[str] = None
    plan_score: Optional[int] = None
    plan_score_source: Optional[str] = None
    plan_score_reason: Optional[str] = None
    runtime_resolution_fingerprint: Optional[str] = None
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
            self.provider_integration_bundle_id,
            self.provider_integration_bundle_fingerprint,
            self.plan_scoring_manifest_id,
            self.plan_scoring_manifest_fingerprint,
            self.plan_scoring_policy_id,
            self.plan_scoring_policy_fingerprint,
            self.plan_score,
            self.plan_score_source,
            self.plan_score_reason,
            self.runtime_resolution_fingerprint,
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
        bundle_fields = (
            self.provider_integration_bundle_id,
            self.provider_integration_bundle_fingerprint,
        )
        if any(item is not None for item in bundle_fields):
            if any(item is None for item in bundle_fields):
                raise SchemaError(
                    "provider integration bundle identity is incomplete"
                )
            if self.runtime_declaration_fingerprint is None:
                raise SchemaError(
                    "provider integration bundle identity requires a runtime "
                    "declaration"
                )
            bundle_id = str(self.provider_integration_bundle_id)
            if not bundle_id.strip() or any(
                item.isspace() for item in bundle_id
            ):
                raise SchemaError(
                    "provider integration bundle id is invalid"
                )
            _require_hash(
                "provider_integration_bundle_fingerprint",
                self.provider_integration_bundle_fingerprint,
            )
        manifest_fields = (
            self.plan_scoring_manifest_id,
            self.plan_scoring_manifest_fingerprint,
            self.plan_scoring_policy_id,
            self.plan_scoring_policy_fingerprint,
        )
        if any(item is not None for item in manifest_fields):
            if any(item is None for item in manifest_fields):
                raise SchemaError(
                    "provider plan scoring manifest identity is incomplete"
                )
            if self.runtime_declaration_fingerprint is None:
                raise SchemaError(
                    "scoring manifest identity requires a runtime declaration"
                )
            manifest_id = str(self.plan_scoring_manifest_id)
            policy_id = str(self.plan_scoring_policy_id)
            if (
                not manifest_id.strip()
                or any(item.isspace() for item in manifest_id)
                or not policy_id.strip()
                or any(item.isspace() for item in policy_id)
            ):
                raise SchemaError("provider plan scoring manifest ids are invalid")
            _require_hash(
                "plan_scoring_manifest_fingerprint",
                self.plan_scoring_manifest_fingerprint,
            )
            _require_hash(
                "plan_scoring_policy_fingerprint",
                self.plan_scoring_policy_fingerprint,
            )
        scoring_fields = (
            self.plan_score,
            self.plan_score_source,
            self.plan_score_reason,
            self.runtime_resolution_fingerprint,
        )
        if any(item is not None for item in scoring_fields):
            if any(item is None for item in scoring_fields):
                raise SchemaError(
                    "provider plan selection scoring evidence is incomplete"
                )
            if not isinstance(self.plan_score, int) or isinstance(
                self.plan_score, bool
            ):
                raise SchemaError("provider plan score must be an integer")
            if not -(2**31) <= self.plan_score <= 2**31 - 1:
                raise SchemaError("provider plan score is out of range")
            if not str(self.plan_score_source).strip() or not str(
                self.plan_score_reason
            ).strip():
                raise SchemaError("provider plan score source/reason is empty")
            _require_hash(
                "runtime_resolution_fingerprint",
                self.runtime_resolution_fingerprint,
            )
        if any(item is not None for item in manifest_fields) and not any(
            item is not None for item in scoring_fields
        ):
            raise SchemaError(
                "scoring manifest identity requires plan scoring evidence"
            )
        if any(item is not None for item in bundle_fields) and not all(
            item is not None for item in manifest_fields
        ):
            raise SchemaError(
                "provider integration bundle identity requires scoring "
                "manifest identity"
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
            "provider_integration_bundle_id": (
                self.provider_integration_bundle_id
            ),
            "provider_integration_bundle_fingerprint": (
                self.provider_integration_bundle_fingerprint
            ),
            "plan_scoring_manifest_id": self.plan_scoring_manifest_id,
            "plan_scoring_manifest_fingerprint": (
                self.plan_scoring_manifest_fingerprint
            ),
            "plan_scoring_policy_id": self.plan_scoring_policy_id,
            "plan_scoring_policy_fingerprint": (
                self.plan_scoring_policy_fingerprint
            ),
            "plan_score": self.plan_score,
            "plan_score_source": self.plan_score_source,
            "plan_score_reason": self.plan_score_reason,
            "runtime_resolution_fingerprint": (
                self.runtime_resolution_fingerprint
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
    provider_integration_bundle_id: Optional[str] = None,
    provider_integration_bundle_fingerprint: Optional[str] = None,
    plan_scoring_manifest_id: Optional[str] = None,
    plan_scoring_manifest_fingerprint: Optional[str] = None,
    plan_scoring_policy_id: Optional[str] = None,
    plan_scoring_policy_fingerprint: Optional[str] = None,
    plan_score: Optional[int] = None,
    plan_score_source: Optional[str] = None,
    plan_score_reason: Optional[str] = None,
    runtime_resolution_fingerprint: Optional[str] = None,
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
    if (
        runtime_resolution_fingerprint
        != active_plan.runtime_resolution_fingerprint
    ):
        raise SchemaError(
            "provider selection runtime resolution differs from active plan"
        )
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
        provider_integration_bundle_id=provider_integration_bundle_id,
        provider_integration_bundle_fingerprint=(
            provider_integration_bundle_fingerprint
        ),
        plan_scoring_manifest_id=plan_scoring_manifest_id,
        plan_scoring_manifest_fingerprint=(
            plan_scoring_manifest_fingerprint
        ),
        plan_scoring_policy_id=plan_scoring_policy_id,
        plan_scoring_policy_fingerprint=plan_scoring_policy_fingerprint,
        plan_score=plan_score,
        plan_score_source=plan_score_source,
        plan_score_reason=plan_score_reason,
        runtime_resolution_fingerprint=runtime_resolution_fingerprint,
    )


__all__ = [
    "ATTENTION_PLAN_SELECTION_VERSION",
    "AttentionPlanSelection",
    "build_provider_plan_selection",
    "build_reference_plan_selection",
]
