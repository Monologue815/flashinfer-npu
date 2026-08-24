"""Wrapper-owned active plans prepared by an Attention operator provider."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from flashinfer_npu.runtime import Backend, SchemaError

from .dispatch import AttentionDispatchReceipt
from .operator_provider import AttentionOperatorProviderSelection
from .planner import AttentionFrameworkPlan, AttentionStateError


ATTENTION_OPERATOR_PLAN_VERSION = 3


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionPreparedOperatorPlan:
    """Provider-owned opaque state with a framework-verifiable identity."""

    provider_id: str
    provider_selection_fingerprint: str
    framework_plan_fingerprint: str
    framework_plan_generation: int
    implementation_id: str
    opaque_plan_token: str
    opaque_state: Any = field(repr=False, compare=False)
    schema_version: int = ATTENTION_OPERATOR_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PLAN_VERSION:
            raise SchemaError("unsupported Attention prepared operator plan version")
        if not self.provider_id:
            raise SchemaError("prepared operator provider_id must be non-empty")
        _require_hash(
            "provider_selection_fingerprint", self.provider_selection_fingerprint
        )
        _require_hash("framework_plan_fingerprint", self.framework_plan_fingerprint)
        if (
            not isinstance(self.framework_plan_generation, int)
            or isinstance(self.framework_plan_generation, bool)
            or self.framework_plan_generation < 1
        ):
            raise SchemaError("prepared operator plan generation must be positive")
        if not self.implementation_id or not self.opaque_plan_token:
            raise SchemaError(
                "prepared operator implementation_id and opaque_plan_token must be non-empty"
            )

    @property
    def fingerprint(self) -> str:
        # Opaque state is deliberately excluded.  The provider is responsible
        # for making opaque_plan_token identify its prepared state generation.
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "provider_id": self.provider_id,
                "provider_selection_fingerprint": self.provider_selection_fingerprint,
                "framework_plan_fingerprint": self.framework_plan_fingerprint,
                "framework_plan_generation": self.framework_plan_generation,
                "implementation_id": self.implementation_id,
                "opaque_plan_token": self.opaque_plan_token,
            }
        )


@runtime_checkable
class AttentionOperatorPlanFactory(Protocol):
    """Execution-side extension implemented by a discovered provider adapter."""

    provider_id: str
    operation_id: str

    def prepare(
        self,
        plan: AttentionFrameworkPlan,
        receipt: AttentionDispatchReceipt,
        selection: AttentionOperatorProviderSelection,
    ) -> AttentionPreparedOperatorPlan:
        """Prepare provider state without publishing it to the wrapper."""


@dataclass(frozen=True)
class AttentionOperatorActivePlan:
    """Complete immutable state consumed by a future wrapper ``run`` call."""

    framework_plan: AttentionFrameworkPlan
    dispatch_receipt: AttentionDispatchReceipt
    provider_selection: AttentionOperatorProviderSelection
    prepared_plan: AttentionPreparedOperatorPlan
    jit_plan_binding_fingerprint: Optional[str] = None
    jit_artifact_binding_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_OPERATOR_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PLAN_VERSION:
            raise SchemaError("unsupported Attention active operator plan version")
        if not isinstance(self.framework_plan, AttentionFrameworkPlan):
            raise TypeError("framework_plan must be AttentionFrameworkPlan")
        if not isinstance(self.dispatch_receipt, AttentionDispatchReceipt):
            raise TypeError("dispatch_receipt must be AttentionDispatchReceipt")
        if not isinstance(
            self.provider_selection, AttentionOperatorProviderSelection
        ):
            raise TypeError(
                "provider_selection must be AttentionOperatorProviderSelection"
            )
        if not isinstance(self.prepared_plan, AttentionPreparedOperatorPlan):
            raise TypeError("prepared_plan must be AttentionPreparedOperatorPlan")
        plan = self.framework_plan
        receipt = self.dispatch_receipt
        selection = self.provider_selection
        prepared = self.prepared_plan
        if (
            receipt.mode != plan.spec.mode
            or receipt.plan_fingerprint != plan.fingerprint
            or receipt.admission_fingerprint != plan.admission_fingerprint
            or receipt.workload_fingerprint != plan.workload.fingerprint
        ):
            raise SchemaError("dispatch receipt does not bind the framework plan")
        if (
            selection.dispatch_receipt_fingerprint != receipt.fingerprint
            or selection.profile_id != receipt.profile_id
            or selection.profile_fingerprint != receipt.profile_fingerprint
            or selection.backend != receipt.backend
        ):
            raise SchemaError("provider selection does not bind the dispatch receipt")
        if (
            prepared.provider_id != selection.provider_id
            or prepared.provider_selection_fingerprint != selection.fingerprint
            or prepared.framework_plan_fingerprint != plan.fingerprint
            or prepared.framework_plan_generation != plan.generation
        ):
            raise SchemaError("prepared operator plan identity is stale")
        if receipt.backend == Backend.ASCENDC_JIT:
            values = (
                self.jit_plan_binding_fingerprint,
                self.jit_artifact_binding_fingerprint,
            )
            if any(
                value is None
                or len(value) != 64
                or any(item not in "0123456789abcdef" for item in value)
                for value in values
            ):
                raise SchemaError(
                    "ascendc_jit active plan requires JIT plan and artifact fingerprints"
                )
        elif (
            self.jit_plan_binding_fingerprint is not None
            or self.jit_artifact_binding_fingerprint is not None
        ):
            raise SchemaError(
                "non-JIT active plan cannot contain JIT binding fingerprints"
            )

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "framework_plan_fingerprint": self.framework_plan.fingerprint,
                "framework_plan_generation": self.framework_plan.generation,
                "dispatch_receipt_fingerprint": self.dispatch_receipt.fingerprint,
                "provider_selection_fingerprint": self.provider_selection.fingerprint,
                "prepared_plan_fingerprint": self.prepared_plan.fingerprint,
                "jit_plan_binding_fingerprint": self.jit_plan_binding_fingerprint,
                "jit_artifact_binding_fingerprint": (
                    self.jit_artifact_binding_fingerprint
                ),
            }
        )


class AttentionOperatorPlanSession:
    """Atomically owns the active provider plan used by a wrapper instance."""

    def __init__(self) -> None:
        self._active_plan = None

    @property
    def is_planned(self) -> bool:
        return self._active_plan is not None

    @property
    def active_plan(self) -> AttentionOperatorActivePlan:
        if self._active_plan is None:
            raise AttentionStateError("Attention operator wrapper has not been planned")
        return self._active_plan

    def plan(
        self,
        factory: AttentionOperatorPlanFactory,
        framework_plan: AttentionFrameworkPlan,
        receipt: AttentionDispatchReceipt,
        selection: AttentionOperatorProviderSelection,
        jit_plan_binding_fingerprint: Optional[str] = None,
        jit_artifact_binding_fingerprint: Optional[str] = None,
    ) -> None:
        """Prepare then publish; a failed re-plan preserves the previous state."""

        if not isinstance(factory, AttentionOperatorPlanFactory):
            raise TypeError("factory must implement AttentionOperatorPlanFactory")
        if factory.provider_id != selection.provider_id:
            raise SchemaError("operator plan factory does not match selected provider")
        prepared = factory.prepare(framework_plan, receipt, selection)
        if prepared.implementation_id != factory.operation_id:
            raise SchemaError("operator plan factory changed its declared operation")
        candidate = AttentionOperatorActivePlan(
            framework_plan=framework_plan,
            dispatch_receipt=receipt,
            provider_selection=selection,
            prepared_plan=prepared,
            jit_plan_binding_fingerprint=jit_plan_binding_fingerprint,
            jit_artifact_binding_fingerprint=jit_artifact_binding_fingerprint,
        )
        self._active_plan = candidate


__all__ = [
    "ATTENTION_OPERATOR_PLAN_VERSION",
    "AttentionOperatorActivePlan",
    "AttentionOperatorPlanFactory",
    "AttentionOperatorPlanSession",
    "AttentionPreparedOperatorPlan",
]
