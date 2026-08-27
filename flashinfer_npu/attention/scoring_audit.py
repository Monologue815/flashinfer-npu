"""Offline verification for manifest-authorized Attention plan selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .holistic import AttentionOperatorRuntimeRegistrySnapshot
from .operator_resolver import (
    AttentionOperatorRuntimeResolutionReport,
)
from .operator_run_receipt import AttentionOperatorRunReceipt
from .operator_scoring import AttentionOperatorPlanScoringManifest
from .plan_selection import AttentionPlanSelection
from .planner import AttentionFrameworkPlan


ATTENTION_PLAN_SCORING_AUDIT_VERSION = 2


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class AttentionPlanScoringAuditError(SchemaError):
    """The supplied offline scoring evidence does not form one valid chain."""


@dataclass(frozen=True)
class AttentionPlanScoringAuditCandidate:
    """One top-tier candidate whose declarative score was replayed."""

    provider_id: str
    operation_id: str
    score: int
    source: str
    reason: str
    policy_id: str
    policy_fingerprint: str

    def __post_init__(self) -> None:
        if not str(self.provider_id) or not str(self.operation_id):
            raise SchemaError("Attention scoring audit candidate identity is empty")
        if not isinstance(self.score, int) or isinstance(self.score, bool) or not (
            -(2**31) <= self.score <= 2**31 - 1
        ):
            raise SchemaError("Attention scoring audit candidate score is invalid")
        if not str(self.source).strip() or not str(self.reason).strip():
            raise SchemaError(
                "Attention scoring audit candidate explanation is empty"
            )
        if not str(self.policy_id).strip() or any(
            item.isspace() for item in str(self.policy_id)
        ):
            raise SchemaError(
                "Attention scoring audit candidate policy id is invalid"
            )
        fingerprint = str(self.policy_fingerprint)
        if len(fingerprint) != 64 or any(
            item not in "0123456789abcdef" for item in fingerprint
        ):
            raise SchemaError(
                "Attention scoring audit candidate policy fingerprint is invalid"
            )
        for name in (
            "provider_id",
            "operation_id",
            "source",
            "reason",
            "policy_id",
            "policy_fingerprint",
        ):
            object.__setattr__(self, name, str(getattr(self, name)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "score": self.score,
            "source": self.source,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "policy_fingerprint": self.policy_fingerprint,
        }


@dataclass(frozen=True)
class AttentionPlanScoringAuditReport:
    """Canonical result of a successful offline scoring-chain verification."""

    registry_generation: int
    device: str
    framework_plan_fingerprint: str
    resolution_report_fingerprint: str
    plan_selection_fingerprint: str
    active_plan_fingerprint: str
    provider_id: str
    operation_id: str
    runtime_declaration_fingerprint: str
    manifest_id: str
    manifest_fingerprint: str
    policy_id: str
    policy_fingerprint: str
    plan_score: int
    replayed_candidates: Tuple[AttentionPlanScoringAuditCandidate, ...]
    provider_integration_bundle_id: Optional[str] = None
    provider_integration_bundle_fingerprint: Optional[str] = None
    run_receipt_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_PLAN_SCORING_AUDIT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PLAN_SCORING_AUDIT_VERSION:
            raise SchemaError("unsupported Attention scoring audit version")
        if not isinstance(self.registry_generation, int) or isinstance(
            self.registry_generation, bool
        ) or self.registry_generation < 0:
            raise SchemaError("Attention scoring audit generation is invalid")
        if not str(self.device) or not str(self.provider_id) or not str(
            self.operation_id
        ):
            raise SchemaError("Attention scoring audit identity is incomplete")
        if not str(self.manifest_id).strip() or any(
            item.isspace() for item in str(self.manifest_id)
        ) or not str(self.policy_id).strip() or any(
            item.isspace() for item in str(self.policy_id)
        ):
            raise SchemaError("Attention scoring audit policy identity is invalid")
        if not isinstance(self.plan_score, int) or isinstance(
            self.plan_score, bool
        ) or not (-(2**31) <= self.plan_score <= 2**31 - 1):
            raise SchemaError("Attention scoring audit selected score is invalid")
        for name in (
            "framework_plan_fingerprint",
            "resolution_report_fingerprint",
            "plan_selection_fingerprint",
            "active_plan_fingerprint",
            "runtime_declaration_fingerprint",
            "manifest_fingerprint",
            "policy_fingerprint",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError("Attention scoring audit %s is invalid" % name)
        if self.run_receipt_fingerprint is not None:
            value = str(self.run_receipt_fingerprint)
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError(
                    "Attention scoring audit run receipt fingerprint is invalid"
                )
        bundle_fields = (
            self.provider_integration_bundle_id,
            self.provider_integration_bundle_fingerprint,
        )
        if any(item is not None for item in bundle_fields):
            if any(item is None for item in bundle_fields):
                raise SchemaError(
                    "Attention scoring audit provider bundle identity is incomplete"
                )
            bundle_id = str(self.provider_integration_bundle_id)
            if not bundle_id.strip() or any(
                item.isspace() for item in bundle_id
            ):
                raise SchemaError(
                    "Attention scoring audit provider bundle id is invalid"
                )
            fingerprint = str(self.provider_integration_bundle_fingerprint)
            if len(fingerprint) != 64 or any(
                item not in "0123456789abcdef" for item in fingerprint
            ):
                raise SchemaError(
                    "Attention scoring audit provider bundle fingerprint is invalid"
                )
        candidates = tuple(self.replayed_candidates)
        if not candidates or any(
            not isinstance(item, AttentionPlanScoringAuditCandidate)
            for item in candidates
        ):
            raise TypeError("replayed_candidates must contain audit candidates")
        identities = tuple(
            (item.provider_id, item.operation_id) for item in candidates
        )
        if len(set(identities)) != len(identities):
            raise SchemaError("Attention scoring audit candidates duplicate")
        selected = tuple(
            item
            for item in candidates
            if (item.provider_id, item.operation_id)
            == (self.provider_id, self.operation_id)
        )
        if len(selected) != 1 or (
            selected[0].score != self.plan_score
            or selected[0].policy_id != self.policy_id
            or selected[0].policy_fingerprint != self.policy_fingerprint
        ):
            raise SchemaError(
                "Attention scoring audit selected candidate differs"
            )
        object.__setattr__(
            self,
            "replayed_candidates",
            tuple(
                sorted(
                    candidates,
                    key=lambda item: (item.provider_id, item.operation_id),
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_plan_scoring_audit",
            "registry_generation": self.registry_generation,
            "device": self.device,
            "framework_plan_fingerprint": self.framework_plan_fingerprint,
            "resolution_report_fingerprint": (
                self.resolution_report_fingerprint
            ),
            "plan_selection_fingerprint": self.plan_selection_fingerprint,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "runtime_declaration_fingerprint": (
                self.runtime_declaration_fingerprint
            ),
            "manifest_id": self.manifest_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "policy_id": self.policy_id,
            "policy_fingerprint": self.policy_fingerprint,
            "plan_score": self.plan_score,
            "provider_integration_bundle_id": (
                self.provider_integration_bundle_id
            ),
            "provider_integration_bundle_fingerprint": (
                self.provider_integration_bundle_fingerprint
            ),
            "replayed_candidates": [
                item.to_dict() for item in self.replayed_candidates
            ],
            "run_receipt_fingerprint": self.run_receipt_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _fail(message: str) -> None:
    raise AttentionPlanScoringAuditError(message)


def verify_attention_plan_scoring_chain(
    *,
    plan: AttentionFrameworkPlan,
    resolution_report: AttentionOperatorRuntimeResolutionReport,
    registry_snapshot: AttentionOperatorRuntimeRegistrySnapshot,
    scoring_manifest: AttentionOperatorPlanScoringManifest,
    plan_selection: AttentionPlanSelection,
    run_receipt: Optional[AttentionOperatorRunReceipt] = None,
) -> AttentionPlanScoringAuditReport:
    """Replay and verify one manifest-bound plan/run chain without providers."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(
        resolution_report, AttentionOperatorRuntimeResolutionReport
    ):
        raise TypeError(
            "resolution_report must be AttentionOperatorRuntimeResolutionReport"
        )
    if not isinstance(
        registry_snapshot, AttentionOperatorRuntimeRegistrySnapshot
    ):
        raise TypeError(
            "registry_snapshot must be AttentionOperatorRuntimeRegistrySnapshot"
        )
    if not isinstance(scoring_manifest, AttentionOperatorPlanScoringManifest):
        raise TypeError(
            "scoring_manifest must be AttentionOperatorPlanScoringManifest"
        )
    if not isinstance(plan_selection, AttentionPlanSelection):
        raise TypeError("plan_selection must be AttentionPlanSelection")
    if run_receipt is not None and not isinstance(
        run_receipt, AttentionOperatorRunReceipt
    ):
        raise TypeError("run_receipt must be AttentionOperatorRunReceipt")

    if plan_selection.route != "provider":
        _fail("Attention scoring audit requires a provider plan selection")
    if (
        plan_selection.framework_plan_fingerprint != plan.fingerprint
        or plan_selection.plan_generation != plan.generation
        or plan_selection.mode != plan.spec.mode
    ):
        _fail("Attention scoring audit framework plan differs from selection")
    if resolution_report.framework_plan_fingerprint != plan.fingerprint:
        _fail("Attention scoring audit resolution report has a different plan")
    if (
        plan_selection.runtime_resolution_fingerprint
        != resolution_report.fingerprint
    ):
        _fail("Attention scoring audit resolution fingerprint differs")
    selected = resolution_report.selected
    if selected is None or selected.plan_score is None:
        _fail("Attention scoring audit resolution has no unique scored winner")
    selected_identity = (selected.provider_id, selected.operation_id)
    if selected_identity != (
        plan_selection.provider_id,
        plan_selection.operation_id,
    ):
        _fail("Attention scoring audit selected provider identity differs")

    if plan_selection.registry_generation != registry_snapshot.generation:
        _fail("Attention scoring audit registry generation differs")
    binding = registry_snapshot.plan_scoring_manifest_binding
    if binding is None or binding != scoring_manifest.binding:
        _fail("Attention scoring audit registry manifest binding differs")
    if (
        plan_selection.plan_scoring_manifest_id != binding.manifest_id
        or plan_selection.plan_scoring_manifest_fingerprint
        != binding.manifest_fingerprint
    ):
        _fail("Attention scoring audit plan manifest identity differs")
    policy_id, policy_fingerprint = binding.policy_binding(*selected_identity)
    if (
        plan_selection.plan_scoring_policy_id != policy_id
        or plan_selection.plan_scoring_policy_fingerprint
        != policy_fingerprint
    ):
        _fail("Attention scoring audit selected policy identity differs")
    declaration_fingerprint = registry_snapshot.declaration_fingerprint(
        *selected_identity
    )
    if (
        declaration_fingerprint is None
        or plan_selection.runtime_declaration_fingerprint
        != declaration_fingerprint
    ):
        _fail("Attention scoring audit runtime declaration differs")
    bundle_binding = registry_snapshot.provider_integration_bundle_binding
    if bundle_binding is None:
        if (
            plan_selection.provider_integration_bundle_id is not None
            or plan_selection.provider_integration_bundle_fingerprint is not None
        ):
            _fail("Attention scoring audit plan provider bundle identity differs")
    else:
        if selected_identity not in bundle_binding.identities:
            _fail("Attention scoring audit selected operation is outside bundle")
        if (
            plan_selection.provider_integration_bundle_id
            != bundle_binding.bundle_id
            or plan_selection.provider_integration_bundle_fingerprint
            != bundle_binding.bundle_fingerprint
        ):
            _fail("Attention scoring audit plan provider bundle identity differs")

    candidate_identities = tuple(
        (item.provider_id, item.operation_id)
        for item in resolution_report.candidates
    )
    if set(candidate_identities) != set(binding.identities):
        _fail("Attention scoring audit resolution candidate set differs")
    replayed = []
    for candidate in resolution_report.candidates:
        if candidate.plan_score is None:
            continue
        policy = scoring_manifest.get(
            candidate.provider_id,
            candidate.operation_id,
        )
        expected_score = policy.score(plan, resolution_report.device)
        if candidate.plan_score.to_dict() != expected_score.to_dict():
            _fail(
                "Attention scoring audit replay differs for %s/%s"
                % (candidate.provider_id, candidate.operation_id)
            )
        replayed.append(
            AttentionPlanScoringAuditCandidate(
                provider_id=candidate.provider_id,
                operation_id=candidate.operation_id,
                score=expected_score.value,
                source=expected_score.source,
                reason=expected_score.reason,
                policy_id=policy.policy_id,
                policy_fingerprint=policy.fingerprint,
            )
        )
    selected_score = selected.plan_score
    if (
        plan_selection.plan_score != selected_score.value
        or plan_selection.plan_score_source != selected_score.source
        or plan_selection.plan_score_reason != selected_score.reason
        or selected_score.policy_id != policy_id
        or selected_score.policy_fingerprint != policy_fingerprint
    ):
        _fail("Attention scoring audit selected score evidence differs")

    run_fingerprint = None
    if run_receipt is not None:
        if (
            run_receipt.active_plan_fingerprint
            != plan_selection.active_plan_fingerprint
            or run_receipt.provider_id != plan_selection.provider_id
            or run_receipt.operation_id != plan_selection.operation_id
            or run_receipt.runtime_declaration_fingerprint
            != plan_selection.runtime_declaration_fingerprint
            or run_receipt.provider_integration_bundle_id
            != plan_selection.provider_integration_bundle_id
            or run_receipt.provider_integration_bundle_fingerprint
            != plan_selection.provider_integration_bundle_fingerprint
            or run_receipt.plan_scoring_manifest_id
            != plan_selection.plan_scoring_manifest_id
            or run_receipt.plan_scoring_manifest_fingerprint
            != plan_selection.plan_scoring_manifest_fingerprint
            or run_receipt.plan_scoring_policy_id
            != plan_selection.plan_scoring_policy_id
            or run_receipt.plan_scoring_policy_fingerprint
            != plan_selection.plan_scoring_policy_fingerprint
        ):
            _fail("Attention scoring audit run receipt differs from selection")
        run_fingerprint = run_receipt.fingerprint

    return AttentionPlanScoringAuditReport(
        registry_generation=registry_snapshot.generation,
        device=resolution_report.device,
        framework_plan_fingerprint=plan.fingerprint,
        resolution_report_fingerprint=resolution_report.fingerprint,
        plan_selection_fingerprint=plan_selection.fingerprint,
        active_plan_fingerprint=plan_selection.active_plan_fingerprint,
        provider_id=selected.provider_id,
        operation_id=selected.operation_id,
        runtime_declaration_fingerprint=declaration_fingerprint,
        manifest_id=binding.manifest_id,
        manifest_fingerprint=binding.manifest_fingerprint,
        policy_id=policy_id,
        policy_fingerprint=policy_fingerprint,
        plan_score=selected_score.value,
        replayed_candidates=tuple(replayed),
        provider_integration_bundle_id=(
            None if bundle_binding is None else bundle_binding.bundle_id
        ),
        provider_integration_bundle_fingerprint=(
            None
            if bundle_binding is None
            else bundle_binding.bundle_fingerprint
        ),
        run_receipt_fingerprint=run_fingerprint,
    )


__all__ = [
    "ATTENTION_PLAN_SCORING_AUDIT_VERSION",
    "AttentionPlanScoringAuditCandidate",
    "AttentionPlanScoringAuditError",
    "AttentionPlanScoringAuditReport",
    "verify_attention_plan_scoring_chain",
]
