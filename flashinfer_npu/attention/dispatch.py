"""Evidence-bearing Attention kernel selection and dispatch receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import (
    Backend,
    DeviceCapability,
    KernelDescriptor,
    SchemaError,
)

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionCapabilityError,
    AttentionCapabilityEvidence,
    AttentionRuntimeEnvironment,
    validate_attention_kernel_bindings,
)
from .corpus import (
    AttentionCoveragePolicy,
    AttentionTraceCorpus,
    framework_attention_coverage_policy,
)
from .corpus_samples import build_framework_attention_corpus
from .numerics import (
    DEFAULT_ATTENTION_NUMERICS_POLICY,
    AttentionNumericsPolicy,
)
from .planner import AttentionFrameworkPlan
from .schema import AttentionMode


ATTENTION_DISPATCH_RECEIPT_VERSION = 1
_SHA256_FIELDS = (
    "plan_fingerprint",
    "admission_fingerprint",
    "workload_fingerprint",
    "numerics_policy_fingerprint",
    "profile_fingerprint",
    "environment_fingerprint",
    "evidence_result_digest",
    "kernel_fingerprint",
    "artifact_fingerprint",
    "launch_abi_fingerprint",
    "binary_abi_fingerprint",
)


class AttentionDispatchError(RuntimeError):
    """Raised when no evidence-bearing Attention kernel can be selected."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _requested_backend(value: Union[str, Backend]) -> Optional[Backend]:
    if value == "auto":
        return None
    try:
        return Backend(value)
    except ValueError as error:
        raise SchemaError("unknown Attention dispatch backend %r" % value) from error


def _device_capability(
    environment: AttentionRuntimeEnvironment,
) -> DeviceCapability:
    """Project an exact Attention environment onto generic constraints.

    Dtype support remains governed by the exact Attention rule.  An empty
    generic dtype set deliberately means "not independently asserted" rather
    than synthesizing a device probe that has not happened.
    """

    return DeviceCapability(
        soc_version=environment.soc_version,
        soc_revision=environment.soc_revision,
        ai_core_count=environment.ai_core_count,
        supported_dtypes=(),
        features=environment.features,
        cann_version=environment.cann_version,
        torch_npu_version=environment.torch_npu_version,
        compiler_version=environment.compiler_version,
    )


@dataclass(frozen=True)
class AttentionDispatchCandidateReport:
    profile_id: str
    rule_id: str
    kernel_id: str
    backend: Backend
    accepted: bool
    reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("profile_id", "rule_id", "kernel_id"):
            if not getattr(self, name):
                raise SchemaError("dispatch candidate %s must be non-empty" % name)
        object.__setattr__(self, "backend", Backend(self.backend))
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        if self.accepted == bool(self.reasons):
            raise SchemaError("accepted dispatch candidate must have no reasons")


@dataclass(frozen=True)
class AttentionDispatchReport:
    candidates: Tuple[AttentionDispatchCandidateReport, ...]

    def __post_init__(self) -> None:
        values = tuple(self.candidates)
        keys = tuple(
            (item.profile_id, item.rule_id, item.kernel_id) for item in values
        )
        if len(keys) != len(set(keys)):
            raise SchemaError("Attention dispatch candidates must be unique")
        object.__setattr__(self, "candidates", values)

    @property
    def accepted(self) -> Tuple[AttentionDispatchCandidateReport, ...]:
        return tuple(item for item in self.candidates if item.accepted)


@dataclass(frozen=True)
class AttentionDispatchReceipt:
    mode: AttentionMode
    plan_fingerprint: str
    admission_fingerprint: str
    workload_fingerprint: str
    numerics_policy_fingerprint: str
    profile_id: str
    profile_fingerprint: str
    rule_id: str
    environment_fingerprint: str
    evidence_id: str
    evidence_result_digest: str
    kernel_id: str
    kernel_fingerprint: str
    artifact_fingerprint: str
    launch_abi_fingerprint: str
    binary_abi_fingerprint: str
    backend: Backend
    float_workspace_bytes: int
    int_workspace_bytes: int
    float_workspace_alignment: int
    int_workspace_alignment: int
    selection_source: str
    requested_backend: str
    schema_version: int = ATTENTION_DISPATCH_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_DISPATCH_RECEIPT_VERSION:
            raise SchemaError("unsupported Attention dispatch receipt version")
        object.__setattr__(self, "mode", AttentionMode(self.mode))
        object.__setattr__(self, "backend", Backend(self.backend))
        if self.backend == Backend.REFERENCE:
            raise SchemaError("Attention dispatch receipts cannot authorize reference")
        for name in _SHA256_FIELDS:
            _require_hash(name, str(getattr(self, name)))
        for name in ("profile_id", "rule_id", "evidence_id", "kernel_id"):
            if not str(getattr(self, name)):
                raise SchemaError("dispatch receipt %s must be non-empty" % name)
        for name in ("float_workspace_bytes", "int_workspace_bytes"):
            if getattr(self, name) < 0:
                raise SchemaError("dispatch %s cannot be negative" % name)
        for name in ("float_workspace_alignment", "int_workspace_alignment"):
            alignment = getattr(self, name)
            if alignment <= 0 or alignment & (alignment - 1):
                raise SchemaError("dispatch workspace alignment must be a power of two")
        if self.selection_source not in {"priority", "tuning"}:
            raise SchemaError("selection_source must be priority or tuning")
        requested = _requested_backend(self.requested_backend)
        if requested is not None and requested != self.backend:
            raise SchemaError("requested backend does not match selected backend")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        result["mode"] = self.mode.value
        result["backend"] = self.backend.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionDispatchReceipt":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionDispatchReceipt fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionDispatchReceipt fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def workspace_bytes(self) -> Tuple[int, int]:
        return self.float_workspace_bytes, self.int_workspace_bytes

    @property
    def workspace_alignments(self) -> Tuple[int, int]:
        return self.float_workspace_alignment, self.int_workspace_alignment

    def validate(
        self,
        plan: AttentionFrameworkPlan,
        profile: AttentionBackendCapabilityProfile,
        descriptor: KernelDescriptor,
        observed_environment: AttentionRuntimeEnvironment,
        *,
        numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
        corpus: Optional[AttentionTraceCorpus] = None,
        coverage_policy: Optional[AttentionCoveragePolicy] = None,
        replay_evidence: bool = False,
    ) -> None:
        if (
            self.mode != plan.spec.mode
            or self.plan_fingerprint != plan.fingerprint
            or self.admission_fingerprint != plan.admission_fingerprint
            or self.workload_fingerprint != plan.workload.fingerprint
        ):
            raise AttentionDispatchError("dispatch receipt does not match Attention plan")
        if self.numerics_policy_fingerprint != numerics_policy.fingerprint:
            raise AttentionDispatchError("dispatch receipt numerics policy mismatch")
        if (
            self.profile_id != profile.profile_id
            or self.profile_fingerprint != profile.fingerprint
            or self.backend != profile.backend
        ):
            raise AttentionDispatchError("dispatch receipt capability profile mismatch")
        if self.environment_fingerprint != observed_environment.fingerprint:
            raise AttentionDispatchError("dispatch receipt environment mismatch")
        capability_report = profile.explain(
            plan.spec,
            plan.metadata,
            observed_environment,
            numerics_policy=numerics_policy,
        )
        if not capability_report.accepted or self.rule_id not in (
            capability_report.matching_rule_ids
        ):
            raise AttentionDispatchError("dispatch receipt capability rule no longer accepts")
        selected_evidence = profile.validate_evidence(
            corpus or build_framework_attention_corpus(),
            coverage_policy or framework_attention_coverage_policy(),
            replay=replay_evidence,
        )
        if (
            self.evidence_id != selected_evidence.evidence_id
            or self.evidence_result_digest != selected_evidence.result_digest
        ):
            raise AttentionDispatchError("dispatch receipt evidence mismatch")
        binding = descriptor.capability_binding
        if (
            self.kernel_id != descriptor.kernel_id
            or self.kernel_fingerprint != descriptor.fingerprint
            or self.backend != descriptor.backend
            or descriptor.op != plan.workload.op
            or binding is None
            or binding.domain != "attention"
            or binding.profile_id != self.profile_id
            or binding.profile_fingerprint != self.profile_fingerprint
            or binding.rule_id != self.rule_id
        ):
            raise AttentionDispatchError("dispatch receipt kernel binding mismatch")
        if (
            descriptor.artifact is None
            or descriptor.launch_abi is None
            or descriptor.binary_abi is None
        ):
            raise AttentionDispatchError("dispatch receipt kernel provenance is missing")
        if (
            self.artifact_fingerprint != descriptor.artifact.fingerprint
            or self.launch_abi_fingerprint != descriptor.launch_abi.fingerprint
            or self.binary_abi_fingerprint != descriptor.binary_abi.fingerprint
        ):
            raise AttentionDispatchError("dispatch receipt artifact/ABI mismatch")
        expected_workspace = (
            descriptor.workspace.size_for(plan.workload),
            descriptor.int_workspace.size_for(plan.workload),
        )
        if (
            self.workspace_bytes != expected_workspace
            or self.workspace_alignments
            != (descriptor.workspace.alignment, descriptor.int_workspace.alignment)
        ):
            raise AttentionDispatchError("dispatch receipt workspace formula mismatch")


def _evaluate_attention_dispatch(
    plan: AttentionFrameworkPlan,
    profiles: Sequence[AttentionBackendCapabilityProfile],
    descriptors: Sequence[KernelDescriptor],
    observed_environment: AttentionRuntimeEnvironment,
    *,
    backend: Union[str, Backend],
    numerics_policy: AttentionNumericsPolicy,
    corpus: AttentionTraceCorpus,
    coverage_policy: AttentionCoveragePolicy,
    replay_evidence: bool,
) -> Tuple[AttentionDispatchReport, Dict[str, AttentionCapabilityEvidence]]:
    profile_values = tuple(profiles)
    descriptor_values = tuple(descriptors)
    if len({item.kernel_id for item in descriptor_values}) != len(descriptor_values):
        raise SchemaError("Attention dispatch descriptors contain duplicate kernel_id")
    validate_attention_kernel_bindings(profile_values, descriptor_values)
    requested = _requested_backend(backend)
    generic_capability = _device_capability(observed_environment)
    profile_map = {item.profile_id: item for item in profile_values}
    evidence: Dict[str, AttentionCapabilityEvidence] = {}
    evidence_errors: Dict[str, str] = {}
    reports = []
    for descriptor in descriptor_values:
        binding = descriptor.capability_binding
        if descriptor.op != plan.workload.op or binding is None:
            continue
        if binding.domain != "attention" or binding.profile_id not in profile_map:
            continue
        profile = profile_map[binding.profile_id]
        capability = profile.explain(
            plan.spec,
            plan.metadata,
            observed_environment,
            numerics_policy=numerics_policy,
        )
        rule_report = next(
            item for item in capability.rules if item.rule_id == binding.rule_id
        )
        reasons = list(capability.global_reasons)
        reasons.extend(rule_report.reasons)
        reasons.extend(
            descriptor.constraints.unsupported_reasons(
                plan.workload, generic_capability
            )
        )
        if profile.backend == Backend.REFERENCE:
            reasons.append("reference profiles cannot authorize device dispatch")
        if requested is not None and descriptor.backend != requested:
            reasons.append("excluded by backend policy %s" % requested.value)
        if profile.profile_id not in evidence and profile.profile_id not in evidence_errors:
            try:
                evidence[profile.profile_id] = profile.validate_evidence(
                    corpus, coverage_policy, replay=replay_evidence
                )
            except AttentionCapabilityError as error:
                evidence_errors[profile.profile_id] = str(error)
        if profile.profile_id in evidence_errors:
            reasons.append(
                "capability evidence invalid: %s" % evidence_errors[profile.profile_id]
            )
        reports.append(
            AttentionDispatchCandidateReport(
                profile.profile_id,
                binding.rule_id,
                descriptor.kernel_id,
                descriptor.backend,
                not reasons,
                tuple(reasons),
            )
        )
    return AttentionDispatchReport(tuple(reports)), evidence


def explain_attention_dispatch(
    plan: AttentionFrameworkPlan,
    profiles: Sequence[AttentionBackendCapabilityProfile],
    descriptors: Sequence[KernelDescriptor],
    observed_environment: AttentionRuntimeEnvironment,
    *,
    backend: Union[str, Backend] = "auto",
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
    corpus: Optional[AttentionTraceCorpus] = None,
    coverage_policy: Optional[AttentionCoveragePolicy] = None,
    replay_evidence: bool = False,
) -> AttentionDispatchReport:
    report, _ = _evaluate_attention_dispatch(
        plan,
        profiles,
        descriptors,
        observed_environment,
        backend=backend,
        numerics_policy=numerics_policy,
        corpus=corpus or build_framework_attention_corpus(),
        coverage_policy=coverage_policy or framework_attention_coverage_policy(),
        replay_evidence=replay_evidence,
    )
    return report


def select_attention_dispatch(
    plan: AttentionFrameworkPlan,
    profiles: Sequence[AttentionBackendCapabilityProfile],
    descriptors: Sequence[KernelDescriptor],
    observed_environment: AttentionRuntimeEnvironment,
    *,
    backend: Union[str, Backend] = "auto",
    tuned_kernel_ids: Sequence[str] = (),
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
    corpus: Optional[AttentionTraceCorpus] = None,
    coverage_policy: Optional[AttentionCoveragePolicy] = None,
    replay_evidence: bool = False,
) -> AttentionDispatchReceipt:
    descriptor_values = tuple(descriptors)
    corpus_value = corpus or build_framework_attention_corpus()
    policy_value = coverage_policy or framework_attention_coverage_policy()
    report, evidence = _evaluate_attention_dispatch(
        plan,
        profiles,
        descriptor_values,
        observed_environment,
        backend=backend,
        numerics_policy=numerics_policy,
        corpus=corpus_value,
        coverage_policy=policy_value,
        replay_evidence=replay_evidence,
    )
    accepted_ids = {item.kernel_id for item in report.accepted}
    selected = next(
        (
            descriptor
            for kernel_id in tuned_kernel_ids
            for descriptor in descriptor_values
            if descriptor.kernel_id == kernel_id and kernel_id in accepted_ids
        ),
        None,
    )
    selection_source = "tuning"
    if selected is None:
        accepted = tuple(
            descriptor
            for descriptor in descriptor_values
            if descriptor.kernel_id in accepted_ids
        )
        if not accepted:
            details = "; ".join(
                "%s: %s" % (item.kernel_id, ", ".join(item.reasons))
                for item in report.candidates
            )
            raise AttentionDispatchError(
                "no evidence-bearing Attention kernel accepted the plan (%s)"
                % (details or "no bound descriptors for mode")
            )
        selected = sorted(
            accepted,
            key=lambda item: (-item.priority, item.kernel_id),
        )[0]
        selection_source = "priority"
    binding = selected.capability_binding
    assert binding is not None
    assert selected.artifact is not None
    assert selected.launch_abi is not None
    assert selected.binary_abi is not None
    profile = next(item for item in profiles if item.profile_id == binding.profile_id)
    selected_evidence = evidence[profile.profile_id]
    requested_backend = (
        "auto" if backend == "auto" else Backend(backend).value
    )
    return AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=numerics_policy.fingerprint,
        profile_id=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        rule_id=binding.rule_id,
        environment_fingerprint=observed_environment.fingerprint,
        evidence_id=selected_evidence.evidence_id,
        evidence_result_digest=selected_evidence.result_digest,
        kernel_id=selected.kernel_id,
        kernel_fingerprint=selected.fingerprint,
        artifact_fingerprint=selected.artifact.fingerprint,
        launch_abi_fingerprint=selected.launch_abi.fingerprint,
        binary_abi_fingerprint=selected.binary_abi.fingerprint,
        backend=selected.backend,
        float_workspace_bytes=selected.workspace.size_for(plan.workload),
        int_workspace_bytes=selected.int_workspace.size_for(plan.workload),
        float_workspace_alignment=selected.workspace.alignment,
        int_workspace_alignment=selected.int_workspace.alignment,
        selection_source=selection_source,
        requested_backend=requested_backend,
    )


__all__ = [
    "ATTENTION_DISPATCH_RECEIPT_VERSION",
    "AttentionDispatchCandidateReport",
    "AttentionDispatchError",
    "AttentionDispatchReceipt",
    "AttentionDispatchReport",
    "explain_attention_dispatch",
    "select_attention_dispatch",
]
