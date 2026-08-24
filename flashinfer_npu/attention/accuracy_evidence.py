"""Bind measured Attention accuracy reports to dispatch authority records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from flashinfer_npu.runtime import Backend, KernelDescriptor, SchemaError

from .accuracy import (
    AttentionAccuracyCase,
    AttentionAccuracyCorpus,
    AttentionAccuracyReport,
    evaluate_attention_accuracy,
)
from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionCapabilityError,
    AttentionRuntimeEnvironment,
)
from .corpus import (
    AttentionCoveragePolicy,
    AttentionTraceCorpus,
    framework_attention_coverage_policy,
)
from .corpus_samples import build_framework_attention_corpus
from .dispatch import AttentionDispatchError, AttentionDispatchReceipt
from .numerics import DEFAULT_ATTENTION_NUMERICS_POLICY, AttentionNumericsPolicy
from .planner import AttentionFrameworkSession
from .reference import ReferenceAttentionResult


ATTENTION_ACCURACY_DISPATCH_BINDING_VERSION = 1
_HASH_FIELDS = (
    "accuracy_corpus_fingerprint",
    "accuracy_case_fingerprint",
    "dense_input_fingerprint",
    "quantized_input_fingerprint",
    "budget_fingerprint",
    "accuracy_report_fingerprint",
    "candidate_result_fingerprint",
    "dispatch_receipt_fingerprint",
    "plan_fingerprint",
    "admission_fingerprint",
    "workload_fingerprint",
    "profile_fingerprint",
    "environment_fingerprint",
    "capability_result_digest",
    "kernel_fingerprint",
    "artifact_fingerprint",
    "launch_abi_fingerprint",
    "binary_abi_fingerprint",
)


class AttentionAccuracyBindingError(RuntimeError):
    """Raised when accuracy and dispatch evidence do not form one chain."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionAccuracyDispatchBinding:
    binding_id: str
    runner: str
    accuracy_corpus_fingerprint: str
    accuracy_case_id: str
    accuracy_case_fingerprint: str
    dense_input_fingerprint: str
    quantized_input_fingerprint: str
    budget_fingerprint: str
    accuracy_report_fingerprint: str
    candidate_result_fingerprint: str
    dispatch_receipt_fingerprint: str
    plan_fingerprint: str
    admission_fingerprint: str
    workload_fingerprint: str
    profile_id: str
    profile_fingerprint: str
    rule_id: str
    environment_fingerprint: str
    capability_evidence_id: str
    capability_result_digest: str
    kernel_id: str
    kernel_fingerprint: str
    artifact_fingerprint: str
    launch_abi_fingerprint: str
    binary_abi_fingerprint: str
    backend: str
    schema_version: int = ATTENTION_ACCURACY_DISPATCH_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_ACCURACY_DISPATCH_BINDING_VERSION:
            raise SchemaError("unsupported Attention accuracy dispatch binding version")
        for name in (
            "binding_id",
            "runner",
            "accuracy_case_id",
            "profile_id",
            "rule_id",
            "capability_evidence_id",
            "kernel_id",
            "backend",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise SchemaError("accuracy dispatch binding %s must be non-empty" % name)
        for name in _HASH_FIELDS:
            _require_hash(name, getattr(self, name))
        try:
            backend = Backend(self.backend)
        except ValueError as error:
            raise SchemaError("accuracy dispatch binding backend is invalid") from error
        if backend == Backend.REFERENCE:
            raise SchemaError("accuracy dispatch binding cannot name reference backend")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionAccuracyDispatchBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionAccuracyDispatchBinding fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError(
                "AttentionAccuracyDispatchBinding fields are invalid"
            ) from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate(
        self,
        accuracy_case: AttentionAccuracyCase,
        accuracy_corpus: AttentionAccuracyCorpus,
        accuracy_report: AttentionAccuracyReport,
        candidate: ReferenceAttentionResult,
        dispatch_receipt: AttentionDispatchReceipt,
        profile: AttentionBackendCapabilityProfile,
        descriptor: KernelDescriptor,
        environment: AttentionRuntimeEnvironment,
        *,
        numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
        conformance_corpus: Optional[AttentionTraceCorpus] = None,
        coverage_policy: Optional[AttentionCoveragePolicy] = None,
        replay_dispatch_evidence: bool = False,
    ) -> None:
        rebuilt = bind_attention_accuracy_dispatch(
            binding_id=self.binding_id,
            runner=self.runner,
            accuracy_case=accuracy_case,
            accuracy_corpus=accuracy_corpus,
            accuracy_report=accuracy_report,
            candidate=candidate,
            dispatch_receipt=dispatch_receipt,
            profile=profile,
            descriptor=descriptor,
            environment=environment,
            numerics_policy=numerics_policy,
            conformance_corpus=conformance_corpus,
            coverage_policy=coverage_policy,
            replay_dispatch_evidence=replay_dispatch_evidence,
        )
        if self != rebuilt:
            raise AttentionAccuracyBindingError(
                "accuracy dispatch binding fields are stale or inconsistent"
            )


def bind_attention_accuracy_dispatch(
    *,
    binding_id: str,
    runner: str,
    accuracy_case: AttentionAccuracyCase,
    accuracy_corpus: AttentionAccuracyCorpus,
    accuracy_report: AttentionAccuracyReport,
    candidate: ReferenceAttentionResult,
    dispatch_receipt: AttentionDispatchReceipt,
    profile: AttentionBackendCapabilityProfile,
    descriptor: KernelDescriptor,
    environment: AttentionRuntimeEnvironment,
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
    conformance_corpus: Optional[AttentionTraceCorpus] = None,
    coverage_policy: Optional[AttentionCoveragePolicy] = None,
    replay_dispatch_evidence: bool = False,
) -> AttentionAccuracyDispatchBinding:
    """Create a revalidatable post-dispatch binding; never authorizes dispatch."""

    if not isinstance(accuracy_case, AttentionAccuracyCase):
        raise TypeError("accuracy_case must be AttentionAccuracyCase")
    if not isinstance(accuracy_corpus, AttentionAccuracyCorpus):
        raise TypeError("accuracy_corpus must be AttentionAccuracyCorpus")
    matching = tuple(
        case for case in accuracy_corpus.cases if case.case_id == accuracy_case.case_id
    )
    if len(matching) != 1 or matching[0].fingerprint != accuracy_case.fingerprint:
        raise AttentionAccuracyBindingError(
            "accuracy case is not the exact case in the declared corpus"
        )
    if not accuracy_case.expect_quantization_pass:
        raise AttentionAccuracyBindingError(
            "expected-rejection accuracy cases cannot bind runnable dispatch"
        )
    recomputed = evaluate_attention_accuracy(
        accuracy_case.dense_trace,
        accuracy_case.quantized_trace,
        budget=accuracy_case.budget,
        candidate=candidate,
    )
    if recomputed != accuracy_report:
        raise AttentionAccuracyBindingError(
            "accuracy report does not match replayed case and candidate"
        )
    if not recomputed.passes:
        raise AttentionAccuracyBindingError(
            "accuracy report must pass quantization and backend budgets"
        )

    trace = accuracy_case.quantized_trace
    plan = AttentionFrameworkSession(trace.spec.mode).plan(trace.spec, trace.metadata)
    corpus_value = conformance_corpus or build_framework_attention_corpus()
    policy_value = coverage_policy or framework_attention_coverage_policy()
    try:
        dispatch_receipt.validate(
            plan,
            profile,
            descriptor,
            environment,
            numerics_policy=numerics_policy,
            corpus=corpus_value,
            coverage_policy=policy_value,
            replay_evidence=replay_dispatch_evidence,
        )
    except (AttentionDispatchError, AttentionCapabilityError, SchemaError) as error:
        raise AttentionAccuracyBindingError(
            "dispatch receipt revalidation failed: %s" % error
        ) from error

    return AttentionAccuracyDispatchBinding(
        binding_id=binding_id,
        runner=runner,
        accuracy_corpus_fingerprint=accuracy_corpus.fingerprint,
        accuracy_case_id=accuracy_case.case_id,
        accuracy_case_fingerprint=accuracy_case.fingerprint,
        dense_input_fingerprint=recomputed.dense_input_fingerprint,
        quantized_input_fingerprint=recomputed.quantized_input_fingerprint,
        budget_fingerprint=recomputed.budget.fingerprint,
        accuracy_report_fingerprint=recomputed.fingerprint,
        candidate_result_fingerprint=recomputed.candidate_result_fingerprint,
        dispatch_receipt_fingerprint=dispatch_receipt.fingerprint,
        plan_fingerprint=dispatch_receipt.plan_fingerprint,
        admission_fingerprint=dispatch_receipt.admission_fingerprint,
        workload_fingerprint=dispatch_receipt.workload_fingerprint,
        profile_id=dispatch_receipt.profile_id,
        profile_fingerprint=dispatch_receipt.profile_fingerprint,
        rule_id=dispatch_receipt.rule_id,
        environment_fingerprint=dispatch_receipt.environment_fingerprint,
        capability_evidence_id=dispatch_receipt.evidence_id,
        capability_result_digest=dispatch_receipt.evidence_result_digest,
        kernel_id=dispatch_receipt.kernel_id,
        kernel_fingerprint=dispatch_receipt.kernel_fingerprint,
        artifact_fingerprint=dispatch_receipt.artifact_fingerprint,
        launch_abi_fingerprint=dispatch_receipt.launch_abi_fingerprint,
        binary_abi_fingerprint=dispatch_receipt.binary_abi_fingerprint,
        backend=dispatch_receipt.backend.value,
    )


__all__ = [
    "ATTENTION_ACCURACY_DISPATCH_BINDING_VERSION",
    "AttentionAccuracyBindingError",
    "AttentionAccuracyDispatchBinding",
    "bind_attention_accuracy_dispatch",
]
