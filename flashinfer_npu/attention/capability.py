"""Versioned, evidence-bearing capability profiles for Attention backends."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import (
    Backend,
    KernelDescriptor,
    QuantSpec,
    SchemaError,
)

from .corpus import (
    AttentionCoveragePolicy,
    AttentionTraceCorpus,
    framework_attention_coverage_policy,
)
from .numerics import (
    DEFAULT_ATTENTION_NUMERICS_POLICY,
    AttentionNumericsPolicy,
)
from .resource_limits import (
    AttentionMetadataLimits,
    AttentionResourceLimitError,
)
from .schema import (
    AttentionMetadata,
    AttentionMode,
    AttentionPlanSpec,
    KVLayout,
    PosEncodingMode,
)


ATTENTION_CAPABILITY_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AttentionCapabilityStatus(str, Enum):
    DRAFT = "draft"
    PROTOCOL = "protocol"
    FUNCTIONAL = "functional"
    OPTIMIZED = "optimized"


_STATUS_RANK = {
    AttentionCapabilityStatus.DRAFT: 0,
    AttentionCapabilityStatus.PROTOCOL: 1,
    AttentionCapabilityStatus.FUNCTIONAL: 2,
    AttentionCapabilityStatus.OPTIMIZED: 3,
}


class AttentionCapabilityError(RuntimeError):
    """Raised when no validated capability rule accepts a workload."""


def _strict_fields(cls, value: Mapping[str, Any], name: str) -> Dict[str, Any]:
    data = dict(value)
    if set(data) != set(cls.__dataclass_fields__):
        raise SchemaError("%s fields do not match schema version 1" % name)
    return data


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _string_tuple(name: str, values: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(str(item) for item in values)
    if not result or any(not item for item in result):
        raise SchemaError("%s must contain at least one non-empty value" % name)
    if len(set(result)) != len(result):
        raise SchemaError("%s cannot contain duplicates" % name)
    return result


@dataclass(frozen=True)
class AttentionRuntimeEnvironment:
    """Exact software/hardware tuple to which a profile is bound."""

    soc_version: str
    soc_revision: str
    driver_version: str
    firmware_version: str
    cann_version: str
    torch_version: str
    torch_npu_version: str
    compiler_version: str
    python_abi: str
    ai_core_count: int = 0
    features: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention runtime environment version")
        for name in (
            "soc_version",
            "soc_revision",
            "driver_version",
            "firmware_version",
            "cann_version",
            "torch_version",
            "torch_npu_version",
            "compiler_version",
            "python_abi",
        ):
            if not str(getattr(self, name)):
                raise SchemaError("%s must be non-empty" % name)
        if self.ai_core_count < 0:
            raise SchemaError("ai_core_count cannot be negative")
        features = tuple(sorted(str(item) for item in self.features))
        if any(not item for item in features) or len(set(features)) != len(features):
            raise SchemaError("environment features must be unique non-empty strings")
        object.__setattr__(self, "features", features)

    @property
    def is_fully_pinned(self) -> bool:
        unresolved = {"unknown", "unavailable", "unset", "none"}
        values = (
            self.soc_version,
            self.soc_revision,
            self.driver_version,
            self.firmware_version,
            self.cann_version,
            self.torch_version,
            self.torch_npu_version,
            self.compiler_version,
            self.python_abi,
        )
        return self.ai_core_count > 0 and all(
            value.strip().lower() not in unresolved for value in values
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "soc_version": self.soc_version,
            "soc_revision": self.soc_revision,
            "driver_version": self.driver_version,
            "firmware_version": self.firmware_version,
            "cann_version": self.cann_version,
            "torch_version": self.torch_version,
            "torch_npu_version": self.torch_npu_version,
            "compiler_version": self.compiler_version,
            "python_abi": self.python_abi,
            "ai_core_count": self.ai_core_count,
            "features": list(self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionRuntimeEnvironment":
        data = _strict_fields(cls, value, "AttentionRuntimeEnvironment")
        data["features"] = tuple(data["features"])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionRuntimeEnvironment fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionCapabilityEvidence:
    evidence_id: str
    level: AttentionCapabilityStatus
    runner: str
    corpus_fingerprint: str
    coverage_policy_name: str
    covered_cells: int
    total_cells: int
    passed_case_ids: Tuple[str, ...]
    result_digest: str
    schema_version: int = ATTENTION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention capability evidence version")
        if not _IDENTIFIER.fullmatch(self.evidence_id):
            raise SchemaError("invalid Attention capability evidence_id")
        object.__setattr__(self, "level", AttentionCapabilityStatus(self.level))
        if self.level == AttentionCapabilityStatus.DRAFT:
            raise SchemaError("evidence level cannot be draft")
        if not self.runner or not self.coverage_policy_name:
            raise SchemaError("evidence runner and coverage policy must be non-empty")
        if not _SHA256.fullmatch(self.corpus_fingerprint):
            raise SchemaError("evidence corpus_fingerprint must be SHA-256")
        if not _SHA256.fullmatch(self.result_digest):
            raise SchemaError("evidence result_digest must be SHA-256")
        if self.covered_cells < 0 or self.total_cells <= 0:
            raise SchemaError("evidence coverage counts are invalid")
        if self.covered_cells > self.total_cells:
            raise SchemaError("covered_cells cannot exceed total_cells")
        case_ids = tuple(str(item) for item in self.passed_case_ids)
        if not case_ids or any(not _IDENTIFIER.fullmatch(item) for item in case_ids):
            raise SchemaError("evidence passed_case_ids are invalid")
        if len(set(case_ids)) != len(case_ids):
            raise SchemaError("evidence passed_case_ids cannot contain duplicates")
        object.__setattr__(self, "passed_case_ids", case_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "level": self.level.value,
            "runner": self.runner,
            "corpus_fingerprint": self.corpus_fingerprint,
            "coverage_policy_name": self.coverage_policy_name,
            "covered_cells": self.covered_cells,
            "total_cells": self.total_cells,
            "passed_case_ids": list(self.passed_case_ids),
            "result_digest": self.result_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionCapabilityEvidence":
        data = _strict_fields(cls, value, "AttentionCapabilityEvidence")
        data["passed_case_ids"] = tuple(data["passed_case_ids"])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionCapabilityEvidence fields are invalid") from error


@dataclass(frozen=True)
class AttentionCapabilityRule:
    """One coherent Attention feature set; quant specs match exactly."""

    rule_id: str
    modes: Tuple[AttentionMode, ...]
    kv_layouts: Tuple[KVLayout, ...]
    dtype_signatures: Tuple[Tuple[str, str, str], ...]
    supports_dense_kv: bool = True
    quant_specs: Tuple[QuantSpec, ...] = ()
    pos_encoding_modes: Tuple[PosEncodingMode, ...] = (PosEncodingMode.NONE,)
    mask_kinds: Tuple[str, ...] = ("none",)
    causal_values: Tuple[bool, ...] = (False, True)
    max_head_dim_qk: int = 0
    max_head_dim_vo: int = 0
    head_dim_qk_multiple: int = 1
    head_dim_vo_multiple: int = 1
    max_gqa_group_size: int = 0
    supports_sliding_window: bool = False
    supports_logits_soft_cap: bool = False
    supports_fp16_qk_reduction: bool = False
    supports_profiler: bool = False
    supports_multi_token_decode: bool = False
    metadata_limits: AttentionMetadataLimits = AttentionMetadataLimits()
    required_features: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention capability rule version")
        if not _IDENTIFIER.fullmatch(self.rule_id):
            raise SchemaError("invalid Attention capability rule_id")
        modes = tuple(AttentionMode(item) for item in self.modes)
        layouts = tuple(KVLayout(item) for item in self.kv_layouts)
        positions = tuple(PosEncodingMode(item) for item in self.pos_encoding_modes)
        if not modes or len(set(modes)) != len(modes):
            raise SchemaError("rule modes must be non-empty and unique")
        if not layouts or len(set(layouts)) != len(layouts):
            raise SchemaError("rule kv_layouts must be non-empty and unique")
        if not positions or len(set(positions)) != len(positions):
            raise SchemaError("rule pos_encoding_modes must be non-empty and unique")
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "kv_layouts", layouts)
        object.__setattr__(self, "pos_encoding_modes", positions)
        signatures = tuple(tuple(str(value) for value in item) for item in self.dtype_signatures)
        if not signatures or any(len(item) != 3 for item in signatures):
            raise SchemaError("dtype_signatures must contain (q, kv, out) triples")
        if len(set(signatures)) != len(signatures):
            raise SchemaError("dtype_signatures cannot contain duplicates")
        object.__setattr__(self, "dtype_signatures", signatures)
        masks = _string_tuple("mask_kinds", self.mask_kinds)
        if any(item not in {"none", "unpacked", "packed"} for item in masks):
            raise SchemaError("mask_kinds contains an unsupported value")
        object.__setattr__(self, "mask_kinds", masks)
        causal = tuple(bool(value) for value in self.causal_values)
        if not causal or len(set(causal)) != len(causal):
            raise SchemaError("causal_values must be non-empty and unique")
        object.__setattr__(self, "causal_values", causal)
        quant_specs = tuple(self.quant_specs)
        if len({item.fingerprint for item in quant_specs}) != len(quant_specs):
            raise SchemaError("rule quant_specs cannot contain duplicates")
        object.__setattr__(self, "quant_specs", quant_specs)
        for name in (
            "max_head_dim_qk",
            "max_head_dim_vo",
            "max_gqa_group_size",
        ):
            if getattr(self, name) < 0:
                raise SchemaError("%s cannot be negative" % name)
        for name in ("head_dim_qk_multiple", "head_dim_vo_multiple"):
            if getattr(self, name) <= 0:
                raise SchemaError("%s must be positive" % name)
        if not isinstance(self.metadata_limits, AttentionMetadataLimits):
            raise TypeError("metadata_limits must be AttentionMetadataLimits")
        features = tuple(sorted(str(item) for item in self.required_features))
        if any(not item for item in features) or len(set(features)) != len(features):
            raise SchemaError("required_features must be unique non-empty strings")
        object.__setattr__(self, "required_features", features)

    def unsupported_reasons(
        self,
        spec: AttentionPlanSpec,
        metadata: AttentionMetadata,
        environment: AttentionRuntimeEnvironment,
    ) -> Tuple[str, ...]:
        reasons = []
        if spec.mode not in self.modes:
            reasons.append("unsupported mode %s" % spec.mode.value)
        if spec.kv_layout not in self.kv_layouts:
            reasons.append("unsupported KV layout %s" % spec.kv_layout.value)
        signature = (spec.q_dtype, str(spec.kv_dtype), str(spec.o_dtype))
        if signature not in self.dtype_signatures:
            reasons.append("unsupported dtype signature %r" % (signature,))
        if spec.kv_quant_spec is None:
            if not self.supports_dense_kv:
                reasons.append("dense KV is unsupported")
        elif spec.kv_quant_spec not in self.quant_specs:
            reasons.append(
                "unsupported exact QuantSpec %s" % spec.kv_quant_spec.fingerprint
            )
        if spec.pos_encoding_mode not in self.pos_encoding_modes:
            reasons.append(
                "unsupported position encoding %s" % spec.pos_encoding_mode.value
            )
        mask_kind = (
            "none"
            if spec.custom_mask is None
            else ("packed" if spec.custom_mask.packed else "unpacked")
        )
        if mask_kind not in self.mask_kinds:
            reasons.append("unsupported mask kind %s" % mask_kind)
        if spec.effective_causal not in self.causal_values:
            reasons.append("unsupported effective causal=%s" % spec.effective_causal)
        if self.max_head_dim_qk and spec.head_dim_qk > self.max_head_dim_qk:
            reasons.append("head_dim_qk exceeds %d" % self.max_head_dim_qk)
        if self.max_head_dim_vo and int(spec.head_dim_vo) > self.max_head_dim_vo:
            reasons.append("head_dim_vo exceeds %d" % self.max_head_dim_vo)
        if spec.head_dim_qk % self.head_dim_qk_multiple:
            reasons.append(
                "head_dim_qk is not a multiple of %d" % self.head_dim_qk_multiple
            )
        if int(spec.head_dim_vo) % self.head_dim_vo_multiple:
            reasons.append(
                "head_dim_vo is not a multiple of %d" % self.head_dim_vo_multiple
            )
        group_size = spec.num_qo_heads // spec.num_kv_heads
        if self.max_gqa_group_size and group_size > self.max_gqa_group_size:
            reasons.append("GQA group size exceeds %d" % self.max_gqa_group_size)
        uses_window = spec.window_left >= 0 or spec.window_right != 0
        if uses_window and not self.supports_sliding_window:
            reasons.append("sliding window is unsupported")
        if float(spec.logits_soft_cap or 0.0) > 0 and not self.supports_logits_soft_cap:
            reasons.append("logits soft cap is unsupported")
        if spec.use_fp16_qk_reduction and not self.supports_fp16_qk_reduction:
            reasons.append("FP16 QK reduction is unsupported")
        if spec.use_profiler and not self.supports_profiler:
            reasons.append("profiler output is unsupported")
        if (
            spec.mode == AttentionMode.BATCH_DECODE_PAGED
            and spec.q_len_per_req > 1
            and not self.supports_multi_token_decode
        ):
            reasons.append("multi-token decode is unsupported")
        missing = tuple(
            feature
            for feature in self.required_features
            if feature not in environment.features
        )
        if missing:
            reasons.append("environment is missing features %r" % (missing,))
        try:
            self.metadata_limits.validate(spec, metadata)
        except AttentionResourceLimitError as error:
            reasons.append(str(error))
        return tuple(reasons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "modes": [item.value for item in self.modes],
            "kv_layouts": [item.value for item in self.kv_layouts],
            "dtype_signatures": [list(item) for item in self.dtype_signatures],
            "supports_dense_kv": self.supports_dense_kv,
            "quant_specs": [item.to_dict() for item in self.quant_specs],
            "pos_encoding_modes": [item.value for item in self.pos_encoding_modes],
            "mask_kinds": list(self.mask_kinds),
            "causal_values": list(self.causal_values),
            "max_head_dim_qk": self.max_head_dim_qk,
            "max_head_dim_vo": self.max_head_dim_vo,
            "head_dim_qk_multiple": self.head_dim_qk_multiple,
            "head_dim_vo_multiple": self.head_dim_vo_multiple,
            "max_gqa_group_size": self.max_gqa_group_size,
            "supports_sliding_window": self.supports_sliding_window,
            "supports_logits_soft_cap": self.supports_logits_soft_cap,
            "supports_fp16_qk_reduction": self.supports_fp16_qk_reduction,
            "supports_profiler": self.supports_profiler,
            "supports_multi_token_decode": self.supports_multi_token_decode,
            "metadata_limits": self.metadata_limits.to_dict(),
            "required_features": list(self.required_features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionCapabilityRule":
        data = _strict_fields(cls, value, "AttentionCapabilityRule")
        for field in (
            "modes",
            "kv_layouts",
            "dtype_signatures",
            "quant_specs",
            "pos_encoding_modes",
            "mask_kinds",
            "causal_values",
            "required_features",
        ):
            data[field] = tuple(data[field])
        data["quant_specs"] = tuple(QuantSpec.from_dict(item) for item in data["quant_specs"])
        data["metadata_limits"] = AttentionMetadataLimits.from_dict(data["metadata_limits"])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionCapabilityRule fields are invalid") from error


@dataclass(frozen=True)
class AttentionCapabilityRuleReport:
    rule_id: str
    accepted: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class AttentionCapabilityReport:
    profile_id: str
    accepted: bool
    global_reasons: Tuple[str, ...]
    rules: Tuple[AttentionCapabilityRuleReport, ...]

    @property
    def matching_rule_ids(self) -> Tuple[str, ...]:
        return tuple(item.rule_id for item in self.rules if item.accepted)


@dataclass(frozen=True)
class AttentionBackendCapabilityProfile:
    profile_id: str
    backend: Backend
    environment: AttentionRuntimeEnvironment
    status: AttentionCapabilityStatus
    numerics_policy: AttentionNumericsPolicy
    rules: Tuple[AttentionCapabilityRule, ...]
    evidence: Tuple[AttentionCapabilityEvidence, ...] = ()
    schema_version: int = ATTENTION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention capability profile version")
        if not _IDENTIFIER.fullmatch(self.profile_id):
            raise SchemaError("invalid Attention capability profile_id")
        object.__setattr__(self, "backend", Backend(self.backend))
        object.__setattr__(self, "status", AttentionCapabilityStatus(self.status))
        if not isinstance(self.environment, AttentionRuntimeEnvironment):
            raise TypeError("environment must be AttentionRuntimeEnvironment")
        if not isinstance(self.numerics_policy, AttentionNumericsPolicy):
            raise TypeError("numerics_policy must be AttentionNumericsPolicy")
        rules = tuple(self.rules)
        if not rules or len({item.rule_id for item in rules}) != len(rules):
            raise SchemaError("profile rules must be non-empty with unique rule_id")
        object.__setattr__(self, "rules", rules)
        evidence = tuple(self.evidence)
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise SchemaError("profile evidence_id values must be unique")
        object.__setattr__(self, "evidence", evidence)
        if self.status in {
            AttentionCapabilityStatus.FUNCTIONAL,
            AttentionCapabilityStatus.OPTIMIZED,
        }:
            if not self.environment.is_fully_pinned:
                raise SchemaError("runnable capability profile requires a fully pinned environment")
            required_rank = _STATUS_RANK[self.status]
            if not any(_STATUS_RANK[item.level] >= required_rank for item in evidence):
                raise SchemaError(
                    "runnable capability profile requires evidence at status %s"
                    % self.status.value
                )

    def explain(
        self,
        spec: AttentionPlanSpec,
        metadata: AttentionMetadata,
        observed_environment: AttentionRuntimeEnvironment,
        *,
        numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
        require_functional: bool = True,
    ) -> AttentionCapabilityReport:
        global_reasons = []
        if observed_environment.fingerprint != self.environment.fingerprint:
            global_reasons.append("runtime environment fingerprint mismatch")
        if numerics_policy.fingerprint != self.numerics_policy.fingerprint:
            global_reasons.append("Attention numerics policy fingerprint mismatch")
        if require_functional and _STATUS_RANK[self.status] < _STATUS_RANK[
            AttentionCapabilityStatus.FUNCTIONAL
        ]:
            global_reasons.append(
                "profile status %s is not runnable" % self.status.value
            )
        reports = tuple(
            AttentionCapabilityRuleReport(
                rule.rule_id,
                not (reasons := rule.unsupported_reasons(spec, metadata, self.environment)),
                reasons,
            )
            for rule in self.rules
        )
        accepted = not global_reasons and any(item.accepted for item in reports)
        return AttentionCapabilityReport(
            self.profile_id, accepted, tuple(global_reasons), reports
        )

    def select_rule(
        self,
        spec: AttentionPlanSpec,
        metadata: AttentionMetadata,
        observed_environment: AttentionRuntimeEnvironment,
        *,
        numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
    ) -> AttentionCapabilityRule:
        report = self.explain(
            spec,
            metadata,
            observed_environment,
            numerics_policy=numerics_policy,
        )
        if not report.accepted:
            reasons = list(report.global_reasons)
            for item in report.rules:
                reasons.extend("%s: %s" % (item.rule_id, reason) for reason in item.reasons)
            raise AttentionCapabilityError(
                "no Attention capability rule accepted the workload: %s"
                % ("; ".join(reasons) or "no rules")
            )
        rule_id = report.matching_rule_ids[0]
        return next(item for item in self.rules if item.rule_id == rule_id)

    def validate_evidence(
        self,
        corpus: AttentionTraceCorpus,
        policy: AttentionCoveragePolicy,
        *,
        replay: bool = False,
    ) -> AttentionCapabilityEvidence:
        if replay:
            corpus.replay_all()
        corpus_cases = {item.case_id: item for item in corpus.cases}
        candidates = tuple(
            item
            for item in self.evidence
            if item.corpus_fingerprint == corpus.fingerprint
            and item.coverage_policy_name == policy.name
        )
        if not candidates:
            raise AttentionCapabilityError("profile has no evidence for this corpus/policy")
        for item in candidates:
            if any(case_id not in corpus_cases for case_id in item.passed_case_ids):
                continue
            selected = AttentionTraceCorpus(
                name=corpus.name + "-evidence-subset",
                description="Capability evidence subset",
                cases=tuple(corpus_cases[case_id] for case_id in item.passed_case_ids),
            )
            report = policy.evaluate(selected)
            if item.covered_cells != report.covered_cells or item.total_cells != len(
                report.requirements
            ):
                continue
            matched_rules = set()
            compatible = True
            for case in selected.cases:
                compatibility = self.explain(
                    case.trace.spec,
                    case.trace.metadata,
                    self.environment,
                    require_functional=False,
                )
                if not compatibility.accepted:
                    compatible = False
                    break
                matched_rules.update(compatibility.matching_rule_ids)
            if compatible and matched_rules == {rule.rule_id for rule in self.rules}:
                return item
        raise AttentionCapabilityError("profile evidence does not match corpus coverage/cases")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "backend": self.backend.value,
            "environment": self.environment.to_dict(),
            "status": self.status.value,
            "numerics_policy": self.numerics_policy.to_dict(),
            "rules": [item.to_dict() for item in self.rules],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionBackendCapabilityProfile":
        data = _strict_fields(cls, value, "AttentionBackendCapabilityProfile")
        data["environment"] = AttentionRuntimeEnvironment.from_dict(data["environment"])
        data["numerics_policy"] = AttentionNumericsPolicy.from_dict(data["numerics_policy"])
        data["rules"] = tuple(AttentionCapabilityRule.from_dict(item) for item in data["rules"])
        data["evidence"] = tuple(AttentionCapabilityEvidence.from_dict(item) for item in data["evidence"])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionBackendCapabilityProfile fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def load_attention_capability_manifest(
    path: Union[str, Path],
) -> Tuple[AttentionBackendCapabilityProfile, ...]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise SchemaError("Attention capability manifest is unreadable") from error
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "generated_at",
        "profiles",
    }:
        raise SchemaError("Attention capability manifest fields do not match schema")
    if value["schema_version"] != ATTENTION_CAPABILITY_SCHEMA_VERSION:
        raise SchemaError("unsupported Attention capability manifest version")
    if not isinstance(value["generated_at"], str) or not value["generated_at"]:
        raise SchemaError("Attention capability manifest generated_at is invalid")
    if not isinstance(value["profiles"], list):
        raise SchemaError("Attention capability manifest profiles must be an array")
    profiles = tuple(
        AttentionBackendCapabilityProfile.from_dict(item)
        for item in value["profiles"]
    )
    if len({item.profile_id for item in profiles}) != len(profiles):
        raise SchemaError("Attention capability manifest has duplicate profile_id")
    runnable = tuple(
        item
        for item in profiles
        if _STATUS_RANK[item.status]
        >= _STATUS_RANK[AttentionCapabilityStatus.FUNCTIONAL]
    )
    if runnable:
        # Import lazily to avoid making the schema module depend on corpus
        # construction at ordinary import time.
        from .corpus_samples import build_framework_attention_corpus

        corpus = build_framework_attention_corpus()
        policy = framework_attention_coverage_policy()
        for profile in runnable:
            try:
                profile.validate_evidence(corpus, policy)
            except AttentionCapabilityError as error:
                raise SchemaError(
                    "runnable Attention profile %s has invalid packaged evidence"
                    % profile.profile_id
                ) from error
    return profiles


def validate_attention_kernel_bindings(
    profiles: Sequence[AttentionBackendCapabilityProfile],
    descriptors: Sequence[KernelDescriptor],
) -> int:
    """Cross-check runnable Attention profiles and concrete kernel artifacts.

    Returns the number of validated Attention descriptor bindings. Empty
    profiles plus empty Attention descriptors is valid and means unsupported.
    """

    profile_values = tuple(profiles)
    descriptor_values = tuple(descriptors)
    profile_map = {item.profile_id: item for item in profile_values}
    if len(profile_map) != len(profile_values):
        raise SchemaError("Attention capability profiles contain duplicate profile_id")
    validated = []
    for descriptor in descriptor_values:
        is_attention = descriptor.op.startswith("attention.")
        binding = descriptor.capability_binding
        if is_attention and descriptor.backend != Backend.REFERENCE and binding is None:
            raise SchemaError(
                "Attention kernel %s requires a capability binding"
                % descriptor.kernel_id
            )
        if binding is None:
            continue
        if binding.domain != "attention":
            if is_attention:
                raise SchemaError(
                    "Attention kernel %s must bind domain='attention'"
                    % descriptor.kernel_id
                )
            continue
        if not is_attention:
            raise SchemaError(
                "non-Attention kernel %s cannot use an Attention capability binding"
                % descriptor.kernel_id
            )
        profile = profile_map.get(binding.profile_id)
        if profile is None:
            raise SchemaError(
                "Attention kernel %s references unknown profile %s"
                % (descriptor.kernel_id, binding.profile_id)
            )
        if profile.fingerprint != binding.profile_fingerprint:
            raise SchemaError(
                "Attention kernel %s profile fingerprint mismatch"
                % descriptor.kernel_id
            )
        if _STATUS_RANK[profile.status] < _STATUS_RANK[
            AttentionCapabilityStatus.FUNCTIONAL
        ]:
            raise SchemaError(
                "Attention kernel %s binds non-runnable profile %s"
                % (descriptor.kernel_id, profile.profile_id)
            )
        if descriptor.backend != profile.backend:
            raise SchemaError(
                "Attention kernel %s backend does not match capability profile"
                % descriptor.kernel_id
            )
        if (
            descriptor.artifact is None
            or descriptor.launch_abi is None
            or descriptor.binary_abi is None
        ):
            raise SchemaError(
                "Attention kernel %s lacks artifact provenance or launch ABI"
                % descriptor.kernel_id
            )
        if descriptor.artifact.target_soc != profile.environment.soc_version:
            raise SchemaError(
                "Attention kernel %s artifact target does not match profile SoC"
                % descriptor.kernel_id
            )
        from .launch_contract import (
            ATTENTION_LAUNCH_ARGUMENT_NAMES,
            attention_kernel_binary_abi,
            attention_kernel_binary_abi_v2,
        )

        if (
            descriptor.launch_abi.argument_names != ATTENTION_LAUNCH_ARGUMENT_NAMES
            or descriptor.launch_abi.stream_argument != "stream"
            or (
                descriptor.launch_abi.abi_name,
                descriptor.binary_abi.fingerprint,
            )
            not in {
                (
                    "flashinfer_npu.attention.v1",
                    attention_kernel_binary_abi().fingerprint,
                ),
                (
                    "flashinfer_npu.attention.v2",
                    attention_kernel_binary_abi_v2().fingerprint,
                ),
            }
        ):
            raise SchemaError(
                "Attention kernel %s launch ABI is incompatible"
                % descriptor.kernel_id
            )
        rule = next(
            (item for item in profile.rules if item.rule_id == binding.rule_id),
            None,
        )
        if rule is None:
            raise SchemaError(
                "Attention kernel %s references unknown rule %s"
                % (descriptor.kernel_id, binding.rule_id)
            )
        if any(
            item.physical_layout != "logical" for item in rule.quant_specs
        ) and descriptor.binary_abi.fingerprint != attention_kernel_binary_abi_v2().fingerprint:
            raise SchemaError(
                "Attention kernel %s requires KV POD v2 for a non-logical layout"
                % descriptor.kernel_id
            )
        try:
            mode = AttentionMode(descriptor.op[len("attention.") :])
        except ValueError as error:
            raise SchemaError(
                "Attention kernel %s has an unknown mode op" % descriptor.kernel_id
            ) from error
        if mode not in rule.modes:
            raise SchemaError(
                "Attention kernel %s op is outside its capability rule"
                % descriptor.kernel_id
            )

        constraints = descriptor.constraints
        if set(constraints.supported_socs) != {profile.environment.soc_version}:
            raise SchemaError(
                "Attention kernel %s must constrain the exact profile SoC"
                % descriptor.kernel_id
            )
        if not constraints.dtype_signatures or any(
            signature not in rule.dtype_signatures
            for signature in constraints.dtype_signatures
        ):
            raise SchemaError(
                "Attention kernel %s dtype constraints exceed its rule"
                % descriptor.kernel_id
            )
        allowed_layouts = {(layout.value,) for layout in rule.kv_layouts}
        if not constraints.layout_signatures or any(
            signature not in allowed_layouts
            for signature in constraints.layout_signatures
        ):
            raise SchemaError(
                "Attention kernel %s layout constraints exceed its rule"
                % descriptor.kernel_id
            )
        rule_quant_dtypes = {
            item.storage_dtype for item in rule.quant_specs
        }
        if rule_quant_dtypes:
            if not constraints.quant_storage_dtypes or any(
                item not in rule_quant_dtypes
                for item in constraints.quant_storage_dtypes
            ):
                raise SchemaError(
                    "Attention kernel %s quant dtype constraints exceed its rule"
                    % descriptor.kernel_id
                )
        if not set(rule.required_features) <= set(constraints.required_features):
            raise SchemaError(
                "Attention kernel %s omits rule required features"
                % descriptor.kernel_id
            )
        validated.append((profile.profile_id, rule.rule_id, mode, descriptor.kernel_id))

    bound_keys = {(profile_id, rule_id, mode) for profile_id, rule_id, mode, _ in validated}
    for profile in profile_values:
        if _STATUS_RANK[profile.status] < _STATUS_RANK[
            AttentionCapabilityStatus.FUNCTIONAL
        ]:
            continue
        for rule in profile.rules:
            for mode in rule.modes:
                if (profile.profile_id, rule.rule_id, mode) not in bound_keys:
                    raise SchemaError(
                        "runnable Attention profile %s rule %s mode %s has no kernel descriptor"
                        % (profile.profile_id, rule.rule_id, mode.value)
                    )
    return len(validated)


__all__ = [
    "ATTENTION_CAPABILITY_SCHEMA_VERSION",
    "AttentionBackendCapabilityProfile",
    "AttentionCapabilityError",
    "AttentionCapabilityEvidence",
    "AttentionCapabilityReport",
    "AttentionCapabilityRule",
    "AttentionCapabilityRuleReport",
    "AttentionCapabilityStatus",
    "AttentionRuntimeEnvironment",
    "load_attention_capability_manifest",
    "validate_attention_kernel_bindings",
]
