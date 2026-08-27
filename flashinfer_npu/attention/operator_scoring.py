"""Declarative, host-only provider preference rules for Attention plans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    AttentionJsonEnvelopeUsage,
    decode_attention_json,
)
from .operator_resolver import (
    ATTENTION_OPERATOR_PLAN_SCORE_MAX,
    ATTENTION_OPERATOR_PLAN_SCORE_MIN,
    AttentionOperatorRuntimePlanScore,
)
from .planner import AttentionFrameworkPlan
from .schema import AttentionMode, KVLayout


ATTENTION_OPERATOR_SCORING_VERSION = 1
ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION = 1

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUANTIZATION_KINDS = {"any", "dense", "quantized"}
_RULE_SEQUENCE_FIELDS = (
    "modes",
    "kv_layouts",
    "dtype_signatures",
    "quant_spec_fingerprints",
    "page_sizes",
    "head_dim_qk_values",
    "head_dim_vo_values",
    "gqa_group_sizes",
    "causal_values",
    "workload_fingerprints",
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _bounded_integer(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaError("%s must be an integer" % name)
    if not (
        ATTENTION_OPERATOR_PLAN_SCORE_MIN
        <= value
        <= ATTENTION_OPERATOR_PLAN_SCORE_MAX
    ):
        raise SchemaError("%s is out of range" % name)
    return value


def _optional_non_negative(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError("%s must be a non-negative integer or None" % name)
    return value


def _enum_tuple(name: str, values: Sequence, enum_type) -> Tuple:
    try:
        result = tuple(
            sorted(
                (enum_type(item) for item in values),
                key=lambda item: item.value,
            )
        )
    except (TypeError, ValueError) as error:
        raise SchemaError("%s contains an invalid value" % name) from error
    if len(set(result)) != len(result):
        raise SchemaError("%s cannot contain duplicates" % name)
    return result


def _string_tuple(name: str, values: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(sorted(str(item) for item in values))
    if any(not item for item in result) or len(set(result)) != len(result):
        raise SchemaError("%s must contain unique non-empty values" % name)
    return result


def _integer_tuple(
    name: str, values: Sequence[int], *, allow_zero: bool
) -> Tuple[int, ...]:
    result = tuple(values)
    minimum = 0 if allow_zero else 1
    if any(
        not isinstance(item, int)
        or isinstance(item, bool)
        or item < minimum
        for item in result
    ):
        raise SchemaError("%s contains an invalid integer" % name)
    if len(set(result)) != len(result):
        raise SchemaError("%s cannot contain duplicates" % name)
    return tuple(sorted(result))


def _dtype_signatures(
    values: Sequence[Sequence[str]],
) -> Tuple[Tuple[str, str, str], ...]:
    try:
        result = tuple(
            sorted(tuple(str(item) for item in signature) for signature in values)
        )
    except TypeError as error:
        raise SchemaError("dtype_signatures must contain dtype triples") from error
    if any(len(item) != 3 or any(not value for value in item) for item in result):
        raise SchemaError("dtype_signatures must contain non-empty dtype triples")
    if len(set(result)) != len(result):
        raise SchemaError("dtype_signatures cannot contain duplicates")
    return result


@dataclass(frozen=True)
class AttentionOperatorPlanScoreRule:
    """One preference bucket; this is not a capability admission rule."""

    rule_id: str
    precedence: int
    score: int
    reason: str
    modes: Tuple[AttentionMode, ...] = ()
    kv_layouts: Tuple[KVLayout, ...] = ()
    dtype_signatures: Tuple[Tuple[str, str, str], ...] = ()
    quantization: str = "any"
    quant_spec_fingerprints: Tuple[str, ...] = ()
    page_sizes: Tuple[int, ...] = ()
    head_dim_qk_values: Tuple[int, ...] = ()
    head_dim_vo_values: Tuple[int, ...] = ()
    gqa_group_sizes: Tuple[int, ...] = ()
    causal_values: Tuple[bool, ...] = ()
    min_batch_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    min_total_qo_tokens: Optional[int] = None
    max_total_qo_tokens: Optional[int] = None
    min_total_kv_tokens: Optional[int] = None
    max_total_kv_tokens: Optional[int] = None
    workload_fingerprints: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_OPERATOR_SCORING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_SCORING_VERSION:
            raise SchemaError("unsupported Attention plan score rule version")
        if not _IDENTIFIER.fullmatch(str(self.rule_id)):
            raise SchemaError("invalid Attention plan score rule_id")
        object.__setattr__(self, "rule_id", str(self.rule_id))
        object.__setattr__(
            self, "precedence", _bounded_integer("rule precedence", self.precedence)
        )
        object.__setattr__(self, "score", _bounded_integer("rule score", self.score))
        if not str(self.reason).strip():
            raise SchemaError("Attention plan score rule reason is empty")
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self, "modes", _enum_tuple("modes", self.modes, AttentionMode)
        )
        object.__setattr__(
            self,
            "kv_layouts",
            _enum_tuple("kv_layouts", self.kv_layouts, KVLayout),
        )
        object.__setattr__(
            self, "dtype_signatures", _dtype_signatures(self.dtype_signatures)
        )
        quantization = str(self.quantization)
        if quantization not in _QUANTIZATION_KINDS:
            raise SchemaError("invalid Attention score rule quantization kind")
        object.__setattr__(self, "quantization", quantization)
        quant_fingerprints = _string_tuple(
            "quant_spec_fingerprints", self.quant_spec_fingerprints
        )
        if any(not _SHA256.fullmatch(item) for item in quant_fingerprints):
            raise SchemaError("quant_spec_fingerprints must be lowercase SHA-256")
        if quant_fingerprints and quantization != "quantized":
            raise SchemaError(
                "quant_spec_fingerprints require quantization='quantized'"
            )
        object.__setattr__(self, "quant_spec_fingerprints", quant_fingerprints)
        for name, allow_zero in (
            ("page_sizes", True),
            ("head_dim_qk_values", False),
            ("head_dim_vo_values", False),
            ("gqa_group_sizes", False),
        ):
            object.__setattr__(
                self,
                name,
                _integer_tuple(name, getattr(self, name), allow_zero=allow_zero),
            )
        causal_values = tuple(self.causal_values)
        if any(not isinstance(item, bool) for item in causal_values):
            raise SchemaError("causal_values must contain booleans")
        if len(set(causal_values)) != len(causal_values):
            raise SchemaError("causal_values cannot contain duplicates")
        object.__setattr__(self, "causal_values", tuple(sorted(causal_values)))
        for name in (
            "min_batch_size",
            "max_batch_size",
            "min_total_qo_tokens",
            "max_total_qo_tokens",
            "min_total_kv_tokens",
            "max_total_kv_tokens",
        ):
            object.__setattr__(
                self, name, _optional_non_negative(name, getattr(self, name))
            )
        for minimum_name, maximum_name in (
            ("min_batch_size", "max_batch_size"),
            ("min_total_qo_tokens", "max_total_qo_tokens"),
            ("min_total_kv_tokens", "max_total_kv_tokens"),
        ):
            minimum = getattr(self, minimum_name)
            maximum = getattr(self, maximum_name)
            if minimum is not None and maximum is not None and minimum > maximum:
                raise SchemaError("Attention score rule range is inverted")
        workload_fingerprints = _string_tuple(
            "workload_fingerprints", self.workload_fingerprints
        )
        if any(not _SHA256.fullmatch(item) for item in workload_fingerprints):
            raise SchemaError("workload_fingerprints must be lowercase SHA-256")
        object.__setattr__(self, "workload_fingerprints", workload_fingerprints)
        if not self._has_match_constraint:
            raise SchemaError(
                "Attention plan score rule must declare a match constraint"
            )

    @property
    def _has_match_constraint(self) -> bool:
        return bool(
            self.modes
            or self.kv_layouts
            or self.dtype_signatures
            or self.quantization != "any"
            or self.page_sizes
            or self.head_dim_qk_values
            or self.head_dim_vo_values
            or self.gqa_group_sizes
            or self.causal_values
            or self.min_batch_size is not None
            or self.max_batch_size is not None
            or self.min_total_qo_tokens is not None
            or self.max_total_qo_tokens is not None
            or self.min_total_kv_tokens is not None
            or self.max_total_kv_tokens is not None
            or self.workload_fingerprints
        )

    @staticmethod
    def _in_range(value: int, minimum: Optional[int], maximum: Optional[int]) -> bool:
        return (minimum is None or value >= minimum) and (
            maximum is None or value <= maximum
        )

    def matches(self, plan: AttentionFrameworkPlan) -> bool:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        spec = plan.spec
        workload = plan.workload
        if len(workload.static_dims) < 5 or len(workload.dynamic_bounds) < 3:
            raise SchemaError("Attention plan scoring workload is incomplete")
        page_size = workload.static_dims[4]
        total_kv_tokens = workload.dynamic_bounds[2]
        quant_spec = spec.kv_quant_spec
        if self.modes and spec.mode not in self.modes:
            return False
        if self.kv_layouts and spec.kv_layout not in self.kv_layouts:
            return False
        signature = (spec.q_dtype, str(spec.kv_dtype), str(spec.o_dtype))
        if self.dtype_signatures and signature not in self.dtype_signatures:
            return False
        if self.quantization == "dense" and quant_spec is not None:
            return False
        if self.quantization == "quantized" and quant_spec is None:
            return False
        if self.quant_spec_fingerprints and (
            quant_spec is None
            or quant_spec.fingerprint not in self.quant_spec_fingerprints
        ):
            return False
        if self.page_sizes and page_size not in self.page_sizes:
            return False
        if (
            self.head_dim_qk_values
            and spec.head_dim_qk not in self.head_dim_qk_values
        ):
            return False
        if (
            self.head_dim_vo_values
            and int(spec.head_dim_vo) not in self.head_dim_vo_values
        ):
            return False
        group_size = spec.num_qo_heads // spec.num_kv_heads
        if self.gqa_group_sizes and group_size not in self.gqa_group_sizes:
            return False
        if self.causal_values and spec.effective_causal not in self.causal_values:
            return False
        if not self._in_range(
            plan.batch_size, self.min_batch_size, self.max_batch_size
        ):
            return False
        if not self._in_range(
            plan.total_qo_tokens,
            self.min_total_qo_tokens,
            self.max_total_qo_tokens,
        ):
            return False
        if not self._in_range(
            total_kv_tokens,
            self.min_total_kv_tokens,
            self.max_total_kv_tokens,
        ):
            return False
        if (
            self.workload_fingerprints
            and workload.fingerprint not in self.workload_fingerprints
        ):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "precedence": self.precedence,
            "score": self.score,
            "reason": self.reason,
            "modes": [item.value for item in self.modes],
            "kv_layouts": [item.value for item in self.kv_layouts],
            "dtype_signatures": [list(item) for item in self.dtype_signatures],
            "quantization": self.quantization,
            "quant_spec_fingerprints": list(self.quant_spec_fingerprints),
            "page_sizes": list(self.page_sizes),
            "head_dim_qk_values": list(self.head_dim_qk_values),
            "head_dim_vo_values": list(self.head_dim_vo_values),
            "gqa_group_sizes": list(self.gqa_group_sizes),
            "causal_values": list(self.causal_values),
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "min_total_qo_tokens": self.min_total_qo_tokens,
            "max_total_qo_tokens": self.max_total_qo_tokens,
            "min_total_kv_tokens": self.min_total_kv_tokens,
            "max_total_kv_tokens": self.max_total_kv_tokens,
            "workload_fingerprints": list(self.workload_fingerprints),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionOperatorPlanScoreRule":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention plan score rule fields are invalid")
        for name in (
            "modes",
            "kv_layouts",
            "dtype_signatures",
            "quant_spec_fingerprints",
            "page_sizes",
            "head_dim_qk_values",
            "head_dim_vo_values",
            "gqa_group_sizes",
            "causal_values",
            "workload_fingerprints",
        ):
            data[name] = tuple(data[name])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention plan score rule fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionOperatorPlanScoringError(RuntimeError):
    """Raised when a declarative scorer cannot make one deterministic choice."""


@dataclass(frozen=True)
class AttentionOperatorPlanScoringPolicy:
    """Identity-bound scorer composed only from serializable plan predicates."""

    policy_id: str
    provider_id: str
    operation_id: str
    rules: Tuple[AttentionOperatorPlanScoreRule, ...] = ()
    default_score: int = 0
    default_reason: str = "no declarative scoring rule matched"
    schema_version: int = ATTENTION_OPERATOR_SCORING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_SCORING_VERSION:
            raise SchemaError("unsupported Attention plan scoring policy version")
        if not _IDENTIFIER.fullmatch(str(self.policy_id)):
            raise SchemaError("invalid Attention scoring policy policy_id")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid Attention scoring policy provider_id")
        if not str(self.operation_id) or any(
            item.isspace() for item in str(self.operation_id)
        ):
            raise SchemaError("invalid Attention scoring policy operation_id")
        object.__setattr__(self, "policy_id", str(self.policy_id))
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))
        rules = tuple(self.rules)
        if any(not isinstance(item, AttentionOperatorPlanScoreRule) for item in rules):
            raise TypeError("scoring policy rules must contain plan score rules")
        if len({item.rule_id for item in rules}) != len(rules):
            raise SchemaError("scoring policy rule_id values must be unique")
        object.__setattr__(
            self,
            "rules",
            tuple(sorted(rules, key=lambda item: (-item.precedence, item.rule_id))),
        )
        object.__setattr__(
            self,
            "default_score",
            _bounded_integer("default plan score", self.default_score),
        )
        if not str(self.default_reason).strip():
            raise SchemaError("default plan score reason is empty")
        object.__setattr__(self, "default_reason", str(self.default_reason))

    def score(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionOperatorRuntimePlanScore:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        if not str(device):
            raise SchemaError("Attention plan scoring device must be non-empty")
        matches = tuple(rule for rule in self.rules if rule.matches(plan))
        if not matches:
            return AttentionOperatorRuntimePlanScore(
                self.default_score,
                "declarative:%s:%s:default" % (self.policy_id, self.fingerprint),
                self.default_reason,
            )
        precedence = max(item.precedence for item in matches)
        finalists = tuple(item for item in matches if item.precedence == precedence)
        if len(finalists) != 1:
            raise AttentionOperatorPlanScoringError(
                "ambiguous Attention plan score rules at precedence %d: %s"
                % (precedence, ", ".join(item.rule_id for item in finalists))
            )
        selected = finalists[0]
        return AttentionOperatorRuntimePlanScore(
            selected.score,
            "declarative:%s:%s:%s"
            % (self.policy_id, self.fingerprint, selected.rule_id),
            selected.reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "rules": [item.to_dict() for item in self.rules],
            "default_score": self.default_score,
            "default_reason": self.default_reason,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionOperatorPlanScoringPolicy":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention plan scoring policy fields are invalid")
        try:
            data["rules"] = tuple(
                AttentionOperatorPlanScoreRule.from_dict(item)
                for item in data["rules"]
            )
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention plan scoring policy fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorPlanScoringManifestLimits:
    """Semantic construction limits layered over the generic JSON envelope."""

    max_policies: int = 64
    max_rules_per_policy: int = 2048
    max_total_rules: int = 8192
    max_values_per_predicate: int = 4096
    max_total_predicate_values: int = 262144
    schema_version: int = ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION:
            raise SchemaError("unsupported Attention scoring manifest limits version")
        for name in self.__dataclass_fields__:
            if name == "schema_version":
                continue
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SchemaError("%s must be a positive integer" % name)


DEFAULT_ATTENTION_OPERATOR_PLAN_SCORING_MANIFEST_LIMITS = (
    AttentionOperatorPlanScoringManifestLimits()
)


def _validate_manifest_shape(
    value: Mapping[str, Any],
    limits: AttentionOperatorPlanScoringManifestLimits,
) -> None:
    policies = value.get("policies")
    if not isinstance(policies, list):
        raise SchemaError("Attention scoring manifest policies must be an array")
    if len(policies) > limits.max_policies:
        raise SchemaError(
            "Attention scoring manifest policies exceed limit %d"
            % limits.max_policies
        )
    total_rules = 0
    total_predicate_values = 0
    for policy_index, policy in enumerate(policies):
        if not isinstance(policy, Mapping):
            raise SchemaError(
                "Attention scoring manifest policy %d must be an object"
                % policy_index
            )
        rules = policy.get("rules")
        if not isinstance(rules, list):
            raise SchemaError(
                "Attention scoring manifest policy rules must be an array"
            )
        if len(rules) > limits.max_rules_per_policy:
            raise SchemaError(
                "Attention scoring policy rules exceed per-policy limit %d"
                % limits.max_rules_per_policy
            )
        total_rules += len(rules)
        if total_rules > limits.max_total_rules:
            raise SchemaError(
                "Attention scoring manifest rules exceed total limit %d"
                % limits.max_total_rules
            )
        for rule_index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                raise SchemaError(
                    "Attention scoring rule %d must be an object" % rule_index
                )
            for field_name in _RULE_SEQUENCE_FIELDS:
                values = rule.get(field_name)
                if not isinstance(values, list):
                    raise SchemaError(
                        "Attention scoring rule %s must be an array" % field_name
                    )
                if len(values) > limits.max_values_per_predicate:
                    raise SchemaError(
                        "Attention scoring predicate %s exceeds limit %d"
                        % (field_name, limits.max_values_per_predicate)
                    )
                total_predicate_values += len(values)
                if (
                    total_predicate_values
                    > limits.max_total_predicate_values
                ):
                    raise SchemaError(
                        "Attention scoring predicate values exceed total limit %d"
                        % limits.max_total_predicate_values
                    )


@dataclass(frozen=True)
class AttentionOperatorPlanScoringManifest:
    """Bounded collection with at most one policy per provider operation."""

    manifest_id: str
    policies: Tuple[AttentionOperatorPlanScoringPolicy, ...]
    schema_version: int = ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION
    kind: str = field(
        default="attention_operator_plan_scoring_manifest",
        init=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION:
            raise SchemaError("unsupported Attention plan scoring manifest version")
        if self.kind != "attention_operator_plan_scoring_manifest":
            raise SchemaError("Attention plan scoring manifest kind is invalid")
        if not _IDENTIFIER.fullmatch(str(self.manifest_id)):
            raise SchemaError("invalid Attention plan scoring manifest_id")
        policies = tuple(self.policies)
        if not policies or any(
            not isinstance(item, AttentionOperatorPlanScoringPolicy)
            for item in policies
        ):
            raise TypeError("Attention scoring manifest must contain policies")
        policy_ids = tuple(item.policy_id for item in policies)
        identities = tuple(
            (item.provider_id, item.operation_id) for item in policies
        )
        if len(set(policy_ids)) != len(policy_ids):
            raise SchemaError("Attention scoring manifest policy_id values duplicate")
        if len(set(identities)) != len(identities):
            raise SchemaError(
                "Attention scoring manifest provider operation identities duplicate"
            )
        object.__setattr__(self, "manifest_id", str(self.manifest_id))
        object.__setattr__(
            self,
            "policies",
            tuple(
                sorted(
                    policies,
                    key=lambda item: (
                        item.provider_id,
                        item.operation_id,
                        item.policy_id,
                    ),
                )
            ),
        )

    @property
    def policy_ids(self) -> Tuple[str, ...]:
        return tuple(item.policy_id for item in self.policies)

    def get(
        self, provider_id: str, operation_id: str
    ) -> AttentionOperatorPlanScoringPolicy:
        identity = (str(provider_id), str(operation_id))
        for policy in self.policies:
            if (policy.provider_id, policy.operation_id) == identity:
                return policy
        raise SchemaError(
            "unknown Attention scoring policy identity %r" % (identity,)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "manifest_id": self.manifest_id,
            "policies": [item.to_dict() for item in self.policies],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        limits: AttentionOperatorPlanScoringManifestLimits = (
            DEFAULT_ATTENTION_OPERATOR_PLAN_SCORING_MANIFEST_LIMITS
        ),
    ) -> "AttentionOperatorPlanScoringManifest":
        if not isinstance(limits, AttentionOperatorPlanScoringManifestLimits):
            raise TypeError(
                "limits must be AttentionOperatorPlanScoringManifestLimits"
            )
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention plan scoring manifest fields are invalid")
        if data.pop("kind") != "attention_operator_plan_scoring_manifest":
            raise SchemaError("Attention plan scoring manifest kind is invalid")
        _validate_manifest_shape(data, limits)
        try:
            data["policies"] = tuple(
                AttentionOperatorPlanScoringPolicy.from_dict(item)
                for item in data["policies"]
            )
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "Attention plan scoring manifest fields are invalid"
            ) from error

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def binding(self) -> "AttentionOperatorPlanScoringManifestBinding":
        return AttentionOperatorPlanScoringManifestBinding.from_manifest(self)


@dataclass(frozen=True)
class AttentionOperatorPlanScoringManifestBinding:
    """Non-executable scoring authority stored with a registry generation."""

    manifest_id: str
    manifest_fingerprint: str
    policy_bindings: Tuple[Tuple[str, str, str, str], ...]
    schema_version: int = ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION:
            raise SchemaError(
                "unsupported Attention scoring manifest binding version"
            )
        if not _IDENTIFIER.fullmatch(str(self.manifest_id)):
            raise SchemaError("invalid Attention scoring manifest binding id")
        if not _SHA256.fullmatch(str(self.manifest_fingerprint)):
            raise SchemaError(
                "Attention scoring manifest binding fingerprint is invalid"
            )
        bindings = tuple(tuple(item) for item in self.policy_bindings)
        if not bindings or any(len(item) != 4 for item in bindings):
            raise SchemaError(
                "Attention scoring manifest policy bindings are invalid"
            )
        normalized = []
        for provider_id, operation_id, policy_id, policy_fingerprint in bindings:
            provider_id = str(provider_id)
            operation_id = str(operation_id)
            policy_id = str(policy_id)
            policy_fingerprint = str(policy_fingerprint)
            if not _PROVIDER_ID.fullmatch(provider_id):
                raise SchemaError(
                    "Attention scoring manifest binding provider is invalid"
                )
            if not operation_id or any(item.isspace() for item in operation_id):
                raise SchemaError(
                    "Attention scoring manifest binding operation is invalid"
                )
            if not _IDENTIFIER.fullmatch(policy_id):
                raise SchemaError(
                    "Attention scoring manifest binding policy id is invalid"
                )
            if not _SHA256.fullmatch(policy_fingerprint):
                raise SchemaError(
                    "Attention scoring manifest policy fingerprint is invalid"
                )
            normalized.append(
                (provider_id, operation_id, policy_id, policy_fingerprint)
            )
        identities = tuple(item[:2] for item in normalized)
        policy_ids = tuple(item[2] for item in normalized)
        if len(set(identities)) != len(identities):
            raise SchemaError(
                "Attention scoring manifest binding identities duplicate"
            )
        if len(set(policy_ids)) != len(policy_ids):
            raise SchemaError(
                "Attention scoring manifest binding policy ids duplicate"
            )
        object.__setattr__(self, "manifest_id", str(self.manifest_id))
        object.__setattr__(
            self,
            "manifest_fingerprint",
            str(self.manifest_fingerprint),
        )
        object.__setattr__(
            self,
            "policy_bindings",
            tuple(sorted(normalized, key=lambda item: item[:3])),
        )

    @classmethod
    def from_manifest(
        cls, manifest: AttentionOperatorPlanScoringManifest
    ) -> "AttentionOperatorPlanScoringManifestBinding":
        if not isinstance(manifest, AttentionOperatorPlanScoringManifest):
            raise TypeError(
                "manifest must be AttentionOperatorPlanScoringManifest"
            )
        return cls(
            manifest_id=manifest.manifest_id,
            manifest_fingerprint=manifest.fingerprint,
            policy_bindings=tuple(
                (
                    policy.provider_id,
                    policy.operation_id,
                    policy.policy_id,
                    policy.fingerprint,
                )
                for policy in manifest.policies
            ),
        )

    @property
    def identities(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(item[:2] for item in self.policy_bindings)

    def policy_fingerprint(self, provider_id: str, operation_id: str) -> str:
        identity = (str(provider_id), str(operation_id))
        for candidate_provider, candidate_operation, _, fingerprint in (
            self.policy_bindings
        ):
            if (candidate_provider, candidate_operation) == identity:
                return fingerprint
        raise SchemaError(
            "unknown Attention scoring manifest binding identity %r"
            % (identity,)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "policy_bindings": [list(item) for item in self.policy_bindings],
        }


def load_attention_operator_plan_scoring_manifest(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    manifest_limits: AttentionOperatorPlanScoringManifestLimits = (
        DEFAULT_ATTENTION_OPERATOR_PLAN_SCORING_MANIFEST_LIMITS
    ),
) -> Tuple[AttentionOperatorPlanScoringManifest, AttentionJsonEnvelopeUsage]:
    """Decode bounded strict JSON before constructing policy/rule objects."""

    if not isinstance(
        manifest_limits, AttentionOperatorPlanScoringManifestLimits
    ):
        raise TypeError(
            "manifest_limits must be AttentionOperatorPlanScoringManifestLimits"
        )
    decoded, usage = decode_attention_json(value, limits=limits)
    if not isinstance(decoded, Mapping):
        raise SchemaError("Attention scoring manifest root must be an object")
    return (
        AttentionOperatorPlanScoringManifest.from_dict(
            decoded,
            limits=manifest_limits,
        ),
        usage,
    )


__all__ = [
    "ATTENTION_OPERATOR_SCORING_MANIFEST_VERSION",
    "ATTENTION_OPERATOR_SCORING_VERSION",
    "DEFAULT_ATTENTION_OPERATOR_PLAN_SCORING_MANIFEST_LIMITS",
    "AttentionOperatorPlanScoreRule",
    "AttentionOperatorPlanScoringError",
    "AttentionOperatorPlanScoringManifest",
    "AttentionOperatorPlanScoringManifestBinding",
    "AttentionOperatorPlanScoringManifestLimits",
    "AttentionOperatorPlanScoringPolicy",
    "load_attention_operator_plan_scoring_manifest",
]
