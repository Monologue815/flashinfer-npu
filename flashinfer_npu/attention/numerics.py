"""Versioned numerical edge policy for Attention conformance oracles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from flashinfer_npu.runtime import SchemaError


ATTENTION_NUMERICS_POLICY_VERSION = 1


@dataclass(frozen=True)
class AttentionNumericsPolicy:
    """Fixed v1 row-wise softmax behavior for exceptional logits.

    This is a conformance contract, not a claim about an untested device kernel.
    Backends must report a compatibility gap until they pass these cases.
    """

    nan_logits: str = "propagate_row"
    positive_infinity_logits: str = "uniform_over_positive_infinity"
    negative_infinity_row: str = "zero_output_negative_infinity_lse"
    empty_row: str = "zero_output_negative_infinity_lse"
    schema_version: int = ATTENTION_NUMERICS_POLICY_VERSION

    def __post_init__(self) -> None:
        expected = {
            "nan_logits": "propagate_row",
            "positive_infinity_logits": "uniform_over_positive_infinity",
            "negative_infinity_row": "zero_output_negative_infinity_lse",
            "empty_row": "zero_output_negative_infinity_lse",
        }
        if self.schema_version != ATTENTION_NUMERICS_POLICY_VERSION:
            raise SchemaError("unsupported Attention numerics policy version")
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise SchemaError(
                    "Attention numerics policy v1 requires %s=%r" % (name, value)
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nan_logits": self.nan_logits,
            "positive_infinity_logits": self.positive_infinity_logits,
            "negative_infinity_row": self.negative_infinity_row,
            "empty_row": self.empty_row,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionNumericsPolicy":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "AttentionNumericsPolicy fields do not match schema version 1"
            )
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionNumericsPolicy fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_ATTENTION_NUMERICS_POLICY = AttentionNumericsPolicy()


def normalize_attention_logits(
    logits: Sequence[float],
    policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
) -> Tuple[Tuple[float, ...], float]:
    """Return deterministic probabilities and LSE under policy v1.

    An empty probability tuple represents a zero-output row.  NaN rows return
    NaN probabilities so the value reduction deterministically taints every
    output component, independent of where the NaN appeared in ``logits``.
    """

    # Validate an injected policy even though v1 currently has one legal value.
    if not isinstance(policy, AttentionNumericsPolicy):
        raise TypeError("policy must be AttentionNumericsPolicy")
    values = tuple(float(value) for value in logits)
    if not values:
        return (), float("-inf")
    if any(math.isnan(value) for value in values):
        return (float("nan"),) * len(values), float("nan")
    positive_infinity = tuple(
        index for index, value in enumerate(values) if value == float("inf")
    )
    if positive_infinity:
        probability = 1.0 / len(positive_infinity)
        selected = set(positive_infinity)
        return (
            tuple(
                probability if index in selected else 0.0
                for index in range(len(values))
            ),
            float("inf"),
        )
    max_logit = max(values)
    if max_logit == float("-inf"):
        return (), float("-inf")
    exponentials = tuple(math.exp(value - max_logit) for value in values)
    denominator = sum(exponentials)
    return (
        tuple(value / denominator for value in exponentials),
        max_logit + math.log(denominator),
    )


__all__ = [
    "ATTENTION_NUMERICS_POLICY_VERSION",
    "AttentionNumericsPolicy",
    "DEFAULT_ATTENTION_NUMERICS_POLICY",
    "normalize_attention_logits",
]
