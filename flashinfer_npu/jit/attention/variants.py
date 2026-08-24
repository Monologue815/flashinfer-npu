"""Attention specialization identity for generated Ascend modules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from flashinfer_npu.attention.schema import AttentionPlanSpec
from flashinfer_npu.runtime import SchemaError


ATTENTION_JIT_VARIANT_SCHEMA_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionJitVariant:
    """Static Attention choices that may affect generated code."""

    mode: str
    q_dtype: str
    kv_dtype: str
    o_dtype: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    kv_layout: str
    pos_encoding_mode: str
    mask_mode: str
    window_left: int
    window_right: int
    q_len_per_req: int
    logits_soft_cap_enabled: bool
    use_fp16_qk_reduction: bool
    use_profiler: bool
    kv_quant_spec_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_JIT_VARIANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_VARIANT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported Attention JIT variant schema version"
            )
        for name in (
            "mode",
            "q_dtype",
            "kv_dtype",
            "o_dtype",
            "kv_layout",
            "pos_encoding_mode",
            "mask_mode",
        ):
            if not str(getattr(self, name)):
                raise SchemaError("Attention JIT variant %s must be non-empty" % name)
        for name in (
            "num_qo_heads",
            "num_kv_heads",
            "head_dim_qk",
            "head_dim_vo",
            "q_len_per_req",
        ):
            if getattr(self, name) <= 0:
                raise SchemaError("Attention JIT variant %s must be positive" % name)
        if self.mask_mode not in {"none", "causal", "custom", "custom_packed"}:
            raise SchemaError("unsupported Attention JIT mask mode")
        if self.kv_quant_spec_fingerprint is not None and (
            len(self.kv_quant_spec_fingerprint) != 64
            or any(
                item not in "0123456789abcdef"
                for item in self.kv_quant_spec_fingerprint
            )
        ):
            raise SchemaError("Attention JIT quant fingerprint must be lowercase SHA-256")

    @classmethod
    def from_plan_spec(cls, spec: AttentionPlanSpec) -> "AttentionJitVariant":
        if not isinstance(spec, AttentionPlanSpec):
            raise TypeError("Attention JIT variant requires AttentionPlanSpec")
        if spec.custom_mask is not None:
            mask_mode = "custom_packed" if spec.custom_mask.packed else "custom"
        elif spec.effective_causal:
            mask_mode = "causal"
        else:
            mask_mode = "none"
        return cls(
            mode=spec.mode.value,
            q_dtype=spec.q_dtype,
            kv_dtype=str(spec.kv_dtype),
            o_dtype=str(spec.o_dtype),
            num_qo_heads=spec.num_qo_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim_qk=spec.head_dim_qk,
            head_dim_vo=int(spec.head_dim_vo),
            kv_layout=spec.kv_layout.value,
            pos_encoding_mode=spec.pos_encoding_mode.value,
            mask_mode=mask_mode,
            window_left=spec.window_left,
            window_right=spec.window_right,
            q_len_per_req=spec.q_len_per_req,
            logits_soft_cap_enabled=float(spec.logits_soft_cap or 0.0) > 0,
            use_fp16_qk_reduction=spec.use_fp16_qk_reduction,
            use_profiler=spec.use_profiler,
            kv_quant_spec_fingerprint=(
                spec.kv_quant_spec.fingerprint
                if spec.kv_quant_spec is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = ["ATTENTION_JIT_VARIANT_SCHEMA_VERSION", "AttentionJitVariant"]
