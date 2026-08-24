"""Explicit plan-time resource admission limits for Attention metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .schema import (
    AttentionMetadata,
    AttentionPlanSpec,
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVMetadata,
    SingleAttentionMetadata,
)


ATTENTION_RESOURCE_LIMITS_VERSION = 1


class AttentionResourceLimitError(SchemaError):
    """Raised when valid metadata exceeds an explicit admission limit."""


@dataclass(frozen=True)
class AttentionResourceUsage:
    batch_size: int
    total_qo_tokens: int
    total_kv_tokens: int
    total_pages: int
    max_pages_per_request: int
    page_size: int
    max_qo_tokens_per_request: int
    max_kv_tokens_per_request: int
    custom_mask_bytes: int

    def to_dict(self) -> Dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


def _maximum(values: Tuple[int, ...]) -> int:
    return max(values, default=0)


def measure_attention_resources(
    spec: AttentionPlanSpec, metadata: AttentionMetadata
) -> AttentionResourceUsage:
    """Measure logical metadata resources without allocating proportional data."""

    spec.validate_metadata(metadata)
    page_counts: Tuple[int, ...] = ()
    page_size = 0
    if isinstance(metadata, SingleAttentionMetadata):
        qo_lengths = (metadata.qo_len,)
        kv_lengths = (metadata.kv_len,)
    elif isinstance(metadata, PagedKVMetadata):
        qo_lengths = (spec.q_len_per_req,) * metadata.batch_size
        kv_lengths = metadata.sequence_lengths
        page_counts = metadata.page_counts
        page_size = metadata.page_size
    elif isinstance(metadata, PagedPrefillMetadata):
        qo_lengths = metadata.qo_lengths
        kv_lengths = metadata.paged_kv.sequence_lengths
        page_counts = metadata.paged_kv.page_counts
        page_size = metadata.paged_kv.page_size
    elif isinstance(metadata, RaggedKVMetadata):
        qo_lengths = metadata.qo_lengths
        kv_lengths = metadata.kv_lengths
    elif isinstance(metadata, MixedPagedKVMetadata):
        qo_lengths = metadata.qo_lengths
        kv_lengths = metadata.kv_len_arr
        page_counts = metadata.page_counts
        page_size = metadata.page_size
    else:  # pragma: no cover - closed union defense
        raise TypeError("unsupported Attention metadata")

    mask_bytes = 0
    if spec.custom_mask is not None:
        # bool and uint8 both occupy one byte in the frozen frontend contract.
        mask_bytes = int(spec.custom_mask.numel)
    return AttentionResourceUsage(
        batch_size=len(qo_lengths),
        total_qo_tokens=sum(qo_lengths),
        total_kv_tokens=sum(kv_lengths),
        total_pages=sum(page_counts),
        max_pages_per_request=_maximum(page_counts),
        page_size=page_size,
        max_qo_tokens_per_request=_maximum(qo_lengths),
        max_kv_tokens_per_request=_maximum(kv_lengths),
        custom_mask_bytes=mask_bytes,
    )


@dataclass(frozen=True)
class AttentionMetadataLimits:
    """Versioned, backend-supplied metadata admission limits.

    ``None`` means unbounded for that dimension.  The framework intentionally
    provides no arbitrary global production limit; a backend or deployment
    profile must supply the values it can safely support.
    """

    max_batch_size: Optional[int] = None
    max_total_qo_tokens: Optional[int] = None
    max_total_kv_tokens: Optional[int] = None
    max_total_pages: Optional[int] = None
    max_pages_per_request: Optional[int] = None
    max_page_size: Optional[int] = None
    max_qo_tokens_per_request: Optional[int] = None
    max_kv_tokens_per_request: Optional[int] = None
    max_custom_mask_bytes: Optional[int] = None
    schema_version: int = ATTENTION_RESOURCE_LIMITS_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_RESOURCE_LIMITS_VERSION:
            raise SchemaError("unsupported Attention metadata limits version")
        for name in self.__dataclass_fields__:
            if name == "schema_version":
                continue
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise SchemaError("%s must be an integer or None" % name)
                if value < 0:
                    raise SchemaError("%s cannot be negative" % name)

    def validate(
        self, spec: AttentionPlanSpec, metadata: AttentionMetadata
    ) -> AttentionResourceUsage:
        usage = measure_attention_resources(spec, metadata)
        pairs = (
            ("batch_size", self.max_batch_size),
            ("total_qo_tokens", self.max_total_qo_tokens),
            ("total_kv_tokens", self.max_total_kv_tokens),
            ("total_pages", self.max_total_pages),
            ("max_pages_per_request", self.max_pages_per_request),
            ("page_size", self.max_page_size),
            ("max_qo_tokens_per_request", self.max_qo_tokens_per_request),
            ("max_kv_tokens_per_request", self.max_kv_tokens_per_request),
            ("custom_mask_bytes", self.max_custom_mask_bytes),
        )
        for usage_name, limit in pairs:
            actual = getattr(usage, usage_name)
            if limit is not None and actual > limit:
                raise AttentionResourceLimitError(
                    "Attention resource %s=%d exceeds limit %d"
                    % (usage_name, actual, limit)
                )
        return usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionMetadataLimits":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "AttentionMetadataLimits fields do not match schema version 1"
            )
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionMetadataLimits fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


UNBOUNDED_ATTENTION_METADATA_LIMITS = AttentionMetadataLimits()


__all__ = [
    "ATTENTION_RESOURCE_LIMITS_VERSION",
    "AttentionMetadataLimits",
    "AttentionResourceLimitError",
    "AttentionResourceUsage",
    "UNBOUNDED_ATTENTION_METADATA_LIMITS",
    "measure_attention_resources",
]
