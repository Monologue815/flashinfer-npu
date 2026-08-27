"""Versioned Attention trace corpora and auditable coverage policies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .reference import ReferenceAttentionResult, ReferenceQuantizedKVData
from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    decode_attention_json,
)
from .schema import (
    AttentionMode,
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVMetadata,
)
from .trace import AttentionTrace


ATTENTION_CORPUS_SCHEMA_VERSION = 1
ATTENTION_CORPUS_FLOAT_SIGNIFICANT_DIGITS = 15
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


def _canonicalize_corpus_numbers(value: Any) -> Any:
    """Remove non-semantic Host/libm noise from finite corpus floats."""

    if isinstance(value, Mapping):
        return {
            key: _canonicalize_corpus_numbers(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_corpus_numbers(item) for item in value]
    if isinstance(value, float) and math.isfinite(value):
        normalized = float(
            format(value, ".%dg" % ATTENTION_CORPUS_FLOAT_SIGNIFICANT_DIGITS)
        )
        return 0.0 if normalized == 0.0 else normalized
    return value


def _canonical_json(value: Mapping[str, Any], *, indent: Optional[int] = None) -> str:
    return json.dumps(
        _canonicalize_corpus_numbers(value),
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        ensure_ascii=True,
        allow_nan=False,
        indent=indent,
    )


@dataclass(frozen=True)
class AttentionTraceCase:
    case_id: str
    trace: AttentionTrace
    description: str = ""

    def __post_init__(self) -> None:
        if not _CASE_ID.fullmatch(self.case_id):
            raise SchemaError(
                "Attention trace case_id must match %s" % _CASE_ID.pattern
            )
        if self.trace.expected_output is None:
            raise SchemaError("corpus cases must contain expected output")
        if self.trace.return_lse and self.trace.expected_lse is None:
            raise SchemaError("corpus cases with return_lse require expected LSE")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "trace": self.trace.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionTraceCase":
        data = dict(value)
        if set(data) != {"case_id", "description", "trace"}:
            raise SchemaError("Attention trace case fields do not match schema")
        if not isinstance(data["trace"], Mapping):
            raise SchemaError("Attention trace case trace must be a dictionary")
        return cls(
            case_id=str(data["case_id"]),
            description=str(data["description"]),
            trace=AttentionTrace.from_dict(data["trace"]),
        )


@dataclass(frozen=True)
class AttentionTraceCorpus:
    name: str
    cases: Tuple[AttentionTraceCase, ...]
    description: str = ""
    schema_version: int = ATTENTION_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_CORPUS_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention corpus schema version")
        if not self.name:
            raise SchemaError("Attention corpus name must be non-empty")
        object.__setattr__(self, "cases", tuple(self.cases))
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise SchemaError("Attention corpus case_id values must be unique")
        input_ids = tuple(case.trace.input_fingerprint for case in self.cases)
        if len(set(input_ids)) != len(input_ids):
            raise SchemaError("Attention corpus inputs must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_conformance_corpus",
            "name": self.name,
            "description": self.description,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionTraceCorpus":
        data = dict(value)
        if set(data) != {
            "schema_version",
            "kind",
            "name",
            "description",
            "cases",
        }:
            raise SchemaError("Attention corpus fields do not match schema")
        if data["kind"] != "attention_conformance_corpus":
            raise SchemaError("unsupported Attention corpus kind")
        if not isinstance(data["cases"], (list, tuple)):
            raise SchemaError("Attention corpus cases must be an array")
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            cases=tuple(AttentionTraceCase.from_dict(case) for case in data["cases"]),
            schema_version=int(data["schema_version"]),
        )

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return _canonical_json(self.to_dict(), indent=indent)

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    ) -> "AttentionTraceCorpus":
        decoded, _usage = decode_attention_json(value, limits=limits)
        if not isinstance(decoded, Mapping):
            raise SchemaError("Attention corpus JSON root must be an object")
        return cls.from_dict(decoded)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def replay_all(
        self, *, atol: float = 1e-6, rtol: float = 1e-6
    ) -> Tuple[Tuple[str, ReferenceAttentionResult], ...]:
        return tuple(
            (
                case.case_id,
                case.trace.replay(atol=atol, rtol=rtol),
            )
            for case in self.cases
        )


def attention_trace_features(trace: AttentionTrace) -> Tuple[Tuple[str, str], ...]:
    """Return canonical, data-independent coverage features for one trace."""

    spec = trace.spec
    metadata = trace.metadata
    if spec.mode in {AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE}:
        cache_kind = "single"
        has_empty_request = False
    elif isinstance(metadata, RaggedKVMetadata):
        cache_kind = "ragged"
        has_empty_request = any(length == 0 for length in metadata.kv_lengths)
    else:
        cache_kind = "paged"
        if isinstance(metadata, PagedPrefillMetadata):
            lengths = metadata.paged_kv.sequence_lengths
        elif isinstance(metadata, PagedKVMetadata):
            lengths = metadata.sequence_lengths
        elif isinstance(metadata, MixedPagedKVMetadata):
            lengths = metadata.kv_len_arr
        else:
            lengths = ()
        has_empty_request = any(length == 0 for length in lengths)

    if isinstance(trace.kv_data, ReferenceQuantizedKVData):
        quant = trace.kv_data.spec.quant_spec
        if quant is None:  # Construction already prevents this; keep extraction total.
            raise SchemaError("quantized trace KV is missing QuantSpec")
        storage = quant.storage_dtype
        quant_scheme = quant.scheme
        quant_granularity = quant.granularity
        quantized = "true"
    else:
        storage = "dense"
        quant_scheme = "none"
        quant_granularity = "none"
        quantized = "false"

    if spec.custom_mask is None:
        mask = "none"
    else:
        mask = "packed" if spec.custom_mask.packed else "unpacked"

    features = {
        "mode": spec.mode.value,
        "kv_layout": spec.kv_layout.value,
        "cache_kind": cache_kind,
        "kv_storage": storage,
        "quant_scheme": quant_scheme,
        "quant_granularity": quant_granularity,
        "quantized": quantized,
        "causal": "true" if spec.causal else "false",
        "position_encoding": spec.pos_encoding_mode.value,
        "mask": mask,
        "head_mapping": (
            "mha" if spec.num_qo_heads == spec.num_kv_heads else "gqa"
        ),
        "head_dim_relation": (
            "same" if spec.head_dim_qk == spec.head_dim_vo else "different"
        ),
        "empty_request": "true" if has_empty_request else "false",
        "return_lse": "true" if trace.return_lse else "false",
        "window": "sliding" if spec.window_left >= 0 else "full",
        "soft_cap": (
            "runtime"
            if trace.logits_soft_cap > 0
            else "planned"
            if spec.logits_soft_cap > 0
            else "disabled"
        ),
        "kv_runtime_scale": (
            "per_head"
            if isinstance(trace.k_scale, tuple)
            or isinstance(trace.v_scale, tuple)
            else "scalar"
        ),
    }
    if trace.expected_lse is None:
        numerical_edge = "unobserved"
    elif any(math.isnan(value) for value in trace.expected_lse.data):
        numerical_edge = "nan"
    elif any(value == float("inf") for value in trace.expected_lse.data):
        numerical_edge = "positive_infinity"
    elif any(value == float("-inf") for value in trace.expected_lse.data):
        if has_empty_request:
            numerical_edge = "empty_request"
        elif spec.custom_mask is not None:
            numerical_edge = "zero_support_mask"
        else:
            numerical_edge = "negative_infinity"
    else:
        numerical_edge = "finite"
    features["numerical_edge"] = numerical_edge
    return tuple(sorted(features.items()))


@dataclass(frozen=True)
class AttentionCoverageCell:
    name: str
    selectors: Tuple[Tuple[str, str], ...]
    min_cases: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("coverage cell name must be non-empty")
        object.__setattr__(
            self,
            "selectors",
            tuple((str(key), str(value)) for key, value in self.selectors),
        )
        if not self.selectors:
            raise SchemaError("coverage cell requires at least one selector")
        if len({key for key, _ in self.selectors}) != len(self.selectors):
            raise SchemaError("coverage cell selector keys must be unique")
        if self.min_cases < 1:
            raise SchemaError("coverage cell min_cases must be positive")

    def matches(self, features: Mapping[str, str]) -> bool:
        return all(features.get(key) == value for key, value in self.selectors)


@dataclass(frozen=True)
class AttentionCoveragePolicy:
    name: str
    cells: Tuple[AttentionCoverageCell, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("coverage policy name must be non-empty")
        object.__setattr__(self, "cells", tuple(self.cells))
        names = tuple(cell.name for cell in self.cells)
        if len(set(names)) != len(names):
            raise SchemaError("coverage policy cell names must be unique")

    def evaluate(self, corpus: AttentionTraceCorpus) -> "AttentionCoverageReport":
        case_features = tuple(
            (case.case_id, dict(attention_trace_features(case.trace)))
            for case in corpus.cases
        )
        matches = tuple(
            (
                cell.name,
                tuple(
                    case_id
                    for case_id, features in case_features
                    if cell.matches(features)
                ),
            )
            for cell in self.cells
        )
        return AttentionCoverageReport(
            corpus_name=corpus.name,
            corpus_fingerprint=corpus.fingerprint,
            policy_name=self.name,
            matches=matches,
            requirements=tuple((cell.name, cell.min_cases) for cell in self.cells),
        )


@dataclass(frozen=True)
class AttentionCoverageReport:
    corpus_name: str
    corpus_fingerprint: str
    policy_name: str
    matches: Tuple[Tuple[str, Tuple[str, ...]], ...]
    requirements: Tuple[Tuple[str, int], ...]

    @property
    def missing_cells(self) -> Tuple[str, ...]:
        counts = {name: len(case_ids) for name, case_ids in self.matches}
        return tuple(
            name
            for name, minimum in self.requirements
            if counts.get(name, 0) < minimum
        )

    @property
    def is_complete(self) -> bool:
        return not self.missing_cells

    @property
    def covered_cells(self) -> int:
        return len(self.requirements) - len(self.missing_cells)

    def to_dict(self) -> Dict[str, Any]:
        minimums = dict(self.requirements)
        return {
            "corpus_name": self.corpus_name,
            "corpus_fingerprint": self.corpus_fingerprint,
            "policy_name": self.policy_name,
            "covered_cells": self.covered_cells,
            "total_cells": len(self.requirements),
            "complete": self.is_complete,
            "cells": [
                {
                    "name": name,
                    "min_cases": minimums[name],
                    "case_ids": list(case_ids),
                    "covered": len(case_ids) >= minimums[name],
                }
                for name, case_ids in self.matches
            ],
        }

    def format(self) -> str:
        lines = [
            "Attention conformance coverage",
            "  corpus: %s" % self.corpus_name,
            "  policy: %s" % self.policy_name,
            "  cells:  %d/%d" % (self.covered_cells, len(self.requirements)),
            "  complete: %s" % ("yes" if self.is_complete else "no"),
        ]
        if self.missing_cells:
            lines.append("  missing:")
            lines.extend("    - %s" % name for name in self.missing_cells)
        return "\n".join(lines)


def framework_attention_coverage_policy() -> AttentionCoveragePolicy:
    """P0 framework cells; each is independently auditable, not a cross-product."""

    cells = []

    def add_dimension(dimension: str, values: Tuple[str, ...]) -> None:
        for value in values:
            cells.append(
                AttentionCoverageCell(
                    "%s=%s" % (dimension, value), ((dimension, value),)
                )
            )

    add_dimension("mode", tuple(mode.value for mode in AttentionMode))
    add_dimension("kv_layout", ("NHD", "HND"))
    add_dimension("cache_kind", ("single", "ragged", "paged"))
    add_dimension("kv_storage", ("dense", "int8", "uint8", "int4_packed"))
    add_dimension("quant_scheme", ("symmetric", "asymmetric"))
    add_dimension("quant_granularity", ("tensor", "token", "group", "channel"))
    add_dimension("causal", ("false", "true"))
    add_dimension("position_encoding", ("NONE", "ROPE_LLAMA", "ALIBI"))
    add_dimension("mask", ("none", "unpacked", "packed"))
    add_dimension("head_mapping", ("mha", "gqa"))
    add_dimension("head_dim_relation", ("same", "different"))
    add_dimension("empty_request", ("true",))
    add_dimension("window", ("sliding",))
    add_dimension("soft_cap", ("runtime",))
    add_dimension("kv_runtime_scale", ("per_head",))
    add_dimension(
        "numerical_edge",
        (
            "finite",
            "nan",
            "positive_infinity",
            "negative_infinity",
            "zero_support_mask",
            "empty_request",
        ),
    )
    cells.extend(
        (
            AttentionCoverageCell(
                "quantized-single",
                (("cache_kind", "single"), ("quantized", "true")),
            ),
            AttentionCoverageCell(
                "quantized-ragged",
                (("cache_kind", "ragged"), ("quantized", "true")),
            ),
            AttentionCoverageCell(
                "quantized-paged",
                (("cache_kind", "paged"), ("quantized", "true")),
            ),
            AttentionCoverageCell(
                "paged-int4",
                (("cache_kind", "paged"), ("kv_storage", "int4_packed")),
            ),
            AttentionCoverageCell(
                "groupwise-decode",
                (
                    ("mode", AttentionMode.BATCH_DECODE_PAGED.value),
                    ("quant_granularity", "group"),
                ),
            ),
            AttentionCoverageCell(
                "paged-quantized-gqa-distinct-dims",
                (
                    ("cache_kind", "paged"),
                    ("quantized", "true"),
                    ("head_mapping", "gqa"),
                    ("head_dim_relation", "different"),
                ),
            ),
            AttentionCoverageCell(
                "mixed-int4-window-softcap-per-head-scale",
                (
                    ("mode", AttentionMode.BATCH_MIXED_PAGED.value),
                    ("kv_storage", "int4_packed"),
                    ("window", "sliding"),
                    ("soft_cap", "runtime"),
                    ("kv_runtime_scale", "per_head"),
                ),
            ),
            AttentionCoverageCell(
                "mixed-asymmetric-uint8-channel-per-head-scale",
                (
                    ("mode", AttentionMode.BATCH_MIXED_PAGED.value),
                    ("kv_storage", "uint8"),
                    ("quant_scheme", "asymmetric"),
                    ("quant_granularity", "channel"),
                    ("kv_runtime_scale", "per_head"),
                ),
            ),
        )
    )
    return AttentionCoveragePolicy("attention-framework-v4", tuple(cells))


__all__ = [
    "ATTENTION_CORPUS_SCHEMA_VERSION",
    "AttentionCoverageCell",
    "AttentionCoveragePolicy",
    "AttentionCoverageReport",
    "AttentionTraceCase",
    "AttentionTraceCorpus",
    "attention_trace_features",
    "framework_attention_coverage_policy",
]
