"""Versioned accuracy budgets separating quantization and backend error."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .reference import (
    ReferenceAttentionResult,
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceTensor,
)
from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    decode_attention_json,
)
from .trace import AttentionTrace


ATTENTION_ACCURACY_SCHEMA_VERSION = 1
ATTENTION_ACCURACY_CORPUS_SCHEMA_VERSION = 1
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


class AttentionAccuracyExpectationError(AssertionError):
    """Raised when replay does not produce a case's declared verdict."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _encode_float(value: float) -> Any:
    number = float(value)
    if math.isnan(number):
        return {"nonfinite": "nan"}
    if number == float("inf"):
        return {"nonfinite": "+inf"}
    if number == float("-inf"):
        return {"nonfinite": "-inf"}
    return number


def _same_number(left: float, right: float) -> bool:
    if math.isnan(left):
        return math.isnan(right)
    return left == right


@dataclass(frozen=True)
class AttentionErrorTolerance:
    """Element-wise ``atol + rtol * abs(reference)`` acceptance rule."""

    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        for name in ("atol", "rtol"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise SchemaError("%s must be finite and non-negative" % name)
            object.__setattr__(self, name, value)

    def to_dict(self) -> Dict[str, float]:
        return {"atol": self.atol, "rtol": self.rtol}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionErrorTolerance":
        data = dict(value)
        if set(data) != {"atol", "rtol"}:
            raise SchemaError("AttentionErrorTolerance fields do not match schema")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionErrorTolerance fields are invalid") from error


@dataclass(frozen=True)
class AttentionAccuracyBudget:
    """Independent tolerances for storage quantization and backend execution."""

    quantization_output: AttentionErrorTolerance = field(
        default_factory=AttentionErrorTolerance
    )
    quantization_lse: AttentionErrorTolerance = field(
        default_factory=AttentionErrorTolerance
    )
    backend_output: AttentionErrorTolerance = field(
        default_factory=AttentionErrorTolerance
    )
    backend_lse: AttentionErrorTolerance = field(
        default_factory=AttentionErrorTolerance
    )
    schema_version: int = ATTENTION_ACCURACY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_ACCURACY_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention accuracy budget version")
        for name in (
            "quantization_output",
            "quantization_lse",
            "backend_output",
            "backend_lse",
        ):
            if not isinstance(getattr(self, name), AttentionErrorTolerance):
                raise SchemaError("%s must be AttentionErrorTolerance" % name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "quantization_output": self.quantization_output.to_dict(),
            "quantization_lse": self.quantization_lse.to_dict(),
            "backend_output": self.backend_output.to_dict(),
            "backend_lse": self.backend_lse.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAccuracyBudget":
        data = dict(value)
        expected = {
            "schema_version",
            "quantization_output",
            "quantization_lse",
            "backend_output",
            "backend_lse",
        }
        if set(data) != expected:
            raise SchemaError("AttentionAccuracyBudget fields do not match schema")
        try:
            return cls(
                schema_version=data["schema_version"],
                quantization_output=AttentionErrorTolerance.from_dict(
                    data["quantization_output"]
                ),
                quantization_lse=AttentionErrorTolerance.from_dict(
                    data["quantization_lse"]
                ),
                backend_output=AttentionErrorTolerance.from_dict(
                    data["backend_output"]
                ),
                backend_lse=AttentionErrorTolerance.from_dict(
                    data["backend_lse"]
                ),
            )
        except (KeyError, TypeError) as error:
            raise SchemaError("AttentionAccuracyBudget fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class AttentionTensorErrorMetrics:
    numel: int
    finite_pairs: int
    nonfinite_matches: int
    nonfinite_mismatches: int
    tolerance_violations: int
    max_abs_error: float
    max_relative_error: float
    rmse: float

    def __post_init__(self) -> None:
        counts = (
            self.numel,
            self.finite_pairs,
            self.nonfinite_matches,
            self.nonfinite_mismatches,
            self.tolerance_violations,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise SchemaError("accuracy metric counts must be non-negative integers")
        if self.finite_pairs + self.nonfinite_matches + self.nonfinite_mismatches != self.numel:
            raise SchemaError("accuracy metric counts do not sum to numel")
        for name in ("max_abs_error", "max_relative_error", "rmse"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise SchemaError("%s must be finite and non-negative" % name)
            object.__setattr__(self, name, value)

    @property
    def within_budget(self) -> bool:
        return self.tolerance_violations == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numel": self.numel,
            "finite_pairs": self.finite_pairs,
            "nonfinite_matches": self.nonfinite_matches,
            "nonfinite_mismatches": self.nonfinite_mismatches,
            "tolerance_violations": self.tolerance_violations,
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
            "rmse": self.rmse,
            "within_budget": self.within_budget,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionTensorErrorMetrics":
        data = dict(value)
        expected = {
            "numel",
            "finite_pairs",
            "nonfinite_matches",
            "nonfinite_mismatches",
            "tolerance_violations",
            "max_abs_error",
            "max_relative_error",
            "rmse",
            "within_budget",
        }
        if set(data) != expected:
            raise SchemaError("AttentionTensorErrorMetrics fields do not match schema")
        declared = data.pop("within_budget")
        try:
            result = cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionTensorErrorMetrics fields are invalid") from error
        if type(declared) is not bool or declared != result.within_budget:
            raise SchemaError("AttentionTensorErrorMetrics within_budget is inconsistent")
        return result


@dataclass(frozen=True)
class AttentionResultErrorMetrics:
    output: AttentionTensorErrorMetrics
    lse: Optional[AttentionTensorErrorMetrics]

    @property
    def within_budget(self) -> bool:
        return self.output.within_budget and (
            self.lse is None or self.lse.within_budget
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output.to_dict(),
            "lse": self.lse.to_dict() if self.lse is not None else None,
            "within_budget": self.within_budget,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionResultErrorMetrics":
        data = dict(value)
        if set(data) != {"output", "lse", "within_budget"}:
            raise SchemaError("AttentionResultErrorMetrics fields do not match schema")
        if not isinstance(data["output"], Mapping):
            raise SchemaError("Attention result output metrics must be an object")
        lse_value = data["lse"]
        if lse_value is not None and not isinstance(lse_value, Mapping):
            raise SchemaError("Attention result LSE metrics must be an object or null")
        result = cls(
            output=AttentionTensorErrorMetrics.from_dict(data["output"]),
            lse=(
                AttentionTensorErrorMetrics.from_dict(lse_value)
                if lse_value is not None
                else None
            ),
        )
        declared = data["within_budget"]
        if type(declared) is not bool or declared != result.within_budget:
            raise SchemaError("AttentionResultErrorMetrics within_budget is inconsistent")
        return result


@dataclass(frozen=True)
class AttentionAccuracyReport:
    dense_input_fingerprint: str
    quantized_input_fingerprint: str
    candidate_result_fingerprint: str
    budget: AttentionAccuracyBudget
    quantization: AttentionResultErrorMetrics
    backend: AttentionResultErrorMetrics
    schema_version: int = ATTENTION_ACCURACY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_ACCURACY_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention accuracy report version")
        for name in (
            "dense_input_fingerprint",
            "quantized_input_fingerprint",
            "candidate_result_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise SchemaError("%s must be a lowercase SHA-256 digest" % name)

    @property
    def quantization_within_budget(self) -> bool:
        return self.quantization.within_budget

    @property
    def backend_within_budget(self) -> bool:
        return self.backend.within_budget

    @property
    def passes(self) -> bool:
        return self.quantization_within_budget and self.backend_within_budget

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dense_input_fingerprint": self.dense_input_fingerprint,
            "quantized_input_fingerprint": self.quantized_input_fingerprint,
            "candidate_result_fingerprint": self.candidate_result_fingerprint,
            "budget": self.budget.to_dict(),
            "quantization": self.quantization.to_dict(),
            "backend": self.backend.to_dict(),
            "quantization_within_budget": self.quantization_within_budget,
            "backend_within_budget": self.backend_within_budget,
            "passes": self.passes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAccuracyReport":
        data = dict(value)
        expected = {
            "schema_version",
            "dense_input_fingerprint",
            "quantized_input_fingerprint",
            "candidate_result_fingerprint",
            "budget",
            "quantization",
            "backend",
            "quantization_within_budget",
            "backend_within_budget",
            "passes",
        }
        if set(data) != expected:
            raise SchemaError("AttentionAccuracyReport fields do not match schema")
        for name in ("budget", "quantization", "backend"):
            if not isinstance(data[name], Mapping):
                raise SchemaError("AttentionAccuracyReport %s must be an object" % name)
        result = cls(
            schema_version=data["schema_version"],
            dense_input_fingerprint=data["dense_input_fingerprint"],
            quantized_input_fingerprint=data["quantized_input_fingerprint"],
            candidate_result_fingerprint=data["candidate_result_fingerprint"],
            budget=AttentionAccuracyBudget.from_dict(data["budget"]),
            quantization=AttentionResultErrorMetrics.from_dict(
                data["quantization"]
            ),
            backend=AttentionResultErrorMetrics.from_dict(data["backend"]),
        )
        derived = {
            "quantization_within_budget": result.quantization_within_budget,
            "backend_within_budget": result.backend_within_budget,
            "passes": result.passes,
        }
        for name, actual in derived.items():
            declared = data[name]
            if type(declared) is not bool or declared != actual:
                raise SchemaError("AttentionAccuracyReport %s is inconsistent" % name)
        return result

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class AttentionAccuracyCase:
    case_id: str
    dense_trace: AttentionTrace
    quantized_trace: AttentionTrace
    budget: AttentionAccuracyBudget
    expect_quantization_pass: bool
    description: str = ""

    def __post_init__(self) -> None:
        if not _CASE_ID.fullmatch(self.case_id):
            raise SchemaError("Attention accuracy case_id must match %s" % _CASE_ID.pattern)
        if type(self.expect_quantization_pass) is not bool:
            raise SchemaError("expect_quantization_pass must be bool")
        _validate_trace_pair(self.dense_trace, self.quantized_trace)

    def evaluate(self) -> AttentionAccuracyReport:
        report = evaluate_attention_accuracy(
            self.dense_trace, self.quantized_trace, budget=self.budget
        )
        if report.quantization_within_budget != self.expect_quantization_pass:
            raise AttentionAccuracyExpectationError(
                "accuracy case %s expected quantization pass=%s, observed %s"
                % (
                    self.case_id,
                    self.expect_quantization_pass,
                    report.quantization_within_budget,
                )
            )
        if not report.backend_within_budget:
            raise AttentionAccuracyExpectationError(
                "accuracy case %s self-reference backend comparison failed"
                % self.case_id
            )
        return report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "dense_trace": self.dense_trace.to_dict(),
            "quantized_trace": self.quantized_trace.to_dict(),
            "budget": self.budget.to_dict(),
            "expect_quantization_pass": self.expect_quantization_pass,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAccuracyCase":
        data = dict(value)
        expected = {
            "case_id",
            "description",
            "dense_trace",
            "quantized_trace",
            "budget",
            "expect_quantization_pass",
        }
        if set(data) != expected:
            raise SchemaError("AttentionAccuracyCase fields do not match schema")
        for name in ("dense_trace", "quantized_trace", "budget"):
            if not isinstance(data[name], Mapping):
                raise SchemaError("AttentionAccuracyCase %s must be an object" % name)
        return cls(
            case_id=str(data["case_id"]),
            description=str(data["description"]),
            dense_trace=AttentionTrace.from_dict(data["dense_trace"]),
            quantized_trace=AttentionTrace.from_dict(data["quantized_trace"]),
            budget=AttentionAccuracyBudget.from_dict(data["budget"]),
            expect_quantization_pass=data["expect_quantization_pass"],
        )


@dataclass(frozen=True)
class AttentionAccuracyCorpus:
    name: str
    cases: Tuple[AttentionAccuracyCase, ...]
    description: str = ""
    schema_version: int = ATTENTION_ACCURACY_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_ACCURACY_CORPUS_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention accuracy corpus version")
        if not self.name:
            raise SchemaError("Attention accuracy corpus name must be non-empty")
        object.__setattr__(self, "cases", tuple(self.cases))
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise SchemaError("Attention accuracy corpus case_id values must be unique")
        pairs = tuple(
            (case.dense_trace.input_fingerprint, case.quantized_trace.input_fingerprint)
            for case in self.cases
        )
        if len(set(pairs)) != len(pairs):
            raise SchemaError("Attention accuracy corpus trace pairs must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_accuracy_corpus",
            "name": self.name,
            "description": self.description,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAccuracyCorpus":
        data = dict(value)
        if set(data) != {"schema_version", "kind", "name", "description", "cases"}:
            raise SchemaError("AttentionAccuracyCorpus fields do not match schema")
        if data["kind"] != "attention_accuracy_corpus":
            raise SchemaError("unsupported Attention accuracy corpus kind")
        if not isinstance(data["cases"], (list, tuple)):
            raise SchemaError("Attention accuracy corpus cases must be an array")
        return cls(
            schema_version=int(data["schema_version"]),
            name=str(data["name"]),
            description=str(data["description"]),
            cases=tuple(AttentionAccuracyCase.from_dict(case) for case in data["cases"]),
        )

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    ) -> "AttentionAccuracyCorpus":
        decoded, _usage = decode_attention_json(value, limits=limits)
        if not isinstance(decoded, Mapping):
            raise SchemaError("Attention accuracy corpus JSON root must be an object")
        return cls.from_dict(decoded)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def replay_all(self) -> Tuple[Tuple[str, AttentionAccuracyReport], ...]:
        return tuple((case.case_id, case.evaluate()) for case in self.cases)


def _compare_tensor(
    expected: ReferenceTensor,
    observed: ReferenceTensor,
    tolerance: AttentionErrorTolerance,
    name: str,
) -> AttentionTensorErrorMetrics:
    if expected.shape != observed.shape:
        raise SchemaError("%s shape mismatch" % name)
    if expected.dtype != observed.dtype:
        raise SchemaError("%s dtype mismatch" % name)

    finite_pairs = 0
    nonfinite_matches = 0
    nonfinite_mismatches = 0
    violations = 0
    max_abs = 0.0
    max_relative = 0.0
    squared_error = 0.0
    for reference, candidate in zip(expected.data, observed.data):
        if not math.isfinite(reference) or not math.isfinite(candidate):
            if _same_number(reference, candidate):
                nonfinite_matches += 1
            else:
                nonfinite_mismatches += 1
                violations += 1
            continue
        finite_pairs += 1
        error = abs(candidate - reference)
        denominator = max(abs(reference), abs(candidate))
        relative = error / denominator if denominator else 0.0
        max_abs = max(max_abs, error)
        max_relative = max(max_relative, relative)
        squared_error += error * error
        if error > tolerance.atol + tolerance.rtol * abs(reference):
            violations += 1
    return AttentionTensorErrorMetrics(
        numel=len(expected.data),
        finite_pairs=finite_pairs,
        nonfinite_matches=nonfinite_matches,
        nonfinite_mismatches=nonfinite_mismatches,
        tolerance_violations=violations,
        max_abs_error=max_abs,
        max_relative_error=max_relative,
        rmse=math.sqrt(squared_error / finite_pairs) if finite_pairs else 0.0,
    )


def compare_attention_results(
    expected: ReferenceAttentionResult,
    observed: ReferenceAttentionResult,
    *,
    output_tolerance: AttentionErrorTolerance = AttentionErrorTolerance(),
    lse_tolerance: AttentionErrorTolerance = AttentionErrorTolerance(),
) -> AttentionResultErrorMetrics:
    """Compare two results without requiring their tensors to share a device."""

    if not isinstance(expected, ReferenceAttentionResult) or not isinstance(
        observed, ReferenceAttentionResult
    ):
        raise TypeError("expected and observed must be ReferenceAttentionResult")
    output = _compare_tensor(
        expected.output, observed.output, output_tolerance, "Attention output"
    )
    if (expected.lse is None) != (observed.lse is None):
        raise SchemaError("Attention LSE presence mismatch")
    lse = (
        _compare_tensor(expected.lse, observed.lse, lse_tolerance, "Attention LSE")
        if expected.lse is not None and observed.lse is not None
        else None
    )
    return AttentionResultErrorMetrics(output, lse)


def _tensor_identity_equal(left: ReferenceTensor, right: ReferenceTensor) -> bool:
    return (
        left.spec == right.spec
        and len(left.data) == len(right.data)
        and all(_same_number(a, b) for a, b in zip(left.data, right.data))
    )


def _validate_trace_pair(dense: AttentionTrace, quantized: AttentionTrace) -> None:
    if not isinstance(dense.kv_data, ReferenceKVData):
        raise SchemaError("dense trace must contain unquantized ReferenceKVData")
    if not isinstance(quantized.kv_data, ReferenceQuantizedKVData):
        raise SchemaError("quantized trace must contain ReferenceQuantizedKVData")

    dense_spec = dense.spec.to_dict()
    quantized_spec = quantized.spec.to_dict()
    for field_name in ("kv_dtype", "kv_quant_spec"):
        dense_spec.pop(field_name)
        quantized_spec.pop(field_name)
    if dense_spec != quantized_spec:
        raise SchemaError("dense and quantized Attention plan semantics do not match")
    if dense.metadata.to_dict() != quantized.metadata.to_dict():
        raise SchemaError("dense and quantized Attention metadata do not match")
    if not _tensor_identity_equal(dense.q, quantized.q):
        raise SchemaError("dense and quantized Attention query inputs do not match")

    dense_cache = dense.kv_data.spec.to_dict()
    quantized_cache = quantized.kv_data.spec.to_dict()
    for field_name in ("dtype", "quant_spec"):
        dense_cache.pop(field_name)
        quantized_cache.pop(field_name)
    if dense_cache != quantized_cache:
        raise SchemaError("dense and quantized KV logical cache specs do not match")

    auxiliary_names = (
        "custom_mask_data",
        "alibi_slopes",
        "q_scale",
        "k_scale",
        "v_scale",
        "logits_soft_cap",
        "return_lse",
    )
    if any(getattr(dense, name) != getattr(quantized, name) for name in auxiliary_names):
        raise SchemaError("dense and quantized Attention runtime inputs do not match")


def attention_result_fingerprint(result: ReferenceAttentionResult) -> str:
    """Return a JSON-safe identity for an Attention result, including non-finites."""

    if not isinstance(result, ReferenceAttentionResult):
        raise TypeError("result must be ReferenceAttentionResult")

    def tensor_dict(tensor: Optional[ReferenceTensor]) -> Any:
        if tensor is None:
            return None
        return {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "data": [_encode_float(value) for value in tensor.data],
        }

    value = {"output": tensor_dict(result.output), "lse": tensor_dict(result.lse)}
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def evaluate_attention_accuracy(
    dense_trace: AttentionTrace,
    quantized_trace: AttentionTrace,
    *,
    budget: AttentionAccuracyBudget,
    candidate: Optional[ReferenceAttentionResult] = None,
) -> AttentionAccuracyReport:
    """Evaluate quantization drift and candidate backend drift independently."""

    if not isinstance(dense_trace, AttentionTrace) or not isinstance(
        quantized_trace, AttentionTrace
    ):
        raise TypeError("dense_trace and quantized_trace must be AttentionTrace")
    if not isinstance(budget, AttentionAccuracyBudget):
        raise TypeError("budget must be AttentionAccuracyBudget")
    _validate_trace_pair(dense_trace, quantized_trace)
    dense_result = dense_trace.replay()
    quantized_result = quantized_trace.replay()
    candidate_result = quantized_result if candidate is None else candidate
    quantization_metrics = compare_attention_results(
        dense_result,
        quantized_result,
        output_tolerance=budget.quantization_output,
        lse_tolerance=budget.quantization_lse,
    )
    backend_metrics = compare_attention_results(
        quantized_result,
        candidate_result,
        output_tolerance=budget.backend_output,
        lse_tolerance=budget.backend_lse,
    )
    return AttentionAccuracyReport(
        dense_input_fingerprint=dense_trace.input_fingerprint,
        quantized_input_fingerprint=quantized_trace.input_fingerprint,
        candidate_result_fingerprint=attention_result_fingerprint(candidate_result),
        budget=budget,
        quantization=quantization_metrics,
        backend=backend_metrics,
    )


__all__ = [
    "ATTENTION_ACCURACY_SCHEMA_VERSION",
    "ATTENTION_ACCURACY_CORPUS_SCHEMA_VERSION",
    "AttentionAccuracyBudget",
    "AttentionAccuracyCase",
    "AttentionAccuracyCorpus",
    "AttentionAccuracyExpectationError",
    "AttentionAccuracyReport",
    "AttentionErrorTolerance",
    "AttentionResultErrorMetrics",
    "AttentionTensorErrorMetrics",
    "compare_attention_results",
    "attention_result_fingerprint",
    "evaluate_attention_accuracy",
]
