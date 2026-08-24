"""Versioned, JSON-safe Attention conformance traces for Host replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    decode_attention_json,
)
from .planner import AttentionFrameworkSession
from .reference import (
    ReferenceAttentionExecutor,
    ReferenceAttentionResult,
    ReferenceKVData,
    ReferenceKVInput,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
)
from .schema import (
    AttentionMetadata,
    AttentionPlanSpec,
    KVCacheSpec,
    PagedKVCacheSpec,
    PosEncodingMode,
    RaggedKVCacheSpec,
    attention_metadata_from_dict,
)


ATTENTION_TRACE_SCHEMA_VERSION = 1
TraceScale = Union[float, Tuple[float, ...]]


class AttentionTraceMismatchError(AssertionError):
    """Raised when replayed output differs from the recorded oracle output."""


def _canonical_json(value: Mapping[str, Any], *, indent: Optional[int] = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        ensure_ascii=True,
        allow_nan=False,
        indent=indent,
    )


def _encode_float(value: float) -> Union[float, Dict[str, str]]:
    number = float(value)
    if math.isnan(number):
        return {"nonfinite": "nan"}
    if number == float("inf"):
        return {"nonfinite": "+inf"}
    if number == float("-inf"):
        return {"nonfinite": "-inf"}
    return number


def _decode_float(value: Any, name: str) -> float:
    if isinstance(value, Mapping):
        if set(value) != {"nonfinite"}:
            raise SchemaError("%s has an invalid non-finite encoding" % name)
        encoded = value["nonfinite"]
        decoded = {
            "nan": float("nan"),
            "+inf": float("inf"),
            "-inf": float("-inf"),
        }.get(encoded)
        if decoded is None:
            raise SchemaError("%s has an unknown non-finite value" % name)
        return decoded
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SchemaError("%s must be numeric" % name) from error


def _tensor_to_dict(value: ReferenceTensor) -> Dict[str, Any]:
    return {
        "shape": list(value.shape),
        "data": [_encode_float(item) for item in value.data],
        "dtype": value.dtype,
        "device": value.device,
    }


def _tensor_from_dict(value: Mapping[str, Any], name: str) -> ReferenceTensor:
    data = dict(value)
    if set(data) != {"shape", "data", "dtype", "device"}:
        raise SchemaError("%s tensor fields do not match the trace schema" % name)
    shape = data["shape"]
    values = data["data"]
    if not isinstance(shape, (list, tuple)) or not isinstance(values, (list, tuple)):
        raise SchemaError("%s shape/data must be arrays" % name)
    return ReferenceTensor(
        tuple(shape),
        tuple(_decode_float(item, "%s.data" % name) for item in values),
        str(data["dtype"]),
        str(data["device"]),
    )


def _cache_spec_from_dict(value: Mapping[str, Any]) -> KVCacheSpec:
    kind = value.get("kind")
    if kind == "paged":
        return PagedKVCacheSpec.from_dict(value)
    if kind == "ragged":
        return RaggedKVCacheSpec.from_dict(value)
    raise SchemaError("unsupported KV cache kind in trace: %r" % kind)


def _quantized_tensor_to_dict(value: ReferenceQuantizedTensor) -> Dict[str, Any]:
    return {
        "logical_shape": list(value.logical_shape),
        "storage": _tensor_to_dict(value.storage),
        "scale": _tensor_to_dict(value.scale),
        "zero_point": (
            _tensor_to_dict(value.zero_point) if value.zero_point is not None else None
        ),
        "quant_spec": value.quant_spec.to_dict(),
    }


def _quantized_tensor_from_dict(
    value: Mapping[str, Any], name: str
) -> ReferenceQuantizedTensor:
    data = dict(value)
    expected_fields = {
        "logical_shape",
        "storage",
        "scale",
        "zero_point",
        "quant_spec",
    }
    if set(data) != expected_fields:
        raise SchemaError("%s quantized tensor fields do not match the schema" % name)
    for field in ("storage", "scale", "quant_spec"):
        if not isinstance(data[field], Mapping):
            raise SchemaError("%s.%s must be a dictionary" % (name, field))
    zero_value = data["zero_point"]
    if zero_value is not None and not isinstance(zero_value, Mapping):
        raise SchemaError("%s.zero_point must be a dictionary or null" % name)
    return ReferenceQuantizedTensor(
        logical_shape=tuple(data["logical_shape"]),
        storage=_tensor_from_dict(data["storage"], "%s.storage" % name),
        scale=_tensor_from_dict(data["scale"], "%s.scale" % name),
        zero_point=(
            _tensor_from_dict(zero_value, "%s.zero_point" % name)
            if zero_value is not None
            else None
        ),
        quant_spec=QuantSpec.from_dict(data["quant_spec"]),
    )


def _kv_to_dict(value: ReferenceKVInput) -> Dict[str, Any]:
    if isinstance(value, ReferenceQuantizedKVData):
        return {
            "kind": "quantized",
            "spec": value.spec.to_dict(),
            "key": _quantized_tensor_to_dict(value.key_data),
            "value": _quantized_tensor_to_dict(value.value_data),
        }
    return {
        "kind": "dense",
        "spec": value.spec.to_dict(),
        "tensors": [_tensor_to_dict(item) for item in value.tensors],
    }


def _kv_from_dict(value: Mapping[str, Any]) -> ReferenceKVInput:
    data = dict(value)
    kind = data.get("kind")
    spec_value = data.get("spec")
    if not isinstance(spec_value, Mapping):
        raise SchemaError("trace KV data requires a cache spec dictionary")
    spec = _cache_spec_from_dict(spec_value)
    if kind == "dense":
        if set(data) != {"kind", "spec", "tensors"}:
            raise SchemaError("dense KV trace fields do not match the schema")
        tensors = data["tensors"]
        if not isinstance(tensors, (list, tuple)):
            raise SchemaError("dense KV tensors must be an array")
        return ReferenceKVData(
            spec,
            tuple(
                _tensor_from_dict(item, "kv.tensors[%d]" % index)
                for index, item in enumerate(tensors)
            ),
        )
    if kind == "quantized":
        if set(data) != {"kind", "spec", "key", "value"}:
            raise SchemaError("quantized KV trace fields do not match the schema")
        if not isinstance(data["key"], Mapping) or not isinstance(
            data["value"], Mapping
        ):
            raise SchemaError("quantized KV key/value must be dictionaries")
        return ReferenceQuantizedKVData(
            spec,
            _quantized_tensor_from_dict(data["key"], "kv.key"),
            _quantized_tensor_from_dict(data["value"], "kv.value"),
        )
    raise SchemaError("unsupported KV data kind in trace: %r" % kind)


def _normalize_scale(value: Union[float, Sequence[float]], name: str) -> TraceScale:
    if isinstance(value, (int, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise SchemaError("%s must be finite" % name)
        return scalar
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise SchemaError("%s must be scalar or numeric array" % name) from error
    if any(not math.isfinite(item) for item in result):
        raise SchemaError("%s must be finite" % name)
    return result


def _scale_to_dict(value: TraceScale) -> Any:
    if isinstance(value, float):
        return _encode_float(value)
    return [_encode_float(item) for item in value]


def _scale_from_dict(value: Any, name: str) -> TraceScale:
    if isinstance(value, (list, tuple)):
        return tuple(_decode_float(item, name) for item in value)
    return _decode_float(value, name)


def _assert_tensor_close(
    name: str,
    actual: ReferenceTensor,
    expected: ReferenceTensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    if actual.spec != expected.spec:
        raise AttentionTraceMismatchError(
            "%s spec mismatch: %r != %r" % (name, actual.spec, expected.spec)
        )
    for index, (observed, wanted) in enumerate(zip(actual.data, expected.data)):
        if math.isnan(wanted):
            matches = math.isnan(observed)
        elif math.isinf(wanted):
            matches = observed == wanted
        else:
            matches = math.isfinite(observed) and abs(observed - wanted) <= (
                atol + rtol * abs(wanted)
            )
        if not matches:
            raise AttentionTraceMismatchError(
                "%s data mismatch at flat index %d" % (name, index)
            )


@dataclass(frozen=True)
class AttentionTrace:
    """A small, backend-independent Attention case with optional oracle output."""

    spec: AttentionPlanSpec
    metadata: AttentionMetadata
    q: ReferenceTensor
    kv_data: ReferenceKVInput
    custom_mask_data: Optional[Tuple[Union[bool, int], ...]] = None
    alibi_slopes: Optional[Tuple[float, ...]] = None
    q_scale: TraceScale = 1.0
    k_scale: TraceScale = 1.0
    v_scale: TraceScale = 1.0
    logits_soft_cap: float = 0.0
    return_lse: bool = True
    expected_output: Optional[ReferenceTensor] = None
    expected_lse: Optional[ReferenceTensor] = None
    schema_version: int = ATTENTION_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_TRACE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention trace schema version")
        object.__setattr__(
            self,
            "custom_mask_data",
            (
                tuple(self.custom_mask_data)
                if self.custom_mask_data is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "alibi_slopes",
            tuple(self.alibi_slopes) if self.alibi_slopes is not None else None,
        )
        object.__setattr__(self, "q_scale", _normalize_scale(self.q_scale, "q_scale"))
        object.__setattr__(self, "k_scale", _normalize_scale(self.k_scale, "k_scale"))
        object.__setattr__(self, "v_scale", _normalize_scale(self.v_scale, "v_scale"))
        runtime_cap = float(self.logits_soft_cap)
        if not math.isfinite(runtime_cap) or runtime_cap < 0:
            raise SchemaError("trace logits_soft_cap must be finite and non-negative")
        object.__setattr__(self, "logits_soft_cap", runtime_cap)

        if self.spec.use_profiler:
            raise NotImplementedError("Host Attention traces do not replay profiler data")
        plan = AttentionFrameworkSession(self.spec.mode).plan(self.spec, self.metadata)
        output_spec = plan.validate_run(
            self.q.spec,
            self.kv_data.spec,
            return_lse=self.return_lse,
            logits_soft_cap=self.logits_soft_cap,
        )
        self._validate_auxiliary_inputs()
        if self.expected_output is not None and self.expected_output.spec != output_spec.output:
            raise SchemaError("trace expected_output spec does not match the plan")
        if self.return_lse:
            if self.expected_output is not None and self.expected_lse is None:
                raise SchemaError("captured trace with return_lse requires expected_lse")
            if (
                self.expected_lse is not None
                and self.expected_lse.spec != output_spec.lse
            ):
                raise SchemaError("trace expected_lse spec does not match the plan")
        elif self.expected_lse is not None:
            raise SchemaError("trace without return_lse cannot contain expected_lse")

    def _validate_auxiliary_inputs(self) -> None:
        mask = self.spec.custom_mask
        if mask is None and self.custom_mask_data is not None:
            raise SchemaError("custom_mask_data provided without a mask plan")
        if mask is not None:
            if self.custom_mask_data is None:
                raise SchemaError("mask plan requires custom_mask_data")
            if len(self.custom_mask_data) != mask.numel:
                raise SchemaError("custom_mask_data length does not match the plan")
        if self.spec.pos_encoding_mode == PosEncodingMode.ALIBI:
            if (
                self.alibi_slopes is not None
                and len(self.alibi_slopes) != self.spec.num_qo_heads
            ):
                raise SchemaError("ALiBi slope count must match query heads")
        elif self.alibi_slopes is not None:
            raise SchemaError("alibi_slopes require ALIBI position encoding")
        for name, value, heads in (
            ("q_scale", self.q_scale, self.spec.num_qo_heads),
            ("k_scale", self.k_scale, self.spec.num_kv_heads),
            ("v_scale", self.v_scale, self.spec.num_kv_heads),
        ):
            if isinstance(value, tuple) and len(value) != heads:
                raise SchemaError("%s array length must match its head count" % name)

    @classmethod
    def capture(cls, **kwargs: Any) -> "AttentionTrace":
        """Execute a trace once and attach its Host oracle output."""

        trace = cls(**kwargs)
        result = trace.replay(validate_expected=False)
        return replace(
            trace,
            expected_output=result.output,
            expected_lse=result.lse,
        )

    def replay(
        self,
        *,
        validate_expected: bool = True,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> ReferenceAttentionResult:
        if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
            raise ValueError("atol and rtol must be finite and non-negative")
        plan = AttentionFrameworkSession(self.spec.mode).plan(self.spec, self.metadata)
        result = ReferenceAttentionExecutor().execute(
            plan,
            self.q,
            self.kv_data,
            return_lse=self.return_lse,
            custom_mask_data=self.custom_mask_data,
            alibi_slopes=self.alibi_slopes,
            q_scale=self.q_scale,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            logits_soft_cap=self.logits_soft_cap,
        )
        if validate_expected and self.expected_output is not None:
            _assert_tensor_close(
                "output", result.output, self.expected_output, atol=atol, rtol=rtol
            )
            if self.return_lse:
                if result.lse is None or self.expected_lse is None:
                    raise AttentionTraceMismatchError("LSE output is missing")
                _assert_tensor_close(
                    "lse", result.lse, self.expected_lse, atol=atol, rtol=rtol
                )
        return result

    def to_dict(self, *, include_expected: bool = True) -> Dict[str, Any]:
        expected = None
        if include_expected and self.expected_output is not None:
            expected = {
                "output": _tensor_to_dict(self.expected_output),
                "lse": (
                    _tensor_to_dict(self.expected_lse)
                    if self.expected_lse is not None
                    else None
                ),
            }
        return {
            "schema_version": self.schema_version,
            "kind": "attention_conformance",
            "plan_spec": self.spec.to_dict(),
            "metadata": self.metadata.to_dict(),
            "inputs": {
                "q": _tensor_to_dict(self.q),
                "kv_data": _kv_to_dict(self.kv_data),
                "custom_mask_data": (
                    list(self.custom_mask_data)
                    if self.custom_mask_data is not None
                    else None
                ),
                "alibi_slopes": (
                    [_encode_float(item) for item in self.alibi_slopes]
                    if self.alibi_slopes is not None
                    else None
                ),
                "q_scale": _scale_to_dict(self.q_scale),
                "k_scale": _scale_to_dict(self.k_scale),
                "v_scale": _scale_to_dict(self.v_scale),
                "logits_soft_cap": _encode_float(self.logits_soft_cap),
                "return_lse": self.return_lse,
            },
            "expected": expected,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionTrace":
        data = dict(value)
        if set(data) != {
            "schema_version",
            "kind",
            "plan_spec",
            "metadata",
            "inputs",
            "expected",
        }:
            raise SchemaError("Attention trace fields do not match the schema")
        if data["kind"] != "attention_conformance":
            raise SchemaError("unsupported trace kind: %r" % data["kind"])
        for name in ("plan_spec", "metadata", "inputs"):
            if not isinstance(data[name], Mapping):
                raise SchemaError("trace %s must be a dictionary" % name)
        inputs = dict(data["inputs"])
        expected_input_fields = {
            "q",
            "kv_data",
            "custom_mask_data",
            "alibi_slopes",
            "q_scale",
            "k_scale",
            "v_scale",
            "logits_soft_cap",
            "return_lse",
        }
        if set(inputs) != expected_input_fields:
            raise SchemaError("Attention trace input fields do not match the schema")
        if not isinstance(inputs["q"], Mapping) or not isinstance(
            inputs["kv_data"], Mapping
        ):
            raise SchemaError("trace q/kv_data must be dictionaries")

        expected_output = None
        expected_lse = None
        expected = data["expected"]
        if expected is not None:
            if not isinstance(expected, Mapping) or set(expected) != {"output", "lse"}:
                raise SchemaError("trace expected fields do not match the schema")
            if not isinstance(expected["output"], Mapping):
                raise SchemaError("trace expected output must be a dictionary")
            expected_output = _tensor_from_dict(expected["output"], "expected.output")
            if expected["lse"] is not None:
                if not isinstance(expected["lse"], Mapping):
                    raise SchemaError("trace expected LSE must be a dictionary or null")
                expected_lse = _tensor_from_dict(expected["lse"], "expected.lse")

        mask_value = inputs["custom_mask_data"]
        if mask_value is not None and not isinstance(mask_value, (list, tuple)):
            raise SchemaError("custom_mask_data must be an array or null")
        slope_value = inputs["alibi_slopes"]
        if slope_value is not None and not isinstance(slope_value, (list, tuple)):
            raise SchemaError("alibi_slopes must be an array or null")
        if not isinstance(inputs["return_lse"], bool):
            raise SchemaError("return_lse must be boolean")
        return cls(
            spec=AttentionPlanSpec.from_dict(data["plan_spec"]),
            metadata=attention_metadata_from_dict(data["metadata"]),
            q=_tensor_from_dict(inputs["q"], "q"),
            kv_data=_kv_from_dict(inputs["kv_data"]),
            custom_mask_data=(tuple(mask_value) if mask_value is not None else None),
            alibi_slopes=(
                tuple(_decode_float(item, "alibi_slopes") for item in slope_value)
                if slope_value is not None
                else None
            ),
            q_scale=_scale_from_dict(inputs["q_scale"], "q_scale"),
            k_scale=_scale_from_dict(inputs["k_scale"], "k_scale"),
            v_scale=_scale_from_dict(inputs["v_scale"], "v_scale"),
            logits_soft_cap=_decode_float(
                inputs["logits_soft_cap"], "logits_soft_cap"
            ),
            return_lse=inputs["return_lse"],
            expected_output=expected_output,
            expected_lse=expected_lse,
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
    ) -> "AttentionTrace":
        decoded, _usage = decode_attention_json(value, limits=limits)
        if not isinstance(decoded, Mapping):
            raise SchemaError("Attention trace JSON root must be an object")
        return cls.from_dict(decoded)

    @property
    def input_fingerprint(self) -> str:
        value = self.to_dict(include_expected=False)
        return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


__all__ = [
    "ATTENTION_TRACE_SCHEMA_VERSION",
    "AttentionTrace",
    "AttentionTraceMismatchError",
    "TraceScale",
]
