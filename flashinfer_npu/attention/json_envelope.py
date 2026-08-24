"""Bounded JSON decoding for untrusted Attention traces and corpora."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import SchemaError


ATTENTION_JSON_ENVELOPE_VERSION = 1


@dataclass(frozen=True)
class AttentionJsonEnvelopeLimits:
    """Pre-construction limits for serialized Attention inputs.

    These bounds protect parsing and Python object construction. They are
    intentionally independent from plan-time ``AttentionMetadataLimits``.
    """

    max_bytes: int = 16 * 1024 * 1024
    max_nesting_depth: int = 64
    max_nodes: int = 2_000_000
    max_array_items: int = 1_000_000
    max_object_fields: int = 128
    max_string_bytes: int = 1 * 1024 * 1024
    max_cases: int = 4096
    max_tensors: int = 32768
    max_tensor_rank: int = 16
    max_tensor_elements: int = 8_000_000
    max_total_tensor_elements: int = 16_000_000
    schema_version: int = ATTENTION_JSON_ENVELOPE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JSON_ENVELOPE_VERSION:
            raise SchemaError("unsupported Attention JSON envelope version")
        for name in self.__dataclass_fields__:
            if name == "schema_version":
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SchemaError("%s must be a positive integer" % name)

    def with_max_bytes(self, value: int) -> "AttentionJsonEnvelopeLimits":
        return replace(self, max_bytes=int(value))


DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS = AttentionJsonEnvelopeLimits()


@dataclass(frozen=True)
class AttentionJsonEnvelopeUsage:
    encoded_bytes: int
    max_nesting_depth: int
    nodes: int
    array_items: int
    object_fields: int
    strings: int
    cases: int
    tensors: int
    total_tensor_elements: int


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError("duplicate JSON object key %r" % key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant %s" % value)


def _preflight_nesting(value: str, limit: int) -> int:
    """Bound lexical nesting before recursive ``json.loads`` runs."""

    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            maximum = max(maximum, depth)
            if depth > limit:
                raise SchemaError(
                    "Attention JSON nesting depth exceeds limit %d" % limit
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                # Let the JSON parser report the malformed syntax uniformly.
                return maximum
    return maximum


def _tensor_numel(value: Mapping[str, Any], limits: AttentionJsonEnvelopeLimits) -> int:
    shape = value.get("shape")
    data = value.get("data")
    if not isinstance(shape, list) or not isinstance(data, list):
        return 0
    if len(shape) > limits.max_tensor_rank:
        raise SchemaError(
            "Attention JSON tensor rank exceeds limit %d" % limits.max_tensor_rank
        )
    dimensions: List[int] = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SchemaError("Attention JSON tensor shape must contain integers")
        if dimension < 0:
            raise SchemaError("Attention JSON tensor shape cannot be negative")
        dimensions.append(dimension)
    claimed = reduce(mul, dimensions, 1)
    if claimed > limits.max_tensor_elements:
        raise SchemaError(
            "Attention JSON tensor elements exceed limit %d"
            % limits.max_tensor_elements
        )
    if len(data) > limits.max_tensor_elements:
        raise SchemaError(
            "Attention JSON tensor data exceeds limit %d"
            % limits.max_tensor_elements
        )
    # Existing tensor schemas prove equality. Charge the larger value here so
    # malformed under-filled tensors cannot bypass the aggregate envelope.
    return max(claimed, len(data))


def _measure_decoded(
    decoded: Any,
    encoded_bytes: int,
    lexical_depth: int,
    limits: AttentionJsonEnvelopeLimits,
) -> AttentionJsonEnvelopeUsage:
    nodes = 0
    array_items = 0
    object_fields = 0
    strings = 0
    tensors = 0
    tensor_elements = 0
    cases = 0
    maximum_depth = 0
    stack: List[Tuple[Any, int]] = [(decoded, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if nodes > limits.max_nodes:
            raise SchemaError("Attention JSON nodes exceed limit %d" % limits.max_nodes)
        if isinstance(current, Mapping):
            field_count = len(current)
            object_fields += field_count
            if field_count > limits.max_object_fields:
                raise SchemaError(
                    "Attention JSON object fields exceed per-object limit %d"
                    % limits.max_object_fields
                )
            if (
                set(("shape", "data", "dtype", "device"))
                <= set(current)
            ):
                tensors += 1
                if tensors > limits.max_tensors:
                    raise SchemaError(
                        "Attention JSON tensors exceed limit %d" % limits.max_tensors
                    )
                tensor_elements += _tensor_numel(current, limits)
                if tensor_elements > limits.max_total_tensor_elements:
                    raise SchemaError(
                        "Attention JSON total tensor elements exceed limit %d"
                        % limits.max_total_tensor_elements
                    )
            if current.get("kind") in {
                "attention_conformance_corpus",
                "attention_protocol_corpus",
            }:
                case_values = current.get("cases")
                if isinstance(case_values, list):
                    cases = len(case_values)
                    if cases > limits.max_cases:
                        raise SchemaError(
                            "Attention JSON corpus cases exceed limit %d"
                            % limits.max_cases
                        )
            for key, item in current.items():
                strings += 1
                if len(str(key).encode("utf-8")) > limits.max_string_bytes:
                    raise SchemaError("Attention JSON key exceeds string byte limit")
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            item_count = len(current)
            array_items += item_count
            if item_count > limits.max_array_items:
                raise SchemaError(
                    "Attention JSON array exceeds per-array limit %d"
                    % limits.max_array_items
                )
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            strings += 1
            if len(current.encode("utf-8")) > limits.max_string_bytes:
                raise SchemaError(
                    "Attention JSON string exceeds byte limit %d"
                    % limits.max_string_bytes
                )
    if maximum_depth > limits.max_nesting_depth:
        raise SchemaError(
            "Attention JSON nesting depth exceeds limit %d"
            % limits.max_nesting_depth
        )
    return AttentionJsonEnvelopeUsage(
        encoded_bytes,
        max(lexical_depth, maximum_depth),
        nodes,
        array_items,
        object_fields,
        strings,
        cases,
        tensors,
        tensor_elements,
    )


def decode_attention_json(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
) -> Tuple[Any, AttentionJsonEnvelopeUsage]:
    """Decode strict JSON and enforce bounds before domain construction."""

    if not isinstance(limits, AttentionJsonEnvelopeLimits):
        raise TypeError("limits must be AttentionJsonEnvelopeLimits")
    if not isinstance(value, str):
        raise SchemaError("Attention JSON input must be text")
    encoded_bytes = len(value.encode("utf-8"))
    if encoded_bytes > limits.max_bytes:
        raise SchemaError(
            "Attention JSON bytes exceed limit %d" % limits.max_bytes
        )
    lexical_depth = _preflight_nesting(value, limits.max_nesting_depth)
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as error:
        raise SchemaError("Attention JSON contains %s" % error) from error
    except (TypeError, ValueError, RecursionError) as error:
        raise SchemaError("Attention JSON is not valid strict JSON") from error
    usage = _measure_decoded(
        decoded, encoded_bytes, lexical_depth, limits
    )
    return decoded, usage


__all__ = [
    "ATTENTION_JSON_ENVELOPE_VERSION",
    "AttentionJsonEnvelopeLimits",
    "AttentionJsonEnvelopeUsage",
    "DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS",
    "decode_attention_json",
]
