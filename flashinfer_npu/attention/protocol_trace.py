"""Versioned Attention protocol traces for lifecycle and ownership replay.

This schema is deliberately separate from :mod:`flashinfer_npu.attention.trace`:
numerical traces prove Attention semantics, while protocol traces prove that a
JIT call or provider launch followed an allowed state path without changing its
stream or resource ownership claim.
"""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    decode_attention_json,
)


ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION = 1


class AttentionProtocolRoute(str, Enum):
    SINGLE_JIT = "single_jit"
    PROVIDER = "provider"


class AttentionProtocolState(str, Enum):
    PREPARED = "prepared"
    INVOKED = "invoked"
    SUBMIT_UNKNOWN = "submit_unknown"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED_SYNC = "failed_sync"
    FAILED_ASYNC = "failed_async"
    RUNTIME_QUIESCED = "runtime_quiesced"
    RELEASED = "released"


def _canonical_json(value: Mapping[str, Any], *, indent=None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        ensure_ascii=True,
        allow_nan=False,
        indent=indent,
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def attention_protocol_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash a JSON-safe protocol subject, owner set, or evidence record."""

    if not isinstance(value, Mapping):
        raise TypeError("Attention protocol fingerprint input must be a mapping")
    try:
        return _canonical_hash(value)
    except (TypeError, ValueError) as error:
        raise SchemaError("Attention protocol fingerprint input is not JSON-safe") from error


def attention_protocol_evidence_fingerprint(value: Any) -> str:
    """Normalize an evidence object without trusting a boolean success claim."""

    if isinstance(value, str):
        _require_hash("evidence_fingerprint", value)
        return value
    if isinstance(value, Mapping):
        return attention_protocol_fingerprint(value)
    fingerprint = getattr(value, "fingerprint", None)
    if isinstance(fingerprint, str):
        _require_hash("evidence_fingerprint", fingerprint)
        return fingerprint
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        encoded = to_dict()
        if not isinstance(encoded, Mapping):
            raise SchemaError("Attention protocol evidence to_dict must return a mapping")
        return attention_protocol_fingerprint(encoded)
    raise TypeError("Attention protocol evidence must be a hash, mapping, or wire record")


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionProtocolEvent:
    sequence: int
    state: AttentionProtocolState
    stream_context_fingerprint: str
    ownership_fingerprint: str
    evidence_fingerprint: str
    schema_version: int = ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention protocol event version")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise SchemaError("protocol event sequence must be a non-negative integer")
        try:
            object.__setattr__(self, "state", AttentionProtocolState(self.state))
        except ValueError as error:
            raise SchemaError("Attention protocol event state is invalid") from error
        for name in (
            "stream_context_fingerprint",
            "ownership_fingerprint",
            "evidence_fingerprint",
        ):
            _require_hash(name, getattr(self, name))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "state": self.state.value,
            "stream_context_fingerprint": self.stream_context_fingerprint,
            "ownership_fingerprint": self.ownership_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProtocolEvent":
        if not isinstance(value, Mapping):
            raise SchemaError("AttentionProtocolEvent must be an object")
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProtocolEvent fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionProtocolEvent fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


_JIT_TRANSITIONS = {
    AttentionProtocolState.PREPARED: {
        AttentionProtocolState.INVOKED,
        AttentionProtocolState.RELEASED,
    },
    AttentionProtocolState.INVOKED: {
        AttentionProtocolState.COMPLETED,
        AttentionProtocolState.FAILED_SYNC,
    },
    AttentionProtocolState.COMPLETED: {AttentionProtocolState.RELEASED},
    AttentionProtocolState.FAILED_SYNC: {AttentionProtocolState.RELEASED},
}

_PROVIDER_TRANSITIONS = {
    AttentionProtocolState.PREPARED: {
        AttentionProtocolState.SUBMIT_UNKNOWN,
        AttentionProtocolState.SUBMITTED,
        AttentionProtocolState.FAILED_SYNC,
        AttentionProtocolState.RELEASED,
    },
    AttentionProtocolState.SUBMIT_UNKNOWN: {
        AttentionProtocolState.SUBMIT_UNKNOWN,
        AttentionProtocolState.PREPARED,
        AttentionProtocolState.SUBMITTED,
        AttentionProtocolState.COMPLETED,
        AttentionProtocolState.FAILED_ASYNC,
        AttentionProtocolState.RUNTIME_QUIESCED,
    },
    AttentionProtocolState.SUBMITTED: {
        AttentionProtocolState.COMPLETED,
        AttentionProtocolState.FAILED_ASYNC,
        AttentionProtocolState.RUNTIME_QUIESCED,
    },
    AttentionProtocolState.COMPLETED: {AttentionProtocolState.RELEASED},
    AttentionProtocolState.FAILED_SYNC: {AttentionProtocolState.RELEASED},
    AttentionProtocolState.FAILED_ASYNC: {AttentionProtocolState.RELEASED},
    AttentionProtocolState.RUNTIME_QUIESCED: {AttentionProtocolState.RELEASED},
}


@dataclass(frozen=True)
class AttentionProtocolTrace:
    trace_id: str
    route: AttentionProtocolRoute
    subject_fingerprint: str
    events: Tuple[AttentionProtocolEvent, ...]
    schema_version: int = ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention protocol trace version")
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise SchemaError("Attention protocol trace_id must be non-empty")
        try:
            object.__setattr__(self, "route", AttentionProtocolRoute(self.route))
        except ValueError as error:
            raise SchemaError("Attention protocol route is invalid") from error
        _require_hash("subject_fingerprint", self.subject_fingerprint)
        events = tuple(self.events)
        if not events:
            raise SchemaError("Attention protocol trace requires events")
        if any(not isinstance(event, AttentionProtocolEvent) for event in events):
            raise TypeError("Attention protocol events must be AttentionProtocolEvent")
        object.__setattr__(self, "events", events)
        self._validate_path()

    def _validate_path(self) -> None:
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence:
                raise SchemaError("Attention protocol event sequence is not contiguous")
        if self.events[0].state != AttentionProtocolState.PREPARED:
            raise SchemaError("Attention protocol trace must start prepared")
        if self.events[-1].state != AttentionProtocolState.RELEASED:
            raise SchemaError("Attention protocol trace must end released")
        stream = self.events[0].stream_context_fingerprint
        owner = self.events[0].ownership_fingerprint
        if any(event.stream_context_fingerprint != stream for event in self.events):
            raise SchemaError("Attention protocol stream ownership drifted")
        if any(event.ownership_fingerprint != owner for event in self.events):
            raise SchemaError("Attention protocol resource ownership drifted")
        transitions = (
            _JIT_TRANSITIONS
            if self.route == AttentionProtocolRoute.SINGLE_JIT
            else _PROVIDER_TRANSITIONS
        )
        for left, right in zip(self.events, self.events[1:]):
            if right.state not in transitions.get(left.state, set()):
                raise SchemaError(
                    "invalid %s protocol transition: %s -> %s"
                    % (self.route.value, left.state.value, right.state.value)
                )

    @property
    def stream_context_fingerprint(self) -> str:
        return self.events[0].stream_context_fingerprint

    @property
    def ownership_fingerprint(self) -> str:
        return self.events[0].ownership_fingerprint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "route": self.route.value,
            "subject_fingerprint": self.subject_fingerprint,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProtocolTrace":
        if not isinstance(value, Mapping):
            raise SchemaError("AttentionProtocolTrace must be an object")
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProtocolTrace fields are invalid")
        events = data.get("events")
        if not isinstance(events, (list, tuple)):
            raise SchemaError("Attention protocol events must be an array")
        data["events"] = tuple(AttentionProtocolEvent.from_dict(item) for item in events)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionProtocolTrace fields are invalid") from error

    def to_json(self, *, indent=None) -> str:
        return _canonical_json(self.to_dict(), indent=indent)

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    ) -> "AttentionProtocolTrace":
        decoded, _ = decode_attention_json(value, limits=limits)
        if not isinstance(decoded, Mapping):
            raise SchemaError("Attention protocol trace JSON must contain an object")
        return cls.from_dict(decoded)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionProtocolRecorder:
    """Append-only builder that only publishes a terminal, validated trace."""

    def __init__(
        self,
        trace_id: str,
        route: AttentionProtocolRoute,
        subject_fingerprint: str,
        stream_context_fingerprint: str,
        ownership_fingerprint: str,
        *,
        on_complete=None,
    ) -> None:
        if not isinstance(trace_id, str) or not trace_id:
            raise SchemaError("Attention protocol recorder trace_id must be non-empty")
        try:
            self.route = AttentionProtocolRoute(route)
        except ValueError as error:
            raise SchemaError("Attention protocol recorder route is invalid") from error
        for name, value in (
            ("subject_fingerprint", subject_fingerprint),
            ("stream_context_fingerprint", stream_context_fingerprint),
            ("ownership_fingerprint", ownership_fingerprint),
        ):
            _require_hash(name, value)
        if on_complete is not None and not callable(on_complete):
            raise TypeError("Attention protocol completion callback must be callable")
        self.trace_id = trace_id
        self.subject_fingerprint = subject_fingerprint
        self.stream_context_fingerprint = stream_context_fingerprint
        self.ownership_fingerprint = ownership_fingerprint
        self._events = []
        self._trace: Optional[AttentionProtocolTrace] = None
        self._on_complete = on_complete
        self._lock = RLock()

    def record(self, state: AttentionProtocolState, evidence: Any) -> None:
        completed_trace = None
        completion_callback = None
        with self._lock:
            if self._trace is not None:
                raise SchemaError("Attention protocol recorder is already complete")
            try:
                normalized_state = AttentionProtocolState(state)
            except ValueError as error:
                raise SchemaError("Attention protocol recorder state is invalid") from error
            event = AttentionProtocolEvent(
                len(self._events),
                normalized_state,
                self.stream_context_fingerprint,
                self.ownership_fingerprint,
                attention_protocol_evidence_fingerprint(evidence),
            )
            candidate = tuple(self._events) + (event,)
            self._validate_increment(candidate)
            self._events.append(event)
            if normalized_state == AttentionProtocolState.RELEASED:
                self._trace = AttentionProtocolTrace(
                    self.trace_id,
                    self.route,
                    self.subject_fingerprint,
                    tuple(self._events),
                )
                completed_trace = self._trace
                completion_callback = self._on_complete
        if completion_callback is not None:
            completion_callback(completed_trace)

    def _validate_increment(self, events: Tuple[AttentionProtocolEvent, ...]) -> None:
        if len(events) == 1:
            if events[0].state != AttentionProtocolState.PREPARED:
                raise SchemaError("Attention protocol recorder must start prepared")
            return
        transitions = (
            _JIT_TRANSITIONS
            if self.route == AttentionProtocolRoute.SINGLE_JIT
            else _PROVIDER_TRANSITIONS
        )
        left, right = events[-2:]
        if right.state not in transitions.get(left.state, set()):
            raise SchemaError(
                "invalid %s protocol transition: %s -> %s"
                % (self.route.value, left.state.value, right.state.value)
            )

    @property
    def events(self) -> Tuple[AttentionProtocolEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def trace(self) -> Optional[AttentionProtocolTrace]:
        with self._lock:
            return self._trace


_CURRENT_ATTENTION_PROTOCOL_CAPTURE: ContextVar[Optional["AttentionProtocolCapture"]] = (
    ContextVar("flashinfer_npu_attention_protocol_capture", default=None)
)


class AttentionProtocolCapture:
    """Context-local collector for automatically instrumented protocol calls."""

    def __init__(self, capture_id: str) -> None:
        if not isinstance(capture_id, str) or not capture_id:
            raise SchemaError("Attention protocol capture_id must be non-empty")
        self.capture_id = capture_id
        self._recorders = []
        self._token: Optional[Token] = None
        self._lock = RLock()

    def __enter__(self) -> "AttentionProtocolCapture":
        if self._token is not None:
            raise RuntimeError("Attention protocol capture cannot be re-entered")
        self._token = _CURRENT_ATTENTION_PROTOCOL_CAPTURE.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._token is not None
        _CURRENT_ATTENTION_PROTOCOL_CAPTURE.reset(self._token)
        self._token = None

    def start(
        self,
        route: AttentionProtocolRoute,
        subject_fingerprint: str,
        stream_context_fingerprint: str,
        ownership_fingerprint: str,
    ) -> AttentionProtocolRecorder:
        with self._lock:
            trace_id = "%s:%d" % (self.capture_id, len(self._recorders) + 1)
            recorder = AttentionProtocolRecorder(
                trace_id,
                route,
                subject_fingerprint,
                stream_context_fingerprint,
                ownership_fingerprint,
            )
            self._recorders.append(recorder)
            return recorder

    @property
    def recorders(self) -> Tuple[AttentionProtocolRecorder, ...]:
        with self._lock:
            return tuple(self._recorders)

    @property
    def traces(self) -> Tuple[AttentionProtocolTrace, ...]:
        with self._lock:
            completed = tuple(recorder.trace for recorder in self._recorders)
            return tuple(trace for trace in completed if trace is not None)

    @property
    def incomplete_count(self) -> int:
        with self._lock:
            return sum(recorder.trace is None for recorder in self._recorders)

    def to_corpus(self, corpus_id: str) -> "AttentionProtocolTraceCorpus":
        with self._lock:
            if any(recorder.trace is None for recorder in self._recorders):
                raise SchemaError(
                    "cannot publish protocol corpus with incomplete recorders"
                )
            return AttentionProtocolTraceCorpus(
                corpus_id,
                tuple(
                    AttentionProtocolTraceCase(trace.trace_id, trace)
                    for trace in (
                        recorder.trace for recorder in self._recorders
                    )
                    if trace is not None
                ),
            )


@dataclass(frozen=True)
class AttentionProtocolTraceCase:
    case_id: str
    trace: AttentionProtocolTrace
    numerical_case_id: Optional[str] = None
    numerical_input_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention protocol trace case version")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise SchemaError("Attention protocol case_id must be non-empty")
        if not isinstance(self.trace, AttentionProtocolTrace):
            raise TypeError("Attention protocol case trace has the wrong type")
        bound = self.numerical_case_id is not None
        if bound != (self.numerical_input_fingerprint is not None):
            raise SchemaError(
                "numerical case id and input fingerprint must be provided together"
            )
        if bound:
            if not isinstance(self.numerical_case_id, str) or not self.numerical_case_id:
                raise SchemaError("numerical_case_id must be non-empty when bound")
            _require_hash(
                "numerical_input_fingerprint",
                str(self.numerical_input_fingerprint),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "trace": self.trace.to_dict(),
            "numerical_case_id": self.numerical_case_id,
            "numerical_input_fingerprint": self.numerical_input_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProtocolTraceCase":
        if not isinstance(value, Mapping):
            raise SchemaError("AttentionProtocolTraceCase must be an object")
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProtocolTraceCase fields are invalid")
        trace = data.get("trace")
        if not isinstance(trace, Mapping):
            raise SchemaError("Attention protocol case trace must be an object")
        data["trace"] = AttentionProtocolTrace.from_dict(trace)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionProtocolTraceCase fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionProtocolTraceCorpus:
    corpus_id: str
    cases: Tuple[AttentionProtocolTraceCase, ...]
    schema_version: int = ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention protocol corpus version")
        if not isinstance(self.corpus_id, str) or not self.corpus_id:
            raise SchemaError("Attention protocol corpus_id must be non-empty")
        cases = tuple(self.cases)
        if not cases:
            raise SchemaError("Attention protocol corpus requires at least one case")
        if any(not isinstance(case, AttentionProtocolTraceCase) for case in cases):
            raise TypeError("Attention protocol corpus cases have the wrong type")
        if len({case.case_id for case in cases}) != len(cases):
            raise SchemaError("Attention protocol corpus case ids must be unique")
        if len({case.trace.trace_id for case in cases}) != len(cases):
            raise SchemaError("Attention protocol corpus trace ids must be unique")
        if len({case.trace.fingerprint for case in cases}) != len(cases):
            raise SchemaError("Attention protocol corpus traces must be unique")
        object.__setattr__(self, "cases", cases)

    @property
    def route_counts(self) -> Dict[str, int]:
        counts = {route.value: 0 for route in AttentionProtocolRoute}
        for case in self.cases:
            counts[case.trace.route.value] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_protocol_corpus",
            "corpus_id": self.corpus_id,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProtocolTraceCorpus":
        if not isinstance(value, Mapping):
            raise SchemaError("AttentionProtocolTraceCorpus must be an object")
        data = dict(value)
        if set(data) != {"schema_version", "kind", "corpus_id", "cases"}:
            raise SchemaError("AttentionProtocolTraceCorpus fields are invalid")
        if data.pop("kind") != "attention_protocol_corpus":
            raise SchemaError("Attention protocol corpus kind is invalid")
        cases = data.get("cases")
        if not isinstance(cases, (list, tuple)):
            raise SchemaError("Attention protocol corpus cases must be an array")
        data["cases"] = tuple(AttentionProtocolTraceCase.from_dict(case) for case in cases)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionProtocolTraceCorpus fields are invalid") from error

    def to_json(self, *, indent=None) -> str:
        return _canonical_json(self.to_dict(), indent=indent)

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    ) -> "AttentionProtocolTraceCorpus":
        decoded, _ = decode_attention_json(value, limits=limits)
        if not isinstance(decoded, Mapping):
            raise SchemaError("Attention protocol corpus JSON must contain an object")
        return cls.from_dict(decoded)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def capture_attention_protocol(capture_id: str) -> AttentionProtocolCapture:
    """Create a context manager used by instrumented JIT/provider paths."""

    return AttentionProtocolCapture(capture_id)


def start_attention_protocol_recording(
    route: AttentionProtocolRoute,
    subject_fingerprint: str,
    stream_context_fingerprint: str,
    ownership_fingerprint: str,
) -> Optional[AttentionProtocolRecorder]:
    """Start a recorder only when the current context explicitly opted in."""

    capture = _CURRENT_ATTENTION_PROTOCOL_CAPTURE.get()
    if capture is None:
        return None
    return capture.start(
        route,
        subject_fingerprint,
        stream_context_fingerprint,
        ownership_fingerprint,
    )


__all__ = [
    "ATTENTION_PROTOCOL_TRACE_SCHEMA_VERSION",
    "AttentionProtocolCapture",
    "AttentionProtocolEvent",
    "AttentionProtocolRecorder",
    "AttentionProtocolRoute",
    "AttentionProtocolState",
    "AttentionProtocolTrace",
    "AttentionProtocolTraceCase",
    "AttentionProtocolTraceCorpus",
    "attention_protocol_evidence_fingerprint",
    "attention_protocol_fingerprint",
    "capture_attention_protocol",
    "start_attention_protocol_recording",
]
