"""Process-local retention of opaque calls until injected completion proof.

No device events, synchronization, package imports or background polling are
implemented here. The integrating runtime must keep the registry alive, record
events on the execution stream and poll them; public wrappers do not opt in yet.
"""

from dataclasses import dataclass, field
from threading import RLock
from typing import Optional, Protocol, runtime_checkable
from uuid import uuid4

from .operator_run import AttentionLoweredOperatorCall


class AttentionCallRetentionError(RuntimeError):
    """A lifecycle operation cannot safely release retained resources."""


@dataclass(frozen=True)
class AttentionRetainedCallToken:
    invocation_id: str
    provider_id: str
    operation_id: str
    active_plan_fingerprint: str


@runtime_checkable
class AttentionRetainedCallEvent(Protocol):
    token: AttentionRetainedCallToken

    def query(self) -> bool:
        """Non-blocking, exact completion proof for the token's execution."""


@runtime_checkable
class AttentionRetainedCallEventRecorder(Protocol):
    def record(self, token: AttentionRetainedCallToken) -> AttentionRetainedCallEvent:
        """Record after the invocation's last queued use, including partial failure."""


@dataclass
class _PendingCall:
    token: AttentionRetainedCallToken
    call: AttentionLoweredOperatorCall = field(repr=False)
    event: Optional[AttentionRetainedCallEvent] = field(default=None, repr=False)
    querying: bool = False


class AttentionOperatorCallRetention:
    """Long-lived owner of calls across replans, failures and executor replacement.

    Tokens contain diagnostics only; they do not own resources. Dropping this
    registry is not a supported shutdown path. close() requires an empty registry;
    no force-release or timeout-based release operation is provided.
    """

    def __init__(self):
        self._lock = RLock()
        self._pending = {}
        self._closed = False

    @property
    def pending_tokens(self):
        with self._lock:
            return tuple(record.token for record in self._pending.values())

    def retain(self, call):
        if not isinstance(call, AttentionLoweredOperatorCall):
            raise TypeError("call must be AttentionLoweredOperatorCall")
        with self._lock:
            if self._closed:
                raise AttentionCallRetentionError("call retention registry is closed")
            token = AttentionRetainedCallToken(
                uuid4().hex, call.provider_id, call.operation_id, call.active_plan_fingerprint)
            self._pending[token.invocation_id] = _PendingCall(token, call)
            return token

    def _record(self, token):
        if not isinstance(token, AttentionRetainedCallToken):
            raise TypeError("token must be AttentionRetainedCallToken")
        record = self._pending.get(token.invocation_id)
        if record is None or record.token != token:
            raise AttentionCallRetentionError("unknown or mismatched retained call token")
        return record

    def bind_event(self, token, event):
        if not isinstance(event, AttentionRetainedCallEvent):
            raise TypeError("event must implement AttentionRetainedCallEvent")
        with self._lock:
            record = self._record(token)
            if record.event is not None:
                raise AttentionCallRetentionError("retained call already has a completion event")
            if event.token != record.token:
                raise AttentionCallRetentionError("completion event does not bind this invocation")
            if any(other.event is event for other in self._pending.values()):
                raise AttentionCallRetentionError("completion event is already bound to another invocation")
            record.event = event

    def poll(self, token):
        """Release one call only on strict True, preserving it on every error.

        The external query runs outside the registry lock. Concurrent or reentrant
        queries for the same invocation are rejected; independent calls may progress.
        A completed token is removed, and cannot be polled or rebound again.
        """
        with self._lock:
            record = self._record(token)
            if record.event is None:
                raise AttentionCallRetentionError("retained call has no completion event")
            if record.querying:
                raise AttentionCallRetentionError("completion query is already in progress")
            event = record.event
            record.querying = True
        try:
            if event.token != record.token:
                raise AttentionCallRetentionError("completion event invocation identity changed")
            completed = event.query()
            if event.token != record.token:
                raise AttentionCallRetentionError("completion event invocation identity changed")
            if not isinstance(completed, bool):
                raise AttentionCallRetentionError("completion query must return an exact boolean")
        except BaseException:
            with self._lock:
                record.querying = False
            raise
        with self._lock:
            record.querying = False
            if completed:
                del self._pending[token.invocation_id]
            return completed

    def close(self):
        with self._lock:
            if self._pending:
                raise AttentionCallRetentionError("cannot close while calls still retain resources")
            self._closed = True


class AttentionRetainedInvocationError(RuntimeError):
    """Execution/recording failed; the token still owns its call in the registry."""

    def __init__(self, token, invocation_error=None, recording_error=None):
        super().__init__("retained Attention invocation failed; completion must be resolved")
        self.token = token
        self.invocation_error = invocation_error
        self.recording_error = recording_error


def execute_attention_retained_call(registry, call, invoke, recorder):
    """Wrap an already-authorized executor, returning (unchanged result, token).

    Retention starts before invoke(call). Event recording is attempted afterward,
    even if invocation failed after potentially queuing device work. On failure,
    the raised error exposes the token for recovery; no retry or release is implied.
    A record failure can be recovered only by supplying valid completion proof to
    bind_event(), not by assuming that the invocation did not submit work.
    """
    if not isinstance(registry, AttentionOperatorCallRetention):
        raise TypeError("registry must be AttentionOperatorCallRetention")
    if not callable(invoke):
        raise TypeError("invoke must be an already-authorized executor callable")
    if not isinstance(recorder, AttentionRetainedCallEventRecorder):
        raise TypeError("recorder must implement AttentionRetainedCallEventRecorder")
    token = registry.retain(call)
    try:
        result = invoke(call)
    except BaseException as invocation_error:
        try:
            registry.bind_event(token, recorder.record(token))
        except BaseException as recording_error:
            raise AttentionRetainedInvocationError(
                token, invocation_error, recording_error) from invocation_error
        raise AttentionRetainedInvocationError(token, invocation_error) from invocation_error
    try:
        registry.bind_event(token, recorder.record(token))
    except BaseException as recording_error:
        raise AttentionRetainedInvocationError(token, recording_error=recording_error) from recording_error
    return result, token
