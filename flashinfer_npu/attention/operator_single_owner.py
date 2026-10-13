"""Explicit ownership for ephemeral, completion-tracked Attention runtimes.

This internal primitive is not connected to public single facades or bootstrap.
No device work, background polling or forced release is implemented here.
"""

from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from .planner import AttentionStateError
from .schema import AttentionMode


@dataclass
class _SingleRequest:
    runtime: object = field(repr=False)
    sealed: bool = False


@dataclass(frozen=True)
class AttentionSingleOwnerReport:
    released: tuple
    preparing: tuple
    pending: tuple
    failures: tuple


class AttentionSingleOwnerClosePending(RuntimeError):
    def __init__(self, report):
        super().__init__("single runtime owner close is pending; requests remain owned")
        self.report = report


class AttentionSingleRuntimeOwner:
    """Service-owned requests, sealed by their facade after submission ends.

    Callers must not plan or run after sealing. Collection only closes sealed
    requests, so reentrant collection cannot close a preparing request. The
    integration must serialize planning/submission and lifecycle operations.
    """

    def __init__(self):
        self._lock = RLock()
        self._requests = {}
        self._closing = False
        self._closed = False
        self._collecting = False

    @property
    def owned_runtime_ids(self):
        with self._lock:
            return tuple(self._requests)

    @property
    def is_closed(self):
        with self._lock:
            return self._closed

    def get_runtime(self, runtime_id):
        with self._lock:
            return self._requests[runtime_id].runtime

    def adopt(self, runtime):
        from .operator_resolver import AttentionOperatorRuntime

        if not isinstance(runtime, AttentionOperatorRuntime):
            raise TypeError("owner requires AttentionOperatorRuntime")
        if runtime.mode not in (AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE):
            raise AttentionStateError("owner requires a single Attention runtime")
        if not runtime.completion_tracking_enabled:
            raise AttentionStateError("owner requires a completion-tracked runtime")
        with self._lock:
            if self._closing or self._closed:
                raise AttentionStateError("single runtime owner is closing or closed")
            if runtime.is_closing or runtime.is_closed:
                raise AttentionStateError("cannot adopt a closing or closed runtime")
            if any(request.runtime is runtime for request in self._requests.values()):
                raise AttentionStateError("runtime is already owned")
            runtime_id = uuid4().hex
            self._requests[runtime_id] = _SingleRequest(runtime)
            return runtime_id

    def seal(self, runtime_id):
        """Mark submission ended, including failed preparation; never release.

        Allowed during shutdown so an in-flight facade can finish its finally
        path. Repeated sealing of an owned request is harmless.
        """
        with self._lock:
            self._requests[runtime_id].sealed = True

    def collect_completed(self):
        return self._drain(shutdown=False)

    def close(self):
        report = self._drain(shutdown=True)
        if report.preparing or report.pending:
            raise AttentionSingleOwnerClosePending(report)

    def _drain(self, *, shutdown):
        with self._lock:
            if self._collecting:
                raise AttentionStateError("single runtime owner collection is already running")
            if shutdown and self._closed:
                return AttentionSingleOwnerReport((), (), (), ())
            if not shutdown and (self._closing or self._closed):
                raise AttentionStateError("single runtime owner is closing or closed")
            if shutdown:
                self._closing = True
            self._collecting = True
            # Capture eligibility now: newly sealed/adopted requests wait for
            # the next pass, even if an event query changes owner state.
            requests = tuple((key, request.runtime) for key, request in self._requests.items()
                             if request.sealed)
        released, failures = [], []
        try:
            for runtime_id, runtime in requests:
                try:
                    runtime.close()
                    if not runtime.is_closed:
                        raise AttentionStateError("runtime close did not establish a closed state")
                except Exception as error:
                    try:
                        message = str(error)
                    except Exception:
                        message = "error text unavailable"
                    failures.append((runtime_id, type(error).__name__, message))
                else:
                    with self._lock:
                        del self._requests[runtime_id]
                    released.append(runtime_id)
            with self._lock:
                preparing = tuple(key for key, request in self._requests.items() if not request.sealed)
                pending = tuple(key for key, request in self._requests.items() if request.sealed)
                if shutdown:
                    self._closed = not self._requests
            return AttentionSingleOwnerReport(tuple(released), preparing, pending, tuple(failures))
        finally:
            with self._lock:
                self._collecting = False
