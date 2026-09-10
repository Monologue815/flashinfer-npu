"""Explicit service ownership of event-tracked batch runtimes.

No destructor, background polling, device package or force-release path is used.
The integrating service must retain this owner until close() succeeds.
"""

from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from .planner import AttentionStateError
from .schema import AttentionMode


@dataclass(frozen=True)
class AttentionRuntimeOwnerCloseReport:
    remaining_runtime_ids: tuple
    failures: tuple


class AttentionRuntimeOwnerClosePending(RuntimeError):
    def __init__(self, report):
        super().__init__("batch runtime owner close is pending; runtimes remain owned")
        self.report = report


class AttentionBatchRuntimeOwner:
    """Strong ownership independent of model-facing wrapper references."""

    def __init__(self):
        self._lock = RLock()
        self._runtimes = {}
        self._closing = False
        self._closed = False
        self._collecting = False

    @property
    def owned_runtime_ids(self):
        with self._lock:
            return tuple(self._runtimes)

    @property
    def is_closed(self):
        with self._lock:
            return self._closed

    def get_runtime(self, runtime_id):
        """Internal recovery access, not an executable model-facing plan handle."""
        with self._lock:
            return self._runtimes[runtime_id]

    def adopt(self, runtime):
        from .operator_resolver import AttentionOperatorRuntime

        if not isinstance(runtime, AttentionOperatorRuntime):
            raise TypeError("owner requires AttentionOperatorRuntime")
        if runtime.mode not in (
            AttentionMode.BATCH_PREFILL_PAGED, AttentionMode.BATCH_PREFILL_RAGGED,
            AttentionMode.BATCH_DECODE_PAGED, AttentionMode.BATCH_MIXED_PAGED,
        ):
            raise AttentionStateError("owner requires a batch Attention runtime")
        if not runtime.completion_tracking_enabled:
            raise AttentionStateError("owner requires a completion-tracked runtime")
        with self._lock:
            if self._closing or self._closed:
                raise AttentionStateError("batch runtime owner is closing or closed")
            if runtime.is_closing or runtime.is_closed:
                raise AttentionStateError("cannot adopt a closing or closed runtime")
            if any(value is runtime for value in self._runtimes.values()):
                raise AttentionStateError("runtime is already owned")
            runtime_id = uuid4().hex
            self._runtimes[runtime_id] = runtime
            return runtime_id

    def close(self):
        """Close all owned runtimes once; retain each failed/pending runtime.

        Independent completed runtimes are released even if another close fails.
        Repeated attempts are non-waiting and never infer completion from time.
        """
        with self._lock:
            if self._closed:
                return
            if self._collecting:
                raise AttentionStateError("batch runtime owner close is already running")
            self._closing = True
            self._collecting = True
            runtimes = tuple(self._runtimes.items())
        failures = []
        try:
            for runtime_id, runtime in runtimes:
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
                        del self._runtimes[runtime_id]
            with self._lock:
                report = AttentionRuntimeOwnerCloseReport(tuple(self._runtimes), tuple(failures))
                self._closed = not self._runtimes
            if report.remaining_runtime_ids:
                raise AttentionRuntimeOwnerClosePending(report)
        finally:
            with self._lock:
                self._collecting = False
