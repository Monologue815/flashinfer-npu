"""Process-local single-request lifetime configuration, not an operator loader."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .operator_retention import AttentionRetainedCallEventRecorder
from .operator_single_owner import AttentionSingleRuntimeOwner
from .schema import AttentionMode


@runtime_checkable
class AttentionSingleCompletionEventRecorderFactory(Protocol):
    def create(self, *, device: str, mode) -> AttentionRetainedCallEventRecorder:
        """Prepare an invocation recorder without submitting device work."""


@dataclass(frozen=True, eq=False)
class AttentionSingleRuntimeResources:
    """Indivisible factory/owner pair; creation itself has no runtime effects."""

    recorder_factory: AttentionSingleCompletionEventRecorderFactory = field(repr=False, compare=False)
    owner: AttentionSingleRuntimeOwner = field(repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.owner, AttentionSingleRuntimeOwner):
            raise TypeError("single resources require AttentionSingleRuntimeOwner")
        if (not isinstance(self.recorder_factory, AttentionSingleCompletionEventRecorderFactory)
                or not callable(getattr(self.recorder_factory, "create", None))):
            raise TypeError("single completion recorder factory must implement create")

    @contextmanager
    def submission(self, snapshot, *, device, mode):
        """Adopt before plan/run and seal on every exit, without collecting.

        The scope yields an internal runtime, not a model-facing plan handle.
        The service must keep the owner alive and schedule collection separately.
        """
        from .holistic import AttentionOperatorRuntimeRegistrySnapshot
        from .operator_resolver import AttentionOperatorRuntime

        if not isinstance(snapshot, AttentionOperatorRuntimeRegistrySnapshot):
            raise TypeError("single submission requires a runtime registry snapshot")
        if not isinstance(mode, AttentionMode) or mode not in (
                AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE):
            raise ValueError("single submission requires a single Attention mode")
        if not isinstance(device, str) or device.split(":", 1)[0] != "npu":
            raise ValueError("single submission requires an NPU device")
        recorder = self.recorder_factory.create(device=device, mode=mode)
        if (not isinstance(recorder, AttentionRetainedCallEventRecorder)
                or not callable(getattr(recorder, "record", None))):
            raise TypeError("single completion recorder must implement record")
        runtime = AttentionOperatorRuntime(
            device, snapshot.registry, snapshot.operation_catalog, mode=mode,
            runtime_declaration_bindings=snapshot.runtime_declaration_binding_tuples,
            plan_scoring_manifest_binding=snapshot.plan_scoring_manifest_binding,
            provider_integration_bundle_binding=snapshot.provider_integration_bundle_binding,
            completion_event_recorder=recorder)
        runtime_id = self.owner.adopt(runtime)
        try:
            yield runtime
        finally:
            self.owner.seal(runtime_id)
