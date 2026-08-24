"""Bind Attention accuracy evidence to a completed provider launch lifecycle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from flashinfer_npu.runtime import SchemaError

from .accuracy import attention_result_fingerprint
from .accuracy_evidence import AttentionAccuracyDispatchBinding
from .launcher_provider import (
    AttentionProviderCompletionResult,
    AttentionProviderSubmitResult,
    AttentionResolvedLauncher,
)
from .launch_packet import AttentionLaunchPacket
from .protocol_trace import (
    AttentionProtocolRoute,
    AttentionProtocolState,
    AttentionProtocolTrace,
    attention_protocol_evidence_fingerprint,
)
from .reference import ReferenceAttentionResult


ATTENTION_ACCURACY_EXECUTION_BINDING_VERSION = 1
_RESULT_ORIGIN = "runner_declared_post_completion"
_HASH_FIELDS = (
    "accuracy_dispatch_binding_fingerprint",
    "candidate_result_fingerprint",
    "launch_packet_fingerprint",
    "execution_identity_fingerprint",
    "dispatch_receipt_fingerprint",
    "launch_lease_fingerprint",
    "stream_context_fingerprint",
    "provider_probe_fingerprint",
    "resolved_launcher_fingerprint",
    "submit_result_fingerprint",
    "completion_result_fingerprint",
    "protocol_trace_fingerprint",
)


class AttentionAccuracyExecutionBindingError(RuntimeError):
    """Raised when an accuracy result and provider lifecycle cannot be joined."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionAccuracyExecutionBinding:
    binding_id: str
    runner: str
    result_origin: str
    accuracy_dispatch_binding_fingerprint: str
    candidate_result_fingerprint: str
    launch_packet_fingerprint: str
    execution_identity_fingerprint: str
    dispatch_receipt_fingerprint: str
    launch_lease_fingerprint: str
    stream_context_fingerprint: str
    runtime_id: str
    runtime_generation: int
    provider_probe_fingerprint: str
    resolved_launcher_fingerprint: str
    submit_result_fingerprint: str
    completion_result_fingerprint: str
    protocol_trace_fingerprint: str
    submission_id: str
    completion_event_id: str
    schema_version: int = ATTENTION_ACCURACY_EXECUTION_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_ACCURACY_EXECUTION_BINDING_VERSION:
            raise SchemaError("unsupported Attention accuracy execution binding version")
        for name in (
            "binding_id",
            "runner",
            "runtime_id",
            "submission_id",
            "completion_event_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise SchemaError("accuracy execution binding %s must be non-empty" % name)
        if self.result_origin != _RESULT_ORIGIN:
            raise SchemaError("accuracy execution binding result_origin is invalid")
        if (
            type(self.runtime_generation) is not int
            or self.runtime_generation < 1
        ):
            raise SchemaError("accuracy execution runtime_generation must be positive")
        for name in _HASH_FIELDS:
            _require_hash(name, getattr(self, name))

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionAccuracyExecutionBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionAccuracyExecutionBinding fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError(
                "AttentionAccuracyExecutionBinding fields are invalid"
            ) from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate(
        self,
        accuracy_dispatch_binding: AttentionAccuracyDispatchBinding,
        candidate: ReferenceAttentionResult,
        launch_packet: AttentionLaunchPacket,
        resolved_launcher: AttentionResolvedLauncher,
        submit_result: AttentionProviderSubmitResult,
        completion_result: AttentionProviderCompletionResult,
        protocol_trace: AttentionProtocolTrace,
    ) -> None:
        rebuilt = bind_attention_accuracy_execution(
            binding_id=self.binding_id,
            runner=self.runner,
            accuracy_dispatch_binding=accuracy_dispatch_binding,
            candidate=candidate,
            launch_packet=launch_packet,
            resolved_launcher=resolved_launcher,
            submit_result=submit_result,
            completion_result=completion_result,
            protocol_trace=protocol_trace,
        )
        if self != rebuilt:
            raise AttentionAccuracyExecutionBindingError(
                "accuracy execution binding fields are stale or inconsistent"
            )


def _validate_candidate_tensor_contract(
    candidate: ReferenceAttentionResult, packet: AttentionLaunchPacket
) -> None:
    bindings = {item.role: item.view for item in packet.launch_lease.bindings}
    output = bindings.get("out")
    if output is None:
        raise AttentionAccuracyExecutionBindingError(
            "launch packet has no output tensor binding"
        )
    if candidate.output.shape != output.shape or candidate.output.dtype != output.dtype:
        raise AttentionAccuracyExecutionBindingError(
            "candidate output does not match launch packet tensor contract"
        )
    lse = bindings.get("lse")
    if (candidate.lse is None) != (lse is None):
        raise AttentionAccuracyExecutionBindingError(
            "candidate LSE presence does not match launch packet"
        )
    if candidate.lse is not None and lse is not None:
        if candidate.lse.shape != lse.shape or candidate.lse.dtype != lse.dtype:
            raise AttentionAccuracyExecutionBindingError(
                "candidate LSE does not match launch packet tensor contract"
            )


def bind_attention_accuracy_execution(
    *,
    binding_id: str,
    runner: str,
    accuracy_dispatch_binding: AttentionAccuracyDispatchBinding,
    candidate: ReferenceAttentionResult,
    launch_packet: AttentionLaunchPacket,
    resolved_launcher: AttentionResolvedLauncher,
    submit_result: AttentionProviderSubmitResult,
    completion_result: AttentionProviderCompletionResult,
    protocol_trace: AttentionProtocolTrace,
) -> AttentionAccuracyExecutionBinding:
    """Join a passing result declaration to a successful provider lifecycle."""

    if not isinstance(accuracy_dispatch_binding, AttentionAccuracyDispatchBinding):
        raise TypeError(
            "accuracy_dispatch_binding must be AttentionAccuracyDispatchBinding"
        )
    if not isinstance(candidate, ReferenceAttentionResult):
        raise TypeError("candidate must be ReferenceAttentionResult")
    if not isinstance(launch_packet, AttentionLaunchPacket):
        raise TypeError("launch_packet must be AttentionLaunchPacket")
    if not isinstance(resolved_launcher, AttentionResolvedLauncher):
        raise TypeError("resolved_launcher must be AttentionResolvedLauncher")
    if not isinstance(submit_result, AttentionProviderSubmitResult):
        raise TypeError("submit_result must be AttentionProviderSubmitResult")
    if not isinstance(completion_result, AttentionProviderCompletionResult):
        raise TypeError("completion_result must be AttentionProviderCompletionResult")
    if not isinstance(protocol_trace, AttentionProtocolTrace):
        raise TypeError("protocol_trace must be AttentionProtocolTrace")

    candidate_fingerprint = attention_result_fingerprint(candidate)
    if candidate_fingerprint != accuracy_dispatch_binding.candidate_result_fingerprint:
        raise AttentionAccuracyExecutionBindingError(
            "candidate result does not match accuracy dispatch binding"
        )
    packet_fingerprint = launch_packet.fingerprint
    receipt = launch_packet.dispatch_receipt
    identity = launch_packet.execution_identity
    if receipt.fingerprint != accuracy_dispatch_binding.dispatch_receipt_fingerprint:
        raise AttentionAccuracyExecutionBindingError(
            "launch packet receipt does not match accuracy dispatch binding"
        )
    if (
        identity.fingerprint != launch_packet.launch_lease.execution_identity_fingerprint
        or identity.plan_fingerprint != accuracy_dispatch_binding.plan_fingerprint
        or identity.admission_fingerprint != accuracy_dispatch_binding.admission_fingerprint
        or identity.kernel_fingerprint != accuracy_dispatch_binding.kernel_fingerprint
        or identity.capability_profile_fingerprint
        != accuracy_dispatch_binding.profile_fingerprint
    ):
        raise AttentionAccuracyExecutionBindingError(
            "launch execution identity does not match accuracy dispatch binding"
        )
    _validate_candidate_tensor_contract(candidate, launch_packet)
    try:
        resolved_launcher.validate_packet(launch_packet)
    except SchemaError as error:
        raise AttentionAccuracyExecutionBindingError(
            "resolved launcher does not validate launch packet: %s" % error
        ) from error

    if submit_result.return_code != 0:
        raise AttentionAccuracyExecutionBindingError(
            "accuracy execution requires successful provider submission"
        )
    if completion_result.return_code != 0:
        raise AttentionAccuracyExecutionBindingError(
            "accuracy execution requires successful provider completion"
        )
    if (
        submit_result.provider_probe_fingerprint
        != resolved_launcher.provider_probe_fingerprint
        or completion_result.provider_probe_fingerprint
        != resolved_launcher.provider_probe_fingerprint
        or submit_result.resolved_launcher_fingerprint != resolved_launcher.fingerprint
        or completion_result.resolved_launcher_fingerprint != resolved_launcher.fingerprint
        or submit_result.launch_packet_fingerprint != packet_fingerprint
        or completion_result.launch_packet_fingerprint != packet_fingerprint
        or submit_result.completion_event_id != completion_result.completion_event_id
    ):
        raise AttentionAccuracyExecutionBindingError(
            "provider submit/completion identity chain is inconsistent"
        )
    assert submit_result.submission_id is not None
    assert submit_result.completion_event_id is not None

    submit_fingerprint = attention_protocol_evidence_fingerprint(submit_result)
    completion_fingerprint = attention_protocol_evidence_fingerprint(completion_result)
    if protocol_trace.route != AttentionProtocolRoute.PROVIDER:
        raise AttentionAccuracyExecutionBindingError(
            "accuracy execution requires provider protocol route"
        )
    if (
        protocol_trace.subject_fingerprint != packet_fingerprint
        or protocol_trace.stream_context_fingerprint
        != identity.stream_context_fingerprint
    ):
        raise AttentionAccuracyExecutionBindingError(
            "provider protocol subject or stream does not match launch packet"
        )
    states = tuple(event.state for event in protocol_trace.events)
    if (
        len(states) < 4
        or states[-2:] != (
            AttentionProtocolState.COMPLETED,
            AttentionProtocolState.RELEASED,
        )
        or any(
            state
            in {
                AttentionProtocolState.FAILED_SYNC,
                AttentionProtocolState.FAILED_ASYNC,
                AttentionProtocolState.RUNTIME_QUIESCED,
            }
            for state in states
        )
    ):
        raise AttentionAccuracyExecutionBindingError(
            "provider protocol is not a successful completed/released path"
        )
    if not any(
        event.state == AttentionProtocolState.SUBMITTED
        and event.evidence_fingerprint == submit_fingerprint
        for event in protocol_trace.events
    ):
        raise AttentionAccuracyExecutionBindingError(
            "provider protocol does not bind successful submission result"
        )
    if not any(
        event.state == AttentionProtocolState.COMPLETED
        and event.evidence_fingerprint == completion_fingerprint
        for event in protocol_trace.events
    ):
        raise AttentionAccuracyExecutionBindingError(
            "provider protocol does not bind successful completion result"
        )

    stream = launch_packet.stream_binding
    return AttentionAccuracyExecutionBinding(
        binding_id=binding_id,
        runner=runner,
        result_origin=_RESULT_ORIGIN,
        accuracy_dispatch_binding_fingerprint=accuracy_dispatch_binding.fingerprint,
        candidate_result_fingerprint=candidate_fingerprint,
        launch_packet_fingerprint=packet_fingerprint,
        execution_identity_fingerprint=identity.fingerprint,
        dispatch_receipt_fingerprint=receipt.fingerprint,
        launch_lease_fingerprint=launch_packet.launch_lease.fingerprint,
        stream_context_fingerprint=identity.stream_context_fingerprint,
        runtime_id=stream.runtime_id,
        runtime_generation=stream.runtime_generation,
        provider_probe_fingerprint=resolved_launcher.provider_probe_fingerprint,
        resolved_launcher_fingerprint=resolved_launcher.fingerprint,
        submit_result_fingerprint=submit_fingerprint,
        completion_result_fingerprint=completion_fingerprint,
        protocol_trace_fingerprint=protocol_trace.fingerprint,
        submission_id=submit_result.submission_id,
        completion_event_id=submit_result.completion_event_id,
    )


__all__ = [
    "ATTENTION_ACCURACY_EXECUTION_BINDING_VERSION",
    "AttentionAccuracyExecutionBinding",
    "AttentionAccuracyExecutionBindingError",
    "bind_attention_accuracy_execution",
]
