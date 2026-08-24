"""Host-only loader/provider evidence and asynchronous launch state machine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    Backend,
    KernelDescriptor,
    SchemaError,
)

from .dispatch import AttentionDispatchReceipt
from .launch_contract import ATTENTION_KERNEL_ERROR_ABI
from .launch_packet import AttentionLaunchPacket
from .protocol_trace import (
    AttentionProtocolRecorder,
    AttentionProtocolRoute,
    AttentionProtocolState,
    AttentionProtocolTrace,
    attention_protocol_fingerprint,
    start_attention_protocol_recording,
)
from .storage_lease import AttentionLeaseRegistry, AttentionLeaseState


ATTENTION_LAUNCHER_PROVIDER_VERSION = 1


def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class AttentionArtifactVerificationKind(str, Enum):
    BYTES = "bytes"
    BUILTIN_CONTRACT = "builtin_contract"


class AttentionLaunchSessionState(str, Enum):
    PREPARED = "prepared"
    SUBMIT_UNKNOWN = "submit_unknown"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED_SYNC = "failed_sync"
    FAILED_ASYNC = "failed_async"
    RELEASED = "released"


class AttentionResolvedLauncherState(str, Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"


class AttentionUnknownSubmitStatus(str, Enum):
    INDETERMINATE = "indeterminate"
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    COMPLETED = "completed"


class AttentionLauncherProviderError(RuntimeError):
    """Raised when provider evidence or lifecycle behavior violates the ABI."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _error_code(code: int):
    if not isinstance(code, int) or isinstance(code, bool):
        raise SchemaError("Attention provider error code must be an integer")
    try:
        return next(item for item in ATTENTION_KERNEL_ERROR_ABI.codes if item.code == code)
    except StopIteration as error:
        raise SchemaError("unknown Attention kernel error code %r" % code) from error


@dataclass(frozen=True)
class AttentionProviderProbe:
    provider_id: str
    provider_version: str
    backend: Backend
    environment_fingerprint: str
    supported_artifact_formats: Tuple[ArtifactFormat, ...]
    binary_abi_fingerprint: str
    error_abi_fingerprint: str
    pointer_width_bits: int = 64
    calling_convention: str = "c"
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention provider probe version")
        for name in ("provider_id", "provider_version"):
            if not str(getattr(self, name)):
                raise SchemaError("Attention provider probe %s must be non-empty" % name)
        try:
            object.__setattr__(self, "backend", Backend(self.backend))
            formats = tuple(ArtifactFormat(item) for item in self.supported_artifact_formats)
        except ValueError as error:
            raise SchemaError("Attention provider probe enum is invalid") from error
        if not formats or len(set(formats)) != len(formats):
            raise SchemaError("provider artifact formats must be non-empty and unique")
        object.__setattr__(self, "supported_artifact_formats", formats)
        _require_hash("environment_fingerprint", self.environment_fingerprint)
        _require_hash("binary_abi_fingerprint", self.binary_abi_fingerprint)
        _require_hash("error_abi_fingerprint", self.error_abi_fingerprint)
        if self.pointer_width_bits != 64 or self.calling_convention != "c":
            raise SchemaError("Attention provider v1 requires C calling convention/u64 pointers")
        if self.error_abi_fingerprint != ATTENTION_KERNEL_ERROR_ABI.fingerprint:
            raise SchemaError("Attention provider error ABI is not canonical")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "backend": self.backend.value,
            "environment_fingerprint": self.environment_fingerprint,
            "supported_artifact_formats": [
                item.value for item in self.supported_artifact_formats
            ],
            "binary_abi_fingerprint": self.binary_abi_fingerprint,
            "error_abi_fingerprint": self.error_abi_fingerprint,
            "pointer_width_bits": self.pointer_width_bits,
            "calling_convention": self.calling_convention,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProviderProbe":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProviderProbe fields are invalid")
        formats = data.get("supported_artifact_formats")
        if not isinstance(formats, (list, tuple)):
            raise SchemaError("provider artifact formats must be an array")
        data["supported_artifact_formats"] = tuple(formats)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionProviderProbe fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionArtifactLoadEvidence:
    artifact_fingerprint: str
    provider_probe_fingerprint: str
    verification_kind: AttentionArtifactVerificationKind
    verified_digest: str
    verified_size_bytes: Optional[int]
    loader_instance_id: str
    loader_generation: int
    artifact_handle: int
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention artifact load evidence version")
        for name in (
            "artifact_fingerprint",
            "provider_probe_fingerprint",
            "verified_digest",
        ):
            _require_hash(name, getattr(self, name))
        try:
            object.__setattr__(
                self, "verification_kind", AttentionArtifactVerificationKind(self.verification_kind)
            )
        except ValueError as error:
            raise SchemaError("artifact verification kind is invalid") from error
        if not self.loader_instance_id:
            raise SchemaError("loader_instance_id must be non-empty")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.loader_generation, self.artifact_handle)
        ) or self.loader_generation < 1 or self.artifact_handle <= 0:
            raise SchemaError("loader generation/artifact handle must be positive")
        if self.verification_kind == AttentionArtifactVerificationKind.BYTES:
            if (
                not isinstance(self.verified_size_bytes, int)
                or isinstance(self.verified_size_bytes, bool)
                or self.verified_size_bytes < 0
            ):
                raise SchemaError("byte verification requires non-negative size")
        elif self.verified_size_bytes is not None:
            raise SchemaError("builtin contract verification cannot claim byte size")

    @classmethod
    def verify_bytes(
        cls,
        artifact: ArtifactRef,
        payload: bytes,
        probe: AttentionProviderProbe,
        *,
        loader_instance_id: str,
        loader_generation: int,
        artifact_handle: int,
    ) -> "AttentionArtifactLoadEvidence":
        if artifact.kind == ArtifactKind.BUILTIN:
            raise SchemaError("builtin artifact requires provider-contract verification")
        artifact.verify_bytes(payload)
        return cls(
            artifact.fingerprint,
            probe.fingerprint,
            AttentionArtifactVerificationKind.BYTES,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            loader_instance_id,
            loader_generation,
            artifact_handle,
        )

    @classmethod
    def verify_builtin_contract(
        cls,
        artifact: ArtifactRef,
        provider_contract_digest: str,
        probe: AttentionProviderProbe,
        *,
        loader_instance_id: str,
        loader_generation: int,
        artifact_handle: int,
    ) -> "AttentionArtifactLoadEvidence":
        if artifact.kind != ArtifactKind.BUILTIN:
            raise SchemaError("provider-contract verification requires builtin artifact")
        _require_hash("provider_contract_digest", provider_contract_digest)
        if provider_contract_digest != artifact.digest:
            raise SchemaError("builtin provider contract digest does not match artifact")
        return cls(
            artifact.fingerprint,
            probe.fingerprint,
            AttentionArtifactVerificationKind.BUILTIN_CONTRACT,
            provider_contract_digest,
            None,
            loader_instance_id,
            loader_generation,
            artifact_handle,
        )

    def validate(self, artifact: ArtifactRef, probe: AttentionProviderProbe) -> None:
        if (
            self.artifact_fingerprint != artifact.fingerprint
            or self.provider_probe_fingerprint != probe.fingerprint
            or self.verified_digest != artifact.digest
        ):
            raise SchemaError("artifact load evidence is stale")
        if artifact.kind == ArtifactKind.BUILTIN:
            if self.verification_kind != AttentionArtifactVerificationKind.BUILTIN_CONTRACT:
                raise SchemaError("builtin artifact load evidence kind is invalid")
        elif (
            self.verification_kind != AttentionArtifactVerificationKind.BYTES
            or self.verified_size_bytes != artifact.size_bytes
        ):
            raise SchemaError("byte artifact load evidence is invalid")

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["verification_kind"] = self.verification_kind.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionArtifactLoadEvidence":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionArtifactLoadEvidence fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionArtifactLoadEvidence fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionResolvedLauncher:
    provider_probe_fingerprint: str
    load_evidence_fingerprint: str
    dispatch_receipt_fingerprint: str
    kernel_id: str
    kernel_fingerprint: str
    artifact_fingerprint: str
    launch_abi_fingerprint: str
    binary_abi_fingerprint: str
    entry_point: str
    symbol_address: int
    symbol_generation: int
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention resolved launcher version")
        for name in (
            "provider_probe_fingerprint",
            "load_evidence_fingerprint",
            "dispatch_receipt_fingerprint",
            "kernel_fingerprint",
            "artifact_fingerprint",
            "launch_abi_fingerprint",
            "binary_abi_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if not self.kernel_id or not self.entry_point:
            raise SchemaError("resolved launcher kernel/entry point must be non-empty")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.symbol_address, self.symbol_generation)
        ) or self.symbol_address <= 0 or self.symbol_generation < 1:
            raise SchemaError("resolved symbol address/generation must be positive")

    def validate_packet(self, packet: AttentionLaunchPacket) -> None:
        if not isinstance(packet, AttentionLaunchPacket):
            raise TypeError("packet must be AttentionLaunchPacket")
        receipt = packet.dispatch_receipt
        if (
            self.dispatch_receipt_fingerprint != receipt.fingerprint
            or self.kernel_id != receipt.kernel_id
            or self.kernel_fingerprint != receipt.kernel_fingerprint
            or self.artifact_fingerprint != receipt.artifact_fingerprint
            or self.launch_abi_fingerprint != receipt.launch_abi_fingerprint
            or self.binary_abi_fingerprint != receipt.binary_abi_fingerprint
        ):
            raise SchemaError("resolved launcher is stale for launch packet")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionResolvedLauncher":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionResolvedLauncher fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionResolvedLauncher fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def resolve_attention_launcher(
    descriptor: KernelDescriptor,
    receipt: AttentionDispatchReceipt,
    probe: AttentionProviderProbe,
    load_evidence: AttentionArtifactLoadEvidence,
    *,
    symbol_address: int,
    symbol_generation: int,
) -> AttentionResolvedLauncher:
    if not isinstance(descriptor, KernelDescriptor):
        raise TypeError("descriptor must be KernelDescriptor")
    if not isinstance(receipt, AttentionDispatchReceipt):
        raise TypeError("receipt must be AttentionDispatchReceipt")
    if not isinstance(probe, AttentionProviderProbe):
        raise TypeError("probe must be AttentionProviderProbe")
    if not isinstance(load_evidence, AttentionArtifactLoadEvidence):
        raise TypeError("load_evidence must be AttentionArtifactLoadEvidence")
    if descriptor.artifact is None or descriptor.launch_abi is None or descriptor.binary_abi is None:
        raise SchemaError("resolved launcher requires complete descriptor provenance")
    artifact = descriptor.artifact
    if (
        descriptor.kernel_id != receipt.kernel_id
        or descriptor.fingerprint != receipt.kernel_fingerprint
        or artifact.fingerprint != receipt.artifact_fingerprint
        or descriptor.launch_abi.fingerprint != receipt.launch_abi_fingerprint
        or descriptor.binary_abi.fingerprint != receipt.binary_abi_fingerprint
    ):
        raise SchemaError("descriptor is stale for dispatch receipt")
    if (
        probe.backend != descriptor.backend
        or probe.environment_fingerprint != receipt.environment_fingerprint
        or artifact.format not in probe.supported_artifact_formats
        or probe.binary_abi_fingerprint != descriptor.binary_abi.fingerprint
        or probe.error_abi_fingerprint != descriptor.binary_abi.error_abi.fingerprint
    ):
        raise SchemaError("provider probe cannot resolve descriptor")
    load_evidence.validate(artifact, probe)
    return AttentionResolvedLauncher(
        probe.fingerprint,
        load_evidence.fingerprint,
        receipt.fingerprint,
        descriptor.kernel_id,
        descriptor.fingerprint,
        artifact.fingerprint,
        descriptor.launch_abi.fingerprint,
        descriptor.binary_abi.fingerprint,
        descriptor.launch_abi.entry_point,
        symbol_address,
        symbol_generation,
    )


@dataclass(frozen=True)
class AttentionProviderSubmitResult:
    provider_probe_fingerprint: str
    resolved_launcher_fingerprint: str
    launch_packet_fingerprint: str
    return_code: int
    submission_id: Optional[str] = None
    completion_event_id: Optional[str] = None
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention provider submit result version")
        for name in (
            "provider_probe_fingerprint",
            "resolved_launcher_fingerprint",
            "launch_packet_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        code = _error_code(self.return_code)
        if code.asynchronous:
            raise SchemaError("asynchronous error code cannot be a synchronous return")
        if self.return_code == 0:
            if not self.submission_id or not self.completion_event_id:
                raise SchemaError("successful submission requires submission/event ids")
        elif self.submission_id is not None or self.completion_event_id is not None:
            raise SchemaError("rejected submission cannot claim submission/event ids")

    @property
    def code_name(self) -> str:
        return _error_code(self.return_code).name

    @property
    def retryable(self) -> bool:
        return _error_code(self.return_code).retryable

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProviderSubmitResult":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProviderSubmitResult fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionProviderSubmitResult fields are invalid") from error


@dataclass(frozen=True)
class AttentionProviderCompletionResult:
    provider_probe_fingerprint: str
    resolved_launcher_fingerprint: str
    launch_packet_fingerprint: str
    completion_event_id: str
    return_code: int
    error_detail_digest: Optional[str] = None
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention provider completion result version")
        for name in (
            "provider_probe_fingerprint",
            "resolved_launcher_fingerprint",
            "launch_packet_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if not self.completion_event_id:
            raise SchemaError("completion event id must be non-empty")
        code = _error_code(self.return_code)
        if self.return_code == 0:
            if self.error_detail_digest is not None:
                raise SchemaError("successful completion cannot claim error detail")
        else:
            if not code.asynchronous:
                raise SchemaError("completion can only report asynchronous error code")
            _require_hash("error_detail_digest", str(self.error_detail_digest))

    @property
    def code_name(self) -> str:
        return _error_code(self.return_code).name

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProviderCompletionResult":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProviderCompletionResult fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionProviderCompletionResult fields are invalid") from error


@dataclass(frozen=True)
class AttentionProviderRecoveryResult:
    provider_probe_fingerprint: str
    resolved_launcher_fingerprint: str
    launch_packet_fingerprint: str
    attempt_number: int
    status: AttentionUnknownSubmitStatus
    submission_id: Optional[str] = None
    completion_event_id: Optional[str] = None
    completion_code: Optional[int] = None
    error_detail_digest: Optional[str] = None
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention provider recovery result version")
        for name in (
            "provider_probe_fingerprint",
            "resolved_launcher_fingerprint",
            "launch_packet_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if (
            not isinstance(self.attempt_number, int)
            or isinstance(self.attempt_number, bool)
            or self.attempt_number < 1
        ):
            raise SchemaError("recovery attempt_number must be positive")
        try:
            object.__setattr__(self, "status", AttentionUnknownSubmitStatus(self.status))
        except ValueError as error:
            raise SchemaError("unknown submit recovery status is invalid") from error
        has_ids = bool(self.submission_id) and bool(self.completion_event_id)
        if self.status in {
            AttentionUnknownSubmitStatus.INDETERMINATE,
            AttentionUnknownSubmitStatus.NOT_SUBMITTED,
        }:
            if any(
                value is not None
                for value in (
                    self.submission_id,
                    self.completion_event_id,
                    self.completion_code,
                    self.error_detail_digest,
                )
            ):
                raise SchemaError("non-submitted recovery status cannot claim launch evidence")
        elif not has_ids:
            raise SchemaError("submitted recovery status requires submission/event ids")
        if self.status == AttentionUnknownSubmitStatus.SUBMITTED:
            if self.completion_code is not None or self.error_detail_digest is not None:
                raise SchemaError("submitted recovery status cannot claim completion")
        elif self.status == AttentionUnknownSubmitStatus.COMPLETED:
            if self.completion_code is None:
                raise SchemaError("completed recovery status requires completion code")
            code = _error_code(self.completion_code)
            if self.completion_code == 0:
                if self.error_detail_digest is not None:
                    raise SchemaError("successful recovery completion cannot claim error detail")
            else:
                if not code.asynchronous:
                    raise SchemaError("recovery completion requires asynchronous error code")
                _require_hash("error_detail_digest", str(self.error_detail_digest))

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionProviderRecoveryResult":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionProviderRecoveryResult fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionProviderRecoveryResult fields are invalid") from error


@dataclass(frozen=True)
class AttentionRuntimeTeardownEvidence:
    provider_probe_fingerprint: str
    environment_fingerprint: str
    runtime_id: str
    runtime_generation: int
    teardown_id: str
    teardown_generation: int
    quiescence_digest: str
    schema_version: int = ATTENTION_LAUNCHER_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_LAUNCHER_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention runtime teardown evidence version")
        for name in (
            "provider_probe_fingerprint",
            "environment_fingerprint",
            "quiescence_digest",
        ):
            _require_hash(name, getattr(self, name))
        for name in ("runtime_id", "teardown_id"):
            if not str(getattr(self, name)):
                raise SchemaError("runtime teardown %s must be non-empty" % name)
        for name in ("runtime_generation", "teardown_generation"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SchemaError("runtime teardown generations must be positive integers")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionRuntimeTeardownEvidence":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionRuntimeTeardownEvidence fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionRuntimeTeardownEvidence fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionLauncherProvider(Protocol):
    def probe(self) -> AttentionProviderProbe:
        ...

    def submit(
        self,
        resolved: AttentionResolvedLauncher,
        packet: AttentionLaunchPacket,
    ) -> AttentionProviderSubmitResult:
        ...

    def query_completion(
        self,
        resolved: AttentionResolvedLauncher,
        packet: AttentionLaunchPacket,
        completion_event_id: str,
    ) -> AttentionProviderCompletionResult:
        ...

    def recover_submit(
        self,
        resolved: AttentionResolvedLauncher,
        packet: AttentionLaunchPacket,
        attempt_number: int,
    ) -> AttentionProviderRecoveryResult:
        ...


@dataclass
class _ResolvedLauncherRecord:
    resolved: AttentionResolvedLauncher
    load_evidence: AttentionArtifactLoadEvidence
    state: AttentionResolvedLauncherState
    active_tokens: Dict[str, None]


class AttentionResolvedLauncherRegistry:
    """Prevent artifact/symbol unload while any launch session owns it."""

    def __init__(self) -> None:
        self._records: Dict[str, _ResolvedLauncherRecord] = {}
        self._tokens: Dict[str, str] = {}
        self._generation = 0
        self._lock = RLock()

    @_synchronized
    def register(
        self,
        resolved: AttentionResolvedLauncher,
        load_evidence: AttentionArtifactLoadEvidence,
    ) -> None:
        if not isinstance(resolved, AttentionResolvedLauncher):
            raise TypeError("resolved must be AttentionResolvedLauncher")
        if not isinstance(load_evidence, AttentionArtifactLoadEvidence):
            raise TypeError("load_evidence must be AttentionArtifactLoadEvidence")
        if resolved.load_evidence_fingerprint != load_evidence.fingerprint:
            raise AttentionLauncherProviderError("resolved launcher load evidence is stale")
        key = resolved.fingerprint
        if key in self._records:
            raise AttentionLauncherProviderError("resolved launcher is already registered")
        self._records[key] = _ResolvedLauncherRecord(
            resolved,
            load_evidence,
            AttentionResolvedLauncherState.LOADED,
            {},
        )

    @_synchronized
    def acquire(self, resolved: AttentionResolvedLauncher) -> str:
        record = self._record(resolved)
        if record.state != AttentionResolvedLauncherState.LOADED:
            raise AttentionLauncherProviderError("resolved launcher is unloaded")
        self._generation += 1
        token = "attention-launcher-%d" % self._generation
        record.active_tokens[token] = None
        self._tokens[token] = resolved.fingerprint
        return token

    @_synchronized
    def release(self, token: str) -> None:
        try:
            key = self._tokens.pop(token)
            record = self._records[key]
            del record.active_tokens[token]
        except KeyError as error:
            raise AttentionLauncherProviderError("unknown resolved launcher token") from error

    @_synchronized
    def unload(self, resolved: AttentionResolvedLauncher) -> None:
        record = self._record(resolved)
        if record.active_tokens:
            raise AttentionLauncherProviderError(
                "cannot unload resolved launcher while launch sessions are active"
            )
        if record.state == AttentionResolvedLauncherState.UNLOADED:
            raise AttentionLauncherProviderError("resolved launcher is already unloaded")
        record.state = AttentionResolvedLauncherState.UNLOADED

    @_synchronized
    def state(
        self, resolved: AttentionResolvedLauncher
    ) -> AttentionResolvedLauncherState:
        return self._record(resolved).state

    @_synchronized
    def active_count(self, resolved: AttentionResolvedLauncher) -> int:
        return len(self._record(resolved).active_tokens)

    def _record(
        self, resolved: AttentionResolvedLauncher
    ) -> _ResolvedLauncherRecord:
        if not isinstance(resolved, AttentionResolvedLauncher):
            raise TypeError("resolved must be AttentionResolvedLauncher")
        try:
            record = self._records[resolved.fingerprint]
        except KeyError as error:
            raise AttentionLauncherProviderError("resolved launcher is not registered") from error
        if record.resolved != resolved:
            raise AttentionLauncherProviderError("resolved launcher registration is stale")
        return record


@dataclass
class _HostBufferRecord:
    packet: AttentionLaunchPacket
    state: AttentionLeaseState
    completion_event_id: Optional[str] = None


class AttentionHostBufferRegistry:
    """Reserve Host descriptor intervals until their launch event completes."""

    def __init__(self) -> None:
        self._records: Dict[str, _HostBufferRecord] = {}
        self._generation = 0
        self._lock = RLock()

    def _active_records(self):
        return tuple(
            (token, record)
            for token, record in self._records.items()
            if record.state in {AttentionLeaseState.ACQUIRED, AttentionLeaseState.SUBMITTED}
        )

    def _assert_available(
        self, packet: AttentionLaunchPacket, *, exclude_token: Optional[str] = None
    ) -> None:
        for token, record in self._active_records():
            if token == exclude_token:
                continue
            for left in packet.host_buffers:
                for right in record.packet.host_buffers:
                    if left.lease.overlaps(right.lease):
                        raise AttentionLauncherProviderError(
                            "Host descriptor interval is already leased by an active launch"
                        )

    @_synchronized
    def acquire(self, packet: AttentionLaunchPacket) -> str:
        if not isinstance(packet, AttentionLaunchPacket):
            raise TypeError("packet must be AttentionLaunchPacket")
        self._assert_available(packet)
        self._generation += 1
        token = "attention-host-buffer-%d" % self._generation
        self._records[token] = _HostBufferRecord(packet, AttentionLeaseState.ACQUIRED)
        return token

    @_synchronized
    def submit(self, token: str, completion_event_id: str) -> None:
        record = self._record(token)
        if record.state not in {AttentionLeaseState.ACQUIRED, AttentionLeaseState.COMPLETED}:
            raise AttentionLauncherProviderError("Host buffer lease is already in flight")
        if not completion_event_id:
            raise AttentionLauncherProviderError("completion event id must be non-empty")
        self._assert_available(record.packet, exclude_token=token)
        record.state = AttentionLeaseState.SUBMITTED
        record.completion_event_id = completion_event_id

    @_synchronized
    def complete(self, token: str, completion_event_id: str) -> None:
        record = self._record(token)
        if record.state != AttentionLeaseState.SUBMITTED:
            raise AttentionLauncherProviderError("Host buffer lease is not submitted")
        if completion_event_id != record.completion_event_id:
            raise AttentionLauncherProviderError("completion event does not own Host buffers")
        record.state = AttentionLeaseState.COMPLETED

    @_synchronized
    def quiesce_after_runtime_teardown(
        self, token: str, teardown_evidence_fingerprint: str
    ) -> None:
        _require_hash("teardown_evidence_fingerprint", teardown_evidence_fingerprint)
        record = self._record(token)
        if record.state == AttentionLeaseState.RELEASED:
            raise AttentionLauncherProviderError("Host buffer lease is already released")
        record.state = AttentionLeaseState.COMPLETED
        record.completion_event_id = (
            "runtime-teardown:%s" % teardown_evidence_fingerprint
        )

    @_synchronized
    def release(self, token: str) -> None:
        record = self._record(token)
        if record.state == AttentionLeaseState.SUBMITTED:
            raise AttentionLauncherProviderError(
                "cannot release Host buffers before completion event"
            )
        if record.state == AttentionLeaseState.RELEASED:
            raise AttentionLauncherProviderError("Host buffer lease is already released")
        record.state = AttentionLeaseState.RELEASED

    @_synchronized
    def state(self, token: str) -> AttentionLeaseState:
        return self._record(token).state

    @_synchronized
    def completion_event_id(self, token: str) -> Optional[str]:
        return self._record(token).completion_event_id

    def _record(self, token: str) -> _HostBufferRecord:
        try:
            return self._records[token]
        except KeyError as error:
            raise AttentionLauncherProviderError("unknown Host buffer lease token") from error

class AttentionLaunchSession:
    """Own one packet lease from preparation through event completion/release."""

    def __init__(
        self,
        packet: AttentionLaunchPacket,
        resolved: AttentionResolvedLauncher,
        registry: AttentionLeaseRegistry,
        host_buffer_registry: AttentionHostBufferRegistry,
        launcher_registry: AttentionResolvedLauncherRegistry,
    ) -> None:
        if not isinstance(packet, AttentionLaunchPacket):
            raise TypeError("packet must be AttentionLaunchPacket")
        if not isinstance(resolved, AttentionResolvedLauncher):
            raise TypeError("resolved must be AttentionResolvedLauncher")
        if not isinstance(registry, AttentionLeaseRegistry):
            raise TypeError("registry must be AttentionLeaseRegistry")
        if not isinstance(host_buffer_registry, AttentionHostBufferRegistry):
            raise TypeError("host_buffer_registry must be AttentionHostBufferRegistry")
        if not isinstance(launcher_registry, AttentionResolvedLauncherRegistry):
            raise TypeError("launcher_registry must be AttentionResolvedLauncherRegistry")
        resolved.validate_packet(packet)
        self.packet = packet
        self.resolved = resolved
        self.registry = registry
        self.lease_token = registry.acquire(packet.launch_lease)
        self.host_buffer_registry = host_buffer_registry
        try:
            self.host_buffer_token = host_buffer_registry.acquire(packet)
        except Exception:
            registry.release(self.lease_token)
            raise
        self.launcher_registry = launcher_registry
        try:
            self.launcher_token = launcher_registry.acquire(resolved)
        except Exception:
            host_buffer_registry.release(self.host_buffer_token)
            registry.release(self.lease_token)
            raise
        self.state = AttentionLaunchSessionState.PREPARED
        self.attempt_count = 0
        self.submission_id: Optional[str] = None
        self.completion_event_id: Optional[str] = None
        self.last_return_code: Optional[int] = None
        self._lock = RLock()
        ownership_fingerprint = attention_protocol_fingerprint(
            {
                "launch_lease": packet.launch_lease.fingerprint,
                "host_buffers": [item.fingerprint for item in packet.host_buffers],
                "resolved_launcher": resolved.fingerprint,
            }
        )
        self._protocol_recorder: Optional[AttentionProtocolRecorder] = (
            start_attention_protocol_recording(
                AttentionProtocolRoute.PROVIDER,
                packet.fingerprint,
                packet.execution_identity.stream_context_fingerprint,
                ownership_fingerprint,
            )
        )
        self._record_protocol(
            AttentionProtocolState.PREPARED,
            {
                "operation": "prepare",
                "packet_fingerprint": packet.fingerprint,
                "resolved_launcher_fingerprint": resolved.fingerprint,
                "device_lease_token": self.lease_token,
                "host_buffer_token": self.host_buffer_token,
                "launcher_token": self.launcher_token,
            },
        )

    @property
    def protocol_trace(self) -> Optional[AttentionProtocolTrace]:
        recorder = self._protocol_recorder
        return None if recorder is None else recorder.trace

    def _record_protocol(self, state: AttentionProtocolState, evidence: Any) -> None:
        if self._protocol_recorder is not None:
            self._protocol_recorder.record(state, evidence)

    def _probe(self, provider: AttentionLauncherProvider) -> AttentionProviderProbe:
        probe = provider.probe()
        if not isinstance(probe, AttentionProviderProbe):
            raise AttentionLauncherProviderError("provider probe returned wrong type")
        if probe.fingerprint != self.resolved.provider_probe_fingerprint:
            raise AttentionLauncherProviderError("provider probe changed after resolution")
        return probe

    @_synchronized
    def submit(
        self, provider: AttentionLauncherProvider
    ) -> AttentionProviderSubmitResult:
        if self.state != AttentionLaunchSessionState.PREPARED:
            raise AttentionLauncherProviderError("Attention launch session is not prepared")
        probe = self._probe(provider)
        self.attempt_count += 1
        try:
            result = provider.submit(self.resolved, self.packet)
        except Exception as error:
            self.state = AttentionLaunchSessionState.SUBMIT_UNKNOWN
            self._record_protocol(
                AttentionProtocolState.SUBMIT_UNKNOWN,
                {
                    "operation": "submit",
                    "attempt": self.attempt_count,
                    "outcome": "exception",
                    "exception_type": "%s.%s"
                    % (type(error).__module__, type(error).__qualname__),
                },
            )
            raise
        if not isinstance(result, AttentionProviderSubmitResult):
            self.state = AttentionLaunchSessionState.SUBMIT_UNKNOWN
            self._record_protocol(
                AttentionProtocolState.SUBMIT_UNKNOWN,
                {
                    "operation": "submit",
                    "attempt": self.attempt_count,
                    "outcome": "wrong_result_type",
                    "result_type": "%s.%s"
                    % (type(result).__module__, type(result).__qualname__),
                },
            )
            raise AttentionLauncherProviderError("provider submit returned wrong type")
        try:
            self._validate_result_identity(
                result.provider_probe_fingerprint,
                result.resolved_launcher_fingerprint,
                result.launch_packet_fingerprint,
                probe,
            )
        except AttentionLauncherProviderError:
            self.state = AttentionLaunchSessionState.SUBMIT_UNKNOWN
            self._record_protocol(
                AttentionProtocolState.SUBMIT_UNKNOWN,
                {
                    "operation": "submit",
                    "attempt": self.attempt_count,
                    "outcome": "identity_mismatch",
                    "result": result.to_dict(),
                },
            )
            raise
        self.last_return_code = result.return_code
        if result.return_code == 0:
            assert result.submission_id is not None
            assert result.completion_event_id is not None
            self.submission_id = result.submission_id
            self.completion_event_id = result.completion_event_id
            self.state = AttentionLaunchSessionState.SUBMIT_UNKNOWN
            self._record_protocol(AttentionProtocolState.SUBMIT_UNKNOWN, result)
            self._ensure_registries_submitted(result.completion_event_id)
            self.state = AttentionLaunchSessionState.SUBMITTED
            self._record_protocol(AttentionProtocolState.SUBMITTED, result)
        elif not result.retryable:
            self.state = AttentionLaunchSessionState.FAILED_SYNC
            self._record_protocol(AttentionProtocolState.FAILED_SYNC, result)
        return result

    @_synchronized
    def recover_unknown(
        self, provider: AttentionLauncherProvider
    ) -> AttentionProviderRecoveryResult:
        if self.state != AttentionLaunchSessionState.SUBMIT_UNKNOWN:
            raise AttentionLauncherProviderError(
                "Attention launch session has no unknown submission to recover"
            )
        probe = self._probe(provider)
        result = provider.recover_submit(
            self.resolved, self.packet, self.attempt_count
        )
        if not isinstance(result, AttentionProviderRecoveryResult):
            raise AttentionLauncherProviderError("provider recovery returned wrong type")
        self._validate_result_identity(
            result.provider_probe_fingerprint,
            result.resolved_launcher_fingerprint,
            result.launch_packet_fingerprint,
            probe,
        )
        if result.attempt_number != self.attempt_count:
            raise AttentionLauncherProviderError("provider recovery attempt is stale")
        if result.status == AttentionUnknownSubmitStatus.INDETERMINATE:
            self._record_protocol(AttentionProtocolState.SUBMIT_UNKNOWN, result)
            return result
        if result.status == AttentionUnknownSubmitStatus.NOT_SUBMITTED:
            if (
                self.registry.state(self.lease_token) != AttentionLeaseState.ACQUIRED
                or self.host_buffer_registry.state(self.host_buffer_token)
                != AttentionLeaseState.ACQUIRED
            ):
                raise AttentionLauncherProviderError(
                    "not-submitted recovery contradicts registered submission"
                )
            self.submission_id = None
            self.completion_event_id = None
            self.state = AttentionLaunchSessionState.PREPARED
            self._record_protocol(AttentionProtocolState.PREPARED, result)
            return result
        assert result.submission_id is not None
        assert result.completion_event_id is not None
        if self.submission_id is not None and self.submission_id != result.submission_id:
            raise AttentionLauncherProviderError("provider recovery submission id changed")
        if (
            self.completion_event_id is not None
            and self.completion_event_id != result.completion_event_id
        ):
            raise AttentionLauncherProviderError("provider recovery event id changed")
        self.submission_id = result.submission_id
        self.completion_event_id = result.completion_event_id
        self._ensure_registries_submitted(result.completion_event_id)
        if result.status == AttentionUnknownSubmitStatus.SUBMITTED:
            self.state = AttentionLaunchSessionState.SUBMITTED
            self._record_protocol(AttentionProtocolState.SUBMITTED, result)
            return result
        assert result.status == AttentionUnknownSubmitStatus.COMPLETED
        assert result.completion_code is not None
        self.registry.complete(self.lease_token, result.completion_event_id)
        self.host_buffer_registry.complete(
            self.host_buffer_token, result.completion_event_id
        )
        self.last_return_code = result.completion_code
        self.state = (
            AttentionLaunchSessionState.COMPLETED
            if result.completion_code == 0
            else AttentionLaunchSessionState.FAILED_ASYNC
        )
        self._record_protocol(AttentionProtocolState(self.state.value), result)
        return result

    @_synchronized
    def poll_completion(
        self, provider: AttentionLauncherProvider
    ) -> AttentionProviderCompletionResult:
        if self.state != AttentionLaunchSessionState.SUBMITTED:
            raise AttentionLauncherProviderError("Attention launch session is not submitted")
        probe = self._probe(provider)
        assert self.completion_event_id is not None
        result = provider.query_completion(
            self.resolved, self.packet, self.completion_event_id
        )
        if not isinstance(result, AttentionProviderCompletionResult):
            raise AttentionLauncherProviderError("provider completion returned wrong type")
        self._validate_result_identity(
            result.provider_probe_fingerprint,
            result.resolved_launcher_fingerprint,
            result.launch_packet_fingerprint,
            probe,
        )
        if result.completion_event_id != self.completion_event_id:
            raise AttentionLauncherProviderError("provider completion event ownership mismatch")
        self.registry.complete(self.lease_token, result.completion_event_id)
        self.host_buffer_registry.complete(
            self.host_buffer_token, result.completion_event_id
        )
        self.last_return_code = result.return_code
        self.state = (
            AttentionLaunchSessionState.COMPLETED
            if result.return_code == 0
            else AttentionLaunchSessionState.FAILED_ASYNC
        )
        self._record_protocol(AttentionProtocolState(self.state.value), result)
        return result

    @_synchronized
    def release(self) -> None:
        if self.state == AttentionLaunchSessionState.RELEASED:
            raise AttentionLauncherProviderError("Attention launch session is already released")
        if self.state == AttentionLaunchSessionState.SUBMIT_UNKNOWN:
            raise AttentionLauncherProviderError(
                "cannot release lease while provider submission outcome is unknown"
            )
        self.registry.release(self.lease_token)
        self.host_buffer_registry.release(self.host_buffer_token)
        self.launcher_registry.release(self.launcher_token)
        self.state = AttentionLaunchSessionState.RELEASED
        self._record_protocol(
            AttentionProtocolState.RELEASED,
            {
                "operation": "release",
                "lease_token": self.lease_token,
                "host_buffer_token": self.host_buffer_token,
                "launcher_token": self.launcher_token,
            },
        )

    @_synchronized
    def release_after_runtime_teardown(
        self, evidence: AttentionRuntimeTeardownEvidence
    ) -> None:
        if self.state not in {
            AttentionLaunchSessionState.SUBMIT_UNKNOWN,
            AttentionLaunchSessionState.SUBMITTED,
        }:
            raise AttentionLauncherProviderError(
                "runtime teardown release requires unknown/submitted session"
            )
        if not isinstance(evidence, AttentionRuntimeTeardownEvidence):
            raise TypeError("evidence must be AttentionRuntimeTeardownEvidence")
        stream = self.packet.stream_binding
        receipt = self.packet.dispatch_receipt
        if (
            evidence.provider_probe_fingerprint
            != self.resolved.provider_probe_fingerprint
            or evidence.environment_fingerprint != receipt.environment_fingerprint
            or evidence.runtime_id != stream.runtime_id
            or evidence.runtime_generation != stream.runtime_generation
        ):
            raise AttentionLauncherProviderError(
                "runtime teardown evidence does not own this session"
            )
        self.registry.quiesce_after_runtime_teardown(
            self.lease_token, evidence.fingerprint
        )
        self.host_buffer_registry.quiesce_after_runtime_teardown(
            self.host_buffer_token, evidence.fingerprint
        )
        self._record_protocol(AttentionProtocolState.RUNTIME_QUIESCED, evidence)
        self.registry.release(self.lease_token)
        self.host_buffer_registry.release(self.host_buffer_token)
        self.launcher_registry.release(self.launcher_token)
        self.state = AttentionLaunchSessionState.RELEASED
        self._record_protocol(
            AttentionProtocolState.RELEASED,
            {
                "operation": "release_after_runtime_teardown",
                "teardown_evidence_fingerprint": evidence.fingerprint,
            },
        )

    def _ensure_registries_submitted(self, completion_event_id: str) -> None:
        device_state = self.registry.state(self.lease_token)
        host_state = self.host_buffer_registry.state(self.host_buffer_token)
        if device_state == AttentionLeaseState.ACQUIRED:
            self.registry.submit(self.lease_token, completion_event_id)
        elif device_state == AttentionLeaseState.SUBMITTED:
            if (
                self.registry.completion_event_id(self.lease_token)
                != completion_event_id
            ):
                raise AttentionLauncherProviderError(
                    "device lease event contradicts recovered submission"
                )
        else:
            raise AttentionLauncherProviderError(
                "device lease state contradicts recovered submission"
            )
        if host_state == AttentionLeaseState.ACQUIRED:
            self.host_buffer_registry.submit(
                self.host_buffer_token, completion_event_id
            )
        elif host_state == AttentionLeaseState.SUBMITTED:
            if (
                self.host_buffer_registry.completion_event_id(
                    self.host_buffer_token
                )
                != completion_event_id
            ):
                raise AttentionLauncherProviderError(
                    "Host buffer event contradicts recovered submission"
                )
        else:
            raise AttentionLauncherProviderError(
                "Host buffer state contradicts recovered submission"
            )

    def _validate_result_identity(
        self,
        provider_probe_fingerprint: str,
        resolved_launcher_fingerprint: str,
        launch_packet_fingerprint: str,
        probe: AttentionProviderProbe,
    ) -> None:
        if (
            provider_probe_fingerprint != probe.fingerprint
            or resolved_launcher_fingerprint != self.resolved.fingerprint
            or launch_packet_fingerprint != self.packet.fingerprint
        ):
            raise AttentionLauncherProviderError("provider result identity is stale")


class AttentionLaunchCoordinator:
    """Share device, Host-buffer, and resolved-symbol ownership across sessions."""

    def __init__(self) -> None:
        self.device_leases = AttentionLeaseRegistry()
        self.host_buffers = AttentionHostBufferRegistry()
        self.launchers = AttentionResolvedLauncherRegistry()

    def register_launcher(
        self,
        resolved: AttentionResolvedLauncher,
        load_evidence: AttentionArtifactLoadEvidence,
    ) -> None:
        self.launchers.register(resolved, load_evidence)

    def prepare(
        self, packet: AttentionLaunchPacket, resolved: AttentionResolvedLauncher
    ) -> AttentionLaunchSession:
        return AttentionLaunchSession(
            packet,
            resolved,
            self.device_leases,
            self.host_buffers,
            self.launchers,
        )

    def unload_launcher(self, resolved: AttentionResolvedLauncher) -> None:
        self.launchers.unload(resolved)


__all__ = [
    "ATTENTION_LAUNCHER_PROVIDER_VERSION",
    "AttentionArtifactLoadEvidence",
    "AttentionArtifactVerificationKind",
    "AttentionHostBufferRegistry",
    "AttentionLaunchSession",
    "AttentionLaunchSessionState",
    "AttentionLaunchCoordinator",
    "AttentionLauncherProvider",
    "AttentionLauncherProviderError",
    "AttentionProviderCompletionResult",
    "AttentionProviderRecoveryResult",
    "AttentionProviderProbe",
    "AttentionProviderSubmitResult",
    "AttentionResolvedLauncher",
    "AttentionResolvedLauncherRegistry",
    "AttentionResolvedLauncherState",
    "AttentionRuntimeTeardownEvidence",
    "AttentionUnknownSubmitStatus",
    "resolve_attention_launcher",
]
