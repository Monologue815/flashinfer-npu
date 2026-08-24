"""Address, allocation-generation, stream, and completion lease contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .tensor_contract import TensorView


ATTENTION_STORAGE_LEASE_VERSION = 1


def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class AttentionStorageLeaseError(RuntimeError):
    """Raised when storage lifetime or in-flight reuse is unsafe."""


class AttentionStorageLifetime(str, Enum):
    RUN = "run"
    PERSISTENT = "persistent"
    CAPTURE = "capture"


class AttentionLeaseState(str, Enum):
    ACQUIRED = "acquired"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    RELEASED = "released"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionStorageLease:
    lease_id: str
    storage_id: str
    device: str
    base_address: int
    capacity_bytes: int
    alignment: int
    allocator_id: str
    allocation_generation: int
    lifetime: AttentionStorageLifetime
    writable: bool
    schema_version: int = ATTENTION_STORAGE_LEASE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_STORAGE_LEASE_VERSION:
            raise SchemaError("unsupported Attention storage lease version")
        for name in ("lease_id", "storage_id", "device", "allocator_id"):
            if not str(getattr(self, name)):
                raise SchemaError("storage lease %s must be non-empty" % name)
        for name in ("base_address", "capacity_bytes", "alignment", "allocation_generation"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SchemaError("storage lease integer fields must be integers")
        if self.capacity_bytes < 0 or self.base_address < 0:
            raise SchemaError("storage lease address/capacity cannot be negative")
        if self.capacity_bytes and self.base_address == 0:
            raise SchemaError("non-empty storage lease requires a non-zero address")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise SchemaError("storage lease alignment must be a power of two")
        if self.base_address and self.base_address % self.alignment:
            raise SchemaError("storage lease base address violates alignment")
        if self.allocation_generation < 1:
            raise SchemaError("allocation_generation must be positive")
        object.__setattr__(self, "lifetime", AttentionStorageLifetime(self.lifetime))
        if not isinstance(self.writable, bool):
            raise SchemaError("storage lease writable must be boolean")

    @property
    def address_interval(self) -> Tuple[int, int]:
        return self.base_address, self.base_address + self.capacity_bytes

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["lifetime"] = self.lifetime.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionStorageLease":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionStorageLease fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionStorageLease fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionHostBufferLease:
    """Process-local ownership for launcher POD/metadata byte buffers.

    This is deliberately separate from :class:`AttentionStorageLease`: host
    descriptor pointers and NPU tensor addresses occupy different memory
    domains and must never be validated as though they were interchangeable.
    """

    lease_id: str
    base_address: int
    capacity_bytes: int
    alignment: int
    owner_id: str
    allocation_generation: int
    lifetime: AttentionStorageLifetime
    writable: bool
    pinned: bool = False
    schema_version: int = ATTENTION_STORAGE_LEASE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_STORAGE_LEASE_VERSION:
            raise SchemaError("unsupported Attention host buffer lease version")
        for name in ("lease_id", "owner_id"):
            if not str(getattr(self, name)):
                raise SchemaError("host buffer lease %s must be non-empty" % name)
        for name in (
            "base_address",
            "capacity_bytes",
            "alignment",
            "allocation_generation",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SchemaError("host buffer lease integer fields must be integers")
        if self.base_address < 0 or self.capacity_bytes < 0:
            raise SchemaError("host buffer lease address/capacity cannot be negative")
        if self.capacity_bytes and self.base_address == 0:
            raise SchemaError("non-empty host buffer lease requires a non-zero address")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise SchemaError("host buffer lease alignment must be a power of two")
        if self.base_address and self.base_address % self.alignment:
            raise SchemaError("host buffer lease base address violates alignment")
        if self.allocation_generation < 1:
            raise SchemaError("host buffer allocation_generation must be positive")
        try:
            object.__setattr__(self, "lifetime", AttentionStorageLifetime(self.lifetime))
        except ValueError as error:
            raise SchemaError("host buffer lease lifetime is invalid") from error
        if not isinstance(self.writable, bool) or not isinstance(self.pinned, bool):
            raise SchemaError("host buffer lease writable/pinned must be boolean")

    @property
    def address_interval(self) -> Tuple[int, int]:
        return self.base_address, self.base_address + self.capacity_bytes

    def overlaps(self, other: "AttentionHostBufferLease") -> bool:
        if self.owner_id != other.owner_id:
            return False
        left_start, left_end = self.address_interval
        right_start, right_end = other.address_interval
        return max(left_start, right_start) < min(left_end, right_end)

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["lifetime"] = self.lifetime.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionHostBufferLease":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionHostBufferLease fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionHostBufferLease fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionAddressBinding:
    role: str
    view: TensorView
    lease: AttentionStorageLease
    schema_version: int = ATTENTION_STORAGE_LEASE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_STORAGE_LEASE_VERSION:
            raise SchemaError("unsupported Attention address binding version")
        if not self.role:
            raise SchemaError("Attention address binding role must be non-empty")
        if not isinstance(self.view, TensorView):
            raise TypeError("view must be TensorView")
        if not isinstance(self.lease, AttentionStorageLease):
            raise TypeError("lease must be AttentionStorageLease")
        if (
            self.view.storage_id != self.lease.storage_id
            or self.view.device != self.lease.device
            or self.view.storage_nbytes != self.lease.capacity_bytes
        ):
            raise SchemaError("TensorView does not match storage lease identity")
        if self.view.writable and not self.lease.writable:
            raise SchemaError("writable TensorView requires writable storage lease")
        if self.data_address and self.data_address % self.view.data_ptr_alignment:
            raise SchemaError("leased data address violates TensorView alignment")

    @property
    def data_address(self) -> int:
        return self.lease.base_address + self.view.storage_offset * self.view.itemsize

    @property
    def absolute_byte_interval(self) -> Tuple[int, int]:
        start, end = self.view.byte_interval
        return self.lease.base_address + start, self.lease.base_address + end

    def overlaps(self, other: "AttentionAddressBinding") -> bool:
        if self.lease.allocator_id != other.lease.allocator_id:
            return False
        left_start, left_end = self.absolute_byte_interval
        right_start, right_end = other.absolute_byte_interval
        return max(left_start, right_start) < min(left_end, right_end)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "view": self.view.to_dict(),
            "lease": self.lease.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionAddressBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionAddressBinding fields are invalid")
        if not isinstance(data.get("view"), Mapping) or not isinstance(
            data.get("lease"), Mapping
        ):
            raise SchemaError("AttentionAddressBinding nested fields are invalid")
        data["view"] = TensorView.from_dict(data["view"])
        data["lease"] = AttentionStorageLease.from_dict(data["lease"])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionAddressBinding fields are invalid") from error


@dataclass(frozen=True)
class AttentionLaunchLeaseContract:
    execution_identity_fingerprint: str
    dispatch_receipt_fingerprint: str
    stream_device: str
    stream_id: str
    bindings: Tuple[AttentionAddressBinding, ...]
    graph_enabled: bool = False
    schema_version: int = ATTENTION_STORAGE_LEASE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_STORAGE_LEASE_VERSION:
            raise SchemaError("unsupported Attention launch lease version")
        _require_hash("execution identity fingerprint", self.execution_identity_fingerprint)
        _require_hash("dispatch receipt fingerprint", self.dispatch_receipt_fingerprint)
        if not self.stream_device or not self.stream_id:
            raise SchemaError("launch lease stream device/id must be non-empty")
        values = tuple(self.bindings)
        if not values or len({item.role for item in values}) != len(values):
            raise SchemaError("launch lease bindings must be non-empty with unique roles")
        if any(item.lease.device != self.stream_device for item in values):
            raise SchemaError("launch lease storage must be on the stream device")
        if not isinstance(self.graph_enabled, bool):
            raise SchemaError("launch lease graph_enabled must be boolean")
        if self.graph_enabled and any(
            item.lease.lifetime
            not in {AttentionStorageLifetime.PERSISTENT, AttentionStorageLifetime.CAPTURE}
            for item in values
        ):
            raise SchemaError("graph launch requires persistent/capture storage leases")
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                if left.overlaps(right) and (
                    left.view.writable or right.view.writable
                ):
                    raise SchemaError("overlapping launch bindings cannot be writable")
        object.__setattr__(self, "bindings", values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_identity_fingerprint": self.execution_identity_fingerprint,
            "dispatch_receipt_fingerprint": self.dispatch_receipt_fingerprint,
            "stream_device": self.stream_device,
            "stream_id": self.stream_id,
            "bindings": [item.to_dict() for item in self.bindings],
            "graph_enabled": self.graph_enabled,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionLaunchLeaseContract":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionLaunchLeaseContract fields are invalid")
        bindings = data.get("bindings")
        if not isinstance(bindings, (list, tuple)):
            raise SchemaError("launch lease bindings must be an array")
        data["bindings"] = tuple(
            AttentionAddressBinding.from_dict(item) for item in bindings
        )
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("AttentionLaunchLeaseContract fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate_reuse(self, candidate: "AttentionLaunchLeaseContract") -> None:
        if self.fingerprint != candidate.fingerprint:
            raise AttentionStorageLeaseError(
                "Attention launch lease changed; address/generation/stream reuse is stale"
            )

    def validate_tensor_contract(
        self,
        tensors,
        *,
        execution_identity_fingerprint: Optional[str] = None,
    ) -> None:
        """Require one exact address lease for every runtime TensorView role."""

        from .tensor_contract import AttentionRunTensorContract

        if not isinstance(tensors, AttentionRunTensorContract):
            raise TypeError("tensors must be AttentionRunTensorContract")
        if execution_identity_fingerprint is not None:
            _require_hash(
                "execution identity fingerprint", execution_identity_fingerprint
            )
            if self.execution_identity_fingerprint != execution_identity_fingerprint:
                raise SchemaError("launch lease is bound to another execution identity")
        if (
            self.stream_device != tensors.stream.device
            or self.stream_id != tensors.stream.stream_id
        ):
            raise SchemaError("launch lease stream does not match tensor contract")
        expected = dict(tensors.named_views)
        actual = {item.role: item.view for item in self.bindings}
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise SchemaError(
                "launch lease roles do not match tensor contract (missing=%r, extra=%r)"
                % (missing, extra)
            )
        if any(actual[role] != view for role, view in expected.items()):
            raise SchemaError("launch lease TensorView does not match tensor contract")


@dataclass
class _LeaseRecord:
    contract: AttentionLaunchLeaseContract
    state: AttentionLeaseState
    completion_event_id: Optional[str] = None


class AttentionLeaseRegistry:
    """Small host-side state machine for future asynchronous launch ownership."""

    def __init__(self) -> None:
        self._records: Dict[str, _LeaseRecord] = {}
        self._generation = 0
        self._lock = RLock()

    def _active_records(self):
        return tuple(
            (token, item)
            for token, item in self._records.items()
            if item.state in {AttentionLeaseState.ACQUIRED, AttentionLeaseState.SUBMITTED}
        )

    def _assert_available(
        self,
        contract: AttentionLaunchLeaseContract,
        *,
        exclude_token: Optional[str] = None,
    ) -> None:
        for token, existing in self._active_records():
            if token == exclude_token:
                continue
            for left in contract.bindings:
                for right in existing.contract.bindings:
                    if left.overlaps(right) and (
                        left.view.writable or right.view.writable
                    ):
                        raise AttentionStorageLeaseError(
                            "storage interval is already leased by an active launch"
                        )

    @_synchronized
    def acquire(self, contract: AttentionLaunchLeaseContract) -> str:
        if not isinstance(contract, AttentionLaunchLeaseContract):
            raise TypeError("contract must be AttentionLaunchLeaseContract")
        self._assert_available(contract)
        self._generation += 1
        token = "attention-lease-%d" % self._generation
        self._records[token] = _LeaseRecord(contract, AttentionLeaseState.ACQUIRED)
        return token

    @_synchronized
    def state(self, token: str) -> AttentionLeaseState:
        try:
            return self._records[token].state
        except KeyError as error:
            raise AttentionStorageLeaseError("unknown Attention lease token") from error

    @_synchronized
    def completion_event_id(self, token: str) -> Optional[str]:
        return self._record(token).completion_event_id

    @_synchronized
    def validate_contract(
        self, token: str, candidate: AttentionLaunchLeaseContract
    ) -> None:
        record = self._record(token)
        record.contract.validate_reuse(candidate)

    @_synchronized
    def submit(self, token: str, completion_event_id: str) -> None:
        record = self._record(token)
        if record.state not in {AttentionLeaseState.ACQUIRED, AttentionLeaseState.COMPLETED}:
            raise AttentionStorageLeaseError("Attention lease is already in flight")
        if not completion_event_id:
            raise AttentionStorageLeaseError("completion event id must be non-empty")
        self._assert_available(record.contract, exclude_token=token)
        record.state = AttentionLeaseState.SUBMITTED
        record.completion_event_id = completion_event_id

    @_synchronized
    def complete(self, token: str, completion_event_id: str) -> None:
        record = self._record(token)
        if record.state != AttentionLeaseState.SUBMITTED:
            raise AttentionStorageLeaseError("Attention lease is not submitted")
        if completion_event_id != record.completion_event_id:
            raise AttentionStorageLeaseError("completion event does not own this lease")
        record.state = AttentionLeaseState.COMPLETED

    @_synchronized
    def quiesce_after_runtime_teardown(
        self, token: str, teardown_evidence_fingerprint: str
    ) -> None:
        """Mark storage safe only after an exact runtime teardown proof."""

        _require_hash("teardown evidence fingerprint", teardown_evidence_fingerprint)
        record = self._record(token)
        if record.state == AttentionLeaseState.RELEASED:
            raise AttentionStorageLeaseError("Attention lease is already released")
        record.state = AttentionLeaseState.COMPLETED
        record.completion_event_id = "runtime-teardown:%s" % teardown_evidence_fingerprint

    @_synchronized
    def release(self, token: str) -> None:
        record = self._record(token)
        if record.state == AttentionLeaseState.SUBMITTED:
            raise AttentionStorageLeaseError(
                "cannot release storage before completion event"
            )
        if record.state == AttentionLeaseState.RELEASED:
            raise AttentionStorageLeaseError("Attention lease is already released")
        record.state = AttentionLeaseState.RELEASED

    def _record(self, token: str) -> _LeaseRecord:
        try:
            return self._records[token]
        except KeyError as error:
            raise AttentionStorageLeaseError("unknown Attention lease token") from error


__all__ = [
    "ATTENTION_STORAGE_LEASE_VERSION",
    "AttentionAddressBinding",
    "AttentionLaunchLeaseContract",
    "AttentionHostBufferLease",
    "AttentionLeaseRegistry",
    "AttentionLeaseState",
    "AttentionStorageLease",
    "AttentionStorageLeaseError",
    "AttentionStorageLifetime",
]
