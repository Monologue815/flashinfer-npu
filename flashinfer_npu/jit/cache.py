"""Pure JIT cache identity and resolution decisions.

No file is read, written, compiled, or loaded here.  Providers will later
materialize verified records behind this contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    SchemaError,
)

from .core import JitSpec
from .env import JitCompilationPolicy


JIT_CACHE_SCHEMA_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if len(str(value)) != 64 or any(
        item not in "0123456789abcdef" for item in str(value)
    ):
        raise SchemaError("JIT cache %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class JitCacheRecord:
    """Verified identity of a compiled artifact already present in a cache."""

    spec_name: str
    spec_fingerprint: str
    environment_fingerprint: str
    artifact: ArtifactRef
    producer_id: str
    build_metadata_fingerprint: str
    schema_version: int = JIT_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_CACHE_SCHEMA_VERSION:
            raise SchemaError("unsupported JIT cache schema version")
        if not self.spec_name or not self.producer_id:
            raise SchemaError("JIT cache spec_name and producer_id must be non-empty")
        for name in (
            "spec_fingerprint",
            "environment_fingerprint",
            "build_metadata_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("JIT cache artifact must be ArtifactRef")
        if self.artifact.kind != ArtifactKind.FILE or self.artifact.format not in {
            ArtifactFormat.ASCENDC_OBJECT,
            ArtifactFormat.SHARED_LIBRARY,
        }:
            raise SchemaError("JIT cache record must identify a compiled file artifact")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "artifact": self.artifact.to_dict(),
            "producer_id": self.producer_id,
            "build_metadata_fingerprint": self.build_metadata_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JitCacheRecord":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("JitCacheRecord fields are invalid")
        artifact = data.get("artifact")
        if not isinstance(artifact, Mapping):
            raise SchemaError("JIT cache artifact must be an object")
        data["artifact"] = ArtifactRef.from_dict(artifact)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("JitCacheRecord fields are invalid") from error

    def validate_spec(self, spec: JitSpec) -> None:
        if (
            self.spec_name != spec.name
            or self.spec_fingerprint != spec.fingerprint
            or self.environment_fingerprint != spec.environment_fingerprint
            or self.artifact.target_soc != spec.target_soc
        ):
            raise SchemaError("JIT cache record does not bind the requested spec")


class JitCacheIndex:
    """Immutable-identity in-memory index used by Host framework tests."""

    def __init__(self, records: Iterable[JitCacheRecord] = ()) -> None:
        self._records: Dict[str, JitCacheRecord] = {}
        for record in records:
            self.publish(record)

    def publish(self, record: JitCacheRecord) -> JitCacheRecord:
        if not isinstance(record, JitCacheRecord):
            raise TypeError("JIT cache accepts JitCacheRecord")
        existing = self._records.get(record.spec_name)
        if existing is not None:
            if existing.fingerprint != record.fingerprint:
                raise SchemaError(
                    "conflicting JIT cache record for %r" % record.spec_name
                )
            return existing
        self._records[record.spec_name] = record
        return record

    def lookup(self, spec: JitSpec) -> Optional[JitCacheRecord]:
        if not isinstance(spec, JitSpec):
            raise TypeError("JIT cache lookup requires JitSpec")
        record = self._records.get(spec.name)
        if record is None:
            return None
        record.validate_spec(spec)
        return record

    @property
    def records(self) -> Tuple[JitCacheRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda item: item.spec_name))


class JitResolutionState(str, Enum):
    CACHE_HIT = "cache_hit"
    BUILD_REQUIRED = "build_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class JitResolution:
    spec_name: str
    spec_fingerprint: str
    state: JitResolutionState
    reason: str
    cache_record_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.spec_name or not self.reason:
            raise SchemaError("JIT resolution spec_name and reason must be non-empty")
        _require_hash("resolution spec_fingerprint", self.spec_fingerprint)
        object.__setattr__(self, "state", JitResolutionState(self.state))
        if self.state == JitResolutionState.CACHE_HIT:
            if self.cache_record_fingerprint is None:
                raise SchemaError("JIT cache hit requires a cache record fingerprint")
            _require_hash(
                "resolution cache_record_fingerprint",
                self.cache_record_fingerprint,
            )
        elif self.cache_record_fingerprint is not None:
            raise SchemaError("JIT cache miss cannot identify a cache record")

    @property
    def ready(self) -> bool:
        return self.state == JitResolutionState.CACHE_HIT


class MissingJITCacheError(RuntimeError):
    """Raised when the policy forbids satisfying a missing JIT artifact."""


def resolve_jit_spec(
    spec: JitSpec,
    cache: JitCacheIndex,
    policy: JitCompilationPolicy,
) -> JitResolution:
    """Return a pure decision; never compile, load, or touch the filesystem."""

    if not isinstance(spec, JitSpec):
        raise TypeError("resolve_jit_spec requires JitSpec")
    if not isinstance(cache, JitCacheIndex):
        raise TypeError("resolve_jit_spec requires JitCacheIndex")
    policy = JitCompilationPolicy(policy)
    record = cache.lookup(spec)
    if record is not None:
        return JitResolution(
            spec.name,
            spec.fingerprint,
            JitResolutionState.CACHE_HIT,
            "verified cache record matches spec and environment",
            record.fingerprint,
        )
    if policy == JitCompilationPolicy.ENABLED:
        return JitResolution(
            spec.name,
            spec.fingerprint,
            JitResolutionState.BUILD_REQUIRED,
            "cache miss requires an authorized build provider",
        )
    return JitResolution(
        spec.name,
        spec.fingerprint,
        JitResolutionState.UNAVAILABLE,
        "JIT cache miss and compilation policy is %s" % policy.value,
    )


def require_jit_cache_hit(resolution: JitResolution) -> None:
    if not isinstance(resolution, JitResolution):
        raise TypeError("resolution must be JitResolution")
    if not resolution.ready:
        raise MissingJITCacheError(resolution.reason)


__all__ = [
    "JIT_CACHE_SCHEMA_VERSION",
    "JitCacheIndex",
    "JitCacheRecord",
    "JitResolution",
    "JitResolutionState",
    "MissingJITCacheError",
    "require_jit_cache_hit",
    "resolve_jit_spec",
]
