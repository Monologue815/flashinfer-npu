"""Verified JIT artifact identities without a built-in filesystem reader.

The framework accepts bytes only through an injected reader.  It verifies the
declared size and SHA-256 before publishing an immutable receipt; it does not
load a shared object, resolve a symbol, initialize a device, or launch code.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from flashinfer_npu.runtime import ArtifactRef, SchemaError

from .cache import JitCacheRecord


JIT_ARTIFACT_VERIFICATION_VERSION = 1
_VERIFIER_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("JIT artifact %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class JitArtifactVerification:
    """Receipt proving that exact cache artifact bytes were reverified."""

    spec_name: str
    spec_fingerprint: str
    cache_record_fingerprint: str
    artifact_fingerprint: str
    payload_digest: str
    payload_size_bytes: int
    verifier_id: str
    schema_version: int = JIT_ARTIFACT_VERIFICATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_ARTIFACT_VERIFICATION_VERSION:
            raise SchemaError("unsupported JIT artifact verification version")
        if not str(self.spec_name):
            raise SchemaError("JIT artifact spec_name must be non-empty")
        if not _VERIFIER_ID.fullmatch(str(self.verifier_id)):
            raise SchemaError("invalid JIT artifact verifier_id")
        for name in (
            "spec_fingerprint",
            "cache_record_fingerprint",
            "artifact_fingerprint",
            "payload_digest",
        ):
            _require_hash(name, getattr(self, name))
        if (
            not isinstance(self.payload_size_bytes, int)
            or isinstance(self.payload_size_bytes, bool)
            or self.payload_size_bytes < 0
        ):
            raise SchemaError("JIT artifact payload size must be non-negative")
        object.__setattr__(self, "spec_name", str(self.spec_name))
        object.__setattr__(self, "verifier_id", str(self.verifier_id))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "cache_record_fingerprint": self.cache_record_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "payload_digest": self.payload_digest,
            "payload_size_bytes": self.payload_size_bytes,
            "verifier_id": self.verifier_id,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate_record(self, record: JitCacheRecord) -> None:
        if not isinstance(record, JitCacheRecord):
            raise TypeError("record must be JitCacheRecord")
        if (
            self.spec_name != record.spec_name
            or self.spec_fingerprint != record.spec_fingerprint
            or self.cache_record_fingerprint != record.fingerprint
            or self.artifact_fingerprint != record.artifact.fingerprint
            or self.payload_digest != record.artifact.digest
            or self.payload_size_bytes != record.artifact.size_bytes
        ):
            raise SchemaError("JIT artifact verification does not bind cache record")


def verify_jit_cache_record_payload(
    record: JitCacheRecord,
    payload: bytes,
    *,
    verifier_id: str,
) -> JitArtifactVerification:
    """Verify injected bytes and return a non-executable identity receipt."""

    if not isinstance(record, JitCacheRecord):
        raise TypeError("record must be JitCacheRecord")
    record.artifact.verify_bytes(payload)
    return JitArtifactVerification(
        spec_name=record.spec_name,
        spec_fingerprint=record.spec_fingerprint,
        cache_record_fingerprint=record.fingerprint,
        artifact_fingerprint=record.artifact.fingerprint,
        payload_digest=hashlib.sha256(payload).hexdigest(),
        payload_size_bytes=len(payload),
        verifier_id=verifier_id,
    )


@runtime_checkable
class JitArtifactPayloadReader(Protocol):
    """Injected byte source; no filesystem implementation is installed."""

    def read(self, artifact: ArtifactRef) -> bytes:
        """Return the bytes for exactly one declared artifact identity."""


@runtime_checkable
class JitArtifactVerifier(Protocol):
    verifier_id: str

    def verify(self, record: JitCacheRecord) -> JitArtifactVerification:
        """Reverify one cache record without loading or executing it."""


class ConfiguredJitArtifactVerifier:
    """Composition of a named verifier and an injected payload reader."""

    def __init__(self, verifier_id: str, reader: JitArtifactPayloadReader) -> None:
        if not _VERIFIER_ID.fullmatch(str(verifier_id)):
            raise SchemaError("invalid JIT artifact verifier_id")
        if not isinstance(reader, JitArtifactPayloadReader):
            raise TypeError("reader must implement JitArtifactPayloadReader")
        self.verifier_id = str(verifier_id)
        self._reader = reader

    def verify(self, record: JitCacheRecord) -> JitArtifactVerification:
        if not isinstance(record, JitCacheRecord):
            raise TypeError("record must be JitCacheRecord")
        payload = self._reader.read(record.artifact)
        if not isinstance(payload, bytes):
            raise TypeError("JIT artifact reader must return bytes")
        return verify_jit_cache_record_payload(
            record, payload, verifier_id=self.verifier_id
        )


__all__ = [
    "JIT_ARTIFACT_VERIFICATION_VERSION",
    "ConfiguredJitArtifactVerifier",
    "JitArtifactPayloadReader",
    "JitArtifactVerification",
    "JitArtifactVerifier",
    "verify_jit_cache_record_payload",
]
