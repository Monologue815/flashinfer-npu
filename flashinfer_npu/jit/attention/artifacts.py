"""Attention binding for an exact byte-verified JIT cache artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from flashinfer_npu.jit.artifacts import JitArtifactVerification, JitArtifactVerifier
from flashinfer_npu.jit.cache import JitCacheIndex
from flashinfer_npu.runtime import SchemaError

from .plan import AttentionJitPlanBinding


ATTENTION_JIT_ARTIFACT_BINDING_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionJitArtifactBinding:
    """Byte-verification receipt frozen against one JIT plan binding."""

    jit_plan_binding_fingerprint: str
    verification: JitArtifactVerification
    schema_version: int = ATTENTION_JIT_ARTIFACT_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_ARTIFACT_BINDING_VERSION:
            raise SchemaError("unsupported Attention JIT artifact binding version")
        if len(str(self.jit_plan_binding_fingerprint)) != 64 or any(
            item not in "0123456789abcdef"
            for item in str(self.jit_plan_binding_fingerprint)
        ):
            raise SchemaError(
                "Attention JIT plan binding fingerprint must be lowercase SHA-256"
            )
        if not isinstance(self.verification, JitArtifactVerification):
            raise TypeError("verification must be JitArtifactVerification")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "jit_plan_binding_fingerprint": self.jit_plan_binding_fingerprint,
                "verification_fingerprint": self.verification.fingerprint,
            }
        )

    def validate_plan_binding(self, binding: AttentionJitPlanBinding) -> None:
        if not isinstance(binding, AttentionJitPlanBinding):
            raise TypeError("binding must be AttentionJitPlanBinding")
        binding.require_ready()
        resolution = binding.resolution
        if (
            self.jit_plan_binding_fingerprint != binding.fingerprint
            or self.verification.spec_name != resolution.spec_name
            or self.verification.spec_fingerprint != resolution.spec_fingerprint
            or self.verification.cache_record_fingerprint
            != resolution.cache_record_fingerprint
        ):
            raise SchemaError("Attention JIT artifact does not bind the plan cache hit")


@runtime_checkable
class AttentionJitArtifactResolver(Protocol):
    def resolve(
        self, binding: AttentionJitPlanBinding
    ) -> AttentionJitArtifactBinding:
        """Verify exact cache bytes without loading or resolving symbols."""


class ConfiguredAttentionJitArtifactResolver:
    """Resolve the cache record and verify its bytes through an injected reader."""

    def __init__(
        self, cache: JitCacheIndex, verifier: JitArtifactVerifier
    ) -> None:
        if not isinstance(cache, JitCacheIndex):
            raise TypeError("cache must be JitCacheIndex")
        if not isinstance(verifier, JitArtifactVerifier):
            raise TypeError("verifier must implement JitArtifactVerifier")
        self._cache = cache
        self._verifier = verifier

    def resolve(
        self, binding: AttentionJitPlanBinding
    ) -> AttentionJitArtifactBinding:
        if not isinstance(binding, AttentionJitPlanBinding):
            raise TypeError("binding must be AttentionJitPlanBinding")
        binding.require_ready()
        record = self._cache.lookup(binding.module_spec.jit_spec)
        if record is None:
            raise SchemaError("Attention JIT cache hit record disappeared")
        if record.fingerprint != binding.resolution.cache_record_fingerprint:
            raise SchemaError("Attention JIT cache record changed after resolution")
        verification = self._verifier.verify(record)
        if not isinstance(verification, JitArtifactVerification):
            raise TypeError("JIT artifact verifier returned an invalid receipt")
        verification.validate_record(record)
        result = AttentionJitArtifactBinding(binding.fingerprint, verification)
        result.validate_plan_binding(binding)
        return result


__all__ = [
    "ATTENTION_JIT_ARTIFACT_BINDING_VERSION",
    "AttentionJitArtifactBinding",
    "AttentionJitArtifactResolver",
    "ConfiguredAttentionJitArtifactResolver",
]
