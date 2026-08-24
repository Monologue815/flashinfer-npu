"""Attention plan binding for a byte-verified, symbol-resolved JIT module."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from flashinfer_npu.jit.cache import JitCacheIndex
from flashinfer_npu.jit.loading import JitLoadedModule, JitModuleLoader
from flashinfer_npu.runtime import SchemaError

from .artifacts import AttentionJitArtifactBinding
from .plan import AttentionJitPlanBinding


ATTENTION_JIT_LOADED_MODULE_BINDING_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionJitLoadedModuleBinding:
    """Loaded module identity frozen against plan and artifact bindings."""

    jit_plan_binding_fingerprint: str
    jit_artifact_binding_fingerprint: str
    loaded_module: JitLoadedModule = field(repr=False, compare=False)
    schema_version: int = ATTENTION_JIT_LOADED_MODULE_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_LOADED_MODULE_BINDING_VERSION:
            raise SchemaError("unsupported Attention JIT loaded module binding version")
        for name in (
            "jit_plan_binding_fingerprint",
            "jit_artifact_binding_fingerprint",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
                raise SchemaError("Attention JIT %s must be lowercase SHA-256" % name)
        if not isinstance(self.loaded_module, JitLoadedModule):
            raise TypeError("loaded_module must be JitLoadedModule")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "jit_plan_binding_fingerprint": self.jit_plan_binding_fingerprint,
                "jit_artifact_binding_fingerprint": self.jit_artifact_binding_fingerprint,
                "loaded_module_fingerprint": self.loaded_module.fingerprint,
            }
        )

    def validate_bindings(
        self,
        plan_binding: AttentionJitPlanBinding,
        artifact_binding: AttentionJitArtifactBinding,
    ) -> None:
        if not isinstance(plan_binding, AttentionJitPlanBinding):
            raise TypeError("plan_binding must be AttentionJitPlanBinding")
        if not isinstance(artifact_binding, AttentionJitArtifactBinding):
            raise TypeError("artifact_binding must be AttentionJitArtifactBinding")
        artifact_binding.validate_plan_binding(plan_binding)
        spec = plan_binding.module_spec.jit_spec
        receipt = self.loaded_module.receipt
        if (
            self.jit_plan_binding_fingerprint != plan_binding.fingerprint
            or self.jit_artifact_binding_fingerprint != artifact_binding.fingerprint
            or receipt.spec_name != spec.name
            or receipt.spec_fingerprint != spec.fingerprint
            or receipt.cache_record_fingerprint
            != plan_binding.resolution.cache_record_fingerprint
            or receipt.artifact_verification_fingerprint
            != artifact_binding.verification.fingerprint
            or receipt.artifact_fingerprint
            != artifact_binding.verification.artifact_fingerprint
            or tuple(item.name for item in receipt.symbols)
            != tuple(sorted(spec.entry_points))
        ):
            raise SchemaError("Attention JIT loaded module bindings are stale")


@runtime_checkable
class AttentionJitModuleResolver(Protocol):
    def resolve(
        self,
        plan_binding: AttentionJitPlanBinding,
        artifact_binding: AttentionJitArtifactBinding,
    ) -> AttentionJitLoadedModuleBinding:
        """Load a verified artifact and resolve its exact entry-point set."""


class ConfiguredAttentionJitModuleResolver:
    """Cache/module-loader composition invoked privately during wrapper plan."""

    def __init__(self, cache: JitCacheIndex, loader: JitModuleLoader) -> None:
        if not isinstance(cache, JitCacheIndex):
            raise TypeError("cache must be JitCacheIndex")
        if not isinstance(loader, JitModuleLoader):
            raise TypeError("loader must implement JitModuleLoader")
        self._cache = cache
        self._loader = loader

    def resolve(
        self,
        plan_binding: AttentionJitPlanBinding,
        artifact_binding: AttentionJitArtifactBinding,
    ) -> AttentionJitLoadedModuleBinding:
        artifact_binding.validate_plan_binding(plan_binding)
        spec = plan_binding.module_spec.jit_spec
        if not spec.entry_points:
            raise SchemaError("Attention JIT module spec has no entry points")
        record = self._cache.lookup(spec)
        if record is None:
            raise SchemaError("Attention JIT cache record disappeared before load")
        if record.fingerprint != plan_binding.resolution.cache_record_fingerprint:
            raise SchemaError("Attention JIT cache record changed before load")
        verification = artifact_binding.verification
        verification.validate_record(record)
        loaded = self._loader.load(record, verification, spec.entry_points)
        if not isinstance(loaded, JitLoadedModule):
            raise TypeError("JIT module loader returned an invalid module")
        if (
            loaded.receipt.loader_id != self._loader.loader_id
            or loaded.receipt.loader_version != self._loader.loader_version
        ):
            raise SchemaError("JIT module loader changed its declared identity")
        loaded.receipt.validate(record, verification, spec.entry_points)
        result = AttentionJitLoadedModuleBinding(
            plan_binding.fingerprint,
            artifact_binding.fingerprint,
            loaded,
        )
        result.validate_bindings(plan_binding, artifact_binding)
        return result


__all__ = [
    "ATTENTION_JIT_LOADED_MODULE_BINDING_VERSION",
    "AttentionJitLoadedModuleBinding",
    "AttentionJitModuleResolver",
    "ConfiguredAttentionJitModuleResolver",
]
