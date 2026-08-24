"""Wrapper-owned Attention JIT plan resolution.

This layer binds one selected ``ascendc_jit`` dispatch to an immutable module
recipe and a pure cache decision.  It cannot build, load, or execute a module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from flashinfer_npu.attention.capability import AttentionRuntimeEnvironment
from flashinfer_npu.attention.dispatch import AttentionDispatchReceipt
from flashinfer_npu.attention.planner import AttentionFrameworkPlan
from flashinfer_npu.jit.cache import (
    JitCacheIndex,
    JitResolution,
    require_jit_cache_hit,
    resolve_jit_spec,
)
from flashinfer_npu.jit.core import JitSpecRegistry
from flashinfer_npu.jit.env import JitCompilationPolicy
from flashinfer_npu.runtime import Backend, SchemaError

from .modules import AttentionJitModuleSpec, gen_attention_jit_module_spec


ATTENTION_JIT_PLAN_BINDING_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionJitPlanBinding:
    """Exact JIT recipe/cache state frozen into a wrapper plan candidate."""

    module_spec: AttentionJitModuleSpec
    resolution: JitResolution
    schema_version: int = ATTENTION_JIT_PLAN_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_PLAN_BINDING_VERSION:
            raise SchemaError("unsupported Attention JIT plan binding version")
        if not isinstance(self.module_spec, AttentionJitModuleSpec):
            raise TypeError("module_spec must be AttentionJitModuleSpec")
        if not isinstance(self.resolution, JitResolution):
            raise TypeError("resolution must be JitResolution")
        if (
            self.resolution.spec_name != self.module_spec.jit_spec.name
            or self.resolution.spec_fingerprint
            != self.module_spec.jit_spec.fingerprint
        ):
            raise SchemaError("Attention JIT resolution does not bind the module spec")

    @property
    def ready(self) -> bool:
        return self.resolution.ready

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "module_spec_fingerprint": self.module_spec.fingerprint,
                "resolution_fingerprint": self.resolution.fingerprint,
            }
        )

    def require_ready(self) -> None:
        require_jit_cache_hit(self.resolution)

    def validate_resolved_runtime(
        self,
        framework_plan_fingerprint: str,
        receipt: AttentionDispatchReceipt,
    ) -> None:
        if not isinstance(receipt, AttentionDispatchReceipt):
            raise TypeError("receipt must be AttentionDispatchReceipt")
        if receipt.backend != Backend.ASCENDC_JIT:
            raise SchemaError("Attention JIT binding requires ascendc_jit receipt")
        if (
            self.module_spec.framework_plan_fingerprint
            != framework_plan_fingerprint
            or self.module_spec.dispatch_receipt_fingerprint
            != receipt.fingerprint
            or self.module_spec.workload_fingerprint
            != receipt.workload_fingerprint
        ):
            raise SchemaError("Attention JIT binding does not match resolved runtime")

    def validate_plan(
        self,
        plan: AttentionFrameworkPlan,
        receipt: AttentionDispatchReceipt,
    ) -> None:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        self.validate_resolved_runtime(plan.fingerprint, receipt)
        if (
            self.module_spec.plan_spec_fingerprint != plan.spec.fingerprint
            or self.module_spec.workload_fingerprint != plan.workload.fingerprint
        ):
            raise SchemaError("Attention JIT binding does not match active plan")


def resolve_attention_jit_plan(
    plan: AttentionFrameworkPlan,
    receipt: AttentionDispatchReceipt,
    runtime_environment: AttentionRuntimeEnvironment,
    cache: JitCacheIndex,
    policy: JitCompilationPolicy,
    *,
    registry: Optional[JitSpecRegistry] = None,
    compiler_id: str = "ascendc",
    build_type: str = "release",
) -> AttentionJitPlanBinding:
    """Generate, register, and resolve one JIT recipe without external effects."""

    module_spec = gen_attention_jit_module_spec(
        plan,
        receipt,
        runtime_environment,
        registry=registry,
        compiler_id=compiler_id,
        build_type=build_type,
    )
    resolution = resolve_jit_spec(
        module_spec.jit_spec,
        cache,
        policy,
    )
    return AttentionJitPlanBinding(module_spec, resolution)


@runtime_checkable
class AttentionJitPlanResolver(Protocol):
    """Private resolver injected into an Attention runtime integration."""

    def resolve(
        self,
        plan: AttentionFrameworkPlan,
        receipt: AttentionDispatchReceipt,
    ) -> AttentionJitPlanBinding:
        """Resolve without compiling, loading, importing a package, or launching."""


class ConfiguredAttentionJitPlanResolver:
    """Explicit environment/cache/policy composition used during wrapper plan."""

    def __init__(
        self,
        runtime_environment: AttentionRuntimeEnvironment,
        cache: JitCacheIndex,
        policy: JitCompilationPolicy,
        *,
        registry: Optional[JitSpecRegistry] = None,
        compiler_id: str = "ascendc",
        build_type: str = "release",
    ) -> None:
        if not isinstance(runtime_environment, AttentionRuntimeEnvironment):
            raise TypeError("runtime_environment must be AttentionRuntimeEnvironment")
        if not isinstance(cache, JitCacheIndex):
            raise TypeError("cache must be JitCacheIndex")
        if registry is not None and not isinstance(registry, JitSpecRegistry):
            raise TypeError("registry must be JitSpecRegistry")
        self._runtime_environment = runtime_environment
        self._cache = cache
        self._policy = JitCompilationPolicy(policy)
        self._registry = registry
        self._compiler_id = str(compiler_id)
        self._build_type = str(build_type)

    def resolve(
        self,
        plan: AttentionFrameworkPlan,
        receipt: AttentionDispatchReceipt,
    ) -> AttentionJitPlanBinding:
        return resolve_attention_jit_plan(
            plan,
            receipt,
            self._runtime_environment,
            self._cache,
            self._policy,
            registry=self._registry,
            compiler_id=self._compiler_id,
            build_type=self._build_type,
        )


__all__ = [
    "ATTENTION_JIT_PLAN_BINDING_VERSION",
    "AttentionJitPlanBinding",
    "AttentionJitPlanResolver",
    "ConfiguredAttentionJitPlanResolver",
    "resolve_attention_jit_plan",
]
