"""FlashInfer-style JIT framework contracts for Ascend NPU.

The current package contains Host-only identities and decisions. It exposes no
compiler or default loader implementation and therefore cannot execute an NPU
operator.
"""

from . import attention as attention
from .cache import (
    JIT_CACHE_SCHEMA_VERSION,
    JitCacheIndex,
    JitCacheRecord,
    JitResolution,
    JitResolutionState,
    MissingJITCacheError,
    require_jit_cache_hit,
    resolve_jit_spec,
)
from .artifacts import (
    JIT_ARTIFACT_VERIFICATION_VERSION,
    ConfiguredJitArtifactVerifier,
    JitArtifactPayloadReader,
    JitArtifactVerification,
    JitArtifactVerifier,
    verify_jit_cache_record_payload,
)
from .loading import (
    JIT_MODULE_LOAD_VERSION,
    JitLoadedModule,
    JitModuleLoadReceipt,
    JitModuleLoader,
    JitResolvedSymbol,
)
from .core import (
    JIT_SPEC_SCHEMA_VERSION,
    JitSpec,
    JitSpecRegistry,
    JitSpecStatus,
    gen_jit_spec,
    jit_spec_registry,
)
from .env import (
    JIT_ENVIRONMENT_SCHEMA_VERSION,
    JitCompilationPolicy,
    JitEnvironment,
)

__all__ = [
    "JIT_ARTIFACT_VERIFICATION_VERSION",
    "JIT_CACHE_SCHEMA_VERSION",
    "JIT_ENVIRONMENT_SCHEMA_VERSION",
    "JIT_MODULE_LOAD_VERSION",
    "JIT_SPEC_SCHEMA_VERSION",
    "ConfiguredJitArtifactVerifier",
    "JitArtifactPayloadReader",
    "JitArtifactVerification",
    "JitArtifactVerifier",
    "JitCacheIndex",
    "JitCacheRecord",
    "JitCompilationPolicy",
    "JitEnvironment",
    "JitLoadedModule",
    "JitModuleLoadReceipt",
    "JitModuleLoader",
    "JitResolution",
    "JitResolutionState",
    "JitResolvedSymbol",
    "JitSpec",
    "JitSpecRegistry",
    "JitSpecStatus",
    "MissingJITCacheError",
    "attention",
    "gen_jit_spec",
    "jit_spec_registry",
    "require_jit_cache_hit",
    "resolve_jit_spec",
    "verify_jit_cache_record_payload",
]
