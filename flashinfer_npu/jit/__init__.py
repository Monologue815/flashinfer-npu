"""FlashInfer-style JIT framework contracts for Ascend NPU.

The current package contains Host-only identities and decisions.  It exposes
no compiler or loader and therefore cannot execute an NPU operator.
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
    "JIT_CACHE_SCHEMA_VERSION",
    "JIT_ENVIRONMENT_SCHEMA_VERSION",
    "JIT_SPEC_SCHEMA_VERSION",
    "JitCacheIndex",
    "JitCacheRecord",
    "JitCompilationPolicy",
    "JitEnvironment",
    "JitResolution",
    "JitResolutionState",
    "JitSpec",
    "JitSpecRegistry",
    "JitSpecStatus",
    "MissingJITCacheError",
    "attention",
    "gen_jit_spec",
    "jit_spec_registry",
    "require_jit_cache_hit",
    "resolve_jit_spec",
]
