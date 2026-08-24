"""Attention-specific JIT recipe generation."""

from .modules import (
    ATTENTION_JIT_GENERATOR_ID,
    ATTENTION_JIT_GENERATOR_VERSION,
    ATTENTION_JIT_MODULE_SCHEMA_VERSION,
    AttentionJitModuleSpec,
    attention_jit_entry_points,
    gen_attention_jit_module_spec,
    jit_environment_from_attention,
)
from .utils import attention_jit_module_name
from .variants import ATTENTION_JIT_VARIANT_SCHEMA_VERSION, AttentionJitVariant
from .plan import (
    ATTENTION_JIT_PLAN_BINDING_VERSION,
    AttentionJitPlanBinding,
    AttentionJitPlanResolver,
    ConfiguredAttentionJitPlanResolver,
    resolve_attention_jit_plan,
)
from .artifacts import (
    ATTENTION_JIT_ARTIFACT_BINDING_VERSION,
    AttentionJitArtifactBinding,
    AttentionJitArtifactResolver,
    ConfiguredAttentionJitArtifactResolver,
)
from .loading import (
    ATTENTION_JIT_LOADED_MODULE_BINDING_VERSION,
    AttentionJitLoadedModuleBinding,
    AttentionJitModuleResolver,
    ConfiguredAttentionJitModuleResolver,
)
from .execution import (
    ATTENTION_JIT_EXECUTOR_BINDING_VERSION,
    AttentionJitExecutorBinder,
    AttentionJitExecutorBinding,
)

__all__ = [
    "ATTENTION_JIT_ARTIFACT_BINDING_VERSION",
    "ATTENTION_JIT_GENERATOR_ID",
    "ATTENTION_JIT_EXECUTOR_BINDING_VERSION",
    "ATTENTION_JIT_GENERATOR_VERSION",
    "ATTENTION_JIT_MODULE_SCHEMA_VERSION",
    "ATTENTION_JIT_LOADED_MODULE_BINDING_VERSION",
    "ATTENTION_JIT_PLAN_BINDING_VERSION",
    "ATTENTION_JIT_VARIANT_SCHEMA_VERSION",
    "AttentionJitArtifactBinding",
    "AttentionJitArtifactResolver",
    "AttentionJitExecutorBinder",
    "AttentionJitExecutorBinding",
    "AttentionJitModuleSpec",
    "AttentionJitLoadedModuleBinding",
    "AttentionJitModuleResolver",
    "AttentionJitPlanBinding",
    "AttentionJitPlanResolver",
    "AttentionJitVariant",
    "ConfiguredAttentionJitArtifactResolver",
    "ConfiguredAttentionJitModuleResolver",
    "ConfiguredAttentionJitPlanResolver",
    "attention_jit_module_name",
    "attention_jit_entry_points",
    "gen_attention_jit_module_spec",
    "jit_environment_from_attention",
    "resolve_attention_jit_plan",
]
