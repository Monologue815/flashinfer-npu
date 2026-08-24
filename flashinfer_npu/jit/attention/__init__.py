"""Attention-specific JIT recipe generation."""

from .modules import (
    ATTENTION_JIT_GENERATOR_ID,
    ATTENTION_JIT_GENERATOR_VERSION,
    ATTENTION_JIT_MODULE_SCHEMA_VERSION,
    AttentionJitModuleSpec,
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

__all__ = [
    "ATTENTION_JIT_GENERATOR_ID",
    "ATTENTION_JIT_GENERATOR_VERSION",
    "ATTENTION_JIT_MODULE_SCHEMA_VERSION",
    "ATTENTION_JIT_PLAN_BINDING_VERSION",
    "ATTENTION_JIT_VARIANT_SCHEMA_VERSION",
    "AttentionJitModuleSpec",
    "AttentionJitPlanBinding",
    "AttentionJitPlanResolver",
    "AttentionJitVariant",
    "ConfiguredAttentionJitPlanResolver",
    "attention_jit_module_name",
    "gen_attention_jit_module_spec",
    "jit_environment_from_attention",
    "resolve_attention_jit_plan",
]
