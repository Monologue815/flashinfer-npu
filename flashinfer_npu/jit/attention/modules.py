"""Attention plan to JIT module-spec binding.

Only an already selected ``ascendc_jit`` dispatch may enter this layer.  The
result is an immutable recipe and plan binding, not compiled code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from flashinfer_npu.attention.capability import AttentionRuntimeEnvironment
from flashinfer_npu.attention.dispatch import AttentionDispatchReceipt
from flashinfer_npu.attention.planner import AttentionFrameworkPlan
from flashinfer_npu.jit.core import JitSpec, JitSpecRegistry, gen_jit_spec
from flashinfer_npu.jit.env import JitEnvironment
from flashinfer_npu.runtime import Backend, SchemaError

from .utils import attention_jit_module_name
from .variants import AttentionJitVariant


ATTENTION_JIT_MODULE_SCHEMA_VERSION = 1
ATTENTION_JIT_GENERATOR_ID = "attention.ascendc"
ATTENTION_JIT_GENERATOR_VERSION = "1"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def jit_environment_from_attention(
    environment: AttentionRuntimeEnvironment,
    *,
    compiler_id: str = "ascendc",
    build_type: str = "release",
) -> JitEnvironment:
    """Project a verified Attention environment into the JIT cache identity."""

    if not isinstance(environment, AttentionRuntimeEnvironment):
        raise TypeError("environment must be AttentionRuntimeEnvironment")
    return JitEnvironment(
        target_soc=environment.soc_version,
        soc_revision=environment.soc_revision,
        cann_version=environment.cann_version,
        compiler_id=compiler_id,
        compiler_version=environment.compiler_version,
        torch_version=environment.torch_version,
        torch_npu_version=environment.torch_npu_version,
        python_abi=environment.python_abi,
        build_type=build_type,
        features=environment.features,
    )


@dataclass(frozen=True)
class AttentionJitModuleSpec:
    """A generic JIT recipe bound to one framework plan and dispatch receipt."""

    variant: AttentionJitVariant
    jit_spec: JitSpec
    plan_spec_fingerprint: str
    framework_plan_fingerprint: str
    workload_fingerprint: str
    dispatch_receipt_fingerprint: str
    kernel_id: str
    schema_version: int = ATTENTION_JIT_MODULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_JIT_MODULE_SCHEMA_VERSION:
            raise SchemaError("unsupported Attention JIT module schema version")
        if not isinstance(self.variant, AttentionJitVariant):
            raise TypeError("variant must be AttentionJitVariant")
        if not isinstance(self.jit_spec, JitSpec):
            raise TypeError("jit_spec must be JitSpec")
        if self.jit_spec.domain != "attention":
            raise SchemaError(
                "Attention module requires an attention JIT spec"
            )
        if self.jit_spec.specialization_fingerprint != self.variant.fingerprint:
            raise SchemaError("Attention JIT spec does not bind its variant")
        for name in (
            "plan_spec_fingerprint",
            "framework_plan_fingerprint",
            "workload_fingerprint",
            "dispatch_receipt_fingerprint",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
                raise SchemaError("Attention JIT %s must be lowercase SHA-256" % name)
        if not self.kernel_id:
            raise SchemaError("Attention JIT kernel_id must be non-empty")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "variant_fingerprint": self.variant.fingerprint,
                "jit_spec_fingerprint": self.jit_spec.fingerprint,
                "plan_spec_fingerprint": self.plan_spec_fingerprint,
                "framework_plan_fingerprint": self.framework_plan_fingerprint,
                "workload_fingerprint": self.workload_fingerprint,
                "dispatch_receipt_fingerprint": self.dispatch_receipt_fingerprint,
                "kernel_id": self.kernel_id,
            }
        )


def gen_attention_jit_module_spec(
    plan: AttentionFrameworkPlan,
    receipt: AttentionDispatchReceipt,
    runtime_environment: AttentionRuntimeEnvironment,
    *,
    registry: Optional[JitSpecRegistry] = None,
    compiler_id: str = "ascendc",
    build_type: str = "release",
) -> AttentionJitModuleSpec:
    """Create a non-executing recipe from an authorized Attention JIT plan."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(receipt, AttentionDispatchReceipt):
        raise TypeError("receipt must be AttentionDispatchReceipt")
    if not isinstance(runtime_environment, AttentionRuntimeEnvironment):
        raise TypeError("runtime_environment must be AttentionRuntimeEnvironment")
    if receipt.backend != Backend.ASCENDC_JIT:
        raise SchemaError("only an ascendc_jit dispatch may generate a JIT module spec")
    if (
        receipt.mode != plan.spec.mode
        or receipt.plan_fingerprint != plan.fingerprint
        or receipt.admission_fingerprint != plan.admission_fingerprint
        or receipt.workload_fingerprint != plan.workload.fingerprint
    ):
        raise SchemaError("Attention JIT dispatch receipt does not bind the plan")
    if receipt.environment_fingerprint != runtime_environment.fingerprint:
        raise SchemaError("Attention JIT runtime environment does not bind the receipt")

    environment = jit_environment_from_attention(
        runtime_environment, compiler_id=compiler_id, build_type=build_type
    )
    variant = AttentionJitVariant.from_plan_spec(plan.spec)
    name = attention_jit_module_name(
        plan.spec.mode, variant.fingerprint, receipt.kernel_fingerprint
    )
    spec = gen_jit_spec(
        name=name,
        domain="attention",
        generator_id=ATTENTION_JIT_GENERATOR_ID,
        generator_version=ATTENTION_JIT_GENERATOR_VERSION,
        target_soc=runtime_environment.soc_version,
        environment_fingerprint=environment.fingerprint,
        specialization_fingerprint=variant.fingerprint,
        input_fingerprints=(
            ("artifact", receipt.artifact_fingerprint),
            ("binary_abi", receipt.binary_abi_fingerprint),
            ("kernel", receipt.kernel_fingerprint),
            ("launch_abi", receipt.launch_abi_fingerprint),
        ),
        entry_points=(),
        registry=registry,
    )
    return AttentionJitModuleSpec(
        variant=variant,
        jit_spec=spec,
        plan_spec_fingerprint=plan.spec.fingerprint,
        framework_plan_fingerprint=plan.fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        dispatch_receipt_fingerprint=receipt.fingerprint,
        kernel_id=receipt.kernel_id,
    )


__all__ = [
    "ATTENTION_JIT_GENERATOR_ID",
    "ATTENTION_JIT_GENERATOR_VERSION",
    "ATTENTION_JIT_MODULE_SCHEMA_VERSION",
    "AttentionJitModuleSpec",
    "gen_attention_jit_module_spec",
    "jit_environment_from_attention",
]
