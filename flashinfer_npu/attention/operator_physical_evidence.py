"""External conformance evidence for provider-native quantized KV layouts.

Host reference traces intentionally decode logical quantized storage only.  A
provider-native physical layout therefore needs a separate, explicit evidence
record before it can authorize dispatch.  This module validates such records;
it does not run a suite, import a provider package, or access a device.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union

from flashinfer_npu.runtime import (
    Backend,
    DeviceCapability,
    KernelDescriptor,
    QuantSpec,
    SchemaError,
)

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionRuntimeEnvironment,
    validate_attention_kernel_bindings,
)
from .dispatch import AttentionDispatchError, AttentionDispatchReceipt
from .launch_contract import attention_kernel_binary_abi_v2
from .numerics import DEFAULT_ATTENTION_NUMERICS_POLICY, AttentionNumericsPolicy
from .operation_catalog import AttentionOperatorOperationSpec
from .planner import AttentionFrameworkPlan
from .quant_physical_layout import QuantPhysicalLayoutCatalog
from .tensor_contract import KVCacheView, QuantizedTensorView


ATTENTION_OPERATOR_PHYSICAL_EVIDENCE_VERSION = 1

_HASH_FIELDS = (
    "suite_fingerprint",
    "result_digest",
    "quant_spec_fingerprint",
    "descriptor_fingerprint",
    "catalog_fingerprint",
    "profile_fingerprint",
    "environment_fingerprint",
    "kernel_fingerprint",
    "artifact_fingerprint",
    "launch_abi_fingerprint",
    "binary_abi_fingerprint",
)


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionOperatorPhysicalLayoutEvidence:
    """One immutable external suite result bound to an exact runtime stack."""

    provider_id: str
    operation_id: str
    evidence_id: str
    runner: str
    suite_id: str
    suite_fingerprint: str
    passed_case_ids: Tuple[str, ...]
    result_digest: str
    quant_spec_fingerprint: str
    descriptor_fingerprint: str
    catalog_fingerprint: str
    profile_id: str
    profile_fingerprint: str
    rule_id: str
    environment_fingerprint: str
    kernel_id: str
    kernel_fingerprint: str
    artifact_fingerprint: str
    launch_abi_fingerprint: str
    binary_abi_fingerprint: str
    schema_version: int = ATTENTION_OPERATOR_PHYSICAL_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PHYSICAL_EVIDENCE_VERSION:
            raise SchemaError("unsupported physical-layout evidence version")
        for name in (
            "provider_id",
            "operation_id",
            "evidence_id",
            "runner",
            "suite_id",
            "profile_id",
            "rule_id",
            "kernel_id",
        ):
            value = str(getattr(self, name))
            if not value:
                raise SchemaError("physical-layout evidence %s must be non-empty" % name)
            object.__setattr__(self, name, value)
        for name in _HASH_FIELDS:
            _require_hash(name, getattr(self, name))
        cases = tuple(str(item) for item in self.passed_case_ids)
        if not cases or any(not item for item in cases) or len(set(cases)) != len(cases):
            raise SchemaError(
                "physical-layout evidence passed_case_ids must be non-empty and unique"
            )
        object.__setattr__(self, "passed_case_ids", tuple(sorted(cases)))

    def to_dict(self):
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["passed_case_ids"] = list(self.passed_case_ids)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("physical-layout evidence fields are invalid")
        cases = data.get("passed_case_ids")
        if not isinstance(cases, (list, tuple)):
            raise SchemaError("physical-layout evidence cases must be an array")
        data["passed_case_ids"] = tuple(cases)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("physical-layout evidence fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate_candidate(
        self,
        operation: AttentionOperatorOperationSpec,
        quant_spec: QuantSpec,
        catalog: QuantPhysicalLayoutCatalog,
        profile: AttentionBackendCapabilityProfile,
        kernel: KernelDescriptor,
        observed_environment: AttentionRuntimeEnvironment,
    ) -> None:
        if quant_spec.physical_layout == "logical":
            raise SchemaError("physical-layout evidence requires non-logical QuantSpec")
        descriptor = catalog.resolve(quant_spec)
        binding = kernel.capability_binding
        rule = next(
            (item for item in profile.rules if item.rule_id == self.rule_id), None
        )
        if (
            self.provider_id != operation.provider_id
            or self.operation_id != operation.operation_id
            or self.quant_spec_fingerprint != quant_spec.fingerprint
            or self.descriptor_fingerprint != descriptor.fingerprint
            or self.catalog_fingerprint != catalog.fingerprint
            or self.profile_id != profile.profile_id
            or self.profile_fingerprint != profile.fingerprint
            or self.environment_fingerprint != observed_environment.fingerprint
            or observed_environment != profile.environment
            or self.kernel_id != kernel.kernel_id
            or self.kernel_fingerprint != kernel.fingerprint
            or kernel.artifact is None
            or self.artifact_fingerprint != kernel.artifact.fingerprint
            or kernel.launch_abi is None
            or self.launch_abi_fingerprint != kernel.launch_abi.fingerprint
            or kernel.binary_abi is None
            or self.binary_abi_fingerprint != kernel.binary_abi.fingerprint
            or kernel.binary_abi.fingerprint
            != attention_kernel_binary_abi_v2().fingerprint
            or binding is None
            or binding.profile_id != profile.profile_id
            or binding.profile_fingerprint != profile.fingerprint
            or binding.rule_id != self.rule_id
            or rule is None
            or quant_spec not in rule.quant_specs
        ):
            raise SchemaError("physical-layout evidence candidate identity is stale")
        features = set(descriptor.required_features)
        if (
            not features
            or not features <= set(rule.required_features)
            or not features <= set(kernel.constraints.required_features)
            or not features <= set(observed_environment.features)
        ):
            raise SchemaError(
                "physical-layout evidence does not cover required features"
            )

    def validate_authority(
        self,
        kv: KVCacheView,
        catalog: QuantPhysicalLayoutCatalog,
        profile: AttentionBackendCapabilityProfile,
        kernel: KernelDescriptor,
        observed_environment: AttentionRuntimeEnvironment,
        receipt: AttentionDispatchReceipt,
    ) -> None:
        if not isinstance(kv, KVCacheView) or not kv.quantized:
            raise SchemaError("physical-layout evidence requires quantized KV")
        quant_spec = kv.spec.quant_spec
        assert quant_spec is not None
        descriptor = catalog.resolve(quant_spec)
        if not isinstance(kv.key, QuantizedTensorView) or (
            kv.key.physical_layout_descriptor != descriptor
            or kv.value.physical_layout_descriptor != descriptor
        ):
            raise SchemaError("physical-layout evidence KV descriptor is stale")
        if (
            receipt.evidence_id != self.evidence_id
            or receipt.evidence_result_digest != self.result_digest
            or receipt.profile_id != self.profile_id
            or receipt.rule_id != self.rule_id
            or receipt.kernel_id != self.kernel_id
        ):
            raise SchemaError("physical-layout evidence receipt identity is stale")
        # Candidate identities except operation were already bound into the receipt.
        # Revalidate the full remaining chain directly to avoid reconstructing an
        # operation catalog entry at run time.
        if (
            self.quant_spec_fingerprint != quant_spec.fingerprint
            or self.descriptor_fingerprint != descriptor.fingerprint
            or self.catalog_fingerprint != catalog.fingerprint
            or self.profile_fingerprint != profile.fingerprint
            or self.environment_fingerprint != observed_environment.fingerprint
            or self.kernel_fingerprint != kernel.fingerprint
            or kernel.artifact is None
            or self.artifact_fingerprint != kernel.artifact.fingerprint
            or kernel.launch_abi is None
            or self.launch_abi_fingerprint != kernel.launch_abi.fingerprint
            or kernel.binary_abi is None
            or self.binary_abi_fingerprint != kernel.binary_abi.fingerprint
        ):
            raise SchemaError("physical-layout evidence authority is stale")


def _device_capability(environment: AttentionRuntimeEnvironment) -> DeviceCapability:
    return DeviceCapability(
        soc_version=environment.soc_version,
        soc_revision=environment.soc_revision,
        ai_core_count=environment.ai_core_count,
        supported_dtypes=(),
        features=environment.features,
        cann_version=environment.cann_version,
        torch_npu_version=environment.torch_npu_version,
        compiler_version=environment.compiler_version,
    )


def select_attention_operator_physical_layout_dispatch(
    plan: AttentionFrameworkPlan,
    operation: AttentionOperatorOperationSpec,
    profiles: Sequence[AttentionBackendCapabilityProfile],
    kernels: Sequence[KernelDescriptor],
    observed_environment: AttentionRuntimeEnvironment,
    catalog: QuantPhysicalLayoutCatalog,
    evidences: Sequence[AttentionOperatorPhysicalLayoutEvidence],
    *,
    backend: Union[str, Backend] = "auto",
    tuned_kernel_ids: Sequence[str] = (),
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
) -> AttentionDispatchReceipt:
    """Select a non-logical kernel only through exact external evidence."""

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    quant_spec = plan.spec.kv_quant_spec
    if quant_spec is None or quant_spec.physical_layout == "logical":
        raise SchemaError("physical-layout dispatch requires a non-logical plan")
    if not isinstance(operation, AttentionOperatorOperationSpec):
        raise TypeError("operation must be AttentionOperatorOperationSpec")
    if not isinstance(observed_environment, AttentionRuntimeEnvironment):
        raise TypeError("observed_environment must be AttentionRuntimeEnvironment")
    if not isinstance(catalog, QuantPhysicalLayoutCatalog):
        raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
    if not isinstance(numerics_policy, AttentionNumericsPolicy):
        raise TypeError("numerics_policy must be AttentionNumericsPolicy")
    profile_values = tuple(profiles)
    kernel_values = tuple(kernels)
    evidence_values = tuple(evidences)
    if not profile_values or any(
        not isinstance(item, AttentionBackendCapabilityProfile)
        for item in profile_values
    ):
        raise TypeError("profiles must contain capability profiles")
    if not kernel_values or any(
        not isinstance(item, KernelDescriptor) for item in kernel_values
    ):
        raise TypeError("kernels must contain KernelDescriptor values")
    validate_attention_kernel_bindings(profile_values, kernel_values)
    if any(
        not isinstance(item, AttentionOperatorPhysicalLayoutEvidence)
        for item in evidence_values
    ):
        raise TypeError("evidences must contain physical-layout evidence")
    if len({item.evidence_id for item in evidence_values}) != len(evidence_values):
        raise SchemaError("physical-layout evidence ids must be unique")
    if backend == "auto":
        requested = None
        requested_name = "auto"
    else:
        try:
            requested = Backend(backend)
        except ValueError as error:
            raise SchemaError("unknown physical-layout backend") from error
        if requested == Backend.REFERENCE:
            raise SchemaError("physical-layout dispatch cannot request reference")
        requested_name = requested.value
    tuned = tuple(str(item) for item in tuned_kernel_ids)
    if any(not item for item in tuned) or len(set(tuned)) != len(tuned):
        raise SchemaError("physical-layout tuned kernel ids must be unique")

    generic = _device_capability(observed_environment)
    accepted = []
    rejected = []
    profiles_by_id = {item.profile_id: item for item in profile_values}
    for kernel in kernel_values:
        binding = kernel.capability_binding
        if kernel.op != plan.workload.op or binding is None:
            continue
        profile = profiles_by_id.get(binding.profile_id)
        if profile is None:
            continue
        report = profile.explain(
            plan.spec,
            plan.metadata,
            observed_environment,
            numerics_policy=numerics_policy,
        )
        rule_report = next(
            item for item in report.rules if item.rule_id == binding.rule_id
        )
        reasons = list(report.global_reasons) + list(rule_report.reasons)
        reasons.extend(kernel.constraints.unsupported_reasons(plan.workload, generic))
        if requested is not None and kernel.backend != requested:
            reasons.append("excluded by backend policy %s" % requested.value)
        matching = tuple(
            item
            for item in evidence_values
            if item.profile_id == profile.profile_id
            and item.rule_id == binding.rule_id
            and item.kernel_id == kernel.kernel_id
            and item.quant_spec_fingerprint == quant_spec.fingerprint
        )
        if len(matching) != 1:
            reasons.append("requires exactly one physical-layout evidence record")
        else:
            try:
                matching[0].validate_candidate(
                    operation,
                    quant_spec,
                    catalog,
                    profile,
                    kernel,
                    observed_environment,
                )
            except SchemaError as error:
                reasons.append(str(error))
        if reasons:
            rejected.append("%s: %s" % (kernel.kernel_id, ", ".join(reasons)))
        else:
            accepted.append((profile, kernel, matching[0]))
    if not accepted:
        raise AttentionDispatchError(
            "no physical-layout evidence-bearing Attention kernel accepted the plan (%s)"
            % ("; ".join(rejected) or "no bound descriptor")
        )
    selected = next(
        (item for kernel_id in tuned for item in accepted if item[1].kernel_id == kernel_id),
        None,
    )
    selection_source = "tuning"
    if selected is None:
        selected = sorted(
            accepted, key=lambda item: (-item[1].priority, item[1].kernel_id)
        )[0]
        selection_source = "priority"
    profile, kernel, evidence = selected
    assert kernel.artifact is not None
    assert kernel.launch_abi is not None
    assert kernel.binary_abi is not None
    binding = kernel.capability_binding
    assert binding is not None
    return AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=numerics_policy.fingerprint,
        profile_id=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        rule_id=binding.rule_id,
        environment_fingerprint=observed_environment.fingerprint,
        evidence_id=evidence.evidence_id,
        evidence_result_digest=evidence.result_digest,
        kernel_id=kernel.kernel_id,
        kernel_fingerprint=kernel.fingerprint,
        artifact_fingerprint=kernel.artifact.fingerprint,
        launch_abi_fingerprint=kernel.launch_abi.fingerprint,
        binary_abi_fingerprint=kernel.binary_abi.fingerprint,
        backend=kernel.backend,
        float_workspace_bytes=kernel.workspace.size_for(plan.workload),
        int_workspace_bytes=kernel.int_workspace.size_for(plan.workload),
        float_workspace_alignment=kernel.workspace.alignment,
        int_workspace_alignment=kernel.int_workspace.alignment,
        selection_source=selection_source,
        requested_backend=requested_name,
    )


__all__ = [
    "ATTENTION_OPERATOR_PHYSICAL_EVIDENCE_VERSION",
    "AttentionOperatorPhysicalLayoutEvidence",
    "select_attention_operator_physical_layout_dispatch",
]
