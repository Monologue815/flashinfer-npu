"""Evidence-bearing authority resolver for package-backed Attention runtimes.

The resolver composes the existing capability profile, conformance evidence,
kernel provenance, launch/binary ABI, and provider-ownership contracts.  It
does not import an operator package, load an artifact, or touch a device.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import Backend, KernelDescriptor, SchemaError

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionRuntimeEnvironment,
    validate_attention_kernel_bindings,
)
from .corpus import AttentionCoveragePolicy, AttentionTraceCorpus
from .dispatch import select_attention_dispatch
from .numerics import (
    DEFAULT_ATTENTION_NUMERICS_POLICY,
    AttentionNumericsPolicy,
)
from .operation_catalog import AttentionOperatorOperationSpec
from .operator_physical_evidence import (
    AttentionOperatorPhysicalLayoutEvidence,
    select_attention_operator_physical_layout_dispatch,
)
from .operator_integration import AttentionOperatorRuntimeAuthority
from .operator_provider import (
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderRecord,
    bind_attention_operator_provider,
)
from .planner import AttentionFrameworkPlan
from .quant_physical_layout import (
    EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
    QuantPhysicalLayoutCatalog,
)


ATTENTION_OPERATOR_AUTHORITY_VERSION = 1
_NPU_DEVICE_PATTERN = re.compile(r"npu(?::[0-9]+)?\Z")


class AttentionEvidenceOperatorRuntimeAuthorityResolver:
    """Authorize one provider operation through the complete dispatch chain."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        profiles: Sequence[AttentionBackendCapabilityProfile],
        descriptors: Sequence[KernelDescriptor],
        observed_environment: AttentionRuntimeEnvironment,
        *,
        backend: Union[str, Backend] = "auto",
        tuned_kernel_ids: Sequence[str] = (),
        numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY,
        corpus: Optional[AttentionTraceCorpus] = None,
        coverage_policy: Optional[AttentionCoveragePolicy] = None,
        replay_evidence: bool = False,
        physical_layout_catalog: QuantPhysicalLayoutCatalog = (
            EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG
        ),
        physical_layout_evidence: Sequence[
            AttentionOperatorPhysicalLayoutEvidence
        ] = (),
    ) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        profile_values = tuple(profiles)
        descriptor_values = tuple(descriptors)
        if not profile_values or any(
            not isinstance(item, AttentionBackendCapabilityProfile)
            for item in profile_values
        ):
            raise TypeError("profiles must contain capability profiles")
        if not descriptor_values or any(
            not isinstance(item, KernelDescriptor) for item in descriptor_values
        ):
            raise TypeError("descriptors must contain KernelDescriptor values")
        if not isinstance(observed_environment, AttentionRuntimeEnvironment):
            raise TypeError(
                "observed_environment must be AttentionRuntimeEnvironment"
            )
        if not isinstance(numerics_policy, AttentionNumericsPolicy):
            raise TypeError("numerics_policy must be AttentionNumericsPolicy")
        if corpus is not None and not isinstance(corpus, AttentionTraceCorpus):
            raise TypeError("corpus must be AttentionTraceCorpus")
        if coverage_policy is not None and not isinstance(
            coverage_policy, AttentionCoveragePolicy
        ):
            raise TypeError("coverage_policy must be AttentionCoveragePolicy")
        if not isinstance(physical_layout_catalog, QuantPhysicalLayoutCatalog):
            raise TypeError(
                "physical_layout_catalog must be QuantPhysicalLayoutCatalog"
            )
        physical_evidence_values = tuple(physical_layout_evidence)
        if any(
            not isinstance(item, AttentionOperatorPhysicalLayoutEvidence)
            for item in physical_evidence_values
        ):
            raise TypeError(
                "physical_layout_evidence must contain physical evidence records"
            )
        if len({item.evidence_id for item in physical_evidence_values}) != len(
            physical_evidence_values
        ):
            raise SchemaError("physical layout evidence ids must be unique")
        if backend != "auto":
            try:
                backend = Backend(backend)
            except ValueError as error:
                raise SchemaError(
                    "unknown operator authority backend: %s" % backend
                ) from error
            if backend == Backend.REFERENCE:
                raise SchemaError("operator authority cannot request reference")
        if len({item.profile_id for item in profile_values}) != len(profile_values):
            raise SchemaError("operator authority profiles contain duplicate ids")
        if len({item.kernel_id for item in descriptor_values}) != len(
            descriptor_values
        ):
            raise SchemaError("operator authority descriptors contain duplicate ids")
        validate_attention_kernel_bindings(profile_values, descriptor_values)
        bound_profile_ids = {
            item.capability_binding.profile_id
            for item in descriptor_values
            if item.capability_binding is not None
            and item.capability_binding.domain == "attention"
        }
        if not bound_profile_ids.intersection(
            item.profile_id for item in profile_values
        ):
            raise SchemaError("operator authority has no bound Attention descriptor")
        tuned_ids = tuple(str(item) for item in tuned_kernel_ids)
        if any(not item for item in tuned_ids) or len(set(tuned_ids)) != len(
            tuned_ids
        ):
            raise SchemaError("tuned kernel ids must be unique and non-empty")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation
        self._profiles = profile_values
        self._descriptors = descriptor_values
        self._observed_environment = observed_environment
        self._backend = backend
        self._tuned_kernel_ids = tuned_ids
        self._numerics_policy = numerics_policy
        self._corpus = corpus
        self._coverage_policy = coverage_policy
        self._replay_evidence = bool(replay_evidence)
        self._physical_layout_catalog = physical_layout_catalog
        self._physical_layout_evidence = physical_evidence_values

    @property
    def profile_ids(self) -> Tuple[str, ...]:
        return tuple(item.profile_id for item in self._profiles)

    @property
    def kernel_ids(self) -> Tuple[str, ...]:
        return tuple(item.kernel_id for item in self._descriptors)

    def authorize(
        self,
        plan: AttentionFrameworkPlan,
        device: str,
        operation: AttentionOperatorOperationSpec,
        provider_probe: AttentionOperatorProviderProbe,
    ) -> AttentionOperatorRuntimeAuthority:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        if _NPU_DEVICE_PATTERN.fullmatch(str(device)) is None:
            raise SchemaError("operator authority requires device npu[:index]")
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(provider_probe, AttentionOperatorProviderProbe):
            raise TypeError("provider_probe must be AttentionOperatorProviderProbe")
        if (
            operation.operation_id != self._operation.operation_id
            or operation.fingerprint != self._operation.fingerprint
            or operation.provider_id != self.provider_id
        ):
            raise SchemaError("operator authority received a different operation")
        if not provider_probe.available or provider_probe.provider_id != self.provider_id:
            raise SchemaError("operator authority requires its available provider probe")
        quant_spec = plan.spec.kv_quant_spec
        if quant_spec is not None and quant_spec.physical_layout != "logical":
            receipt = select_attention_operator_physical_layout_dispatch(
                plan,
                operation,
                self._profiles,
                self._descriptors,
                self._observed_environment,
                self._physical_layout_catalog,
                self._physical_layout_evidence,
                backend=self._backend,
                tuned_kernel_ids=self._tuned_kernel_ids,
                numerics_policy=self._numerics_policy,
            )
        else:
            receipt = select_attention_dispatch(
                plan,
                self._profiles,
                self._descriptors,
                self._observed_environment,
                backend=self._backend,
                tuned_kernel_ids=self._tuned_kernel_ids,
                numerics_policy=self._numerics_policy,
                corpus=self._corpus,
                coverage_policy=self._coverage_policy,
                replay_evidence=self._replay_evidence,
            )
        provider_record = AttentionOperatorProviderRecord(
            probe=provider_probe,
            profiles=self._profiles,
        )
        selection = bind_attention_operator_provider(
            (provider_record,), receipt, provider=self.provider_id
        )
        return AttentionOperatorRuntimeAuthority(
            framework_plan_fingerprint=plan.fingerprint,
            device=str(device),
            provider_probe_fingerprint=provider_probe.fingerprint,
            operation_id=operation.operation_id,
            operation_fingerprint=operation.fingerprint,
            receipt=receipt,
            selection=selection,
        )


__all__ = [
    "ATTENTION_OPERATOR_AUTHORITY_VERSION",
    "AttentionEvidenceOperatorRuntimeAuthorityResolver",
]
