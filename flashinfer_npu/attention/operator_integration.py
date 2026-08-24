"""Composition of package, authority, materialization, and runtime binding.

This is framework orchestration only.  Every external effect is supplied by
an injected package loader, capability authority resolver, tensor materializer,
and callable.  No provider is registered by default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from flashinfer_npu.jit.attention import (
    AttentionJitArtifactBinding,
    AttentionJitArtifactResolver,
    AttentionJitPlanBinding,
    AttentionJitPlanResolver,
)
from flashinfer_npu.runtime import Backend, SchemaError

from .dispatch import AttentionDispatchReceipt
from .operation_catalog import AttentionOperatorOperationSpec
from .operator_materialization import (
    AttentionMaterializingPlanFactory,
    AttentionMaterializingRunAdapter,
    AttentionOperatorTensorMaterializer,
)
from .operator_package import (
    AttentionOperatorPackageResolver,
    AttentionResolvedOperatorPackage,
)
from .operator_plan import AttentionOperatorPlanFactory
from .operator_provider import (
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
)
from .operator_resolver import AttentionResolvedOperatorRuntime
from .operator_run import (
    AttentionOperatorRunAdapter,
    AttentionOperatorRunAdapterFactory,
)
from .planner import AttentionFrameworkPlan


ATTENTION_OPERATOR_INTEGRATION_VERSION = 2


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class AttentionOperatorPlanGate(Protocol):
    """Pure provider/API plan admission before capability authorization."""

    provider_id: str
    operation_id: str

    def rejection_reasons(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> Sequence[str]:
        """Return deterministic reasons without imports or device access."""


@dataclass(frozen=True)
class AttentionOperatorRuntimeAuthority:
    """Capability/evidence authority created before callable import."""

    framework_plan_fingerprint: str
    device: str
    provider_probe_fingerprint: str
    operation_id: str
    operation_fingerprint: str
    receipt: AttentionDispatchReceipt = field(repr=False, compare=False)
    selection: AttentionOperatorProviderSelection = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_INTEGRATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_INTEGRATION_VERSION:
            raise SchemaError("unsupported Attention runtime authority version")
        for name in (
            "framework_plan_fingerprint",
            "provider_probe_fingerprint",
            "operation_fingerprint",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError("%s must be lowercase SHA-256" % name)
        if not str(self.operation_id):
            raise SchemaError("runtime authority operation_id must be non-empty")
        if not str(self.device):
            raise SchemaError("runtime authority device must be non-empty")
        object.__setattr__(self, "device", str(self.device))
        if not isinstance(self.receipt, AttentionDispatchReceipt):
            raise TypeError("receipt must be AttentionDispatchReceipt")
        if not isinstance(self.selection, AttentionOperatorProviderSelection):
            raise TypeError("selection must be AttentionOperatorProviderSelection")
        if (
            self.receipt.plan_fingerprint != self.framework_plan_fingerprint
            or self.selection.dispatch_receipt_fingerprint
            != self.receipt.fingerprint
            or self.selection.provider_probe_fingerprint
            != self.provider_probe_fingerprint
        ):
            raise SchemaError("Attention runtime authority chain is stale")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "framework_plan_fingerprint": self.framework_plan_fingerprint,
            "device": self.device,
            "provider_probe_fingerprint": self.provider_probe_fingerprint,
            "operation_id": self.operation_id,
            "operation_fingerprint": self.operation_fingerprint,
            "dispatch_receipt_fingerprint": self.receipt.fingerprint,
            "provider_selection_fingerprint": self.selection.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@runtime_checkable
class AttentionOperatorRuntimeAuthorityResolver(Protocol):
    """Evidence-bearing dispatch boundary supplied by an integration."""

    provider_id: str
    operation_id: str

    def authorize(
        self,
        plan: AttentionFrameworkPlan,
        device: str,
        operation: AttentionOperatorOperationSpec,
        provider_probe: AttentionOperatorProviderProbe,
    ) -> AttentionOperatorRuntimeAuthority:
        """Authorize an exact plan/provider/API without importing its callable."""


class AttentionOperatorIntegrationError(RuntimeError):
    """Raised before a package implementation can publish runtime state."""


def _unique_reasons(values) -> Tuple[str, ...]:
    try:
        normalized = tuple(str(item) for item in values)
    except TypeError as error:
        raise TypeError("plan gate rejection_reasons must be a sequence") from error
    if any(not item for item in normalized):
        raise SchemaError("plan gate rejection reasons must be non-empty")
    return tuple(dict.fromkeys(normalized))


class AttentionOperatorPackageRuntimeImplementation:
    """Complete auto-selection candidate backed by one external package API."""

    def __init__(
        self,
        *,
        priority: int,
        package_resolver: AttentionOperatorPackageResolver,
        plan_gate: AttentionOperatorPlanGate,
        authority_resolver: AttentionOperatorRuntimeAuthorityResolver,
        logical_factory: AttentionOperatorPlanFactory,
        logical_run_adapter: AttentionOperatorRunAdapter,
        tensor_materializer: AttentionOperatorTensorMaterializer,
        run_adapter_factory: Optional[AttentionOperatorRunAdapterFactory] = None,
        jit_plan_resolver: Optional[AttentionJitPlanResolver] = None,
        jit_artifact_resolver: Optional[AttentionJitArtifactResolver] = None,
    ) -> None:
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise SchemaError("package runtime priority must be an integer")
        if not isinstance(package_resolver, AttentionOperatorPackageResolver):
            raise TypeError("package_resolver must be AttentionOperatorPackageResolver")
        if not isinstance(plan_gate, AttentionOperatorPlanGate):
            raise TypeError("plan_gate must implement AttentionOperatorPlanGate")
        if not isinstance(
            authority_resolver, AttentionOperatorRuntimeAuthorityResolver
        ):
            raise TypeError(
                "authority_resolver must implement "
                "AttentionOperatorRuntimeAuthorityResolver"
            )
        if not isinstance(logical_factory, AttentionOperatorPlanFactory):
            raise TypeError(
                "logical_factory must implement AttentionOperatorPlanFactory"
            )
        if not isinstance(logical_run_adapter, AttentionOperatorRunAdapter):
            raise TypeError(
                "logical_run_adapter must implement AttentionOperatorRunAdapter"
            )
        if not isinstance(tensor_materializer, AttentionOperatorTensorMaterializer):
            raise TypeError(
                "tensor_materializer must implement "
                "AttentionOperatorTensorMaterializer"
            )
        if run_adapter_factory is not None and not isinstance(
            run_adapter_factory, AttentionOperatorRunAdapterFactory
        ):
            raise TypeError(
                "run_adapter_factory must implement "
                "AttentionOperatorRunAdapterFactory"
            )
        if jit_plan_resolver is not None and not isinstance(
            jit_plan_resolver, AttentionJitPlanResolver
        ):
            raise TypeError(
                "jit_plan_resolver must implement AttentionJitPlanResolver"
            )
        if jit_artifact_resolver is not None and not isinstance(
            jit_artifact_resolver, AttentionJitArtifactResolver
        ):
            raise TypeError(
                "jit_artifact_resolver must implement AttentionJitArtifactResolver"
            )
        operation = package_resolver.operation
        identities = (
            (plan_gate.provider_id, plan_gate.operation_id),
            (authority_resolver.provider_id, authority_resolver.operation_id),
            (logical_factory.provider_id, logical_factory.operation_id),
        )
        if any(
            provider_id != operation.provider_id
            or operation_id != operation.operation_id
            for provider_id, operation_id in identities
        ):
            raise SchemaError("package runtime component identities differ")
        if (
            logical_run_adapter.provider_id != operation.provider_id
            or tensor_materializer.provider_id != operation.provider_id
        ):
            raise SchemaError("package runtime provider components differ")
        if run_adapter_factory is not None and (
            run_adapter_factory.provider_id != operation.provider_id
            or run_adapter_factory.operation_id != operation.operation_id
        ):
            raise SchemaError("package runtime run adapter factory differs")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self.priority = priority
        self._package_resolver = package_resolver
        self._plan_gate = plan_gate
        self._authority_resolver = authority_resolver
        self._logical_factory = logical_factory
        self._logical_run_adapter = logical_run_adapter
        self._tensor_materializer = tensor_materializer
        self._run_adapter_factory = run_adapter_factory
        self._jit_plan_resolver = jit_plan_resolver
        self._jit_artifact_resolver = jit_artifact_resolver

    def rejection_reasons(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> Tuple[str, ...]:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        gate_reasons = _unique_reasons(
            self._plan_gate.rejection_reasons(plan, str(device))
        )
        package_report = self._package_resolver.explain()
        package_reasons = tuple(
            "package: %s" % item for item in package_report.reasons
        )
        return _unique_reasons(gate_reasons + package_reasons)

    def _validate_authority(
        self,
        authority: AttentionOperatorRuntimeAuthority,
        plan: AttentionFrameworkPlan,
        package: AttentionResolvedOperatorPackage,
        device: str,
    ) -> None:
        if not isinstance(authority, AttentionOperatorRuntimeAuthority):
            raise TypeError("authority resolver returned an invalid authority")
        operation = self._package_resolver.operation
        if (
            authority.framework_plan_fingerprint != plan.fingerprint
            or authority.device != str(device)
            or authority.provider_probe_fingerprint
            != package.provider_probe.fingerprint
            or authority.operation_id != operation.operation_id
            or authority.operation_fingerprint != operation.fingerprint
            or authority.selection.provider_id != operation.provider_id
        ):
            raise SchemaError("package runtime authority does not bind the candidate")

    def resolve(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionResolvedOperatorRuntime:
        reasons = self.rejection_reasons(plan, str(device))
        if reasons:
            raise AttentionOperatorIntegrationError(
                "Attention package runtime does not accept the plan: %s"
                % "; ".join(reasons)
            )
        metadata_report = self._package_resolver.explain()
        provider_probe = self._package_resolver.provider_probe(metadata_report)
        operation = self._package_resolver.operation
        authority = self._authority_resolver.authorize(
            plan, str(device), operation, provider_probe
        )
        if not isinstance(authority, AttentionOperatorRuntimeAuthority):
            raise TypeError("authority resolver returned an invalid authority")
        if (
            authority.framework_plan_fingerprint != plan.fingerprint
            or authority.device != str(device)
            or authority.provider_probe_fingerprint != provider_probe.fingerprint
            or authority.operation_id != operation.operation_id
            or authority.operation_fingerprint != operation.fingerprint
            or authority.selection.provider_id != operation.provider_id
        ):
            raise SchemaError("pre-import package runtime authority is stale")

        jit_plan_binding = None
        jit_artifact_binding = None
        if authority.receipt.backend == Backend.ASCENDC_JIT:
            if self._jit_plan_resolver is None:
                raise AttentionOperatorIntegrationError(
                    "ascendc_jit runtime requires a configured JIT plan resolver"
                )
            if self._jit_artifact_resolver is None:
                raise AttentionOperatorIntegrationError(
                    "ascendc_jit runtime requires a configured JIT artifact resolver"
                )
            jit_plan_binding = self._jit_plan_resolver.resolve(
                plan, authority.receipt
            )
            if not isinstance(jit_plan_binding, AttentionJitPlanBinding):
                raise TypeError("JIT plan resolver returned an invalid binding")
            jit_plan_binding.validate_resolved_runtime(
                plan.fingerprint,
                authority.receipt,
            )
            # A cache miss cannot become an executable provider runtime until a
            # separately authorized builder has produced and verified an artifact.
            jit_plan_binding.require_ready()
            jit_artifact_binding = self._jit_artifact_resolver.resolve(
                jit_plan_binding
            )
            if not isinstance(jit_artifact_binding, AttentionJitArtifactBinding):
                raise TypeError(
                    "JIT artifact resolver returned an invalid binding"
                )
            jit_artifact_binding.validate_plan_binding(jit_plan_binding)

        package = self._package_resolver.resolve(
            expected_provider_probe=provider_probe
        )
        self._validate_authority(authority, plan, package, str(device))
        factory = AttentionMaterializingPlanFactory(
            self._logical_factory, self._tensor_materializer, str(device)
        )
        run_adapter = AttentionMaterializingRunAdapter(
            self._logical_run_adapter
        )
        if self._run_adapter_factory is not None:
            run_adapter = self._run_adapter_factory.build(
                run_adapter, str(device)
            )
        return AttentionResolvedOperatorRuntime(
            framework_plan_fingerprint=plan.fingerprint,
            factory=factory,
            run_adapter=run_adapter,
            executor=package.executor,
            receipt=authority.receipt,
            selection=authority.selection,
            callable_binding=package.callable_binding,
            jit_plan_binding=jit_plan_binding,
            jit_artifact_binding=jit_artifact_binding,
        )


__all__ = [
    "ATTENTION_OPERATOR_INTEGRATION_VERSION",
    "AttentionOperatorIntegrationError",
    "AttentionOperatorPackageRuntimeImplementation",
    "AttentionOperatorPlanGate",
    "AttentionOperatorRuntimeAuthority",
    "AttentionOperatorRuntimeAuthorityResolver",
]
