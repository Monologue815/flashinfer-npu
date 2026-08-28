"""Declarative bootstrap for package-backed Attention runtimes.

This module is the single composition root for external Attention packages.
Building a registry does not inspect package metadata, import a callable, load
an artifact, initialize a device, or execute an operator.  A runtime becomes a
candidate only when an integration supplies exact package versions and the
already-required capability/evidence/kernel authority inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple, Union

from flashinfer_npu.runtime import Backend, KernelDescriptor, SchemaError
from flashinfer_npu.jit.attention import (
    AttentionJitArtifactResolver,
    AttentionJitExecutorBinder,
    AttentionJitModuleResolver,
    AttentionJitPlanResolver,
    AttentionJitPlannerBinder,
)

from .capability import (
    AttentionBackendCapabilityProfile,
    AttentionRuntimeEnvironment,
)
from .corpus import AttentionCoveragePolicy, AttentionTraceCorpus
from .numerics import (
    DEFAULT_ATTENTION_NUMERICS_POLICY,
    AttentionNumericsPolicy,
)
from .operation_catalog import (
    AttentionOperatorOperationCatalog,
    load_packaged_attention_operator_catalog,
)
from .operator_authority import (
    AttentionEvidenceOperatorRuntimeAuthorityResolver,
)
from .operator_integration import (
    AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorPlanGate,
)
from .operator_completion import AttentionOperatorCompletionValidatorFactory
from .operator_materialization import AttentionOperatorTensorMaterializer
from .operator_package import (
    AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageLoader,
    AttentionOperatorPackageResolver,
    ImportlibAttentionOperatorPackageLoader,
)
from .operator_plan import AttentionOperatorPlanFactory
from .operator_evidence_manifest import (
    AttentionVerifiedOperatorPhysicalEvidenceBundle,
)
from .operator_quantization import (
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizationPlanGate,
    AttentionOperatorQuantizationRunAdapterFactory,
    validate_attention_operator_quantization_bindings,
)
from .nvfp4_packed_kv import (
    AttentionOperatorNvfp4PackedKVBinding,
    AttentionOperatorNvfp4PackedKVRunAdapterFactory,
    validate_attention_operator_nvfp4_packed_kv_bindings,
)
from .quant_physical_layout import (
    EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
    QuantPhysicalLayoutCatalog,
)
from .operator_resolver import (
    EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimePlanScorer,
    AttentionOperatorRuntimeResolverRegistry,
)
from .operator_scoring import (
    AttentionOperatorPlanScoringManifest,
    AttentionOperatorPlanScoringPolicy,
)
from .operator_run import (
    AttentionOperatorCallerBufferRunAdapterFactory,
    AttentionOperatorRunAdapter,
    AttentionOperatorRunAdapterFactoryChain,
    AttentionOperatorRunTensorValidationAdapterFactory,
    AttentionOperatorTensorMetadataInspector,
)
from .tensor_contract import AttentionTensorAccessPolicy


ATTENTION_OPERATOR_BOOTSTRAP_VERSION = 14


@dataclass(frozen=True)
class AttentionOperatorPackageRuntimeSpec:
    """Complete, explicit declaration for one external operation candidate."""

    operation_id: str
    priority: int
    adapter_version: str
    supported_package_versions: Tuple[str, ...]
    profiles: Tuple[AttentionBackendCapabilityProfile, ...]
    descriptors: Tuple[KernelDescriptor, ...]
    observed_environment: AttentionRuntimeEnvironment
    plan_gate: AttentionOperatorPlanGate
    logical_factory: AttentionOperatorPlanFactory
    logical_run_adapter: AttentionOperatorRunAdapter
    tensor_materializer: AttentionOperatorTensorMaterializer
    tensor_metadata_inspector: AttentionOperatorTensorMetadataInspector
    tensor_access_policy: AttentionTensorAccessPolicy
    plan_scorer: Optional[AttentionOperatorRuntimePlanScorer] = None
    validate_provider_results: bool = True
    quantization_bindings: Tuple[AttentionOperatorQuantizationBinding, ...] = ()
    nvfp4_packed_kv_bindings: Tuple[
        AttentionOperatorNvfp4PackedKVBinding, ...
    ] = ()
    quant_physical_layout_catalog: QuantPhysicalLayoutCatalog = (
        EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG
    )
    physical_layout_evidence_bundle: Optional[
        AttentionVerifiedOperatorPhysicalEvidenceBundle
    ] = None
    jit_plan_resolver: Optional[AttentionJitPlanResolver] = None
    jit_artifact_resolver: Optional[AttentionJitArtifactResolver] = None
    jit_module_resolver: Optional[AttentionJitModuleResolver] = None
    jit_planner_binder: Optional[AttentionJitPlannerBinder] = None
    jit_executor_binder: Optional[AttentionJitExecutorBinder] = None
    backend: Union[str, Backend] = "auto"
    tuned_kernel_ids: Tuple[str, ...] = ()
    numerics_policy: AttentionNumericsPolicy = DEFAULT_ATTENTION_NUMERICS_POLICY
    corpus: Optional[AttentionTraceCorpus] = None
    coverage_policy: Optional[AttentionCoveragePolicy] = None
    replay_evidence: bool = False
    schema_version: int = ATTENTION_OPERATOR_BOOTSTRAP_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_BOOTSTRAP_VERSION:
            raise SchemaError("unsupported Attention operator bootstrap version")
        if not str(self.operation_id) or not str(self.adapter_version):
            raise SchemaError("bootstrap operation and adapter ids must be non-empty")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise SchemaError("bootstrap priority must be an integer")
        versions = tuple(str(item) for item in self.supported_package_versions)
        if not versions or any(not item for item in versions):
            raise SchemaError("bootstrap package versions must be non-empty")
        if len(set(versions)) != len(versions):
            raise SchemaError("bootstrap package versions must be unique")
        profiles = tuple(self.profiles)
        descriptors = tuple(self.descriptors)
        if not profiles or any(
            not isinstance(item, AttentionBackendCapabilityProfile)
            for item in profiles
        ):
            raise TypeError("bootstrap profiles must contain capability profiles")
        if not descriptors or any(
            not isinstance(item, KernelDescriptor) for item in descriptors
        ):
            raise TypeError(
                "bootstrap descriptors must contain KernelDescriptor values"
            )
        if not isinstance(self.observed_environment, AttentionRuntimeEnvironment):
            raise TypeError(
                "bootstrap observed_environment must be AttentionRuntimeEnvironment"
            )
        if not isinstance(self.plan_gate, AttentionOperatorPlanGate):
            raise TypeError(
                "bootstrap plan_gate must implement AttentionOperatorPlanGate"
            )
        if self.plan_scorer is not None:
            if not isinstance(
                self.plan_scorer, AttentionOperatorRuntimePlanScorer
            ):
                raise TypeError(
                    "bootstrap plan_scorer must implement "
                    "AttentionOperatorRuntimePlanScorer"
                )
            if (
                self.plan_scorer.provider_id != self.plan_gate.provider_id
                or self.plan_scorer.operation_id != self.operation_id
            ):
                raise SchemaError("bootstrap plan scorer identity differs")
        if not isinstance(self.logical_factory, AttentionOperatorPlanFactory):
            raise TypeError(
                "bootstrap logical_factory must implement AttentionOperatorPlanFactory"
            )
        if not isinstance(self.logical_run_adapter, AttentionOperatorRunAdapter):
            raise TypeError(
                "bootstrap logical_run_adapter must implement "
                "AttentionOperatorRunAdapter"
            )
        if not isinstance(
            self.tensor_materializer, AttentionOperatorTensorMaterializer
        ):
            raise TypeError(
                "bootstrap tensor_materializer must implement "
                "AttentionOperatorTensorMaterializer"
            )
        quantization_bindings = tuple(self.quantization_bindings)
        if any(
            not isinstance(item, AttentionOperatorQuantizationBinding)
            for item in quantization_bindings
        ):
            raise TypeError(
                "bootstrap quantization_bindings must contain quantization bindings"
            )
        nvfp4_bindings = tuple(self.nvfp4_packed_kv_bindings)
        if any(
            not isinstance(item, AttentionOperatorNvfp4PackedKVBinding)
            for item in nvfp4_bindings
        ):
            raise TypeError(
                "bootstrap nvfp4_packed_kv_bindings must contain packed bindings"
            )
        nvfp4_fingerprints = tuple(
            item.quant_spec.fingerprint for item in nvfp4_bindings
        )
        if len(set(nvfp4_fingerprints)) != len(nvfp4_fingerprints):
            raise SchemaError("bootstrap contains duplicate NVFP4 QuantSpec binding")
        general_fingerprints = {
            item.quant_spec.fingerprint for item in quantization_bindings
        }
        if general_fingerprints.intersection(nvfp4_fingerprints):
            raise SchemaError("bootstrap routes one QuantSpec through two bindings")
        if any(
            item.provider_id != self.plan_gate.provider_id
            or item.operation_id != self.operation_id
            for item in nvfp4_bindings
        ):
            raise SchemaError("bootstrap NVFP4 binding identity differs")
        if self.tensor_metadata_inspector is None:
            raise SchemaError(
                "bootstrap provider runtime requires a tensor metadata inspector"
            )
        if self.tensor_metadata_inspector is not None and not isinstance(
            self.tensor_metadata_inspector,
            AttentionOperatorTensorMetadataInspector,
        ):
            raise TypeError(
                "tensor_metadata_inspector must implement "
                "AttentionOperatorTensorMetadataInspector"
            )
        if not isinstance(self.tensor_access_policy, AttentionTensorAccessPolicy):
            raise TypeError(
                "tensor_access_policy must be AttentionTensorAccessPolicy"
            )
        if not isinstance(self.validate_provider_results, bool):
            raise SchemaError("validate_provider_results must be boolean")
        if not isinstance(
            self.quant_physical_layout_catalog, QuantPhysicalLayoutCatalog
        ):
            raise TypeError(
                "quant_physical_layout_catalog must be QuantPhysicalLayoutCatalog"
            )
        if self.physical_layout_evidence_bundle is not None and not isinstance(
            self.physical_layout_evidence_bundle,
            AttentionVerifiedOperatorPhysicalEvidenceBundle,
        ):
            raise TypeError(
                "physical_layout_evidence_bundle must be a verified bundle"
            )
        if self.jit_plan_resolver is not None and not isinstance(
            self.jit_plan_resolver, AttentionJitPlanResolver
        ):
            raise TypeError(
                "jit_plan_resolver must implement AttentionJitPlanResolver"
            )
        if self.jit_artifact_resolver is not None and not isinstance(
            self.jit_artifact_resolver, AttentionJitArtifactResolver
        ):
            raise TypeError(
                "jit_artifact_resolver must implement AttentionJitArtifactResolver"
            )
        if self.jit_module_resolver is not None and not isinstance(
            self.jit_module_resolver, AttentionJitModuleResolver
        ):
            raise TypeError(
                "jit_module_resolver must implement AttentionJitModuleResolver"
            )
        if self.jit_executor_binder is not None and not isinstance(
            self.jit_executor_binder, AttentionJitExecutorBinder
        ):
            raise TypeError(
                "jit_executor_binder must implement AttentionJitExecutorBinder"
            )
        if self.jit_planner_binder is not None and not isinstance(
            self.jit_planner_binder, AttentionJitPlannerBinder
        ):
            raise TypeError(
                "jit_planner_binder must implement AttentionJitPlannerBinder"
            )
        if not isinstance(self.numerics_policy, AttentionNumericsPolicy):
            raise TypeError("bootstrap numerics_policy must be AttentionNumericsPolicy")
        if self.corpus is not None and not isinstance(
            self.corpus, AttentionTraceCorpus
        ):
            raise TypeError("bootstrap corpus must be AttentionTraceCorpus")
        if self.coverage_policy is not None and not isinstance(
            self.coverage_policy, AttentionCoveragePolicy
        ):
            raise TypeError(
                "bootstrap coverage_policy must be AttentionCoveragePolicy"
            )
        tuned_ids = tuple(str(item) for item in self.tuned_kernel_ids)
        if any(not item for item in tuned_ids) or len(set(tuned_ids)) != len(
            tuned_ids
        ):
            raise SchemaError("bootstrap tuned kernel ids must be unique and non-empty")
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(self, "adapter_version", str(self.adapter_version))
        object.__setattr__(self, "supported_package_versions", tuple(sorted(versions)))
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(
            self, "quantization_bindings", quantization_bindings
        )
        object.__setattr__(
            self, "nvfp4_packed_kv_bindings", nvfp4_bindings
        )
        object.__setattr__(self, "tuned_kernel_ids", tuned_ids)
        object.__setattr__(self, "replay_evidence", bool(self.replay_evidence))

    @property
    def provider_id(self) -> str:
        return self.plan_gate.provider_id


def bind_attention_operator_plan_scoring_manifest(
    specs: Sequence[AttentionOperatorPackageRuntimeSpec],
    manifest: AttentionOperatorPlanScoringManifest,
) -> Tuple[AttentionOperatorPackageRuntimeSpec, ...]:
    """Bind one reviewed policy to every runtime spec as an immutable set.

    The manifest and spec identity sets must match exactly.  Validation is
    completed before any replacement spec is returned, so callers cannot
    observe a partly bound bootstrap set.
    """

    values = tuple(specs)
    if any(
        not isinstance(item, AttentionOperatorPackageRuntimeSpec)
        for item in values
    ):
        raise TypeError(
            "specs must contain AttentionOperatorPackageRuntimeSpec values"
        )
    if not isinstance(manifest, AttentionOperatorPlanScoringManifest):
        raise TypeError(
            "manifest must be AttentionOperatorPlanScoringManifest"
        )
    spec_identities = tuple(
        (item.provider_id, item.operation_id) for item in values
    )
    if len(set(spec_identities)) != len(spec_identities):
        raise SchemaError("duplicate Attention runtime spec identity")
    policy_identities = tuple(
        (item.provider_id, item.operation_id) for item in manifest.policies
    )
    missing = tuple(sorted(set(spec_identities).difference(policy_identities)))
    orphan = tuple(sorted(set(policy_identities).difference(spec_identities)))
    if missing or orphan:
        raise SchemaError(
            "Attention scoring manifest identity set differs from runtime specs "
            "(missing=%r, orphan=%r)" % (missing, orphan)
        )
    policies = {
        (item.provider_id, item.operation_id): item
        for item in manifest.policies
    }
    for spec in values:
        current = spec.plan_scorer
        policy = policies[(spec.provider_id, spec.operation_id)]
        if current is None:
            continue
        if not isinstance(current, AttentionOperatorPlanScoringPolicy):
            raise SchemaError(
                "Attention runtime spec already has a non-manifest plan scorer"
            )
        if current.fingerprint != policy.fingerprint:
            raise SchemaError(
                "Attention runtime spec plan scorer differs from manifest policy"
            )
    return tuple(
        replace(
            spec,
            plan_scorer=policies[(spec.provider_id, spec.operation_id)],
        )
        for spec in values
    )


def build_attention_operator_package_runtime(
    spec: AttentionOperatorPackageRuntimeSpec,
    *,
    operation_catalog: Optional[AttentionOperatorOperationCatalog] = None,
    package_loader: Optional[AttentionOperatorPackageLoader] = None,
) -> AttentionOperatorPackageRuntimeImplementation:
    """Compose one implementation without observing or importing its package."""

    if not isinstance(spec, AttentionOperatorPackageRuntimeSpec):
        raise TypeError("spec must be AttentionOperatorPackageRuntimeSpec")
    if operation_catalog is None:
        operation_catalog = load_packaged_attention_operator_catalog()
    if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
        raise TypeError("operation_catalog must be AttentionOperatorOperationCatalog")
    if package_loader is None:
        package_loader = ImportlibAttentionOperatorPackageLoader()
    if not isinstance(package_loader, AttentionOperatorPackageLoader):
        raise TypeError("package_loader must implement AttentionOperatorPackageLoader")
    operation = operation_catalog.get(spec.operation_id)
    if operation.provider_id != spec.provider_id:
        raise SchemaError("bootstrap operation and plan gate providers differ")
    nvfp4_bindings = validate_attention_operator_nvfp4_packed_kv_bindings(
        operation, spec.profiles, spec.nvfp4_packed_kv_bindings
    )
    delegated_quant_specs = tuple(
        item.quant_spec for item in nvfp4_bindings
    )
    quantization_bindings = validate_attention_operator_quantization_bindings(
        operation,
        spec.profiles,
        spec.quantization_bindings,
        delegated_quant_specs=delegated_quant_specs,
    )
    evidence_bundle = spec.physical_layout_evidence_bundle
    if evidence_bundle is not None:
        evidence_bundle.manifest.validate_runtime_spec(
            operation,
            spec.adapter_version,
            spec.supported_package_versions,
            spec.quant_physical_layout_catalog,
        )
        physical_layout_evidence = evidence_bundle.evidences
    else:
        physical_layout_evidence = ()
    plan_gate = AttentionOperatorQuantizationPlanGate(
        spec.plan_gate,
        operation,
        quantization_bindings,
        delegated_quant_specs,
    )
    run_adapter_factories = []
    if quantization_bindings:
        run_adapter_factories.append(
            AttentionOperatorQuantizationRunAdapterFactory(
                operation,
                quantization_bindings,
                spec.tensor_metadata_inspector,
                spec.tensor_access_policy,
                spec.quant_physical_layout_catalog,
                spec.profiles,
                spec.descriptors,
                spec.observed_environment,
                physical_layout_evidence,
                delegated_quant_specs,
            )
        )
    if nvfp4_bindings:
        run_adapter_factories.append(
            AttentionOperatorNvfp4PackedKVRunAdapterFactory(
                operation,
                nvfp4_bindings,
                spec.tensor_metadata_inspector,
                spec.tensor_access_policy,
            )
        )
    run_adapter_factories.append(
        AttentionOperatorCallerBufferRunAdapterFactory(operation)
    )
    run_adapter_factories.append(
        AttentionOperatorRunTensorValidationAdapterFactory(
            operation.provider_id,
            operation.operation_id,
            spec.tensor_metadata_inspector,
            spec.tensor_access_policy,
        )
    )
    run_adapter_factory = AttentionOperatorRunAdapterFactoryChain(
        operation.provider_id,
        operation.operation_id,
        run_adapter_factories,
    )
    compatibility = AttentionOperatorPackageCompatibility(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        adapter_version=spec.adapter_version,
        supported_package_versions=spec.supported_package_versions,
    )
    package_resolver = AttentionOperatorPackageResolver(
        operation_catalog, compatibility, package_loader
    )
    completion_validator_factory = (
        AttentionOperatorCompletionValidatorFactory(
            operation,
            spec.tensor_metadata_inspector,
            spec.tensor_access_policy,
        )
        if spec.validate_provider_results
        else None
    )
    authority_resolver = AttentionEvidenceOperatorRuntimeAuthorityResolver(
        operation,
        spec.profiles,
        spec.descriptors,
        spec.observed_environment,
        backend=spec.backend,
        tuned_kernel_ids=spec.tuned_kernel_ids,
        numerics_policy=spec.numerics_policy,
        corpus=spec.corpus,
        coverage_policy=spec.coverage_policy,
        replay_evidence=spec.replay_evidence,
        physical_layout_catalog=spec.quant_physical_layout_catalog,
        physical_layout_evidence=physical_layout_evidence,
    )
    return AttentionOperatorPackageRuntimeImplementation(
        priority=spec.priority,
        package_resolver=package_resolver,
        plan_gate=plan_gate,
        authority_resolver=authority_resolver,
        logical_factory=spec.logical_factory,
        logical_run_adapter=spec.logical_run_adapter,
        tensor_materializer=spec.tensor_materializer,
        run_adapter_factory=run_adapter_factory,
        completion_validator_factory=completion_validator_factory,
        plan_scorer=spec.plan_scorer,
        jit_plan_resolver=spec.jit_plan_resolver,
        jit_artifact_resolver=spec.jit_artifact_resolver,
        jit_module_resolver=spec.jit_module_resolver,
        jit_planner_binder=spec.jit_planner_binder,
        jit_executor_binder=spec.jit_executor_binder,
    )


def build_attention_operator_runtime_resolvers(
    specs: Sequence[AttentionOperatorPackageRuntimeSpec] = (),
    *,
    operation_catalog: Optional[AttentionOperatorOperationCatalog] = None,
    package_loader: Optional[AttentionOperatorPackageLoader] = None,
    plan_scoring_manifest: Optional[
        AttentionOperatorPlanScoringManifest
    ] = None,
) -> AttentionOperatorRuntimeResolverRegistry:
    """Build the immutable NPU resolver tree from explicit integration specs."""

    values = tuple(specs)
    if any(
        not isinstance(item, AttentionOperatorPackageRuntimeSpec)
        for item in values
    ):
        raise TypeError("specs must contain AttentionOperatorPackageRuntimeSpec values")
    if plan_scoring_manifest is not None:
        values = bind_attention_operator_plan_scoring_manifest(
            values,
            plan_scoring_manifest,
        )
    if not values:
        return EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS
    if operation_catalog is None:
        operation_catalog = load_packaged_attention_operator_catalog()
    if package_loader is None:
        package_loader = ImportlibAttentionOperatorPackageLoader()
    implementations = tuple(
        build_attention_operator_package_runtime(
            item,
            operation_catalog=operation_catalog,
            package_loader=package_loader,
        )
        for item in values
    )
    implementation_registry = AttentionOperatorRuntimeImplementationRegistry(
        implementations
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementation_registry),)
    )


# No real integration is declared until exact package versions, runtime
# environment, capability evidence, kernel provenance, and materializer are
# verified together.  Future provider work extends this private tuple.
_DEFAULT_ATTENTION_OPERATOR_PACKAGE_RUNTIME_SPECS: Tuple[
    AttentionOperatorPackageRuntimeSpec, ...
] = ()


def build_default_attention_operator_runtime_resolvers(
) -> AttentionOperatorRuntimeResolverRegistry:
    """Build the process default; currently the intentionally empty registry."""

    return build_attention_operator_runtime_resolvers(
        _DEFAULT_ATTENTION_OPERATOR_PACKAGE_RUNTIME_SPECS
    )


__all__ = [
    "ATTENTION_OPERATOR_BOOTSTRAP_VERSION",
    "AttentionOperatorPackageRuntimeSpec",
    "bind_attention_operator_plan_scoring_manifest",
    "build_attention_operator_package_runtime",
    "build_attention_operator_runtime_resolvers",
    "build_default_attention_operator_runtime_resolvers",
]
