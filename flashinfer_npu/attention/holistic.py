"""Public Host facade for FlashInfer's holistic mixed ``BatchAttention``."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Optional

from flashinfer_npu.runtime import DispatchError, SchemaError

from .frontend import (
    adapt_head_scale,
    adapt_paged_kv_data,
    canonicalize_dtype_name,
    canonicalize_flashinfer_paged_kv_dtype,
    canonicalize_kv_dtype,
    finalize_reference_result,
    finite_scalar,
    framework_index_values,
    parse_kv_layout,
    reference_index_values,
    require_reference_tensor,
)
from .planner import AttentionFrameworkSession, AttentionStateError
from .operator_resolver import (
    AttentionOperatorBatchRuntime,
    AttentionOperatorRuntimeResolverRegistry,
)
from .plan_selection import (
    AttentionPlanSelection,
    build_provider_plan_selection,
    build_reference_plan_selection,
)
from .operation_catalog import (
    AttentionOperatorOperationCatalog,
    load_packaged_attention_operator_catalog,
)
from .operator_bootstrap import build_default_attention_operator_runtime_resolvers
from .operator_declaration import (
    AttentionDeclaredOperatorPackageRuntimeSpec,
    AttentionOperatorRuntimeDeclarationBinding,
    build_declared_attention_operator_runtime_resolvers,
)
from .operator_scoring import (
    AttentionOperatorPlanScoringManifest,
    AttentionOperatorPlanScoringManifestBinding,
)
from .provider_bundle import (
    AttentionOperatorProviderIntegrationBundleBinding,
)
from .reference import ReferenceAttentionExecutor, ReferenceTensor
from .schema import AttentionMode, AttentionPlanSpec, MixedPagedKVMetadata
from .workspace import AttentionWorkspaceContract
from .tensor_contract import validate_reference_attention_views


# Package integrations replace this immutable registry at bootstrap.  Keeping
# it module-private avoids adding provider controls to the public constructor.
_operator_runtime_resolvers = build_default_attention_operator_runtime_resolvers()
_operator_runtime_operation_catalog = load_packaged_attention_operator_catalog()
_operator_runtime_declarations = ()
_operator_runtime_plan_scoring_manifest_binding = None
_operator_runtime_provider_integration_bundle_binding = None
_operator_runtime_resolvers_generation = 0
_operator_runtime_resolvers_lock = RLock()


@dataclass(frozen=True)
class AttentionOperatorRuntimeRegistrySnapshot:
    """Atomic process-bootstrap snapshot captured by a future NPU wrapper."""

    generation: int
    device_types: tuple
    registry: AttentionOperatorRuntimeResolverRegistry = field(
        repr=False, compare=False
    )
    operation_catalog: AttentionOperatorOperationCatalog = field(
        default=None, repr=False, compare=False
    )
    runtime_declarations: tuple = ()
    plan_scoring_manifest_binding: Optional[
        AttentionOperatorPlanScoringManifestBinding
    ] = None
    provider_integration_bundle_binding: Optional[
        AttentionOperatorProviderIntegrationBundleBinding
    ] = None

    def __post_init__(self):
        if not isinstance(self.generation, int) or isinstance(self.generation, bool):
            raise SchemaError("Attention runtime registry generation must be an integer")
        if self.generation < 0:
            raise SchemaError("Attention runtime registry generation cannot be negative")
        if not isinstance(self.registry, AttentionOperatorRuntimeResolverRegistry):
            raise TypeError(
                "registry must be AttentionOperatorRuntimeResolverRegistry"
            )
        catalog = self.operation_catalog
        if catalog is None:
            catalog = load_packaged_attention_operator_catalog()
            object.__setattr__(self, "operation_catalog", catalog)
        if not isinstance(catalog, AttentionOperatorOperationCatalog):
            raise TypeError(
                "operation_catalog must be AttentionOperatorOperationCatalog"
            )
        device_types = tuple(str(item) for item in self.device_types)
        actual_types = tuple(item[0] for item in self.registry.resolvers)
        if device_types != actual_types:
            raise SchemaError("Attention runtime registry snapshot is stale")
        declarations = tuple(self.runtime_declarations)
        if any(
            not isinstance(item, AttentionOperatorRuntimeDeclarationBinding)
            for item in declarations
        ):
            raise TypeError(
                "runtime_declarations must contain declaration bindings"
            )
        identities = tuple(
            (item.provider_id, item.operation_id) for item in declarations
        )
        if len(set(identities)) != len(identities):
            raise SchemaError("runtime declaration bindings contain duplicates")
        if declarations and "npu" not in device_types:
            raise SchemaError("runtime declarations require an npu resolver")
        for item in declarations:
            operation = catalog.get(item.operation_id)
            if operation.provider_id != item.provider_id:
                raise SchemaError(
                    "runtime declaration binding differs from operation catalog"
                )
        scoring_binding = self.plan_scoring_manifest_binding
        if scoring_binding is not None:
            if not isinstance(
                scoring_binding, AttentionOperatorPlanScoringManifestBinding
            ):
                raise TypeError(
                    "plan_scoring_manifest_binding has the wrong type"
                )
            if not declarations:
                raise SchemaError(
                    "scoring manifest binding requires runtime declarations"
                )
            if set(scoring_binding.identities) != set(identities):
                raise SchemaError(
                    "scoring manifest binding differs from runtime declarations"
                )
        bundle_binding = self.provider_integration_bundle_binding
        if bundle_binding is not None:
            if not isinstance(
                bundle_binding,
                AttentionOperatorProviderIntegrationBundleBinding,
            ):
                raise TypeError(
                    "provider_integration_bundle_binding has the wrong type"
                )
            if not declarations:
                raise SchemaError(
                    "provider integration bundle binding requires runtime "
                    "declarations"
                )
            if scoring_binding is None:
                raise SchemaError(
                    "provider integration bundle binding requires a scoring "
                    "manifest binding"
                )
            if (
                bundle_binding.catalog_name != catalog.name
                or bundle_binding.catalog_fingerprint != catalog.fingerprint
            ):
                raise SchemaError(
                    "provider integration bundle binding differs from operation "
                    "catalog"
                )
            if (
                bundle_binding.scoring_manifest_id
                != scoring_binding.manifest_id
                or bundle_binding.scoring_manifest_fingerprint
                != scoring_binding.manifest_fingerprint
            ):
                raise SchemaError(
                    "provider integration bundle binding differs from scoring "
                    "manifest"
                )
            declaration_bindings = tuple(
                sorted(
                    (
                        item.provider_id,
                        item.operation_id,
                        item.declaration_fingerprint,
                    )
                    for item in declarations
                )
            )
            if bundle_binding.registration_bindings != declaration_bindings:
                raise SchemaError(
                    "provider integration bundle binding differs from runtime "
                    "declarations"
                )
        object.__setattr__(self, "device_types", device_types)
        object.__setattr__(
            self,
            "runtime_declarations",
            tuple(
                sorted(
                    declarations,
                    key=lambda item: (item.provider_id, item.operation_id),
                )
            ),
        )

    @property
    def plan_scoring_manifest_id(self):
        binding = self.plan_scoring_manifest_binding
        return None if binding is None else binding.manifest_id

    @property
    def plan_scoring_manifest_fingerprint(self):
        binding = self.plan_scoring_manifest_binding
        return None if binding is None else binding.manifest_fingerprint

    @property
    def provider_integration_bundle_id(self):
        binding = self.provider_integration_bundle_binding
        return None if binding is None else binding.bundle_id

    @property
    def provider_integration_bundle_fingerprint(self):
        binding = self.provider_integration_bundle_binding
        return None if binding is None else binding.bundle_fingerprint

    def declaration_fingerprint(self, provider_id: str, operation_id: str):
        """Return the reviewed declaration for one selected operation, if any."""

        identity = (str(provider_id), str(operation_id))
        for item in self.runtime_declarations:
            if (item.provider_id, item.operation_id) == identity:
                return item.declaration_fingerprint
        return None

    @property
    def runtime_declaration_binding_tuples(self):
        return tuple(
            (
                item.provider_id,
                item.operation_id,
                item.declaration_fingerprint,
            )
            for item in self.runtime_declarations
        )


def attention_operator_runtime_registry_snapshot(
) -> AttentionOperatorRuntimeRegistrySnapshot:
    """Return one consistent registry generation without probing providers."""

    with _operator_runtime_resolvers_lock:
        registry = _operator_runtime_resolvers
        operation_catalog = _operator_runtime_operation_catalog
        runtime_declarations = _operator_runtime_declarations
        scoring_binding = _operator_runtime_plan_scoring_manifest_binding
        bundle_binding = _operator_runtime_provider_integration_bundle_binding
        generation = _operator_runtime_resolvers_generation
    return AttentionOperatorRuntimeRegistrySnapshot(
        generation=generation,
        device_types=tuple(item[0] for item in registry.resolvers),
        registry=registry,
        operation_catalog=operation_catalog,
        runtime_declarations=runtime_declarations,
        plan_scoring_manifest_binding=scoring_binding,
        provider_integration_bundle_binding=bundle_binding,
    )


def _install_attention_operator_runtime_resolvers(
    registry: AttentionOperatorRuntimeResolverRegistry,
    *,
    operation_catalog: AttentionOperatorOperationCatalog,
    runtime_declarations=(),
    plan_scoring_manifest_binding=None,
    provider_integration_bundle_binding=None,
    expected_generation=None,
) -> AttentionOperatorRuntimeRegistrySnapshot:
    device_types = tuple(item[0] for item in registry.resolvers)
    if any(item != "npu" for item in device_types):
        raise SchemaError("public Attention runtime registry may only route npu")
    if expected_generation is not None and (
        not isinstance(expected_generation, int)
        or isinstance(expected_generation, bool)
        or expected_generation < 0
    ):
        raise SchemaError("expected Attention runtime generation must be non-negative")
    candidate = AttentionOperatorRuntimeRegistrySnapshot(
        generation=0,
        device_types=device_types,
        registry=registry,
        operation_catalog=operation_catalog,
        runtime_declarations=tuple(runtime_declarations),
        plan_scoring_manifest_binding=plan_scoring_manifest_binding,
        provider_integration_bundle_binding=(
            provider_integration_bundle_binding
        ),
    )
    global _operator_runtime_resolvers
    global _operator_runtime_operation_catalog
    global _operator_runtime_declarations
    global _operator_runtime_plan_scoring_manifest_binding
    global _operator_runtime_provider_integration_bundle_binding
    global _operator_runtime_resolvers_generation
    with _operator_runtime_resolvers_lock:
        if (
            expected_generation is not None
            and expected_generation != _operator_runtime_resolvers_generation
        ):
            raise SchemaError("Attention runtime registry generation changed")
        _operator_runtime_resolvers = registry
        _operator_runtime_operation_catalog = operation_catalog
        _operator_runtime_declarations = candidate.runtime_declarations
        _operator_runtime_plan_scoring_manifest_binding = (
            candidate.plan_scoring_manifest_binding
        )
        _operator_runtime_provider_integration_bundle_binding = (
            candidate.provider_integration_bundle_binding
        )
        _operator_runtime_resolvers_generation += 1
        generation = _operator_runtime_resolvers_generation
    return AttentionOperatorRuntimeRegistrySnapshot(
        generation=generation,
        device_types=device_types,
        registry=registry,
        operation_catalog=operation_catalog,
        runtime_declarations=candidate.runtime_declarations,
        plan_scoring_manifest_binding=(
            candidate.plan_scoring_manifest_binding
        ),
        provider_integration_bundle_binding=(
            candidate.provider_integration_bundle_binding
        ),
    )


def install_attention_operator_runtime_resolvers(
    registry: AttentionOperatorRuntimeResolverRegistry,
    *,
    operation_catalog: AttentionOperatorOperationCatalog = None,
    expected_generation=None,
) -> AttentionOperatorRuntimeRegistrySnapshot:
    """Atomically install legacy or synthetic framework integrations."""

    if not isinstance(registry, AttentionOperatorRuntimeResolverRegistry):
        raise TypeError("registry must be AttentionOperatorRuntimeResolverRegistry")
    if operation_catalog is None:
        operation_catalog = load_packaged_attention_operator_catalog()
    if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
        raise TypeError(
            "operation_catalog must be AttentionOperatorOperationCatalog"
        )
    return _install_attention_operator_runtime_resolvers(
        registry,
        operation_catalog=operation_catalog,
        expected_generation=expected_generation,
    )


def _install_declared_attention_operator_runtime_resolvers(
    registrations,
    *,
    operation_catalog: AttentionOperatorOperationCatalog = None,
    package_loader=None,
    plan_scoring_manifest: Optional[
        AttentionOperatorPlanScoringManifest
    ] = None,
    provider_integration_bundle_binding=None,
    expected_generation=None,
) -> AttentionOperatorRuntimeRegistrySnapshot:
    """Internal install root shared by declared and bundled integrations."""

    values = tuple(registrations)
    if any(
        not isinstance(item, AttentionDeclaredOperatorPackageRuntimeSpec)
        for item in values
    ):
        raise TypeError(
            "registrations must contain AttentionDeclaredOperatorPackageRuntimeSpec"
        )
    if operation_catalog is None:
        operation_catalog = load_packaged_attention_operator_catalog()
    registry = build_declared_attention_operator_runtime_resolvers(
        values,
        operation_catalog=operation_catalog,
        package_loader=package_loader,
        plan_scoring_manifest=plan_scoring_manifest,
    )
    return _install_attention_operator_runtime_resolvers(
        registry,
        operation_catalog=operation_catalog,
        runtime_declarations=tuple(item.binding for item in values),
        plan_scoring_manifest_binding=(
            plan_scoring_manifest.binding
            if plan_scoring_manifest is not None
            else None
        ),
        provider_integration_bundle_binding=(
            provider_integration_bundle_binding
        ),
        expected_generation=expected_generation,
    )


def install_declared_attention_operator_runtime_resolvers(
    registrations,
    *,
    operation_catalog: AttentionOperatorOperationCatalog = None,
    package_loader=None,
    plan_scoring_manifest: Optional[
        AttentionOperatorPlanScoringManifest
    ] = None,
    expected_generation=None,
) -> AttentionOperatorRuntimeRegistrySnapshot:
    """Atomically install declaration-bound external integrations."""

    return _install_declared_attention_operator_runtime_resolvers(
        registrations,
        operation_catalog=operation_catalog,
        package_loader=package_loader,
        plan_scoring_manifest=plan_scoring_manifest,
        expected_generation=expected_generation,
    )


class BatchAttention:
    """Holistic mixed prefill/decode Attention with FlashInfer lifecycle."""

    def __init__(self, kv_layout="NHD", device="cuda"):
        if device == "cuda" or str(device).startswith("cuda:"):
            raise DispatchError(
                "CUDA is not a valid flashinfer-npu device; pass device='cpu' for "
                "the Host oracle or device='npu[:index]' for an installed provider"
            )
        self._kv_layout = parse_kv_layout(kv_layout)
        self.device = str(device)
        self._reference_backend = self.device == "cpu"
        self._operator_runtime = None
        if not self._reference_backend:
            if self.device.split(":", 1)[0] != "npu":
                raise DispatchError("BatchAttention device must be cpu or npu[:index]")
            self.float_workspace_buffer = None
            self.int_workspace_buffer = None
            self.page_locked_int_workspace_buffer = None
            self._workspace_contract = None
            self._session = None
            self._executor = None
            snapshot = attention_operator_runtime_registry_snapshot()
            self._operator_runtime_registry_snapshot = snapshot
            self._operator_runtime = AttentionOperatorBatchRuntime(
                self.device,
                snapshot.registry,
                snapshot.operation_catalog,
                runtime_declaration_bindings=(
                    snapshot.runtime_declaration_binding_tuples
                ),
                plan_scoring_manifest_binding=(
                    snapshot.plan_scoring_manifest_binding
                ),
                provider_integration_bundle_binding=(
                    snapshot.provider_integration_bundle_binding
                ),
            )
            return
        # Logical placeholders only; Host scalar execution needs no workspace.
        self.float_workspace_buffer = ReferenceTensor.zeros(
            (0,), dtype="uint8", device=device
        )
        self.int_workspace_buffer = ReferenceTensor.zeros(
            (0,), dtype="uint8", device=device
        )
        self.page_locked_int_workspace_buffer = ReferenceTensor.zeros(
            (0,), dtype="uint8", device="cpu"
        )
        self._workspace_contract = AttentionWorkspaceContract.for_host_reference(
            device=device,
            float_capacity_bytes=0,
            int_capacity_bytes=0,
        )
        self._session = AttentionFrameworkSession(AttentionMode.BATCH_MIXED_PAGED)
        self._executor = ReferenceAttentionExecutor()

    @property
    def plan_state(self):
        if self._operator_runtime is not None:
            return self._operator_runtime.plan_state
        return self._session.plan_state

    @property
    def plan_selection(self) -> AttentionPlanSelection:
        """Describe the selected route without exposing an executable handle."""

        plan = self.plan_state
        if self._operator_runtime is not None:
            active_plan = self._operator_runtime.operator_session.active_plan
            plan_score = self._operator_runtime.runtime_plan_score
            scoring_binding = (
                self._operator_runtime.runtime_plan_scoring_binding
            )
            bundle_binding = (
                self._operator_runtime.runtime_provider_integration_bundle_binding
            )
            return build_provider_plan_selection(
                plan,
                active_plan,
                registry_generation=(
                    self._operator_runtime_registry_snapshot.generation
                ),
                runtime_declaration_fingerprint=(
                    self._operator_runtime_registry_snapshot.declaration_fingerprint(
                        active_plan.provider_selection.provider_id,
                        active_plan.prepared_plan.implementation_id,
                    )
                ),
                provider_integration_bundle_id=(
                    None if bundle_binding is None else bundle_binding[0]
                ),
                provider_integration_bundle_fingerprint=(
                    None if bundle_binding is None else bundle_binding[1]
                ),
                plan_scoring_manifest_id=(
                    None if scoring_binding is None else scoring_binding[0]
                ),
                plan_scoring_manifest_fingerprint=(
                    None if scoring_binding is None else scoring_binding[1]
                ),
                plan_scoring_policy_id=(
                    None if scoring_binding is None else scoring_binding[2]
                ),
                plan_scoring_policy_fingerprint=(
                    None if scoring_binding is None else scoring_binding[3]
                ),
                plan_score=(None if plan_score is None else plan_score.value),
                plan_score_source=(
                    None if plan_score is None else plan_score.source
                ),
                plan_score_reason=(
                    None if plan_score is None else plan_score.reason
                ),
                runtime_resolution_fingerprint=(
                    self._operator_runtime.runtime_resolution_fingerprint
                ),
            )
        return build_reference_plan_selection(plan)

    @property
    def last_run_receipt(self):
        """Return read-only evidence for the latest validated provider run."""

        if self._operator_runtime is None:
            raise AttentionStateError(
                "reference Attention runs do not publish provider run receipts"
            )
        return self._operator_runtime.last_run_receipt

    @property
    def workspace_contract(self):
        return self._workspace_contract

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        page_size,
        causal=False,
        sm_scale=None,
        logits_soft_cap=None,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        use_profiler=False,
    ):
        if head_dim_qk > 256 or head_dim_vo > 256:
            raise ValueError("BatchAttention does not support head_dim > 256")
        index_reader = (
            reference_index_values
            if self._reference_backend
            else framework_index_values
        )
        metadata = MixedPagedKVMetadata(
            qo_indptr=index_reader(qo_indptr, "qo_indptr"),
            kv_indptr=index_reader(kv_indptr, "kv_indptr"),
            kv_indices=index_reader(kv_indices, "kv_indices"),
            kv_len_arr=index_reader(kv_len_arr, "kv_len_arr"),
            page_size=page_size,
        )
        q_dtype = canonicalize_dtype_name(q_data_type)
        kv_dtype, kv_quant_spec = canonicalize_kv_dtype(kv_data_type, q_dtype)
        if self._operator_runtime is not None and kv_quant_spec is None:
            kv_dtype, kv_quant_spec = canonicalize_flashinfer_paged_kv_dtype(
                kv_dtype, q_dtype
            )
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_MIXED_PAGED,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            kv_layout=self._kv_layout,
            causal=bool(causal),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=q_dtype,
            kv_quant_spec=kv_quant_spec,
            sm_scale=(None if sm_scale is None else finite_scalar(sm_scale, "sm_scale")),
            logits_soft_cap=(
                None
                if logits_soft_cap is None
                else finite_scalar(logits_soft_cap, "logits_soft_cap")
            ),
            use_profiler=bool(use_profiler),
        )
        if self._operator_runtime is not None:
            self._operator_runtime.plan(spec, metadata)
        else:
            plan = self._session.plan(spec, metadata)
            self._workspace_contract = self._workspace_contract.bind_plan(
                plan.generation
            )

    def run(
        self,
        q,
        kv_cache,
        out=None,
        lse=None,
        k_scale=None,
        v_scale=None,
        logits_soft_cap=0.0,
        profiler_buffer=None,
        kv_cache_sf=None,
    ):
        if self._operator_runtime is not None:
            return self._operator_runtime.run(
                q,
                kv_cache,
                return_lse=True,
                out=out,
                lse=lse,
                k_scale=k_scale,
                v_scale=v_scale,
                logits_soft_cap=logits_soft_cap,
                profiler_buffer=profiler_buffer,
                kv_cache_sf=kv_cache_sf,
            )
        plan = self.plan_state
        query = require_reference_tensor(q, "q")
        self._workspace_contract.validate_run(
            device=query.device,
            plan_generation=plan.generation,
        )
        if kv_cache_sf is not None:
            raise NotImplementedError("NVFP4 kv_cache_sf is not implemented")
        if plan.spec.use_profiler or profiler_buffer is not None:
            raise NotImplementedError(
                "Host BatchAttention does not emit profiler events"
            )
        kv_data = adapt_paged_kv_data(
            kv_cache,
            page_size=plan.metadata.page_size,
            num_kv_heads=plan.spec.num_kv_heads,
            head_dim_qk=plan.spec.head_dim_qk,
            head_dim_vo=int(plan.spec.head_dim_vo),
            dtype=plan.spec.kv_dtype,
            layout=plan.spec.kv_layout,
            quant_spec=plan.spec.kv_quant_spec,
        )
        validate_reference_attention_views(
            query,
            kv_data,
            out=out,
            lse=lse,
            workspace_float=self.float_workspace_buffer,
            workspace_int=self.int_workspace_buffer,
            plan=plan,
        )
        runtime_cap = finite_scalar(logits_soft_cap, "logits_soft_cap")
        result = self._executor.execute(
            plan,
            query,
            kv_data,
            return_lse=True,
            k_scale=adapt_head_scale(
                k_scale,
                name="k_scale",
                num_heads=plan.spec.num_kv_heads,
                device=self.device,
            ),
            v_scale=adapt_head_scale(
                v_scale,
                name="v_scale",
                num_heads=plan.spec.num_kv_heads,
                device=self.device,
            ),
            logits_soft_cap=runtime_cap,
        )
        return finalize_reference_result(
            result,
            out=out,
            lse=lse,
            return_lse=True,
        )


__all__ = [
    "AttentionOperatorRuntimeRegistrySnapshot",
    "BatchAttention",
    "attention_operator_runtime_registry_snapshot",
    "install_declared_attention_operator_runtime_resolvers",
    "install_attention_operator_runtime_resolvers",
]
