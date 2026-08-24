"""Internal runtime resolution behind the public ``BatchAttention`` facade."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from flashinfer_npu.jit.attention import (
    AttentionJitArtifactBinding,
    AttentionJitExecutorBinding,
    AttentionJitLoadedModuleBinding,
    AttentionJitPlanBinding,
    AttentionJitPlannerBinding,
)
from flashinfer_npu.runtime import Backend, DispatchError, SchemaError

from .operation_catalog import (
    AttentionOperatorOperationCatalog,
    load_packaged_attention_operator_catalog,
)
from .operator_callable import AttentionOperatorCallableBinding
from .operator_execution import AttentionRuntimeBindingAwareExecutor
from .operator_plan import AttentionOperatorPlanFactory
from .operator_provider import AttentionOperatorProviderSelection
from .operator_run import (
    AttentionLoweredOperatorCall,
    AttentionOperatorRunAdapter,
    AttentionOperatorWrapperSession,
)
from .dispatch import AttentionDispatchReceipt
from .planner import (
    AttentionFrameworkPlan,
    AttentionFrameworkSession,
    AttentionStateError,
)
from .schema import AttentionMetadata, AttentionMode, AttentionPlanSpec


ATTENTION_OPERATOR_RESOLVER_VERSION = 6

_DEVICE_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class AttentionLoweredOperatorExecutor(Protocol):
    """Provider-owned execution boundary; no implementation is supplied here."""

    provider_id: str
    operation_id: str

    def execute(self, call: AttentionLoweredOperatorCall) -> Any:
        """Execute an already validated call and return public wrapper results."""


@dataclass(frozen=True)
class AttentionResolvedOperatorRuntime:
    """Complete candidate returned by automatic runtime resolution."""

    framework_plan_fingerprint: str
    factory: AttentionOperatorPlanFactory = field(repr=False, compare=False)
    run_adapter: AttentionOperatorRunAdapter = field(repr=False, compare=False)
    executor: AttentionLoweredOperatorExecutor = field(repr=False, compare=False)
    receipt: AttentionDispatchReceipt = field(repr=False, compare=False)
    selection: AttentionOperatorProviderSelection = field(repr=False, compare=False)
    callable_binding: AttentionOperatorCallableBinding = field(
        repr=False, compare=False
    )
    jit_plan_binding: Optional[AttentionJitPlanBinding] = field(
        default=None, repr=False, compare=False
    )
    jit_artifact_binding: Optional[AttentionJitArtifactBinding] = field(
        default=None, repr=False, compare=False
    )
    jit_module_binding: Optional[AttentionJitLoadedModuleBinding] = field(
        default=None, repr=False, compare=False
    )
    jit_planner_binding: Optional[AttentionJitPlannerBinding] = field(
        default=None, repr=False, compare=False
    )
    jit_executor_binding: Optional[AttentionJitExecutorBinding] = field(
        default=None, repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RESOLVER_VERSION:
            raise SchemaError("unsupported Attention operator resolver version")
        if not isinstance(self.factory, AttentionOperatorPlanFactory):
            raise TypeError("factory must implement AttentionOperatorPlanFactory")
        if not isinstance(self.run_adapter, AttentionOperatorRunAdapter):
            raise TypeError("run_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(self.executor, AttentionLoweredOperatorExecutor):
            raise TypeError("executor must implement AttentionLoweredOperatorExecutor")
        if not isinstance(self.receipt, AttentionDispatchReceipt):
            raise TypeError("receipt must be AttentionDispatchReceipt")
        if not isinstance(self.selection, AttentionOperatorProviderSelection):
            raise TypeError("selection must be AttentionOperatorProviderSelection")
        if not isinstance(self.callable_binding, AttentionOperatorCallableBinding):
            raise TypeError("callable_binding must be AttentionOperatorCallableBinding")
        if self.receipt.backend == Backend.ASCENDC_JIT:
            if not isinstance(self.jit_plan_binding, AttentionJitPlanBinding):
                raise SchemaError(
                    "ascendc_jit runtime requires an Attention JIT plan binding"
                )
            self.jit_plan_binding.validate_resolved_runtime(
                self.framework_plan_fingerprint,
                self.receipt,
            )
            self.jit_plan_binding.require_ready()
            if not isinstance(
                self.jit_artifact_binding, AttentionJitArtifactBinding
            ):
                raise SchemaError(
                    "ascendc_jit runtime requires an Attention JIT artifact binding"
                )
            self.jit_artifact_binding.validate_plan_binding(
                self.jit_plan_binding
            )
            if not isinstance(
                self.jit_module_binding, AttentionJitLoadedModuleBinding
            ):
                raise SchemaError(
                    "ascendc_jit runtime requires an Attention JIT module binding"
                )
            self.jit_module_binding.validate_bindings(
                self.jit_plan_binding,
                self.jit_artifact_binding,
            )
            if not isinstance(
                self.jit_planner_binding, AttentionJitPlannerBinding
            ):
                raise SchemaError(
                    "ascendc_jit runtime requires an Attention JIT planner binding"
                )
            self.jit_planner_binding.validate(self.jit_module_binding)
            if self.jit_planner_binding.factory is not self.factory:
                raise SchemaError(
                    "ascendc_jit runtime factory differs from its JIT planner binding"
                )
            if not isinstance(
                self.jit_executor_binding, AttentionJitExecutorBinding
            ):
                raise SchemaError(
                    "ascendc_jit runtime requires an Attention JIT executor binding"
                )
            self.jit_executor_binding.validate(
                self.jit_module_binding,
                self.callable_binding,
            )
            if self.jit_executor_binding.executor is not self.executor:
                raise SchemaError(
                    "ascendc_jit runtime executor differs from its JIT binding"
                )
        elif (
            self.jit_plan_binding is not None
            or self.jit_artifact_binding is not None
            or self.jit_module_binding is not None
            or self.jit_planner_binding is not None
            or self.jit_executor_binding is not None
        ):
            raise SchemaError(
                "non-JIT Attention runtime cannot publish JIT bindings"
            )
        provider_id = self.selection.provider_id
        operation_id = self.factory.operation_id
        if (
            self.factory.provider_id != provider_id
            or self.run_adapter.provider_id != provider_id
            or self.executor.provider_id != provider_id
        ):
            raise SchemaError("resolved Attention runtime provider identities differ")
        if (
            self.executor.operation_id != operation_id
            or self.callable_binding.operation_id != operation_id
        ):
            raise SchemaError("resolved Attention runtime operation identities differ")
        if self.receipt.plan_fingerprint != self.framework_plan_fingerprint:
            raise SchemaError("resolved Attention runtime does not bind its plan")
        if (
            self.selection.dispatch_receipt_fingerprint != self.receipt.fingerprint
            or self.callable_binding.provider_probe_fingerprint
            != self.selection.provider_probe_fingerprint
        ):
            raise SchemaError("resolved Attention runtime authority chain is stale")


@runtime_checkable
class AttentionOperatorRuntimeResolver(Protocol):
    """Resolve one already validated plan to one complete provider runtime."""

    def resolve(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionResolvedOperatorRuntime:
        """Select without publishing state in the public wrapper."""


@runtime_checkable
class AttentionOperatorRuntimeImplementation(Protocol):
    """One versioned provider operation offered to automatic resolution."""

    provider_id: str
    operation_id: str
    priority: int

    def rejection_reasons(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> Sequence[str]:
        """Pure explanation; must not import a package or initialize a device."""

    def resolve(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionResolvedOperatorRuntime:
        """Build the complete authority chain for the selected implementation."""


@dataclass(frozen=True)
class AttentionOperatorRuntimeImplementationCandidate:
    provider_id: str
    operation_id: str
    priority: int
    accepted: bool
    reasons: Tuple[str, ...]
    schema_version: int = ATTENTION_OPERATOR_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RESOLVER_VERSION:
            raise SchemaError("unsupported Attention runtime candidate version")
        if not str(self.provider_id) or not str(self.operation_id):
            raise SchemaError("runtime candidate identities must be non-empty")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise SchemaError("runtime candidate priority must be an integer")
        reasons = tuple(str(item) for item in self.reasons)
        if any(not item for item in reasons) or len(set(reasons)) != len(reasons):
            raise SchemaError("runtime candidate reasons must be non-empty and unique")
        if self.accepted == bool(reasons):
            raise SchemaError("accepted runtime candidate must have no reasons")
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "priority": self.priority,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AttentionOperatorRuntimeResolutionReport:
    framework_plan_fingerprint: str
    device: str
    candidates: Tuple[AttentionOperatorRuntimeImplementationCandidate, ...]
    schema_version: int = ATTENTION_OPERATOR_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RESOLVER_VERSION:
            raise SchemaError("unsupported Attention runtime resolution report version")
        if len(self.framework_plan_fingerprint) != 64 or any(
            item not in "0123456789abcdef"
            for item in self.framework_plan_fingerprint
        ):
            raise SchemaError("runtime resolution plan fingerprint must be SHA-256")
        if not str(self.device):
            raise SchemaError("runtime resolution device must be non-empty")
        candidates = tuple(self.candidates)
        identities = tuple(
            (item.provider_id, item.operation_id) for item in candidates
        )
        if len(set(identities)) != len(identities):
            raise SchemaError("runtime resolution candidates contain duplicates")
        object.__setattr__(self, "device", str(self.device))
        object.__setattr__(self, "candidates", candidates)

    @property
    def accepted(self) -> Tuple[AttentionOperatorRuntimeImplementationCandidate, ...]:
        return tuple(item for item in self.candidates if item.accepted)

    @property
    def top_priority(self):
        return max((item.priority for item in self.accepted), default=None)

    @property
    def finalists(self) -> Tuple[AttentionOperatorRuntimeImplementationCandidate, ...]:
        top = self.top_priority
        if top is None:
            return ()
        return tuple(item for item in self.accepted if item.priority == top)

    @property
    def selected(self):
        return self.finalists[0] if len(self.finalists) == 1 else None

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "framework_plan_fingerprint": self.framework_plan_fingerprint,
            "device": self.device,
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionOperatorRuntimeResolutionError(RuntimeError):
    def __init__(
        self, message: str, report: AttentionOperatorRuntimeResolutionReport
    ) -> None:
        super().__init__(message)
        self.report = report


class AttentionOperatorRuntimeImplementationRegistry:
    """Deterministic auto resolver over versioned provider implementations."""

    def __init__(
        self, implementations: Sequence[AttentionOperatorRuntimeImplementation] = ()
    ) -> None:
        values = tuple(implementations)
        if any(
            not isinstance(item, AttentionOperatorRuntimeImplementation)
            for item in values
        ):
            raise TypeError(
                "implementations must implement AttentionOperatorRuntimeImplementation"
            )
        identities = tuple((item.provider_id, item.operation_id) for item in values)
        if len(set(identities)) != len(identities):
            raise SchemaError("duplicate Attention runtime implementation identity")
        for item in values:
            if not isinstance(item.priority, int) or isinstance(item.priority, bool):
                raise SchemaError("runtime implementation priority must be an integer")
        self._implementations = tuple(
            sorted(
                values,
                key=lambda item: (-item.priority, item.provider_id, item.operation_id),
            )
        )
        self._implementation_map: Dict[
            Tuple[str, str], AttentionOperatorRuntimeImplementation
        ] = {
            (item.provider_id, item.operation_id): item
            for item in self._implementations
        }

    @property
    def implementation_ids(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(self._implementation_map)

    def explain(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionOperatorRuntimeResolutionReport:
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        candidates = []
        for implementation in self._implementations:
            raw_reasons = implementation.rejection_reasons(plan, str(device))
            try:
                reasons = tuple(str(item) for item in raw_reasons)
            except TypeError as error:
                raise TypeError(
                    "runtime implementation rejection_reasons must be a sequence"
                ) from error
            candidates.append(
                AttentionOperatorRuntimeImplementationCandidate(
                    provider_id=implementation.provider_id,
                    operation_id=implementation.operation_id,
                    priority=implementation.priority,
                    accepted=not reasons,
                    reasons=reasons,
                )
            )
        return AttentionOperatorRuntimeResolutionReport(
            framework_plan_fingerprint=plan.fingerprint,
            device=str(device),
            candidates=tuple(candidates),
        )

    def resolve(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionResolvedOperatorRuntime:
        report = self.explain(plan, device)
        selected = report.selected
        if selected is None:
            if not report.accepted:
                details = "; ".join(
                    "%s/%s: %s"
                    % (
                        item.provider_id,
                        item.operation_id,
                        ", ".join(item.reasons),
                    )
                    for item in report.candidates
                )
                message = (
                    "no Attention runtime implementation accepts the plan (%s)"
                    % (details or "no implementations registered")
                )
            else:
                identities = ", ".join(
                    "%s/%s" % (item.provider_id, item.operation_id)
                    for item in report.finalists
                )
                message = (
                    "ambiguous Attention runtime implementations at priority %d: %s"
                    % (report.top_priority, identities)
                )
            raise AttentionOperatorRuntimeResolutionError(message, report)
        implementation = self._implementation_map[
            (selected.provider_id, selected.operation_id)
        ]
        resolved = implementation.resolve(plan, str(device))
        if not isinstance(resolved, AttentionResolvedOperatorRuntime):
            raise TypeError("runtime implementation returned an invalid runtime")
        if (
            resolved.framework_plan_fingerprint != plan.fingerprint
            or resolved.selection.provider_id != selected.provider_id
            or resolved.factory.operation_id != selected.operation_id
        ):
            raise SchemaError("selected runtime implementation changed its identity")
        return resolved


@dataclass(frozen=True)
class AttentionOperatorRuntimeResolverRegistry:
    """Immutable device-type routing table for package integration resolvers."""

    resolvers: Tuple[Tuple[str, AttentionOperatorRuntimeResolver], ...] = ()
    schema_version: int = ATTENTION_OPERATOR_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RESOLVER_VERSION:
            raise SchemaError("unsupported Attention resolver registry version")
        normalized = []
        for device_type, resolver in self.resolvers:
            device_type = str(device_type)
            if not _DEVICE_TYPE.fullmatch(device_type):
                raise SchemaError("invalid Attention resolver device type")
            if not isinstance(resolver, AttentionOperatorRuntimeResolver):
                raise TypeError(
                    "resolver must implement AttentionOperatorRuntimeResolver"
                )
            normalized.append((device_type, resolver))
        names = tuple(item[0] for item in normalized)
        if len(set(names)) != len(names):
            raise SchemaError("duplicate Attention resolver device type")
        object.__setattr__(
            self, "resolvers", tuple(sorted(normalized, key=lambda item: item[0]))
        )

    def resolve(
        self, plan: AttentionFrameworkPlan, device: str
    ) -> AttentionResolvedOperatorRuntime:
        device_type = str(device).split(":", 1)[0]
        for candidate_type, resolver in self.resolvers:
            if candidate_type == device_type:
                resolved = resolver.resolve(plan, str(device))
                if not isinstance(resolved, AttentionResolvedOperatorRuntime):
                    raise TypeError("Attention resolver returned an invalid runtime")
                if resolved.framework_plan_fingerprint != plan.fingerprint:
                    raise SchemaError("Attention resolver returned a stale runtime")
                return resolved
        raise DispatchError(
            "no Attention operator runtime resolver is installed for device %r"
            % device_type
        )


EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS = (
    AttentionOperatorRuntimeResolverRegistry()
)


class AttentionOperatorBatchRuntime:
    """Transactional provider runtime owned by one public BatchAttention."""

    def __init__(
        self,
        device: str,
        resolver_registry: AttentionOperatorRuntimeResolverRegistry,
        operation_catalog: AttentionOperatorOperationCatalog = None,
    ) -> None:
        if not str(device):
            raise SchemaError("Attention operator device must be non-empty")
        if not isinstance(
            resolver_registry, AttentionOperatorRuntimeResolverRegistry
        ):
            raise TypeError(
                "resolver_registry must be AttentionOperatorRuntimeResolverRegistry"
            )
        if operation_catalog is None:
            operation_catalog = load_packaged_attention_operator_catalog()
        if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
            raise TypeError(
                "operation_catalog must be AttentionOperatorOperationCatalog"
            )
        self.device = str(device)
        self._resolver_registry = resolver_registry
        self._operation_catalog = operation_catalog
        self._framework_session = AttentionFrameworkSession(
            AttentionMode.BATCH_MIXED_PAGED
        )
        self._operator_session = None
        self._executor = None
        self._jit_plan_binding = None
        self._jit_artifact_binding = None
        self._jit_module_binding = None
        self._jit_planner_binding = None
        self._jit_executor_binding = None
        self._last_lowered_call = None

    @property
    def is_planned(self) -> bool:
        return self._operator_session is not None

    @property
    def plan_state(self) -> AttentionFrameworkPlan:
        return self._framework_session.plan_state

    @property
    def operator_session(self) -> AttentionOperatorWrapperSession:
        if self._operator_session is None:
            raise AttentionStateError("Attention operator runtime has not been planned")
        return self._operator_session

    @property
    def last_lowered_call(self) -> AttentionLoweredOperatorCall:
        if self._last_lowered_call is None:
            raise AttentionStateError("Attention operator runtime has not been run")
        return self._last_lowered_call

    @property
    def jit_plan_binding(self):
        if not self.is_planned:
            raise AttentionStateError("Attention operator runtime has not been planned")
        return self._jit_plan_binding

    @property
    def jit_artifact_binding(self):
        if not self.is_planned:
            raise AttentionStateError("Attention operator runtime has not been planned")
        return self._jit_artifact_binding

    @property
    def jit_module_binding(self):
        if not self.is_planned:
            raise AttentionStateError("Attention operator runtime has not been planned")
        return self._jit_module_binding

    @property
    def jit_executor_binding(self):
        if not self.is_planned:
            raise AttentionStateError("Attention operator runtime has not been planned")
        return self._jit_executor_binding

    @property
    def jit_planner_binding(self):
        if not self.is_planned:
            raise AttentionStateError("Attention operator runtime has not been planned")
        return self._jit_planner_binding

    def plan(self, spec: AttentionPlanSpec, metadata: AttentionMetadata) -> None:
        """Resolve and prepare completely, then publish all wrapper state."""

        candidate_plan = self._framework_session.prepare_plan(spec, metadata)
        resolved = self._resolver_registry.resolve(candidate_plan, self.device)
        candidate_operator_session = AttentionOperatorWrapperSession(
            self._operation_catalog
        )
        candidate_operator_session.plan(
            resolved.factory,
            resolved.run_adapter,
            candidate_plan,
            resolved.receipt,
            resolved.selection,
            resolved.callable_binding,
            (
                resolved.jit_plan_binding.fingerprint
                if resolved.jit_plan_binding is not None
                else None
            ),
            (
                resolved.jit_artifact_binding.fingerprint
                if resolved.jit_artifact_binding is not None
                else None
            ),
            (
                resolved.jit_module_binding.fingerprint
                if resolved.jit_module_binding is not None
                else None
            ),
            (
                resolved.jit_planner_binding.fingerprint
                if resolved.jit_planner_binding is not None
                else None
            ),
            (
                resolved.jit_executor_binding.fingerprint
                if resolved.jit_executor_binding is not None
                else None
            ),
        )
        candidate_executor = resolved.executor
        candidate_jit_plan_binding = resolved.jit_plan_binding
        candidate_jit_artifact_binding = resolved.jit_artifact_binding
        candidate_jit_module_binding = resolved.jit_module_binding
        candidate_jit_planner_binding = resolved.jit_planner_binding
        candidate_jit_executor_binding = resolved.jit_executor_binding
        if candidate_jit_plan_binding is not None and (
            candidate_operator_session.active_plan.jit_plan_binding_fingerprint
            != candidate_jit_plan_binding.fingerprint
        ):
            raise SchemaError("active plan did not freeze the JIT binding identity")
        if candidate_jit_artifact_binding is not None and (
            candidate_operator_session.active_plan.jit_artifact_binding_fingerprint
            != candidate_jit_artifact_binding.fingerprint
        ):
            raise SchemaError(
                "active plan did not freeze the JIT artifact identity"
            )
        if candidate_jit_module_binding is not None and (
            candidate_operator_session.active_plan.jit_module_binding_fingerprint
            != candidate_jit_module_binding.fingerprint
        ):
            raise SchemaError("active plan did not freeze the JIT module identity")
        if candidate_jit_planner_binding is not None and (
            candidate_operator_session.active_plan.jit_planner_binding_fingerprint
            != candidate_jit_planner_binding.fingerprint
        ):
            raise SchemaError("active plan did not freeze the JIT planner identity")
        if candidate_jit_planner_binding is not None:
            candidate_jit_planner_binding.validate(candidate_jit_module_binding)
            if candidate_jit_planner_binding.factory is not resolved.factory:
                raise SchemaError(
                    "resolved factory differs from its JIT planner binding"
                )
        if candidate_jit_executor_binding is not None and (
            candidate_operator_session.active_plan.jit_executor_binding_fingerprint
            != candidate_jit_executor_binding.fingerprint
        ):
            raise SchemaError("active plan did not freeze the JIT executor identity")
        if candidate_jit_executor_binding is not None:
            candidate_jit_executor_binding.validate(
                candidate_jit_module_binding,
                resolved.callable_binding,
            )
            if candidate_jit_executor_binding.executor is not candidate_executor:
                raise SchemaError(
                    "resolved executor differs from its JIT executor binding"
                )
        if isinstance(candidate_executor, AttentionRuntimeBindingAwareExecutor):
            candidate_executor = candidate_executor.bind_runtime(
                candidate_operator_session.runtime_binding
            )
            if not isinstance(candidate_executor, AttentionLoweredOperatorExecutor):
                raise TypeError(
                    "runtime-bound Attention executor must implement "
                    "AttentionLoweredOperatorExecutor"
                )
            if (
                candidate_executor.provider_id != resolved.selection.provider_id
                or candidate_executor.operation_id != resolved.factory.operation_id
            ):
                raise SchemaError(
                    "runtime-bound Attention executor changed its identity"
                )
        # commit_prepared_plan cannot fail after the same candidate was prepared;
        # it is deliberately last so resolver/prepare/binding failures preserve
        # the old framework plan and executable runtime as one atomic generation.
        self._framework_session.commit_prepared_plan(candidate_plan)
        self._operator_session = candidate_operator_session
        self._executor = candidate_executor
        self._jit_plan_binding = candidate_jit_plan_binding
        self._jit_artifact_binding = candidate_jit_artifact_binding
        self._jit_module_binding = candidate_jit_module_binding
        self._jit_planner_binding = candidate_jit_planner_binding
        self._jit_executor_binding = candidate_jit_executor_binding
        self._last_lowered_call = None

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
        session = self.operator_session
        if self._executor is None:  # defensive; plan publication is atomic
            raise AttentionStateError("Attention operator executor is not initialized")
        if self._jit_plan_binding is not None:
            if (
                session.active_plan.jit_plan_binding_fingerprint
                != self._jit_plan_binding.fingerprint
            ):
                raise AttentionStateError("Attention JIT active-plan identity is stale")
            self._jit_plan_binding.validate_plan(
                session.active_plan.framework_plan,
                session.active_plan.dispatch_receipt,
            )
            self._jit_plan_binding.require_ready()
            if self._jit_artifact_binding is None:
                raise AttentionStateError(
                    "Attention JIT artifact binding is not initialized"
                )
            if (
                session.active_plan.jit_artifact_binding_fingerprint
                != self._jit_artifact_binding.fingerprint
            ):
                raise AttentionStateError(
                    "Attention JIT artifact active-plan identity is stale"
                )
            self._jit_artifact_binding.validate_plan_binding(
                self._jit_plan_binding
            )
            if self._jit_module_binding is None:
                raise AttentionStateError(
                    "Attention JIT module binding is not initialized"
                )
            if (
                session.active_plan.jit_module_binding_fingerprint
                != self._jit_module_binding.fingerprint
            ):
                raise AttentionStateError(
                    "Attention JIT module active-plan identity is stale"
                )
            self._jit_module_binding.validate_bindings(
                self._jit_plan_binding,
                self._jit_artifact_binding,
            )
            if self._jit_planner_binding is None:
                raise AttentionStateError(
                    "Attention JIT planner binding is not initialized"
                )
            if (
                session.active_plan.jit_planner_binding_fingerprint
                != self._jit_planner_binding.fingerprint
            ):
                raise AttentionStateError(
                    "Attention JIT planner active-plan identity is stale"
                )
            self._jit_planner_binding.validate(self._jit_module_binding)
            if self._jit_executor_binding is None:
                raise AttentionStateError(
                    "Attention JIT executor binding is not initialized"
                )
            if (
                session.active_plan.jit_executor_binding_fingerprint
                != self._jit_executor_binding.fingerprint
            ):
                raise AttentionStateError(
                    "Attention JIT executor active-plan identity is stale"
                )
            self._jit_executor_binding.validate(
                self._jit_module_binding,
                session.callable_binding,
            )
        lowered = session.run(
            q,
            kv_cache,
            out=out,
            lse=lse,
            k_scale=k_scale,
            v_scale=v_scale,
            logits_soft_cap=logits_soft_cap,
            profiler_buffer=profiler_buffer,
            kv_cache_sf=kv_cache_sf,
        )
        result = self._executor.execute(lowered)
        self._last_lowered_call = lowered
        return result


__all__ = [
    "ATTENTION_OPERATOR_RESOLVER_VERSION",
    "EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS",
    "AttentionLoweredOperatorExecutor",
    "AttentionOperatorBatchRuntime",
    "AttentionOperatorRuntimeImplementation",
    "AttentionOperatorRuntimeImplementationCandidate",
    "AttentionOperatorRuntimeImplementationRegistry",
    "AttentionOperatorRuntimeResolutionError",
    "AttentionOperatorRuntimeResolutionReport",
    "AttentionOperatorRuntimeResolver",
    "AttentionOperatorRuntimeResolverRegistry",
    "AttentionResolvedOperatorRuntime",
]
