"""Validated lowering boundary for provider-owned Attention ``run`` calls.

The framework deliberately stops at a call description in this module. A
future package adapter may execute that description only after its real
operator integration is authorized and tested.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .dispatch import AttentionDispatchReceipt
from .operator_plan import (
    AttentionOperatorActivePlan,
    AttentionOperatorPlanFactory,
    AttentionOperatorPlanSession,
)
from .operation_catalog import (
    AttentionOperatorOperationBinding,
    AttentionOperatorOperationCatalog,
    AttentionOperatorOperationSpec,
    bind_attention_operator_operation,
)
from .operator_callable import (
    AttentionOperatorCallableBinding,
    AttentionOperatorRuntimeBinding,
    bind_attention_operator_runtime,
)
from .operator_provider import AttentionOperatorProviderSelection
from .planner import AttentionFrameworkPlan, AttentionStateError


ATTENTION_OPERATOR_RUN_VERSION = 2

ATTENTION_OPERATOR_RUN_REQUEST_FIELDS = (
    "query",
    "kv_cache",
    "return_lse",
    "out",
    "lse",
    "k_scale",
    "v_scale",
    "logits_soft_cap",
    "profiler_buffer",
    "kv_cache_sf",
)

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _argument_pairs(name: str, values) -> Tuple[Tuple[str, Any], ...]:
    try:
        result = tuple((str(argument_name), value) for argument_name, value in values)
    except (TypeError, ValueError) as error:
        raise SchemaError("%s must contain (name, value) pairs" % name) from error
    names = tuple(argument_name for argument_name, _ in result)
    if any(not _ARGUMENT_NAME.fullmatch(argument_name) for argument_name in names):
        raise SchemaError("%s contains an invalid argument name" % name)
    if len(set(names)) != len(names):
        raise SchemaError("%s contains duplicate argument names" % name)
    return result


@dataclass(frozen=True)
class AttentionOperatorRunRequest:
    """Internal mirror of FlashInfer holistic ``BatchAttention.run`` inputs."""

    active_plan_fingerprint: str
    framework_plan_fingerprint: str
    framework_plan_generation: int
    query: Any
    kv_cache: Any
    return_lse: bool = True
    out: Any = None
    lse: Any = None
    k_scale: Any = None
    v_scale: Any = None
    logits_soft_cap: float = 0.0
    profiler_buffer: Any = None
    kv_cache_sf: Any = None
    schema_version: int = ATTENTION_OPERATOR_RUN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUN_VERSION:
            raise SchemaError("unsupported Attention operator run request version")
        _require_hash("active_plan_fingerprint", self.active_plan_fingerprint)
        _require_hash("framework_plan_fingerprint", self.framework_plan_fingerprint)
        if (
            not isinstance(self.framework_plan_generation, int)
            or isinstance(self.framework_plan_generation, bool)
            or self.framework_plan_generation < 1
        ):
            raise SchemaError("operator run plan generation must be positive")
        if self.query is None or self.kv_cache is None:
            raise SchemaError("operator run query and kv_cache must be provided")
        if not isinstance(self.return_lse, bool):
            raise SchemaError("return_lse must be boolean")
        if isinstance(self.logits_soft_cap, bool):
            raise SchemaError("logits_soft_cap must be a finite non-negative scalar")
        try:
            logits_soft_cap = float(self.logits_soft_cap)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "logits_soft_cap must be a finite non-negative scalar"
            ) from error
        if not math.isfinite(logits_soft_cap) or logits_soft_cap < 0.0:
            raise SchemaError("logits_soft_cap must be a finite non-negative scalar")
        object.__setattr__(self, "logits_soft_cap", logits_soft_cap)

    @classmethod
    def from_active_plan(
        cls,
        active_plan: AttentionOperatorActivePlan,
        query: Any,
        kv_cache: Any,
        *,
        return_lse: bool = True,
        out: Any = None,
        lse: Any = None,
        k_scale: Any = None,
        v_scale: Any = None,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Any = None,
        kv_cache_sf: Any = None,
    ) -> "AttentionOperatorRunRequest":
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        return cls(
            active_plan_fingerprint=active_plan.fingerprint,
            framework_plan_fingerprint=active_plan.framework_plan.fingerprint,
            framework_plan_generation=active_plan.framework_plan.generation,
            query=query,
            kv_cache=kv_cache,
            return_lse=return_lse,
            out=out,
            lse=lse,
            k_scale=k_scale,
            v_scale=v_scale,
            logits_soft_cap=logits_soft_cap,
            profiler_buffer=profiler_buffer,
            kv_cache_sf=kv_cache_sf,
        )

    @property
    def consumed_fields(self) -> Tuple[str, ...]:
        """Fields an adapter must explicitly consume or validate during lowering."""

        optional_values = {
            "out": self.out,
            "lse": self.lse,
            "k_scale": self.k_scale,
            "v_scale": self.v_scale,
            "profiler_buffer": self.profiler_buffer,
            "kv_cache_sf": self.kv_cache_sf,
        }
        return tuple(
            field
            for field in ATTENTION_OPERATOR_RUN_REQUEST_FIELDS
            if field in ("query", "kv_cache", "return_lse", "logits_soft_cap")
            or optional_values.get(field) is not None
        )


@dataclass(frozen=True)
class AttentionLoweredOperatorCall:
    """Inspectable, non-executing description of one external package call."""

    provider_id: str
    operation_id: str
    active_plan_fingerprint: str
    positional_arguments: Tuple[Tuple[str, Any], ...]
    keyword_arguments: Tuple[Tuple[str, Any], ...] = ()
    return_names: Tuple[str, ...] = ("output",)
    mutable_argument_names: Tuple[str, ...] = ()
    consumed_request_fields: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_OPERATOR_RUN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUN_VERSION:
            raise SchemaError("unsupported lowered Attention operator call version")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid lowered Attention provider_id")
        if not str(self.operation_id) or any(
            character.isspace() for character in str(self.operation_id)
        ):
            raise SchemaError("Attention operation_id must be non-empty without spaces")
        _require_hash("active_plan_fingerprint", self.active_plan_fingerprint)
        positional = _argument_pairs(
            "positional_arguments", self.positional_arguments
        )
        keyword = _argument_pairs("keyword_arguments", self.keyword_arguments)
        positional_names = tuple(name for name, _ in positional)
        keyword_names = tuple(name for name, _ in keyword)
        if set(positional_names).intersection(keyword_names):
            raise SchemaError("lowered operator argument names must be unique")
        returns = tuple(str(item) for item in self.return_names)
        if not returns or any(not _ARGUMENT_NAME.fullmatch(item) for item in returns):
            raise SchemaError("lowered operator return_names are invalid")
        if len(set(returns)) != len(returns):
            raise SchemaError("lowered operator return_names must be unique")
        mutable = tuple(str(item) for item in self.mutable_argument_names)
        all_arguments = set(positional_names).union(keyword_names)
        if len(set(mutable)) != len(mutable) or not set(mutable).issubset(
            all_arguments
        ):
            raise SchemaError("mutable arguments must name unique call arguments")
        consumed = tuple(str(item) for item in self.consumed_request_fields)
        if len(set(consumed)) != len(consumed) or any(
            item not in ATTENTION_OPERATOR_RUN_REQUEST_FIELDS for item in consumed
        ):
            raise SchemaError("consumed request fields are invalid")
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(self, "positional_arguments", positional)
        object.__setattr__(self, "keyword_arguments", keyword)
        object.__setattr__(self, "return_names", returns)
        object.__setattr__(self, "mutable_argument_names", mutable)
        object.__setattr__(self, "consumed_request_fields", consumed)


@runtime_checkable
class AttentionOperatorRunAdapter(Protocol):
    """Provider extension that lowers but does not execute a planned run."""

    provider_id: str

    def lower(
        self,
        active_plan: AttentionOperatorActivePlan,
        request: AttentionOperatorRunRequest,
    ) -> AttentionLoweredOperatorCall:
        """Lower a run using only the already selected active implementation."""


@runtime_checkable
class AttentionOperatorRunAdapterFactory(Protocol):
    """Late-bind a run adapter to the device selected for one wrapper."""

    provider_id: str
    operation_id: str

    def build(
        self, base_adapter: AttentionOperatorRunAdapter, device: str
    ) -> AttentionOperatorRunAdapter:
        """Decorate ``base_adapter`` without importing or executing an operator."""


def lower_attention_operator_run(
    adapter: AttentionOperatorRunAdapter,
    active_plan: AttentionOperatorActivePlan,
    request: AttentionOperatorRunRequest,
) -> AttentionLoweredOperatorCall:
    """Validate plan reuse and return a provider call description without execution."""

    if not isinstance(adapter, AttentionOperatorRunAdapter):
        raise TypeError("adapter must implement AttentionOperatorRunAdapter")
    if not isinstance(active_plan, AttentionOperatorActivePlan):
        raise TypeError("active_plan must be AttentionOperatorActivePlan")
    if not isinstance(request, AttentionOperatorRunRequest):
        raise TypeError("request must be AttentionOperatorRunRequest")
    if (
        request.active_plan_fingerprint != active_plan.fingerprint
        or request.framework_plan_fingerprint
        != active_plan.framework_plan.fingerprint
        or request.framework_plan_generation != active_plan.framework_plan.generation
    ):
        raise SchemaError("operator run request does not bind the active plan")
    provider_id = active_plan.provider_selection.provider_id
    if adapter.provider_id != provider_id:
        raise SchemaError("operator run adapter does not match the active provider")
    lowered = adapter.lower(active_plan, request)
    if not isinstance(lowered, AttentionLoweredOperatorCall):
        raise TypeError("operator run adapter returned an invalid call description")
    if (
        lowered.provider_id != provider_id
        or lowered.active_plan_fingerprint != active_plan.fingerprint
    ):
        raise SchemaError("lowered operator call does not bind the active plan")
    if lowered.operation_id != active_plan.prepared_plan.implementation_id:
        raise SchemaError("lowered operator call changed the planned implementation")
    if lowered.consumed_request_fields != request.consumed_fields:
        raise SchemaError("lowered operator call did not consume every run field")
    return lowered


def validate_attention_lowered_operator_call(
    operation: AttentionOperatorOperationSpec,
    binding: AttentionOperatorOperationBinding,
    lowered: AttentionLoweredOperatorCall,
) -> AttentionLoweredOperatorCall:
    """Validate one lowered call against its exact versioned operation spec."""

    if not isinstance(operation, AttentionOperatorOperationSpec):
        raise TypeError("operation must be AttentionOperatorOperationSpec")
    if not isinstance(binding, AttentionOperatorOperationBinding):
        raise TypeError("binding must be AttentionOperatorOperationBinding")
    if not isinstance(lowered, AttentionLoweredOperatorCall):
        raise TypeError("lowered must be AttentionLoweredOperatorCall")
    if (
        binding.operation_id != operation.operation_id
        or binding.operation_fingerprint != operation.fingerprint
        or binding.provider_id != operation.provider_id
        or binding.api_version != operation.api_version
    ):
        raise SchemaError("operation binding does not match the catalog signature")
    if (
        lowered.operation_id != operation.operation_id
        or lowered.provider_id != operation.provider_id
        or lowered.active_plan_fingerprint != binding.active_plan_fingerprint
    ):
        raise SchemaError("lowered call does not match the catalog binding")
    positional_names = tuple(name for name, _ in lowered.positional_arguments)
    if positional_names != operation.positional_arguments:
        raise SchemaError("lowered call positional arguments do not match the signature")
    keyword_names = tuple(name for name, _ in lowered.keyword_arguments)
    unknown_keywords = set(keyword_names).difference(operation.keyword_arguments)
    if unknown_keywords:
        raise SchemaError(
            "lowered call contains unknown keyword argument %r"
            % sorted(unknown_keywords)[0]
        )
    unknown_returns = set(lowered.return_names).difference(operation.return_names)
    if unknown_returns:
        raise SchemaError(
            "lowered call contains unknown return name %r"
            % sorted(unknown_returns)[0]
        )
    provided_arguments = set(positional_names).union(keyword_names)
    expected_mutable = tuple(
        name for name in operation.mutable_arguments if name in provided_arguments
    )
    if lowered.mutable_argument_names != expected_mutable:
        raise SchemaError("lowered call mutable arguments do not match the signature")
    if operation.lse_control_argument in keyword_names:
        control_value = dict(lowered.keyword_arguments)[operation.lse_control_argument]
        if bool(control_value) and "softmax_lse" not in lowered.return_names:
            raise SchemaError("lowered call enables LSE but omits the LSE return")
    return lowered


class AttentionOperatorWrapperSession:
    """Wrapper-owned plan/run lifecycle with no public plan or adapter handle."""

    def __init__(
        self, operation_catalog: AttentionOperatorOperationCatalog
    ) -> None:
        if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
            raise TypeError(
                "operation_catalog must be AttentionOperatorOperationCatalog"
            )
        self._operation_catalog = operation_catalog
        self._plan_session = None
        self._run_adapter = None
        self._operation_binding = None
        self._callable_binding = None
        self._runtime_binding = None

    @property
    def is_planned(self) -> bool:
        return self._plan_session is not None

    @property
    def active_plan(self) -> AttentionOperatorActivePlan:
        if self._plan_session is None:
            raise AttentionStateError("Attention operator wrapper has not been planned")
        return self._plan_session.active_plan

    @property
    def operation_binding(self) -> AttentionOperatorOperationBinding:
        if self._operation_binding is None:
            raise AttentionStateError("Attention operator wrapper has not been planned")
        return self._operation_binding

    @property
    def callable_binding(self) -> AttentionOperatorCallableBinding:
        if self._callable_binding is None:
            raise AttentionStateError("Attention operator wrapper has not been planned")
        return self._callable_binding

    @property
    def runtime_binding(self) -> AttentionOperatorRuntimeBinding:
        if self._runtime_binding is None:
            raise AttentionStateError("Attention operator wrapper has not been planned")
        return self._runtime_binding

    def plan(
        self,
        factory: AttentionOperatorPlanFactory,
        run_adapter: AttentionOperatorRunAdapter,
        framework_plan: AttentionFrameworkPlan,
        receipt: AttentionDispatchReceipt,
        selection: AttentionOperatorProviderSelection,
        callable_binding: AttentionOperatorCallableBinding,
        jit_plan_binding_fingerprint: Optional[str] = None,
        jit_artifact_binding_fingerprint: Optional[str] = None,
        jit_module_binding_fingerprint: Optional[str] = None,
        jit_planner_binding_fingerprint: Optional[str] = None,
        jit_executor_binding_fingerprint: Optional[str] = None,
    ) -> None:
        """Prepare a complete runtime candidate, then atomically publish it."""

        if not isinstance(run_adapter, AttentionOperatorRunAdapter):
            raise TypeError("run_adapter must implement AttentionOperatorRunAdapter")
        if run_adapter.provider_id != selection.provider_id:
            raise SchemaError("operator run adapter does not match selected provider")
        if not isinstance(callable_binding, AttentionOperatorCallableBinding):
            raise TypeError(
                "callable_binding must be AttentionOperatorCallableBinding"
            )
        operation = self._operation_catalog.get(factory.operation_id)
        if factory.operation_id != callable_binding.operation_id:
            raise SchemaError("operator plan factory does not match callable binding")
        if (
            operation.provider_id != selection.provider_id
            or operation.fingerprint != callable_binding.operation_fingerprint
            or callable_binding.provider_probe_fingerprint
            != selection.provider_probe_fingerprint
            or framework_plan.spec.mode not in operation.candidate_modes
        ):
            raise SchemaError("callable binding does not authorize the planned operation")
        candidate_session = AttentionOperatorPlanSession()
        candidate_session.plan(
            factory,
            framework_plan,
            receipt,
            selection,
            jit_plan_binding_fingerprint,
            jit_artifact_binding_fingerprint,
            jit_module_binding_fingerprint,
            jit_planner_binding_fingerprint,
            jit_executor_binding_fingerprint,
        )
        candidate_binding = bind_attention_operator_operation(
            self._operation_catalog, candidate_session.active_plan
        )
        candidate_runtime_binding = bind_attention_operator_runtime(
            candidate_session.active_plan,
            candidate_binding,
            callable_binding,
        )
        self._plan_session = candidate_session
        self._run_adapter = run_adapter
        self._operation_binding = candidate_binding
        self._callable_binding = callable_binding
        self._runtime_binding = candidate_runtime_binding

    def run(
        self,
        q,
        kv_cache,
        return_lse=True,
        out=None,
        lse=None,
        k_scale=None,
        v_scale=None,
        logits_soft_cap=0.0,
        profiler_buffer=None,
        kv_cache_sf=None,
    ) -> AttentionLoweredOperatorCall:
        """Lower the public FlashInfer run surface through the active adapter."""

        active_plan = self.active_plan
        if self._run_adapter is None:  # defensive; active publication is atomic
            raise AttentionStateError("Attention operator run adapter is not initialized")
        request = AttentionOperatorRunRequest.from_active_plan(
            active_plan,
            q,
            kv_cache,
            return_lse=return_lse,
            out=out,
            lse=lse,
            k_scale=k_scale,
            v_scale=v_scale,
            logits_soft_cap=logits_soft_cap,
            profiler_buffer=profiler_buffer,
            kv_cache_sf=kv_cache_sf,
        )
        lowered = lower_attention_operator_run(
            self._run_adapter, active_plan, request
        )
        operation = self._operation_catalog.get(self.operation_binding.operation_id)
        if self.runtime_binding.active_plan_fingerprint != active_plan.fingerprint:
            raise AttentionStateError("Attention operator runtime binding is stale")
        return validate_attention_lowered_operator_call(
            operation, self.operation_binding, lowered
        )


__all__ = [
    "ATTENTION_OPERATOR_RUN_REQUEST_FIELDS",
    "ATTENTION_OPERATOR_RUN_VERSION",
    "AttentionLoweredOperatorCall",
    "AttentionOperatorRunAdapter",
    "AttentionOperatorRunAdapterFactory",
    "AttentionOperatorRunRequest",
    "AttentionOperatorWrapperSession",
    "lower_attention_operator_run",
    "validate_attention_lowered_operator_call",
]
