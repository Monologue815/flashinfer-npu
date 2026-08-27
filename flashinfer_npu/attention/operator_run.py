"""Validated lowering boundary for provider-owned Attention ``run`` calls.

The framework deliberately stops at a call description in this module. A
future package adapter may execute that description only after its real
operator integration is authorized and tested.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable

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
from .schema import (
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    SingleAttentionMetadata,
)
from .tensor_contract import (
    AttentionTensorAccessPolicy,
    KVCacheView,
    TensorView,
)


ATTENTION_OPERATOR_RUN_VERSION = 9

ATTENTION_OPERATOR_RUN_REQUEST_FIELDS = (
    "query",
    "kv_cache",
    "return_lse",
    "out",
    "lse",
    "q_scale",
    "k_scale",
    "v_scale",
    "q_head_scale",
    "k_head_scale",
    "v_head_scale",
    "o_scale",
    "logits_soft_cap",
    "profiler_buffer",
    "kv_cache_sf",
)

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VIEW_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


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
    q_scale: Any = None
    k_scale: Any = None
    v_scale: Any = None
    q_head_scale: Any = None
    k_head_scale: Any = None
    v_head_scale: Any = None
    o_scale: Any = None
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
        q_scale: Any = None,
        k_scale: Any = None,
        v_scale: Any = None,
        q_head_scale: Any = None,
        k_head_scale: Any = None,
        v_head_scale: Any = None,
        o_scale: Any = None,
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
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            q_head_scale=q_head_scale,
            k_head_scale=k_head_scale,
            v_head_scale=v_head_scale,
            o_scale=o_scale,
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
            "q_scale": self.q_scale,
            "k_scale": self.k_scale,
            "v_scale": self.v_scale,
            "q_head_scale": self.q_head_scale,
            "k_head_scale": self.k_head_scale,
            "v_head_scale": self.v_head_scale,
            "o_scale": self.o_scale,
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
    validated_input_views: Tuple[Tuple[str, TensorView], ...] = ()
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
        try:
            validated_inputs = tuple(
                (str(name), view) for name, view in self.validated_input_views
            )
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "validated input views must contain (name, TensorView) pairs"
            ) from error
        validated_names = tuple(name for name, _ in validated_inputs)
        if (
            len(set(validated_names)) != len(validated_names)
            or any(not _VIEW_NAME.fullmatch(name) for name in validated_names)
            or any(not isinstance(view, TensorView) for _, view in validated_inputs)
        ):
            raise SchemaError("validated input views are invalid")
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
        object.__setattr__(self, "validated_input_views", validated_inputs)
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


@runtime_checkable
class AttentionOperatorTensorMetadataInspector(Protocol):
    """Read opaque provider tensors without importing or touching device data."""

    def to_view(
        self, tensor: Any, *, name: str, writable: bool = False
    ) -> TensorView:
        """Return a zero-copy metadata view for ``tensor``."""


def inspect_attention_operator_dense_kv_input(
    plan: AttentionFrameworkPlan,
    value: Any,
    inspector: AttentionOperatorTensorMetadataInspector,
    expected_device: str,
) -> KVCacheView:
    """Close one unquantized public KV input against the active plan.

    Paged inputs may use FlashInfer's packed tensor or separate ``(K, V)``
    representation. Ragged and single inputs use separate tensors. The
    returned view contains metadata only; the original provider tensors are
    never copied or replaced.
    """

    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
        raise TypeError(
            "inspector must implement AttentionOperatorTensorMetadataInspector"
        )
    if not str(expected_device):
        raise SchemaError("expected provider tensor device must be non-empty")
    if plan.spec.kv_quant_spec is not None:
        raise SchemaError("dense KV metadata validation requires an unquantized plan")

    if isinstance(value, (tuple, list)):
        if len(value) != 2 or value[0] is None or value[1] is None:
            raise SchemaError(
                "separate provider KV input requires exactly (key, value)"
            )
        tensors = tuple(value)
        names = ("kv.key", "kv.value")
        packed = False
    else:
        if value is None:
            raise SchemaError("provider KV input must be provided")
        tensors = (value,)
        names = ("kv.packed",)
        packed = True

    views = []
    for tensor, name in zip(tensors, names):
        view = inspector.to_view(tensor, name=name, writable=False)
        if not isinstance(view, TensorView):
            raise TypeError("tensor metadata inspector must return TensorView")
        views.append(view)

    metadata = plan.metadata
    common = {
        "num_kv_heads": plan.spec.num_kv_heads,
        "head_dim_qk": plan.spec.head_dim_qk,
        "head_dim_vo": int(plan.spec.head_dim_vo),
        "dtype": str(plan.spec.kv_dtype),
        "layout": plan.spec.kv_layout,
        "device": str(expected_device),
    }
    if isinstance(
        metadata,
        (PagedKVMetadata, PagedPrefillMetadata, MixedPagedKVMetadata),
    ):
        paged = (
            metadata.paged_kv
            if isinstance(metadata, PagedPrefillMetadata)
            else metadata
        )
        if not views[0].shape:
            raise SchemaError("provider KV input must expose a page dimension")
        num_pages = views[0].shape[0]
        if num_pages <= paged.max_page_index:
            raise SchemaError(
                "provider KV input does not contain every referenced page"
            )
        cache_spec = PagedKVCacheSpec(
            num_pages=num_pages,
            page_size=paged.page_size,
            structure="packed" if packed else "separate",
            **common,
        )
    else:
        if packed:
            raise SchemaError(
                "ragged or single provider KV input requires separate (key, value)"
            )
        if isinstance(metadata, RaggedKVMetadata):
            total_kv_tokens = metadata.total_kv_tokens
        elif isinstance(metadata, SingleAttentionMetadata):
            total_kv_tokens = metadata.kv_len
        else:  # guarded by the framework plan schema; retained for future modes
            raise SchemaError("unsupported metadata for provider KV validation")
        cache_spec = RaggedKVCacheSpec(
            total_kv_tokens=total_kv_tokens,
            **common,
        )

    if packed:
        return KVCacheView(
            cache_spec,
            views[0],
            views[0],
            packed=True,
        )
    return KVCacheView(cache_spec, views[0], views[1])


class AttentionOperatorRunTensorValidationAdapter:
    """Close caller-owned run tensors against the immutable framework plan."""

    def __init__(
        self,
        base_adapter: AttentionOperatorRunAdapter,
        inspector: AttentionOperatorTensorMetadataInspector,
        expected_device: str,
        access_policy: AttentionTensorAccessPolicy,
    ) -> None:
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
            raise TypeError(
                "inspector must implement AttentionOperatorTensorMetadataInspector"
            )
        if not str(expected_device):
            raise SchemaError("run tensor validation device must be non-empty")
        if not isinstance(access_policy, AttentionTensorAccessPolicy):
            raise TypeError("access_policy must be AttentionTensorAccessPolicy")
        self.provider_id = base_adapter.provider_id
        self._base_adapter = base_adapter
        self._inspector = inspector
        self._expected_device = str(expected_device)
        self._access_policy = access_policy

    def lower(
        self,
        active_plan: AttentionOperatorActivePlan,
        request: AttentionOperatorRunRequest,
    ) -> AttentionLoweredOperatorCall:
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        if not isinstance(request, AttentionOperatorRunRequest):
            raise TypeError("request must be AttentionOperatorRunRequest")
        query_view = self._inspector.to_view(
            request.query, name="query", writable=False
        )
        if not isinstance(query_view, TensorView):
            raise TypeError("tensor metadata inspector must return TensorView")
        plan = active_plan.framework_plan
        if query_view.shape != plan.expected_query_shape:
            raise SchemaError(
                "query shape must be %r, got %r"
                % (plan.expected_query_shape, query_view.shape)
            )
        if query_view.dtype != plan.spec.q_dtype:
            raise SchemaError(
                "query dtype must be %s, got %s"
                % (plan.spec.q_dtype, query_view.dtype)
            )
        if query_view.device != self._expected_device:
            raise SchemaError("query device must match the planned provider device")
        query_view.require_alignment(
            self._access_policy.required_alignment, "query"
        )
        if (
            self._access_policy.require_contiguous_q
            and not query_view.is_contiguous
        ):
            raise SchemaError("query must be contiguous for provider lowering")
        input_views = [("query", query_view)]
        if plan.spec.kv_quant_spec is None:
            kv_view = inspect_attention_operator_dense_kv_input(
                plan,
                request.kv_cache,
                self._inspector,
                self._expected_device,
            )
            for name, view in kv_view.named_component_views:
                view.require_alignment(
                    self._access_policy.required_alignment, name
                )
                if (
                    self._access_policy.require_contiguous_kv
                    and not view.is_contiguous
                ):
                    raise SchemaError(
                        "%s must be contiguous for provider lowering" % name
                    )
                input_views.append((name, view))
        output_views = []
        if request.out is not None:
            out_view = self._inspector.to_view(
                request.out, name="out", writable=True
            )
            if not isinstance(out_view, TensorView):
                raise TypeError("tensor metadata inspector must return TensorView")
            if out_view.shape != plan.expected_output_shape:
                raise SchemaError(
                    "out shape must be %r, got %r"
                    % (plan.expected_output_shape, out_view.shape)
                )
            if out_view.dtype != plan.spec.o_dtype:
                raise SchemaError(
                    "out dtype must be %s, got %s"
                    % (plan.spec.o_dtype, out_view.dtype)
                )
            if out_view.device != self._expected_device:
                raise SchemaError(
                    "out device must match the planned provider device"
                )
            if not out_view.writable:
                raise SchemaError("out must be writable")
            out_view.require_alignment(
                self._access_policy.required_alignment, "out"
            )
            if (
                self._access_policy.require_contiguous_output
                and not out_view.is_contiguous
            ):
                raise SchemaError("out must be contiguous for provider lowering")
            output_views.append(("out", out_view))
        if request.lse is not None:
            if not request.return_lse:
                raise SchemaError("lse buffer requires return_lse")
            lse_view = self._inspector.to_view(
                request.lse, name="lse", writable=True
            )
            if not isinstance(lse_view, TensorView):
                raise TypeError("tensor metadata inspector must return TensorView")
            if lse_view.shape != plan.expected_lse_shape:
                raise SchemaError(
                    "lse shape must be %r, got %r"
                    % (plan.expected_lse_shape, lse_view.shape)
                )
            if lse_view.dtype != "float32":
                raise SchemaError(
                    "lse dtype must be float32, got %s" % lse_view.dtype
                )
            if lse_view.device != self._expected_device:
                raise SchemaError(
                    "lse device must match the planned provider device"
                )
            if not lse_view.writable:
                raise SchemaError("lse must be writable")
            lse_view.require_alignment(
                self._access_policy.required_alignment, "lse"
            )
            output_views.append(("lse", lse_view))
        if not self._access_policy.permit_output_input_alias:
            for name, output_view in output_views:
                for input_name, input_view in input_views:
                    if output_view.overlaps(input_view):
                        raise SchemaError(
                            "%s cannot alias %s" % (name, input_name)
                        )
        if len(output_views) == 2 and output_views[0][1].overlaps(
            output_views[1][1]
        ):
            raise SchemaError("out and lse cannot alias")
        lowered = self._base_adapter.lower(active_plan, request)
        if not isinstance(lowered, AttentionLoweredOperatorCall):
            raise TypeError("base run adapter returned an invalid call description")
        delegated_inputs = lowered.validated_input_views
        if plan.spec.kv_quant_spec is None and delegated_inputs:
            raise SchemaError(
                "unquantized base adapter cannot supply validated input views"
            )
        combined_inputs = tuple(input_views) + delegated_inputs
        combined_names = tuple(name for name, _ in combined_inputs)
        if len(set(combined_names)) != len(combined_names):
            raise SchemaError("validated input view names overlap")
        return replace(lowered, validated_input_views=combined_inputs)


class AttentionOperatorRunTensorValidationAdapterFactory:
    """Bind caller-owned run tensor validation to one provider and device."""

    def __init__(
        self,
        provider_id: str,
        operation_id: str,
        inspector: AttentionOperatorTensorMetadataInspector,
        access_policy: AttentionTensorAccessPolicy,
    ) -> None:
        if not str(provider_id) or not str(operation_id):
            raise SchemaError("run tensor factory identities must be non-empty")
        if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
            raise TypeError(
                "inspector must implement AttentionOperatorTensorMetadataInspector"
            )
        if not isinstance(access_policy, AttentionTensorAccessPolicy):
            raise TypeError("access_policy must be AttentionTensorAccessPolicy")
        self.provider_id = str(provider_id)
        self.operation_id = str(operation_id)
        self._inspector = inspector
        self._access_policy = access_policy

    def build(
        self, base_adapter: AttentionOperatorRunAdapter, device: str
    ) -> AttentionOperatorRunAdapter:
        if base_adapter.provider_id != self.provider_id:
            raise SchemaError("run tensor validation adapter provider differs")
        return AttentionOperatorRunTensorValidationAdapter(
            base_adapter,
            self._inspector,
            str(device),
            self._access_policy,
        )


# Compatibility aliases for integrations built against the query-only name.
AttentionOperatorQueryValidationRunAdapter = (
    AttentionOperatorRunTensorValidationAdapter
)
AttentionOperatorQueryValidationRunAdapterFactory = (
    AttentionOperatorRunTensorValidationAdapterFactory
)


class AttentionOperatorCallerBufferRunAdapter:
    """Inject caller-owned output buffers using exact catalog arguments."""

    def __init__(
        self,
        base_adapter: AttentionOperatorRunAdapter,
        operation: AttentionOperatorOperationSpec,
    ) -> None:
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if base_adapter.provider_id != operation.provider_id:
            raise SchemaError("caller buffer adapter provider differs")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._base_adapter = base_adapter
        self._operation = operation

    def lower(
        self,
        active_plan: AttentionOperatorActivePlan,
        request: AttentionOperatorRunRequest,
    ) -> AttentionLoweredOperatorCall:
        lowered = self._base_adapter.lower(active_plan, request)
        injected = tuple(
            (argument_name, value)
            for argument_name, value in (
                (self._operation.output_buffer_argument, request.out),
                (self._operation.lse_buffer_argument, request.lse),
            )
            if argument_name is not None and value is not None
        )
        existing_names = {
            name
            for name, _ in (
                lowered.positional_arguments + lowered.keyword_arguments
            )
        }
        collision = existing_names.intersection(name for name, _ in injected)
        if collision:
            raise SchemaError(
                "caller buffer argument collides with provider lowering: %s"
                % sorted(collision)[0]
            )
        keyword_arguments = lowered.keyword_arguments + injected
        provided_names = {
            name
            for name, _ in lowered.positional_arguments + keyword_arguments
        }
        mutable_arguments = tuple(
            name
            for name in self._operation.mutable_arguments
            if name in provided_names
        )
        return replace(
            lowered,
            keyword_arguments=keyword_arguments,
            mutable_argument_names=mutable_arguments,
        )


class AttentionOperatorCallerBufferRunAdapterFactory:
    """Bind exact caller-owned output argument names from an operation spec."""

    def __init__(self, operation: AttentionOperatorOperationSpec) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation

    def build(
        self, base_adapter: AttentionOperatorRunAdapter, device: str
    ) -> AttentionOperatorRunAdapter:
        return AttentionOperatorCallerBufferRunAdapter(
            base_adapter, self._operation
        )


class AttentionOperatorRunAdapterFactoryChain:
    """Compose multiple identity-matched run adapter decorators in order."""

    def __init__(
        self,
        provider_id: str,
        operation_id: str,
        factories: Sequence[AttentionOperatorRunAdapterFactory],
    ) -> None:
        values = tuple(factories)
        if not values or any(
            not isinstance(item, AttentionOperatorRunAdapterFactory)
            for item in values
        ):
            raise TypeError("factories must contain run adapter factories")
        if any(
            item.provider_id != str(provider_id)
            or item.operation_id != str(operation_id)
            for item in values
        ):
            raise SchemaError("run adapter factory chain identities differ")
        self.provider_id = str(provider_id)
        self.operation_id = str(operation_id)
        self._factories = values

    def build(
        self, base_adapter: AttentionOperatorRunAdapter, device: str
    ) -> AttentionOperatorRunAdapter:
        adapter = base_adapter
        for factory in self._factories:
            adapter = factory.build(adapter, str(device))
        return adapter


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
        self._resource_binding = None

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

    @property
    def resource_binding(self):
        if self._resource_binding is None:
            raise AttentionStateError("Attention operator resources are not bound")
        return self._resource_binding

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
        runtime_resolution_fingerprint: Optional[str] = None,
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
            runtime_resolution_fingerprint,
        )
        candidate_binding = bind_attention_operator_operation(
            self._operation_catalog, candidate_session.active_plan
        )
        from .operator_resources import bind_attention_operator_resources

        candidate_resource_binding = bind_attention_operator_resources(
            operation,
            candidate_session.active_plan,
            candidate_binding,
        )
        candidate_runtime_binding = bind_attention_operator_runtime(
            candidate_session.active_plan,
            candidate_binding,
            callable_binding,
            candidate_resource_binding.fingerprint,
        )
        if (
            candidate_runtime_binding.resource_binding_fingerprint
            != candidate_resource_binding.fingerprint
        ):
            raise SchemaError("runtime binding did not freeze provider resources")
        self._plan_session = candidate_session
        self._run_adapter = run_adapter
        self._operation_binding = candidate_binding
        self._callable_binding = callable_binding
        self._runtime_binding = candidate_runtime_binding
        self._resource_binding = candidate_resource_binding

    def run(
        self,
        q,
        kv_cache,
        return_lse=True,
        out=None,
        lse=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        o_scale=None,
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
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            o_scale=o_scale,
            logits_soft_cap=logits_soft_cap,
            profiler_buffer=profiler_buffer,
            kv_cache_sf=kv_cache_sf,
        )
        return self._lower_request(request)

    def _lower_request(
        self, request: AttentionOperatorRunRequest
    ) -> AttentionLoweredOperatorCall:
        """Lower an internal request while keeping mode-specific fields private."""

        if not isinstance(request, AttentionOperatorRunRequest):
            raise TypeError("request must be AttentionOperatorRunRequest")
        active_plan = self.active_plan
        if self._run_adapter is None:  # defensive; active publication is atomic
            raise AttentionStateError("Attention operator run adapter is not initialized")
        if (
            request.active_plan_fingerprint != active_plan.fingerprint
            or request.framework_plan_fingerprint
            != active_plan.framework_plan.fingerprint
            or request.framework_plan_generation
            != active_plan.framework_plan.generation
        ):
            raise AttentionStateError(
                "Attention operator request does not bind the active plan"
            )
        if (
            self.runtime_binding.resource_binding_fingerprint
            != self.resource_binding.fingerprint
        ):
            raise AttentionStateError("Attention operator resource binding is stale")
        self.resource_binding.validate_request(request)
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
    "AttentionOperatorCallerBufferRunAdapter",
    "AttentionOperatorCallerBufferRunAdapterFactory",
    "AttentionOperatorQueryValidationRunAdapter",
    "AttentionOperatorQueryValidationRunAdapterFactory",
    "AttentionOperatorRunTensorValidationAdapter",
    "AttentionOperatorRunTensorValidationAdapterFactory",
    "AttentionOperatorRunAdapter",
    "AttentionOperatorRunAdapterFactory",
    "AttentionOperatorRunAdapterFactoryChain",
    "AttentionOperatorRunRequest",
    "AttentionOperatorTensorMetadataInspector",
    "AttentionOperatorWrapperSession",
    "inspect_attention_operator_dense_kv_input",
    "lower_attention_operator_run",
    "validate_attention_lowered_operator_call",
]
