"""Plan-time materialization of provider tensor recipes.

The framework owns identities and lifecycle; a selected provider owns the
actual tensor constructor.  No torch or device package is imported here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, Mapping, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .operator_plan import (
    AttentionOperatorActivePlan,
    AttentionOperatorPlanFactory,
    AttentionPreparedOperatorPlan,
)
from .operator_run import (
    AttentionLoweredOperatorCall,
    AttentionOperatorRunAdapter,
    AttentionOperatorRunRequest,
)
from .provider_adapters import AttentionOperatorTensorPlan


ATTENTION_OPERATOR_MATERIALIZATION_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class AttentionOperatorTensorMaterializer(Protocol):
    """Provider-specific constructor for one already validated tensor recipe."""

    provider_id: str
    materializer_id: str

    def materialize(self, tensor_plan: AttentionOperatorTensorPlan, device: str) -> Any:
        """Create one provider tensor without invoking an Attention operator."""


@dataclass(frozen=True)
class AttentionMaterializedOperatorTensor:
    provider_id: str
    materializer_id: str
    tensor_plan_fingerprint: str
    role: str
    device: str
    tensor: Any = field(repr=False, compare=False)
    schema_version: int = ATTENTION_OPERATOR_MATERIALIZATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_MATERIALIZATION_VERSION:
            raise SchemaError("unsupported Attention materialized tensor version")
        if not self.provider_id or not self.materializer_id or not self.role:
            raise SchemaError("materialized tensor identities must be non-empty")
        if len(self.tensor_plan_fingerprint) != 64 or any(
            item not in "0123456789abcdef"
            for item in self.tensor_plan_fingerprint
        ):
            raise SchemaError("materialized tensor plan fingerprint must be SHA-256")
        if not str(self.device):
            raise SchemaError("materialized tensor device must be non-empty")
        if self.tensor is None or isinstance(self.tensor, AttentionOperatorTensorPlan):
            raise SchemaError("materializer must return an opaque provider tensor")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "materializer_id": self.materializer_id,
            "tensor_plan_fingerprint": self.tensor_plan_fingerprint,
            "role": self.role,
            "device": self.device,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionMaterializedOperatorPlanState:
    provider_id: str
    logical_state: Any = field(repr=False, compare=False)
    tensors: Tuple[AttentionMaterializedOperatorTensor, ...] = ()
    schema_version: int = ATTENTION_OPERATOR_MATERIALIZATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_MATERIALIZATION_VERSION:
            raise SchemaError("unsupported Attention materialized plan state version")
        if not self.provider_id:
            raise SchemaError("materialized plan provider_id must be non-empty")
        tensors = tuple(self.tensors)
        if not tensors:
            raise SchemaError("materialized provider plan must contain tensors")
        if any(item.provider_id != self.provider_id for item in tensors):
            raise SchemaError("materialized plan contains a foreign provider tensor")
        roles = tuple(item.role for item in tensors)
        if len(set(roles)) != len(roles):
            raise SchemaError("materialized provider tensor roles must be unique")
        object.__setattr__(self, "tensors", tuple(sorted(tensors, key=lambda x: x.role)))

    @property
    def tensor_by_role(self) -> Dict[str, Any]:
        return {item.role: item.tensor for item in self.tensors}

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "tensors": [item.to_dict() for item in self.tensors],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def collect_attention_operator_tensor_plans(value: Any) -> Tuple[AttentionOperatorTensorPlan, ...]:
    """Collect unique tensor recipes from immutable provider logical state."""

    found = []

    def visit(item):
        if isinstance(item, AttentionOperatorTensorPlan):
            found.append(item)
            return
        if is_dataclass(item) and not isinstance(
            item,
            (AttentionMaterializedOperatorTensor, AttentionMaterializedOperatorPlanState),
        ):
            for definition in fields(item):
                visit(getattr(item, definition.name))
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(value)
    by_role = {}
    for tensor_plan in found:
        previous = by_role.get(tensor_plan.role)
        if previous is not None and previous.fingerprint != tensor_plan.fingerprint:
            raise SchemaError("provider plan reuses a tensor role with different recipes")
        by_role[tensor_plan.role] = tensor_plan
    return tuple(by_role[role] for role in sorted(by_role))


def materialize_attention_operator_plan_state(
    provider_id: str,
    logical_state: Any,
    materializer: AttentionOperatorTensorMaterializer,
    device: str,
) -> AttentionMaterializedOperatorPlanState:
    if not isinstance(materializer, AttentionOperatorTensorMaterializer):
        raise TypeError("materializer must implement AttentionOperatorTensorMaterializer")
    if materializer.provider_id != provider_id:
        raise SchemaError("tensor materializer does not match provider")
    if not materializer.materializer_id:
        raise SchemaError("tensor materializer_id must be non-empty")
    tensor_plans = collect_attention_operator_tensor_plans(logical_state)
    if not tensor_plans:
        raise SchemaError("provider logical state has no tensor recipes")
    bindings = []
    for tensor_plan in tensor_plans:
        tensor = materializer.materialize(tensor_plan, str(device))
        bindings.append(
            AttentionMaterializedOperatorTensor(
                provider_id=provider_id,
                materializer_id=materializer.materializer_id,
                tensor_plan_fingerprint=tensor_plan.fingerprint,
                role=tensor_plan.role,
                device=str(device),
                tensor=tensor,
            )
        )
    return AttentionMaterializedOperatorPlanState(
        provider_id=provider_id,
        logical_state=logical_state,
        tensors=tuple(bindings),
    )


class AttentionMaterializingPlanFactory:
    """Compose a logical plan factory with one selected tensor materializer."""

    def __init__(
        self,
        logical_factory: AttentionOperatorPlanFactory,
        materializer: AttentionOperatorTensorMaterializer,
        device: str,
    ) -> None:
        if not isinstance(logical_factory, AttentionOperatorPlanFactory):
            raise TypeError("logical_factory must implement AttentionOperatorPlanFactory")
        if not isinstance(materializer, AttentionOperatorTensorMaterializer):
            raise TypeError("materializer must implement AttentionOperatorTensorMaterializer")
        if logical_factory.provider_id != materializer.provider_id:
            raise SchemaError("logical factory and tensor materializer providers differ")
        if not str(device):
            raise SchemaError("materialization device must be non-empty")
        self.provider_id = logical_factory.provider_id
        self.operation_id = logical_factory.operation_id
        self._logical_factory = logical_factory
        self._materializer = materializer
        self._device = str(device)

    def prepare(self, plan, receipt, selection):
        logical = self._logical_factory.prepare(plan, receipt, selection)
        if not isinstance(logical, AttentionPreparedOperatorPlan):
            raise TypeError("logical plan factory returned an invalid prepared plan")
        materialized_state = materialize_attention_operator_plan_state(
            self.provider_id,
            logical.opaque_state,
            self._materializer,
            self._device,
        )
        return replace(
            logical,
            opaque_plan_token=(
                "%s-materialized-%s"
                % (logical.opaque_plan_token, materialized_state.fingerprint)
            ),
            opaque_state=materialized_state,
        )


class AttentionMaterializingRunAdapter:
    """Replace tensor recipes in a logical lowered call with plan-owned tensors."""

    def __init__(self, logical_adapter: AttentionOperatorRunAdapter) -> None:
        if not isinstance(logical_adapter, AttentionOperatorRunAdapter):
            raise TypeError("logical_adapter must implement AttentionOperatorRunAdapter")
        self.provider_id = logical_adapter.provider_id
        self._logical_adapter = logical_adapter

    def lower(
        self,
        active_plan: AttentionOperatorActivePlan,
        request: AttentionOperatorRunRequest,
    ) -> AttentionLoweredOperatorCall:
        state = active_plan.prepared_plan.opaque_state
        if not isinstance(state, AttentionMaterializedOperatorPlanState):
            raise SchemaError("materializing adapter requires materialized plan state")
        if state.provider_id != self.provider_id:
            raise SchemaError("materialized plan state provider is stale")
        logical_prepared = replace(
            active_plan.prepared_plan, opaque_state=state.logical_state
        )
        logical_active = replace(active_plan, prepared_plan=logical_prepared)
        logical_call = self._logical_adapter.lower(logical_active, request)
        tensors = state.tensor_by_role

        def replace_recipe(value):
            if isinstance(value, AttentionOperatorTensorPlan):
                if value.role not in tensors:
                    raise SchemaError("lowered call references an unmaterialized tensor")
                return tensors[value.role]
            if isinstance(value, tuple):
                return tuple(replace_recipe(item) for item in value)
            if isinstance(value, list):
                return [replace_recipe(item) for item in value]
            return value

        positional = tuple(
            (name, replace_recipe(value))
            for name, value in logical_call.positional_arguments
        )
        keyword = tuple(
            (name, replace_recipe(value))
            for name, value in logical_call.keyword_arguments
        )
        return replace(
            logical_call,
            positional_arguments=positional,
            keyword_arguments=keyword,
        )


__all__ = [
    "ATTENTION_OPERATOR_MATERIALIZATION_VERSION",
    "AttentionMaterializedOperatorPlanState",
    "AttentionMaterializedOperatorTensor",
    "AttentionMaterializingPlanFactory",
    "AttentionMaterializingRunAdapter",
    "AttentionOperatorTensorMaterializer",
    "collect_attention_operator_tensor_plans",
    "materialize_attention_operator_plan_state",
]
