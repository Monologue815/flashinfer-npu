"""Plan-bound workspace and output ownership for provider operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import (
    AttentionOperatorOperationBinding,
    AttentionOperatorOperationSpec,
)
from .operator_plan import AttentionOperatorActivePlan
from .operator_run import AttentionOperatorRunRequest
from .workspace import AttentionWorkspaceContract


ATTENTION_OPERATOR_RESOURCE_VERSION = 1


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionOperatorResourceBinding:
    """Resource behavior proven by one exact catalog operation.

    The current documented package APIs return tensors and do not expose the
    FlashInfer wrapper workspace as an argument.  Their package-internal
    temporary memory is therefore outside the caller-owned workspace contract.
    """

    active_plan_fingerprint: str
    operation_fingerprint: str
    provider_id: str
    operation_id: str
    workspace_ownership: str
    required_float_workspace_bytes: int
    required_int_workspace_bytes: int
    output_binding: str
    lse_binding: str
    schema_version: int = ATTENTION_OPERATOR_RESOURCE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RESOURCE_VERSION:
            raise SchemaError("unsupported Attention operator resource version")
        for name in ("active_plan_fingerprint", "operation_fingerprint"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError("%s must be lowercase SHA-256" % name)
        if not self.provider_id or not self.operation_id:
            raise SchemaError("resource binding identities must be non-empty")
        if self.workspace_ownership not in ("package_managed", "caller_managed"):
            raise SchemaError("invalid Attention workspace ownership")
        for name in (
            "required_float_workspace_bytes",
            "required_int_workspace_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SchemaError("resource workspace sizes must be non-negative")
        if self.workspace_ownership == "package_managed" and (
            self.required_float_workspace_bytes
            or self.required_int_workspace_bytes
        ):
            raise SchemaError(
                "package-managed operations cannot require wrapper workspace bytes"
            )
        if self.output_binding != "returned":
            raise SchemaError("unsupported Attention output binding")
        if self.lse_binding not in ("returned", "unsupported"):
            raise SchemaError("unsupported Attention LSE binding")

    def validate_request(self, request: AttentionOperatorRunRequest) -> None:
        if not isinstance(request, AttentionOperatorRunRequest):
            raise TypeError("request must be AttentionOperatorRunRequest")
        if request.active_plan_fingerprint != self.active_plan_fingerprint:
            raise SchemaError("resource binding does not match the active run plan")
        if request.out is not None:
            raise NotImplementedError(
                "provider operation has no caller-owned output-buffer binding"
            )
        if request.lse is not None:
            raise NotImplementedError(
                "provider operation has no caller-owned LSE-buffer binding"
            )
        if request.return_lse and self.lse_binding == "unsupported":
            raise NotImplementedError(
                "provider operation has no LSE return binding"
            )

    def bind_workspace_contract(
        self,
        contract: AttentionWorkspaceContract,
        *,
        plan_generation: int,
    ) -> AttentionWorkspaceContract:
        if not isinstance(contract, AttentionWorkspaceContract):
            raise TypeError("contract must be AttentionWorkspaceContract")
        return contract.bind_requirements(
            required_float_bytes=self.required_float_workspace_bytes,
            required_int_bytes=self.required_int_workspace_bytes,
            plan_generation=plan_generation,
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "operation_fingerprint": self.operation_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "workspace_ownership": self.workspace_ownership,
            "required_float_workspace_bytes": self.required_float_workspace_bytes,
            "required_int_workspace_bytes": self.required_int_workspace_bytes,
            "output_binding": self.output_binding,
            "lse_binding": self.lse_binding,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def bind_attention_operator_resources(
    operation: AttentionOperatorOperationSpec,
    active_plan: AttentionOperatorActivePlan,
    operation_binding: AttentionOperatorOperationBinding,
) -> AttentionOperatorResourceBinding:
    """Derive the conservative resource contract from an exact API signature."""

    if not isinstance(operation, AttentionOperatorOperationSpec):
        raise TypeError("operation must be AttentionOperatorOperationSpec")
    if not isinstance(active_plan, AttentionOperatorActivePlan):
        raise TypeError("active_plan must be AttentionOperatorActivePlan")
    if not isinstance(operation_binding, AttentionOperatorOperationBinding):
        raise TypeError(
            "operation_binding must be AttentionOperatorOperationBinding"
        )
    if (
        operation.operation_id != active_plan.prepared_plan.implementation_id
        or operation.provider_id != active_plan.provider_selection.provider_id
        or operation_binding.active_plan_fingerprint != active_plan.fingerprint
        or operation_binding.operation_id != operation.operation_id
        or operation_binding.operation_fingerprint != operation.fingerprint
    ):
        raise SchemaError("resource operation does not match the active plan")
    argument_names = set(operation.positional_arguments).union(
        operation.keyword_arguments
    )
    if argument_names.intersection(
        {
            "workspace",
            "workspace_buffer",
            "float_workspace_buffer",
            "int_workspace_buffer",
        }
    ):
        raise SchemaError(
            "operation workspace argument requires an explicit resource binder"
        )
    if set(operation.mutable_arguments).intersection(
        {"out", "output", "lse", "softmax_lse"}
    ):
        raise SchemaError(
            "operation mutable output requires an explicit resource binder"
        )
    if "output" not in operation.return_names:
        raise SchemaError("return-only resource binding requires an output return")
    return AttentionOperatorResourceBinding(
        active_plan_fingerprint=active_plan.fingerprint,
        operation_fingerprint=operation.fingerprint,
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        workspace_ownership="package_managed",
        required_float_workspace_bytes=0,
        required_int_workspace_bytes=0,
        output_binding="returned",
        lse_binding=("returned" if operation.supports_lse else "unsupported"),
    )


__all__ = [
    "ATTENTION_OPERATOR_RESOURCE_VERSION",
    "AttentionOperatorResourceBinding",
    "bind_attention_operator_resources",
]
