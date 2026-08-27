"""Plan-bound metadata validation for completed provider Attention calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import AttentionOperatorOperationSpec
from .operator_plan import AttentionOperatorActivePlan
from .operator_run import (
    AttentionLoweredOperatorCall,
    AttentionOperatorTensorMetadataInspector,
)
from .tensor_contract import AttentionTensorAccessPolicy, TensorView


ATTENTION_OPERATOR_COMPLETION_VERSION = 2


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class AttentionOperatorCompletionReceipt:
    """Evidence that public result tensors match one active plan."""

    active_plan_fingerprint: str
    framework_plan_fingerprint: str
    operation_fingerprint: str
    access_policy_fingerprint: str
    provider_id: str
    operation_id: str
    expected_device: str
    return_names: Tuple[str, ...]
    input_view_fingerprints: Tuple[Tuple[str, str], ...]
    result_view_fingerprints: Tuple[Tuple[str, str], ...]
    schema_version: int = ATTENTION_OPERATOR_COMPLETION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_COMPLETION_VERSION:
            raise SchemaError("unsupported Attention completion receipt version")
        for name in (
            "active_plan_fingerprint",
            "framework_plan_fingerprint",
            "operation_fingerprint",
            "access_policy_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if not self.provider_id or not self.operation_id or not self.expected_device:
            raise SchemaError("Attention completion identities must be non-empty")
        return_names = tuple(str(item) for item in self.return_names)
        inputs = tuple(
            (str(name), str(fingerprint))
            for name, fingerprint in self.input_view_fingerprints
        )
        views = tuple(
            (str(name), str(fingerprint))
            for name, fingerprint in self.result_view_fingerprints
        )
        if not return_names or len(set(return_names)) != len(return_names):
            raise SchemaError("Attention completion return names must be unique")
        input_names = tuple(name for name, _ in inputs)
        if (
            not inputs
            or len(set(input_names)) != len(input_names)
            or "query" not in input_names
        ):
            raise SchemaError("Attention completion input views are invalid")
        if tuple(name for name, _ in views) != return_names:
            raise SchemaError("Attention completion result views are out of order")
        for _, fingerprint in inputs + views:
            _require_hash("result view fingerprint", fingerprint)
        object.__setattr__(self, "return_names", return_names)
        object.__setattr__(self, "input_view_fingerprints", inputs)
        object.__setattr__(self, "result_view_fingerprints", views)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "framework_plan_fingerprint": self.framework_plan_fingerprint,
            "operation_fingerprint": self.operation_fingerprint,
            "access_policy_fingerprint": self.access_policy_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "expected_device": self.expected_device,
            "return_names": list(self.return_names),
            "input_view_fingerprints": [
                list(item) for item in self.input_view_fingerprints
            ],
            "result_view_fingerprints": [list(item) for item in self.result_view_fingerprints],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionOperatorCompletionValidator:
    """Validate provider-produced output/LSE tensors without reading device data."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        active_plan: AttentionOperatorActivePlan,
        inspector: AttentionOperatorTensorMetadataInspector,
        access_policy: AttentionTensorAccessPolicy,
        expected_device: str,
    ) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
            raise TypeError(
                "inspector must implement AttentionOperatorTensorMetadataInspector"
            )
        if not isinstance(access_policy, AttentionTensorAccessPolicy):
            raise TypeError("access_policy must be AttentionTensorAccessPolicy")
        if not str(expected_device):
            raise SchemaError("completion expected_device must be non-empty")
        if (
            active_plan.provider_selection.provider_id != operation.provider_id
            or active_plan.prepared_plan.implementation_id != operation.operation_id
        ):
            raise SchemaError("completion validator operation differs from active plan")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation
        self._active_plan = active_plan
        self._inspector = inspector
        self._access_policy = access_policy
        self._expected_device = str(expected_device)

    @property
    def operation_fingerprint(self) -> str:
        return self._operation.fingerprint

    @property
    def access_policy_fingerprint(self) -> str:
        return self._access_policy.fingerprint

    def validate(
        self, call: AttentionLoweredOperatorCall, result: Any
    ) -> AttentionOperatorCompletionReceipt:
        if not isinstance(call, AttentionLoweredOperatorCall):
            raise TypeError("call must be AttentionLoweredOperatorCall")
        if (
            call.provider_id != self.provider_id
            or call.operation_id != self.operation_id
            or call.active_plan_fingerprint != self._active_plan.fingerprint
        ):
            raise SchemaError("completed Attention call differs from active plan")
        if not set(call.return_names).issubset(self._operation.return_names):
            raise SchemaError("completed Attention return names differ from operation")
        if any(name not in ("output", "softmax_lse") for name in call.return_names):
            raise SchemaError("unsupported public Attention completion result")
        input_views = call.validated_input_views
        input_names = tuple(name for name, _ in input_views)
        if not input_views or "query" not in input_names:
            raise SchemaError(
                "completed Attention call has no validated query/input views"
            )

        if len(call.return_names) == 1:
            if isinstance(result, (tuple, list)):
                raise SchemaError("completed Attention return arity differs from call")
            values = (result,)
        else:
            if not isinstance(result, (tuple, list)) or len(result) != len(
                call.return_names
            ):
                raise SchemaError("completed Attention return arity differs from call")
            values = tuple(result)
        if any(value is None for value in values):
            raise SchemaError("completed Attention public result cannot be missing")

        plan = self._active_plan.framework_plan
        named_views = []
        for name, value in zip(call.return_names, values):
            view = self._inspector.to_view(value, name=name, writable=True)
            if not isinstance(view, TensorView):
                raise TypeError("tensor metadata inspector must return TensorView")
            if name == "output":
                expected_shape = plan.expected_output_shape
                expected_dtype = plan.spec.o_dtype
            else:
                expected_shape = plan.expected_lse_shape
                expected_dtype = "float32"
            if view.shape != expected_shape:
                raise SchemaError(
                    "%s result shape must be %r, got %r"
                    % (name, expected_shape, view.shape)
                )
            if view.dtype != expected_dtype:
                raise SchemaError(
                    "%s result dtype must be %s, got %s"
                    % (name, expected_dtype, view.dtype)
                )
            if view.device != self._expected_device:
                raise SchemaError(
                    "%s result device must match the planned provider device" % name
                )
            if not view.writable:
                raise SchemaError("%s result must be writable" % name)
            view.require_alignment(self._access_policy.required_alignment, name)
            if (
                self._access_policy.require_contiguous_output
                and not view.is_contiguous
            ):
                raise SchemaError("%s result must be contiguous" % name)
            named_views.append((name, view))

        if len(named_views) == 2 and named_views[0][1].overlaps(named_views[1][1]):
            raise SchemaError("output and softmax_lse results cannot alias")
        if not self._access_policy.permit_output_input_alias:
            for result_name, result_view in named_views:
                for input_name, input_view in input_views:
                    if result_view.overlaps(input_view):
                        raise SchemaError(
                            "%s result cannot alias %s" % (result_name, input_name)
                        )
        return AttentionOperatorCompletionReceipt(
            active_plan_fingerprint=self._active_plan.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            operation_fingerprint=self._operation.fingerprint,
            access_policy_fingerprint=self._access_policy.fingerprint,
            provider_id=self.provider_id,
            operation_id=self.operation_id,
            expected_device=self._expected_device,
            return_names=call.return_names,
            input_view_fingerprints=tuple(
                (name, view.fingerprint) for name, view in input_views
            ),
            result_view_fingerprints=tuple(
                (name, view.fingerprint) for name, view in named_views
            ),
        )


class AttentionOperatorCompletionValidatorFactory:
    """Bind one provider's result metadata contract to each active plan."""

    def __init__(
        self,
        operation: AttentionOperatorOperationSpec,
        inspector: AttentionOperatorTensorMetadataInspector,
        access_policy: AttentionTensorAccessPolicy,
    ) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
            raise TypeError(
                "inspector must implement AttentionOperatorTensorMetadataInspector"
            )
        if not isinstance(access_policy, AttentionTensorAccessPolicy):
            raise TypeError("access_policy must be AttentionTensorAccessPolicy")
        self.provider_id = operation.provider_id
        self.operation_id = operation.operation_id
        self._operation = operation
        self._inspector = inspector
        self._access_policy = access_policy

    @property
    def operation_fingerprint(self) -> str:
        return self._operation.fingerprint

    @property
    def access_policy_fingerprint(self) -> str:
        return self._access_policy.fingerprint

    def build(
        self, active_plan: AttentionOperatorActivePlan, expected_device: str
    ) -> AttentionOperatorCompletionValidator:
        return AttentionOperatorCompletionValidator(
            self._operation,
            active_plan,
            self._inspector,
            self._access_policy,
            expected_device,
        )


__all__ = [
    "ATTENTION_OPERATOR_COMPLETION_VERSION",
    "AttentionOperatorCompletionReceipt",
    "AttentionOperatorCompletionValidator",
    "AttentionOperatorCompletionValidatorFactory",
]
