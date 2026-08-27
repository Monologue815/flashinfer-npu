"""Atomic evidence for one provider Attention invocation and completion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Optional

from flashinfer_npu.jit.attention import AttentionJitRuntimeExecutorBinding
from flashinfer_npu.runtime import SchemaError

from .operator_completion import AttentionOperatorCompletionReceipt
from .operator_execution import AttentionOperatorExecutionReceipt


ATTENTION_OPERATOR_RUN_RECEIPT_VERSION = 4


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionOperatorRunReceipt:
    """Join execution authority and result acceptance for one public run."""

    execution: AttentionOperatorExecutionReceipt = field(repr=False, compare=False)
    completion: AttentionOperatorCompletionReceipt = field(
        repr=False, compare=False
    )
    jit_runtime_executor: Optional[AttentionJitRuntimeExecutorBinding] = field(
        default=None, repr=False, compare=False
    )
    runtime_declaration_fingerprint: Optional[str] = None
    provider_integration_bundle_id: Optional[str] = None
    provider_integration_bundle_fingerprint: Optional[str] = None
    plan_scoring_manifest_id: Optional[str] = None
    plan_scoring_manifest_fingerprint: Optional[str] = None
    plan_scoring_policy_id: Optional[str] = None
    plan_scoring_policy_fingerprint: Optional[str] = None
    schema_version: int = ATTENTION_OPERATOR_RUN_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUN_RECEIPT_VERSION:
            raise SchemaError("unsupported Attention run receipt version")
        if not isinstance(self.execution, AttentionOperatorExecutionReceipt):
            raise TypeError("execution must be AttentionOperatorExecutionReceipt")
        if not isinstance(self.completion, AttentionOperatorCompletionReceipt):
            raise TypeError("completion must be AttentionOperatorCompletionReceipt")
        if self.runtime_declaration_fingerprint is not None and (
            len(self.runtime_declaration_fingerprint) != 64
            or any(
                item not in "0123456789abcdef"
                for item in self.runtime_declaration_fingerprint
            )
        ):
            raise SchemaError(
                "runtime_declaration_fingerprint must be lowercase SHA-256"
            )
        scoring_fields = (
            self.plan_scoring_manifest_id,
            self.plan_scoring_manifest_fingerprint,
            self.plan_scoring_policy_id,
            self.plan_scoring_policy_fingerprint,
        )
        if any(item is not None for item in scoring_fields):
            if any(item is None for item in scoring_fields):
                raise SchemaError(
                    "Attention run scoring manifest identity is incomplete"
                )
            if self.runtime_declaration_fingerprint is None:
                raise SchemaError(
                    "run scoring manifest identity requires a declaration"
                )
            manifest_id = str(self.plan_scoring_manifest_id)
            policy_id = str(self.plan_scoring_policy_id)
            if (
                not manifest_id.strip()
                or any(item.isspace() for item in manifest_id)
                or not policy_id.strip()
                or any(item.isspace() for item in policy_id)
            ):
                raise SchemaError("Attention run scoring manifest ids are invalid")
            for name, value in (
                (
                    "plan_scoring_manifest_fingerprint",
                    self.plan_scoring_manifest_fingerprint,
                ),
                (
                    "plan_scoring_policy_fingerprint",
                    self.plan_scoring_policy_fingerprint,
                ),
            ):
                if len(str(value)) != 64 or any(
                    item not in "0123456789abcdef" for item in str(value)
                ):
                    raise SchemaError("%s must be lowercase SHA-256" % name)
        bundle_fields = (
            self.provider_integration_bundle_id,
            self.provider_integration_bundle_fingerprint,
        )
        if any(item is not None for item in bundle_fields):
            if any(item is None for item in bundle_fields):
                raise SchemaError(
                    "Attention run provider bundle identity is incomplete"
                )
            if self.runtime_declaration_fingerprint is None or not all(
                item is not None for item in scoring_fields
            ):
                raise SchemaError(
                    "Attention run provider bundle identity requires declaration "
                    "and scoring manifest identities"
                )
            bundle_id = str(self.provider_integration_bundle_id)
            if not bundle_id.strip() or any(
                item.isspace() for item in bundle_id
            ):
                raise SchemaError("Attention run provider bundle id is invalid")
            fingerprint = str(self.provider_integration_bundle_fingerprint)
            if len(fingerprint) != 64 or any(
                item not in "0123456789abcdef" for item in fingerprint
            ):
                raise SchemaError(
                    "provider_integration_bundle_fingerprint must be lowercase "
                    "SHA-256"
                )
        execution = self.execution
        completion = self.completion
        if (
            execution.active_plan_fingerprint
            != completion.active_plan_fingerprint
            or execution.provider_id != completion.provider_id
            or execution.operation_id != completion.operation_id
            or execution.return_names != completion.return_names
        ):
            raise SchemaError("Attention execution and completion receipts differ")
        if self.jit_runtime_executor is not None:
            if not isinstance(
                self.jit_runtime_executor, AttentionJitRuntimeExecutorBinding
            ):
                raise TypeError(
                    "jit_runtime_executor must be "
                    "AttentionJitRuntimeExecutorBinding"
                )
            jit = self.jit_runtime_executor
            if (
                jit.active_plan_fingerprint != execution.active_plan_fingerprint
                or jit.runtime_binding_fingerprint
                != execution.runtime_binding_fingerprint
                or jit.provider_id != execution.provider_id
                or jit.operation_id != execution.operation_id
            ):
                raise SchemaError("Attention JIT runtime receipt differs from execution")

    @property
    def active_plan_fingerprint(self) -> str:
        return self.execution.active_plan_fingerprint

    @property
    def provider_id(self) -> str:
        return self.execution.provider_id

    @property
    def operation_id(self) -> str:
        return self.execution.operation_id

    @property
    def return_names(self):
        return self.execution.return_names

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "return_names": list(self.return_names),
            "execution_receipt_fingerprint": self.execution.fingerprint,
            "completion_receipt_fingerprint": self.completion.fingerprint,
            "jit_runtime_executor_binding_fingerprint": (
                self.jit_runtime_executor.fingerprint
                if self.jit_runtime_executor is not None
                else None
            ),
            "runtime_declaration_fingerprint": (
                self.runtime_declaration_fingerprint
            ),
            "provider_integration_bundle_id": (
                self.provider_integration_bundle_id
            ),
            "provider_integration_bundle_fingerprint": (
                self.provider_integration_bundle_fingerprint
            ),
            "plan_scoring_manifest_id": self.plan_scoring_manifest_id,
            "plan_scoring_manifest_fingerprint": (
                self.plan_scoring_manifest_fingerprint
            ),
            "plan_scoring_policy_id": self.plan_scoring_policy_id,
            "plan_scoring_policy_fingerprint": (
                self.plan_scoring_policy_fingerprint
            ),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "ATTENTION_OPERATOR_RUN_RECEIPT_VERSION",
    "AttentionOperatorRunReceipt",
]
