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


ATTENTION_OPERATOR_RUN_RECEIPT_VERSION = 1


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
    schema_version: int = ATTENTION_OPERATOR_RUN_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUN_RECEIPT_VERSION:
            raise SchemaError("unsupported Attention run receipt version")
        if not isinstance(self.execution, AttentionOperatorExecutionReceipt):
            raise TypeError("execution must be AttentionOperatorExecutionReceipt")
        if not isinstance(self.completion, AttentionOperatorCompletionReceipt):
            raise TypeError("completion must be AttentionOperatorCompletionReceipt")
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
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "ATTENTION_OPERATOR_RUN_RECEIPT_VERSION",
    "AttentionOperatorRunReceipt",
]
