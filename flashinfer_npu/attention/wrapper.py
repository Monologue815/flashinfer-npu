"""Executor-injected Attention wrapper for framework validation."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, Union

from .planner import AttentionFrameworkPlan, AttentionFrameworkSession
from .reference import (
    ReferenceAttentionResult,
    ReferenceKVInput,
    ReferenceTensor,
    ScaleInput,
)
from .resource_limits import AttentionMetadataLimits
from .schema import AttentionMetadata, AttentionMode, AttentionPlanSpec


class AttentionExecutor(Protocol):
    def execute(
        self,
        plan: AttentionFrameworkPlan,
        q: ReferenceTensor,
        kv_data: ReferenceKVInput,
        *,
        return_lse: bool = False,
        custom_mask_data: Optional[Sequence[Union[bool, int]]] = None,
        alibi_slopes: Optional[Sequence[float]] = None,
        q_scale: ScaleInput = 1.0,
        k_scale: ScaleInput = 1.0,
        v_scale: ScaleInput = 1.0,
        logits_soft_cap: float = 0.0,
    ) -> ReferenceAttentionResult: ...


class AttentionWrapper:
    """Backend-neutral wrapper with FlashInfer-style plan/run lifecycle."""

    def __init__(
        self,
        mode: AttentionMode,
        executor: AttentionExecutor,
        *,
        graph_enabled: bool = False,
        fixed_batch_size: Optional[int] = None,
        metadata_limits: Optional[AttentionMetadataLimits] = None,
    ) -> None:
        self._session = AttentionFrameworkSession(
            mode,
            graph_enabled=graph_enabled,
            fixed_batch_size=fixed_batch_size,
            metadata_limits=metadata_limits,
        )
        self._executor = executor

    @property
    def plan_state(self) -> AttentionFrameworkPlan:
        return self._session.plan_state

    def plan(
        self, spec: AttentionPlanSpec, metadata: AttentionMetadata
    ) -> AttentionFrameworkPlan:
        return self._session.plan(spec, metadata)

    def run(
        self,
        q: ReferenceTensor,
        kv_data: ReferenceKVInput,
        *,
        return_lse: bool = False,
        custom_mask_data: Optional[Sequence[Union[bool, int]]] = None,
        alibi_slopes: Optional[Sequence[float]] = None,
        q_scale: ScaleInput = 1.0,
        k_scale: ScaleInput = 1.0,
        v_scale: ScaleInput = 1.0,
        logits_soft_cap: float = 0.0,
    ) -> ReferenceAttentionResult:
        return self._executor.execute(
            self.plan_state,
            q,
            kv_data,
            return_lse=return_lse,
            custom_mask_data=custom_mask_data,
            alibi_slopes=alibi_slopes,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            logits_soft_cap=logits_soft_cap,
        )
