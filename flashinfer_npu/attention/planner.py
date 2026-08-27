"""Attention plan/run state machine for host-only conformance tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

from flashinfer_npu.runtime import SchemaError, WorkloadSpec

from .schema import (
    AttentionMetadata,
    AttentionMode,
    AttentionPlanSpec,
    KVCacheSpec,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    MixedPagedKVMetadata,
    SingleAttentionMetadata,
    TensorSpec,
)
from .resource_limits import (
    AttentionMetadataLimits,
    AttentionResourceUsage,
    UNBOUNDED_ATTENTION_METADATA_LIMITS,
)


class AttentionStateError(RuntimeError):
    """Raised when plan/run lifecycle rules are violated."""


@dataclass(frozen=True)
class AttentionOutputSpec:
    output: TensorSpec
    lse: Optional[TensorSpec]


@dataclass(frozen=True)
class AttentionFrameworkPlan:
    spec: AttentionPlanSpec
    metadata: AttentionMetadata
    workload: WorkloadSpec
    generation: int
    resource_limits: AttentionMetadataLimits = UNBOUNDED_ATTENTION_METADATA_LIMITS
    resource_usage: Optional[AttentionResourceUsage] = None

    @property
    def batch_size(self) -> int:
        return self.workload.dynamic_bounds[0]

    @property
    def total_qo_tokens(self) -> int:
        return self.workload.dynamic_bounds[1]

    @property
    def fingerprint(self) -> str:
        value = "%s:%s" % (
            self.workload.fingerprint,
            self.metadata.fingerprint,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def admission_fingerprint(self) -> str:
        """Identity of semantic plan plus its resource admission profile."""

        value = "%s:%s" % (self.fingerprint, self.resource_limits.fingerprint)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _expected_query_shape(self) -> Tuple[int, ...]:
        if self.spec.mode == AttentionMode.SINGLE_DECODE:
            return (self.spec.num_qo_heads, self.spec.head_dim_qk)
        if self.spec.mode == AttentionMode.BATCH_DECODE_PAGED:
            return (
                self.total_qo_tokens,
                self.spec.num_qo_heads,
                self.spec.head_dim_qk,
            )
        return (
            self.total_qo_tokens,
            self.spec.num_qo_heads,
            self.spec.head_dim_qk,
        )

    @property
    def expected_query_shape(self) -> Tuple[int, ...]:
        """Exact public query shape bound to this reusable plan."""

        return self._expected_query_shape()

    def _expected_output_shape(self) -> Tuple[int, ...]:
        q_shape = self._expected_query_shape()
        return q_shape[:-1] + (int(self.spec.head_dim_vo),)

    def _expected_lse_shape(self) -> Tuple[int, ...]:
        return self._expected_query_shape()[:-1]

    @property
    def expected_output_shape(self) -> Tuple[int, ...]:
        """Exact caller-owned output shape bound to this plan."""

        return self._expected_output_shape()

    @property
    def expected_lse_shape(self) -> Tuple[int, ...]:
        """Exact caller-owned log-sum-exp shape bound to this plan."""

        return self._expected_lse_shape()

    def validate_run(
        self,
        q: TensorSpec,
        kv_cache: KVCacheSpec,
        out: Optional[TensorSpec] = None,
        lse: Optional[TensorSpec] = None,
        return_lse: bool = False,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Optional[TensorSpec] = None,
    ) -> AttentionOutputSpec:
        expected_q = self._expected_query_shape()
        if q.shape != expected_q:
            raise SchemaError("q shape must be %r, got %r" % (expected_q, q.shape))
        if q.dtype != self.spec.q_dtype:
            raise SchemaError(
                "q dtype must be %s, got %s" % (self.spec.q_dtype, q.dtype)
            )
        if q.device != kv_cache.device:
            raise SchemaError("q and KV cache must be on the same device")
        if kv_cache.layout != self.spec.kv_layout:
            raise SchemaError("KV cache layout does not match the plan")
        if kv_cache.num_kv_heads != self.spec.num_kv_heads:
            raise SchemaError("KV head count does not match the plan")
        if kv_cache.head_dim_qk != self.spec.head_dim_qk:
            raise SchemaError("key head dimension does not match the plan")
        if kv_cache.head_dim_vo != self.spec.head_dim_vo:
            raise SchemaError("value head dimension does not match the plan")
        if kv_cache.dtype != self.spec.kv_dtype:
            raise SchemaError("KV dtype does not match the plan")
        if kv_cache.quant_spec != self.spec.kv_quant_spec:
            raise SchemaError("KV quantization spec does not match the plan")
        if isinstance(
            self.metadata,
            (PagedKVMetadata, PagedPrefillMetadata, MixedPagedKVMetadata),
        ):
            if not isinstance(kv_cache, PagedKVCacheSpec):
                raise SchemaError("paged metadata requires a paged KV cache")
            paged_metadata = (
                self.metadata.paged_kv
                if isinstance(self.metadata, PagedPrefillMetadata)
                else self.metadata
            )
            if kv_cache.page_size != paged_metadata.page_size:
                raise SchemaError("KV page size does not match metadata")
            if kv_cache.num_pages <= paged_metadata.max_page_index:
                raise SchemaError("KV cache does not contain every referenced page")
        elif isinstance(self.metadata, (RaggedKVMetadata, SingleAttentionMetadata)):
            if not isinstance(kv_cache, RaggedKVCacheSpec):
                raise SchemaError("ragged metadata requires a ragged KV cache")
            expected_kv_tokens = (
                self.metadata.total_kv_tokens
                if isinstance(self.metadata, RaggedKVMetadata)
                else self.metadata.kv_len
            )
            if kv_cache.total_kv_tokens != expected_kv_tokens:
                raise SchemaError("ragged KV token count does not match metadata")
        if logits_soft_cap < 0:
            raise SchemaError("run-time logits_soft_cap cannot be negative")
        if logits_soft_cap > 0 and float(self.spec.logits_soft_cap or 0.0) <= 0:
            raise SchemaError(
                "non-zero run-time logits_soft_cap requires a capped plan"
            )
        if self.spec.use_profiler and profiler_buffer is None:
            raise SchemaError("profiler_buffer is required by the planned workload")

        output_shape = self._expected_output_shape()
        if out is None:
            output = TensorSpec(output_shape, self.spec.o_dtype or q.dtype, q.device)
        else:
            if out.shape != output_shape:
                raise SchemaError("out shape must be %r" % (output_shape,))
            if out.dtype != self.spec.o_dtype or out.device != q.device:
                raise SchemaError("out dtype/device does not match the plan")
            output = out

        wants_lse = return_lse or lse is not None or self.spec.mode == AttentionMode.BATCH_MIXED_PAGED
        lse_output = None
        if wants_lse:
            lse_shape = self._expected_lse_shape()
            if lse is None:
                lse_output = TensorSpec(lse_shape, "float32", q.device)
            else:
                if lse.shape != lse_shape or lse.dtype != "float32":
                    raise SchemaError("lse must be float32 with shape %r" % (lse_shape,))
                if lse.device != q.device:
                    raise SchemaError("lse and q must be on the same device")
                lse_output = lse
        return AttentionOutputSpec(output=output, lse=lse_output)


class AttentionFrameworkSession:
    """Stateful plan/run validator mirroring FlashInfer wrapper lifecycle.

    The session performs no tensor computation. ``infer_run`` proves shape,
    layout, cache-capacity, option-consistency, and lifecycle behavior before a
    device backend is introduced.
    """

    def __init__(
        self,
        mode: AttentionMode,
        graph_enabled: bool = False,
        fixed_batch_size: Optional[int] = None,
        metadata_limits: Optional[AttentionMetadataLimits] = None,
    ) -> None:
        self._mode = AttentionMode(mode)
        self._graph_enabled = bool(graph_enabled)
        self._fixed_batch_size = fixed_batch_size
        self._fixed_q_len_per_req: Optional[int] = None
        self._metadata_limits = (
            UNBOUNDED_ATTENTION_METADATA_LIMITS
            if metadata_limits is None
            else metadata_limits
        )
        if not isinstance(self._metadata_limits, AttentionMetadataLimits):
            raise TypeError("metadata_limits must be AttentionMetadataLimits")
        if self._fixed_batch_size is not None and self._fixed_batch_size <= 0:
            raise SchemaError("fixed_batch_size must be positive")
        self._plan: Optional[AttentionFrameworkPlan] = None
        self._generation = 0

    @property
    def is_planned(self) -> bool:
        return self._plan is not None

    @property
    def graph_enabled(self) -> bool:
        return self._graph_enabled

    @property
    def metadata_limits(self) -> AttentionMetadataLimits:
        return self._metadata_limits

    @property
    def plan_state(self) -> AttentionFrameworkPlan:
        if self._plan is None:
            raise AttentionStateError("attention wrapper has not been planned")
        return self._plan

    def plan(
        self, spec: AttentionPlanSpec, metadata: AttentionMetadata
    ) -> AttentionFrameworkPlan:
        candidate = self.prepare_plan(spec, metadata)
        self.commit_prepared_plan(candidate)
        return candidate

    def prepare_plan(
        self, spec: AttentionPlanSpec, metadata: AttentionMetadata
    ) -> AttentionFrameworkPlan:
        """Validate and build the next plan without publishing session state."""

        if spec.mode != self._mode:
            raise SchemaError(
                "session mode %s cannot plan %s" % (self._mode.value, spec.mode.value)
            )
        workload = spec.to_workload_spec(metadata)
        resource_usage = self._metadata_limits.validate(spec, metadata)
        batch_size = workload.dynamic_bounds[0]
        if self._graph_enabled:
            if (
                self._fixed_batch_size is not None
                and batch_size != self._fixed_batch_size
            ):
                raise AttentionStateError(
                    "graph-enabled wrapper requires a fixed batch size"
                )
            if (
                self._fixed_q_len_per_req is not None
                and spec.q_len_per_req != self._fixed_q_len_per_req
            ):
                raise AttentionStateError(
                    "graph-enabled wrapper requires a fixed q_len_per_req"
                )
        return AttentionFrameworkPlan(
            spec=spec,
            metadata=metadata,
            workload=workload,
            generation=self._generation + 1,
            resource_limits=self._metadata_limits,
            resource_usage=resource_usage,
        )

    def commit_prepared_plan(self, candidate: AttentionFrameworkPlan) -> None:
        """Atomically publish a plan returned by :meth:`prepare_plan`."""

        if not isinstance(candidate, AttentionFrameworkPlan):
            raise TypeError("candidate must be AttentionFrameworkPlan")
        expected = self.prepare_plan(candidate.spec, candidate.metadata)
        if candidate != expected:
            raise SchemaError("prepared Attention framework plan is stale")
        batch_size = candidate.batch_size
        if self._graph_enabled:
            if self._fixed_batch_size is None:
                self._fixed_batch_size = batch_size
            if self._fixed_q_len_per_req is None:
                self._fixed_q_len_per_req = candidate.spec.q_len_per_req
        self._generation = candidate.generation
        self._plan = candidate

    def infer_run(
        self,
        q: TensorSpec,
        kv_cache: KVCacheSpec,
        out: Optional[TensorSpec] = None,
        lse: Optional[TensorSpec] = None,
        return_lse: bool = False,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Optional[TensorSpec] = None,
    ) -> AttentionOutputSpec:
        return self.plan_state.validate_run(
            q,
            kv_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            logits_soft_cap=logits_soft_cap,
            profiler_buffer=profiler_buffer,
        )
