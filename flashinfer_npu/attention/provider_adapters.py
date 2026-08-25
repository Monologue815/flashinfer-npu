"""Pure framework lowering for documented Ascend Attention providers.

This module intentionally imports neither ``torch_npu`` nor
``flash_attn``.  It turns an already-authorized framework plan into immutable
provider state and later describes the exact external Python call.  Device
tensor materialization and operator execution remain outside this checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .dispatch import AttentionDispatchReceipt
from .operator_plan import AttentionOperatorActivePlan, AttentionPreparedOperatorPlan
from .operator_provider import AttentionOperatorProviderSelection
from .operator_run import AttentionLoweredOperatorCall, AttentionOperatorRunRequest
from .planner import AttentionFrameworkPlan
from .schema import (
    AttentionMode,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
)


ATTENTION_PROVIDER_ADAPTER_VERSION = 1

CANN_V2_OPERATION_ID = (
    "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"
)
FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID = (
    "flash_attention_npu.flash_attn_with_kvcache@v3"
)

_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_INT32_MAX = (1 << 31) - 1
_CANN_CAUSAL_MASK_SIZE = 2048


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _shape_numel(shape: Tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


@dataclass(frozen=True)
class AttentionOperatorTensorPlan:
    """Immutable recipe for a provider tensor created from plan metadata.

    ``explicit`` recipes carry row-major integer values.  The causal mask is
    represented symbolically so the framework never allocates a 2048x2048
    tensor during host-only conformance tests.
    """

    role: str
    shape: Tuple[int, ...]
    dtype: str
    materialization: str
    values: Tuple[int, ...] = ()
    device_policy: str = "same_as_query"
    schema_version: int = ATTENTION_PROVIDER_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROVIDER_ADAPTER_VERSION:
            raise SchemaError("unsupported Attention provider tensor plan version")
        if not _ROLE.fullmatch(str(self.role)):
            raise SchemaError("invalid provider tensor role")
        shape = tuple(int(item) for item in self.shape)
        if not shape or any(item < 0 for item in shape):
            raise SchemaError("provider tensor shape must be non-empty and non-negative")
        values = tuple(int(item) for item in self.values)
        if self.materialization == "explicit_int32":
            if self.dtype != "int32":
                raise SchemaError("explicit_int32 tensor plans require int32 dtype")
            if len(values) != _shape_numel(shape):
                raise SchemaError("explicit provider tensor values do not match shape")
            if any(item < 0 or item > _INT32_MAX for item in values):
                raise SchemaError("explicit provider tensor values must fit int32")
        elif self.materialization == "right_down_causal_2048":
            if self.dtype != "bool" or shape != (
                _CANN_CAUSAL_MASK_SIZE,
                _CANN_CAUSAL_MASK_SIZE,
            ):
                raise SchemaError("CANN causal mask recipe has an invalid contract")
            if values:
                raise SchemaError("symbolic causal mask cannot carry explicit values")
        else:
            raise SchemaError("unknown provider tensor materialization recipe")
        if self.device_policy != "same_as_query":
            raise SchemaError("unsupported provider tensor device policy")
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "materialization": self.materialization,
            "values": list(self.values),
            "device_policy": self.device_policy,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionDensePageTablePlan:
    """Dense provider page table derived from FlashInfer CSR metadata."""

    tensor: AttentionOperatorTensorPlan
    row_page_counts: Tuple[int, ...]
    padding_page_id: int = 0
    schema_version: int = ATTENTION_PROVIDER_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROVIDER_ADAPTER_VERSION:
            raise SchemaError("unsupported Attention dense page table version")
        if not isinstance(self.tensor, AttentionOperatorTensorPlan):
            raise TypeError("tensor must be AttentionOperatorTensorPlan")
        if self.tensor.role not in {"block_table", "page_table"}:
            raise SchemaError("dense page table has an invalid tensor role")
        if len(self.tensor.shape) != 2:
            raise SchemaError("dense page table tensor must be two-dimensional")
        counts = tuple(int(item) for item in self.row_page_counts)
        if len(counts) != self.tensor.shape[0]:
            raise SchemaError("page table row counts do not match batch size")
        if any(item < 0 or item > self.tensor.shape[1] for item in counts):
            raise SchemaError("page table row count exceeds its dense width")
        if self.padding_page_id < 0 or self.padding_page_id > _INT32_MAX:
            raise SchemaError("page table padding id must fit int32")
        width = self.tensor.shape[1]
        for row, count in enumerate(counts):
            start = row * width + count
            end = (row + 1) * width
            if any(
                item != self.padding_page_id
                for item in self.tensor.values[start:end]
            ):
                raise SchemaError("page table padding cells are not deterministic")
        object.__setattr__(self, "row_page_counts", counts)

    @property
    def rows(self) -> Tuple[Tuple[int, ...], ...]:
        width = self.tensor.shape[1]
        return tuple(
            self.tensor.values[row * width : (row + 1) * width]
            for row in range(self.tensor.shape[0])
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "tensor": self.tensor.to_dict(),
            "row_page_counts": list(self.row_page_counts),
            "padding_page_id": self.padding_page_id,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def build_attention_dense_page_table(
    indptr: Tuple[int, ...],
    indices: Tuple[int, ...],
    *,
    role: str,
    padding_page_id: int = 0,
) -> AttentionDensePageTablePlan:
    """Convert FlashInfer CSR page metadata into a deterministic dense table."""

    indptr = tuple(int(item) for item in indptr)
    indices = tuple(int(item) for item in indices)
    if len(indptr) < 2 or indptr[0] != 0:
        raise SchemaError("page table indptr must start at zero")
    if any(left > right for left, right in zip(indptr, indptr[1:])):
        raise SchemaError("page table indptr must be monotonic")
    if indptr[-1] != len(indices):
        raise SchemaError("page table indices length must equal indptr[-1]")
    if any(item < 0 or item > _INT32_MAX for item in indices):
        raise SchemaError("page table indices must fit int32")
    counts = tuple(right - left for left, right in zip(indptr, indptr[1:]))
    width = max(counts, default=0)
    if width == 0:
        raise SchemaError("provider paged Attention requires at least one page")
    values = []
    for row, count in enumerate(counts):
        values.extend(indices[indptr[row] : indptr[row + 1]])
        values.extend([padding_page_id] * (width - count))
    return AttentionDensePageTablePlan(
        tensor=AttentionOperatorTensorPlan(
            role=role,
            shape=(len(counts), width),
            dtype="int32",
            materialization="explicit_int32",
            values=tuple(values),
        ),
        row_page_counts=counts,
        padding_page_id=padding_page_id,
    )


def _explicit_int32_tensor(
    role: str, values: Tuple[int, ...]
) -> AttentionOperatorTensorPlan:
    return AttentionOperatorTensorPlan(
        role=role,
        shape=(len(values),),
        dtype="int32",
        materialization="explicit_int32",
        values=values,
    )


def _cumulative(lengths: Tuple[int, ...], *, include_zero: bool) -> Tuple[int, ...]:
    total = 0
    result = [0] if include_zero else []
    for length in lengths:
        total += int(length)
        result.append(total)
    return tuple(result)


def _paged_metadata(
    plan: AttentionFrameworkPlan,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], int]:
    metadata = plan.metadata
    if isinstance(metadata, PagedPrefillMetadata):
        return (
            metadata.qo_lengths,
            metadata.paged_kv.sequence_lengths,
            metadata.paged_kv.indptr,
            metadata.paged_kv.indices,
            metadata.paged_kv.page_size,
        )
    if isinstance(metadata, PagedKVMetadata):
        return (
            (plan.spec.q_len_per_req,) * metadata.batch_size,
            metadata.sequence_lengths,
            metadata.indptr,
            metadata.indices,
            metadata.page_size,
        )
    if isinstance(metadata, MixedPagedKVMetadata):
        return (
            metadata.qo_lengths,
            metadata.kv_len_arr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.page_size,
        )
    raise SchemaError("provider adapter requires paged Attention metadata")


def _common_paged_plan_rejection_reasons(
    plan: AttentionFrameworkPlan,
) -> Tuple[str, ...]:
    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    reasons = []
    if plan.spec.mode not in {
        AttentionMode.BATCH_PREFILL_PAGED,
        AttentionMode.BATCH_DECODE_PAGED,
        AttentionMode.BATCH_MIXED_PAGED,
    }:
        reasons.append("provider adapter supports paged prefill/decode/mixed only")
    if plan.spec.pos_encoding_mode != PosEncodingMode.NONE:
        reasons.append("provider adapter does not support positional encoding")
    if plan.spec.custom_mask is not None:
        reasons.append("provider adapter does not support a custom FlashInfer mask")
    if float(plan.spec.logits_soft_cap or 0.0) != 0.0:
        reasons.append("provider adapter does not support logits soft cap")
    if plan.spec.use_profiler:
        reasons.append("provider adapter does not support profiler buffers")
    if plan.spec.kv_quant_spec is not None:
        reasons.append(
            "provider operation has no verified paged KV quantization binding"
        )
    try:
        query_lengths, kv_lengths, _, _, _ = _paged_metadata(plan)
    except SchemaError as error:
        reasons.append(str(error))
        return tuple(dict.fromkeys(reasons))
    if not query_lengths or any(item <= 0 for item in query_lengths):
        reasons.append("provider adapter requires positive query lengths")
    if not kv_lengths or any(item <= 0 for item in kv_lengths):
        reasons.append("provider adapter requires positive KV lengths")
    return tuple(dict.fromkeys(reasons))


def explain_cann_v2_paged_plan(
    plan: AttentionFrameworkPlan,
) -> Tuple[str, ...]:
    """Pure CANN v2 plan admission shared by auto-selection and prepare."""

    reasons = list(_common_paged_plan_rejection_reasons(plan))
    spec = plan.spec
    if spec.kv_layout != KVLayout.HND:
        reasons.append("CANN v2 paged lowering requires HND KV cache")
    if spec.q_dtype not in {"float16", "bfloat16"}:
        reasons.append("CANN v2 TND query requires float16 or bfloat16")
    if spec.kv_dtype != spec.q_dtype or spec.o_dtype != spec.q_dtype:
        reasons.append("CANN v2 dense TND lowering requires matching dtypes")
    if spec.head_dim_qk not in {128, 192} or spec.head_dim_vo not in {128, 192}:
        reasons.append("CANN v2 TND lowering requires documented head dimensions")
    if spec.num_qo_heads // spec.num_kv_heads > 64:
        reasons.append("CANN v2 GQA ratio cannot exceed 64")
    if spec.window_left != -1 or spec.window_right != 0:
        reasons.append("CANN v2 adapter has no verified sliding-window binding")
    try:
        query_lengths, _, _, _, page_size = _paged_metadata(plan)
    except SchemaError:
        query_lengths = ()
        page_size = 0
    if page_size and (
        page_size < 128 or page_size > 512 or page_size % 128
    ):
        reasons.append(
            "CANN v2 PageAttention block size must be 128..512 by 128"
        )
    if len(query_lengths) > 4096:
        reasons.append("CANN v2 TND batch size cannot exceed 4096")
    return tuple(dict.fromkeys(reasons))


def explain_flash_attention_npu_v3_paged_plan(
    plan: AttentionFrameworkPlan,
) -> Tuple[str, ...]:
    """Pure flash-attention-npu v3 admission shared by selection/prepare."""

    reasons = list(_common_paged_plan_rejection_reasons(plan))
    spec = plan.spec
    if spec.kv_layout != KVLayout.NHD:
        reasons.append(
            "flash-attention-npu v3 paged lowering requires NHD KV cache"
        )
    if spec.q_dtype not in {"float16", "bfloat16"}:
        reasons.append("flash-attention-npu v3 requires float16 or bfloat16")
    if spec.kv_dtype != spec.q_dtype or spec.o_dtype != spec.q_dtype:
        reasons.append(
            "flash-attention-npu v3 lowering requires matching dense dtypes"
        )
    return tuple(dict.fromkeys(reasons))


class CannV2PagedPlanGate:
    """Side-effect-free auto-selection gate for the documented CANN API."""

    provider_id = "cann"
    operation_id = CANN_V2_OPERATION_ID

    def rejection_reasons(self, plan, device):
        return explain_cann_v2_paged_plan(plan)


class FlashAttentionNpuV3PagedPlanGate:
    """Side-effect-free auto-selection gate for flash-attention-npu v3."""

    provider_id = "flash_attention_npu"
    operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID

    def rejection_reasons(self, plan, device):
        return explain_flash_attention_npu_v3_paged_plan(plan)


def _validate_prepare_authority(
    provider_id: str,
    plan: AttentionFrameworkPlan,
    receipt: AttentionDispatchReceipt,
    selection: AttentionOperatorProviderSelection,
) -> None:
    if not isinstance(receipt, AttentionDispatchReceipt):
        raise TypeError("receipt must be AttentionDispatchReceipt")
    if not isinstance(selection, AttentionOperatorProviderSelection):
        raise TypeError("selection must be AttentionOperatorProviderSelection")
    if selection.provider_id != provider_id:
        raise SchemaError("provider plan factory does not match selection")
    if receipt.plan_fingerprint != plan.fingerprint:
        raise SchemaError("provider plan factory received a stale dispatch receipt")


@dataclass(frozen=True)
class CannV2PagedPlanState:
    query_cumulative_lengths: Tuple[int, ...]
    kv_sequence_lengths: Tuple[int, ...]
    block_table: AttentionDensePageTablePlan
    causal_mask: Optional[AttentionOperatorTensorPlan]
    num_query_heads: int
    num_key_value_heads: int
    softmax_scale: float
    sparse_mode: int
    block_size: int
    minimum_kv_block_pool_size: int
    input_layout: str = "TND"
    inner_precise: int = 0
    schema_version: int = ATTENTION_PROVIDER_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROVIDER_ADAPTER_VERSION:
            raise SchemaError("unsupported CANN v2 plan state version")
        if self.input_layout != "TND":
            raise SchemaError("CANN v2 paged plan must use TND query layout")
        if len(self.query_cumulative_lengths) != len(self.kv_sequence_lengths):
            raise SchemaError("CANN sequence metadata batch sizes differ")
        if self.block_table.tensor.shape[0] != len(self.kv_sequence_lengths):
            raise SchemaError("CANN block table batch size differs")
        if any(
            left > right
            for left, right in zip(
                self.query_cumulative_lengths, self.query_cumulative_lengths[1:]
            )
        ):
            raise SchemaError("CANN TND query lengths must be cumulative")
        if self.sparse_mode == 3 and self.causal_mask is None:
            raise SchemaError("CANN causal plan requires the optimized mask recipe")
        if self.sparse_mode == 0 and self.causal_mask is not None:
            raise SchemaError("CANN non-causal plan cannot carry a causal mask")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "query_cumulative_lengths": list(self.query_cumulative_lengths),
            "kv_sequence_lengths": list(self.kv_sequence_lengths),
            "block_table": self.block_table.to_dict(),
            "causal_mask": (
                self.causal_mask.to_dict() if self.causal_mask is not None else None
            ),
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "softmax_scale": self.softmax_scale,
            "sparse_mode": self.sparse_mode,
            "block_size": self.block_size,
            "minimum_kv_block_pool_size": self.minimum_kv_block_pool_size,
            "input_layout": self.input_layout,
            "inner_precise": self.inner_precise,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class CannV2PagedPlanFactory:
    """Prepare CANN FIA v2 TND+PageAttention state without importing CANN."""

    provider_id = "cann"
    operation_id = CANN_V2_OPERATION_ID

    def prepare(self, plan, receipt, selection):
        _validate_prepare_authority(self.provider_id, plan, receipt, selection)
        reasons = explain_cann_v2_paged_plan(plan)
        if reasons:
            raise SchemaError(reasons[0])
        spec = plan.spec
        query_lengths, kv_lengths, indptr, indices, page_size = _paged_metadata(plan)
        causal_mask = None
        sparse_mode = 0
        if spec.effective_causal:
            causal_mask = AttentionOperatorTensorPlan(
                role="atten_mask",
                shape=(_CANN_CAUSAL_MASK_SIZE, _CANN_CAUSAL_MASK_SIZE),
                dtype="bool",
                materialization="right_down_causal_2048",
            )
            sparse_mode = 3
        state = CannV2PagedPlanState(
            query_cumulative_lengths=_cumulative(
                query_lengths, include_zero=False
            ),
            kv_sequence_lengths=kv_lengths,
            block_table=build_attention_dense_page_table(
                indptr, indices, role="block_table"
            ),
            causal_mask=causal_mask,
            num_query_heads=spec.num_qo_heads,
            num_key_value_heads=spec.num_kv_heads,
            softmax_scale=float(spec.sm_scale),
            sparse_mode=sparse_mode,
            block_size=page_size,
            minimum_kv_block_pool_size=sum(
                right - left for left, right in zip(indptr, indptr[1:])
            ),
        )
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="cann-v2-paged-%s" % state.fingerprint,
            opaque_state=state,
        )


class CannV2PagedRunAdapter:
    provider_id = "cann"

    def lower(self, active_plan, request):
        state = active_plan.prepared_plan.opaque_state
        if not isinstance(state, CannV2PagedPlanState):
            raise SchemaError("CANN v2 adapter received incompatible prepared state")
        key, value = _separate_kv_cache(request.kv_cache)
        _reject_unbound_run_options(request)
        keywords = (
            ("atten_mask", state.causal_mask),
            ("actual_seq_qlen", list(state.query_cumulative_lengths)),
            ("actual_seq_kvlen", list(state.kv_sequence_lengths)),
            ("block_table", state.block_table.tensor),
            ("num_query_heads", state.num_query_heads),
            ("num_key_value_heads", state.num_key_value_heads),
            ("softmax_scale", state.softmax_scale),
            ("pre_tokens", _INT32_MAX),
            ("next_tokens", _INT32_MAX),
            ("input_layout", state.input_layout),
            ("sparse_mode", state.sparse_mode),
            ("block_size", state.block_size),
            ("query_quant_mode", 0),
            ("key_quant_mode", 0),
            ("value_quant_mode", 0),
            ("inner_precise", state.inner_precise),
            ("return_softmax_lse", request.return_lse),
        )
        return AttentionLoweredOperatorCall(
            provider_id=self.provider_id,
            operation_id=CANN_V2_OPERATION_ID,
            active_plan_fingerprint=active_plan.fingerprint,
            positional_arguments=(
                ("query", request.query),
                ("key", key),
                ("value", value),
            ),
            keyword_arguments=keywords,
            return_names=(
                ("output", "softmax_lse")
                if request.return_lse
                else ("output",)
            ),
            consumed_request_fields=request.consumed_fields,
        )


@dataclass(frozen=True)
class FlashAttentionNpuV3PagedPlanState:
    cu_seqlens_q: AttentionOperatorTensorPlan
    cache_seqlens: AttentionOperatorTensorPlan
    page_table: AttentionDensePageTablePlan
    max_seqlen_q: int
    softmax_scale: float
    causal: bool
    window_size: Tuple[int, int]
    schema_version: int = ATTENTION_PROVIDER_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PROVIDER_ADAPTER_VERSION:
            raise SchemaError("unsupported flash-attention-npu v3 plan state version")
        if self.cu_seqlens_q.role != "cu_seqlens_q":
            raise SchemaError("invalid flash-attention-npu query offsets")
        if self.cache_seqlens.role != "cache_seqlens":
            raise SchemaError("invalid flash-attention-npu cache lengths")
        if self.page_table.tensor.role != "page_table":
            raise SchemaError("invalid flash-attention-npu page table")
        if self.max_seqlen_q <= 0:
            raise SchemaError("max_seqlen_q must be positive")
        if len(self.window_size) != 2:
            raise SchemaError("window_size must contain left and right bounds")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "cu_seqlens_q": self.cu_seqlens_q.to_dict(),
            "cache_seqlens": self.cache_seqlens.to_dict(),
            "page_table": self.page_table.to_dict(),
            "max_seqlen_q": self.max_seqlen_q,
            "softmax_scale": self.softmax_scale,
            "causal": self.causal,
            "window_size": list(self.window_size),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class FlashAttentionNpuV3PagedPlanFactory:
    """Prepare v3 paged-KV state without importing flash-attention-npu."""

    provider_id = "flash_attention_npu"
    operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID

    def prepare(self, plan, receipt, selection):
        _validate_prepare_authority(self.provider_id, plan, receipt, selection)
        reasons = explain_flash_attention_npu_v3_paged_plan(plan)
        if reasons:
            raise SchemaError(reasons[0])
        spec = plan.spec
        query_lengths, kv_lengths, indptr, indices, _ = _paged_metadata(plan)
        if spec.window_left == -1:
            window_size = (-1, -1)
        else:
            window_size = (spec.window_left, spec.window_right)
        state = FlashAttentionNpuV3PagedPlanState(
            cu_seqlens_q=_explicit_int32_tensor(
                "cu_seqlens_q", _cumulative(query_lengths, include_zero=True)
            ),
            cache_seqlens=_explicit_int32_tensor("cache_seqlens", kv_lengths),
            page_table=build_attention_dense_page_table(
                indptr, indices, role="page_table"
            ),
            max_seqlen_q=max(query_lengths),
            softmax_scale=float(spec.sm_scale),
            causal=spec.effective_causal,
            window_size=window_size,
        )
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="flash-attention-npu-v3-paged-%s" % state.fingerprint,
            opaque_state=state,
        )


class FlashAttentionNpuV3PagedRunAdapter:
    provider_id = "flash_attention_npu"

    def lower(self, active_plan, request):
        state = active_plan.prepared_plan.opaque_state
        if not isinstance(state, FlashAttentionNpuV3PagedPlanState):
            raise SchemaError(
                "flash-attention-npu v3 adapter received incompatible prepared state"
            )
        key, value = _separate_kv_cache(request.kv_cache)
        _reject_unbound_run_options(request)
        return AttentionLoweredOperatorCall(
            provider_id=self.provider_id,
            operation_id=FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            active_plan_fingerprint=active_plan.fingerprint,
            positional_arguments=(
                ("q", request.query),
                ("k_cache", key),
                ("v_cache", value),
            ),
            keyword_arguments=(
                ("cache_seqlens", state.cache_seqlens),
                ("page_table", state.page_table.tensor),
                ("cu_seqlens_q", state.cu_seqlens_q),
                ("max_seqlen_q", state.max_seqlen_q),
                ("softmax_scale", state.softmax_scale),
                ("causal", state.causal),
                ("window_size", state.window_size),
                ("softcap", 0.0),
                ("num_splits", 0),
                ("return_softmax_lse", request.return_lse),
            ),
            return_names=(
                ("output", "softmax_lse")
                if request.return_lse
                else ("output",)
            ),
            mutable_argument_names=("k_cache", "v_cache"),
            consumed_request_fields=request.consumed_fields,
        )


def _separate_kv_cache(kv_cache: Any) -> Tuple[Any, Any]:
    if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
        raise SchemaError("provider adapter requires a separate (key, value) KV cache")
    if kv_cache[0] is None or kv_cache[1] is None:
        raise SchemaError("provider adapter key and value must be provided")
    return kv_cache[0], kv_cache[1]


def _reject_unbound_run_options(request: AttentionOperatorRunRequest) -> None:
    if request.out is not None or request.lse is not None:
        raise SchemaError("provider adapter has no verified output-buffer binding")
    if request.k_scale is not None or request.v_scale is not None:
        raise SchemaError("provider adapter has no verified KV scale binding")
    if request.profiler_buffer is not None:
        raise SchemaError("provider adapter has no profiler-buffer binding")
    if request.kv_cache_sf is not None:
        raise SchemaError("provider adapter has no NVFP4 scale-factor binding")
    if not math.isclose(request.logits_soft_cap, 0.0, abs_tol=0.0):
        raise SchemaError("provider adapter does not support run-time logits soft cap")


__all__ = [
    "ATTENTION_PROVIDER_ADAPTER_VERSION",
    "CANN_V2_OPERATION_ID",
    "FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID",
    "AttentionDensePageTablePlan",
    "AttentionOperatorTensorPlan",
    "CannV2PagedPlanFactory",
    "CannV2PagedPlanGate",
    "CannV2PagedPlanState",
    "CannV2PagedRunAdapter",
    "FlashAttentionNpuV3PagedPlanFactory",
    "FlashAttentionNpuV3PagedPlanGate",
    "FlashAttentionNpuV3PagedPlanState",
    "FlashAttentionNpuV3PagedRunAdapter",
    "build_attention_dense_page_table",
    "explain_cann_v2_paged_plan",
    "explain_flash_attention_npu_v3_paged_plan",
]
