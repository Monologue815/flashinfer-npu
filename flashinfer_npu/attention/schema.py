"""Framework-independent Attention contracts.

This module models FlashInfer Attention metadata and shape semantics without
importing torch or requiring a device. It is used to validate the framework
layer before an Ascend C backend exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from flashinfer_npu.runtime import QuantSpec, SchemaError, WorkloadSpec


ATTENTION_SCHEMA_VERSION = 1


class KVLayout(str, Enum):
    NHD = "NHD"
    HND = "HND"


class PosEncodingMode(str, Enum):
    NONE = "NONE"
    ROPE_LLAMA = "ROPE_LLAMA"
    ALIBI = "ALIBI"


class AttentionMode(str, Enum):
    SINGLE_PREFILL = "single_prefill"
    SINGLE_DECODE = "single_decode"
    BATCH_PREFILL_PAGED = "batch_prefill_paged"
    BATCH_PREFILL_RAGGED = "batch_prefill_ragged"
    BATCH_DECODE_PAGED = "batch_decode_paged"
    BATCH_MIXED_PAGED = "batch_mixed_paged"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_construct(cls, data: Mapping[str, Any], name: str):
    values = dict(data)
    expected = set(cls.__dataclass_fields__)
    if set(values) != expected:
        raise SchemaError("%s fields do not match schema version 1" % name)
    try:
        return cls(**values)
    except TypeError as error:
        raise SchemaError("%s fields are invalid" % name) from error


def _as_int_tuple(name: str, values: Tuple[int, ...]) -> Tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise SchemaError("%s must contain integers" % name) from error


def _validate_indptr(name: str, values: Tuple[int, ...]) -> Tuple[int, ...]:
    result = _as_int_tuple(name, values)
    if len(result) < 2:
        raise SchemaError("%s must have shape [batch_size + 1]" % name)
    if result[0] != 0:
        raise SchemaError("%s must start at zero" % name)
    if any(current > following for current, following in zip(result, result[1:])):
        raise SchemaError("%s must be monotonically non-decreasing" % name)
    return result


def _segment_lengths(indptr: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(end - start for start, end in zip(indptr, indptr[1:]))


@dataclass(frozen=True)
class TensorSpec:
    """Shape/dtype/device metadata extracted from a frontend tensor."""

    shape: Tuple[int, ...]
    dtype: str
    device: str = "npu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _as_int_tuple("shape", self.shape))
        if any(dim < 0 for dim in self.shape):
            raise SchemaError("tensor shape cannot contain negative dimensions")
        if not self.dtype:
            raise SchemaError("tensor dtype must be non-empty")
        if not self.device:
            raise SchemaError("tensor device must be non-empty")


@dataclass(frozen=True)
class CustomMaskSpec:
    """Flattened custom attention mask passed during planning.

    Unpacked masks use boolean elements. Packed masks use one uint8 value per
    eight bits, with each request segment packed independently as FlashInfer's
    ``segment_packbits`` contract requires. FlashInfer Attention packs and
    consumes these segments with little-endian bit order.
    """

    numel: int
    packed: bool = False
    dtype: Optional[str] = None
    bit_order: Optional[str] = None

    def __post_init__(self) -> None:
        if self.numel < 0:
            raise SchemaError("custom mask numel cannot be negative")
        expected_dtype = "uint8" if self.packed else "bool"
        if self.dtype is None:
            object.__setattr__(self, "dtype", expected_dtype)
        elif self.dtype != expected_dtype:
            raise SchemaError(
                "%s custom mask must use dtype %s"
                % ("packed" if self.packed else "unpacked", expected_dtype)
            )
        if self.packed:
            if self.bit_order is None:
                object.__setattr__(self, "bit_order", "little")
            elif self.bit_order != "little":
                raise SchemaError(
                    "Attention packed custom mask must use little bit order"
                )
        elif self.bit_order is not None:
            raise SchemaError("unpacked custom mask does not define bit order")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numel": self.numel,
            "packed": self.packed,
            "dtype": self.dtype,
            "bit_order": self.bit_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CustomMaskSpec":
        return _strict_construct(cls, value, "CustomMaskSpec")


@dataclass(frozen=True)
class PagedKVMetadata:
    """CSR page table used by batch prefill and decode wrappers."""

    indptr: Tuple[int, ...]
    indices: Tuple[int, ...]
    last_page_len: Tuple[int, ...]
    page_size: int
    schema_version: int = ATTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise SchemaError("unsupported PagedKVMetadata schema version")
        object.__setattr__(self, "indptr", _validate_indptr("indptr", self.indptr))
        object.__setattr__(self, "indices", _as_int_tuple("indices", self.indices))
        object.__setattr__(
            self,
            "last_page_len",
            _as_int_tuple("last_page_len", self.last_page_len),
        )
        if self.page_size <= 0:
            raise SchemaError("page_size must be positive")
        if self.indptr[-1] != len(self.indices):
            raise SchemaError("indices length must equal indptr[-1]")
        if len(self.last_page_len) != self.batch_size:
            raise SchemaError("last_page_len must have shape [batch_size]")
        if any(index < 0 for index in self.indices):
            raise SchemaError("page indices cannot be negative")
        for request, (pages, last_len) in enumerate(
            zip(self.page_counts, self.last_page_len)
        ):
            if pages == 0 and last_len != 0:
                raise SchemaError(
                    "empty request %d must have last_page_len=0" % request
                )
            if pages > 0 and not 1 <= last_len <= self.page_size:
                raise SchemaError(
                    "request %d last_page_len must be in [1, page_size]" % request
                )

    @property
    def batch_size(self) -> int:
        return len(self.indptr) - 1

    @property
    def page_counts(self) -> Tuple[int, ...]:
        return _segment_lengths(self.indptr)

    @property
    def sequence_lengths(self) -> Tuple[int, ...]:
        return tuple(
            0 if pages == 0 else (pages - 1) * self.page_size + last_len
            for pages, last_len in zip(self.page_counts, self.last_page_len)
        )

    @property
    def max_page_index(self) -> int:
        return max(self.indices, default=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "paged",
            "indptr": list(self.indptr),
            "indices": list(self.indices),
            "last_page_len": list(self.last_page_len),
            "page_size": self.page_size,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class RaggedKVMetadata:
    """CSR token offsets used by ragged prefill."""

    qo_indptr: Tuple[int, ...]
    kv_indptr: Tuple[int, ...]
    schema_version: int = ATTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise SchemaError("unsupported RaggedKVMetadata schema version")
        object.__setattr__(
            self, "qo_indptr", _validate_indptr("qo_indptr", self.qo_indptr)
        )
        object.__setattr__(
            self, "kv_indptr", _validate_indptr("kv_indptr", self.kv_indptr)
        )
        if len(self.qo_indptr) != len(self.kv_indptr):
            raise SchemaError("qo_indptr and kv_indptr must have the same batch size")

    @property
    def batch_size(self) -> int:
        return len(self.qo_indptr) - 1

    @property
    def qo_lengths(self) -> Tuple[int, ...]:
        return _segment_lengths(self.qo_indptr)

    @property
    def kv_lengths(self) -> Tuple[int, ...]:
        return _segment_lengths(self.kv_indptr)

    @property
    def total_qo_tokens(self) -> int:
        return self.qo_indptr[-1]

    @property
    def total_kv_tokens(self) -> int:
        return self.kv_indptr[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "ragged",
            "qo_indptr": list(self.qo_indptr),
            "kv_indptr": list(self.kv_indptr),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class PagedPrefillMetadata:
    """Query offsets plus a paged KV table for batch prefill."""

    qo_indptr: Tuple[int, ...]
    paged_kv: PagedKVMetadata
    schema_version: int = ATTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise SchemaError("unsupported PagedPrefillMetadata schema version")
        object.__setattr__(
            self, "qo_indptr", _validate_indptr("qo_indptr", self.qo_indptr)
        )
        if len(self.qo_indptr) - 1 != self.paged_kv.batch_size:
            raise SchemaError("qo_indptr and paged KV metadata must share batch size")

    @property
    def batch_size(self) -> int:
        return len(self.qo_indptr) - 1

    @property
    def qo_lengths(self) -> Tuple[int, ...]:
        return _segment_lengths(self.qo_indptr)

    @property
    def total_qo_tokens(self) -> int:
        return self.qo_indptr[-1]

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "paged_prefill",
            "qo_indptr": list(self.qo_indptr),
            "paged_kv": self.paged_kv.to_dict(),
        }


@dataclass(frozen=True)
class SingleAttentionMetadata:
    """Sequence lengths for single-request prefill or decode."""

    qo_len: int
    kv_len: int
    schema_version: int = ATTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise SchemaError("unsupported SingleAttentionMetadata schema version")
        if self.qo_len <= 0:
            raise SchemaError("qo_len must be positive")
        if self.kv_len <= 0:
            raise SchemaError("kv_len must be positive")

    @property
    def batch_size(self) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "single",
            "qo_len": self.qo_len,
            "kv_len": self.kv_len,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class MixedPagedKVMetadata:
    """Metadata used by FlashInfer's holistic mixed BatchAttention."""

    qo_indptr: Tuple[int, ...]
    kv_indptr: Tuple[int, ...]
    kv_indices: Tuple[int, ...]
    kv_len_arr: Tuple[int, ...]
    page_size: int
    schema_version: int = ATTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise SchemaError("unsupported MixedPagedKVMetadata schema version")
        object.__setattr__(
            self, "qo_indptr", _validate_indptr("qo_indptr", self.qo_indptr)
        )
        object.__setattr__(
            self, "kv_indptr", _validate_indptr("kv_indptr", self.kv_indptr)
        )
        object.__setattr__(
            self, "kv_indices", _as_int_tuple("kv_indices", self.kv_indices)
        )
        object.__setattr__(
            self, "kv_len_arr", _as_int_tuple("kv_len_arr", self.kv_len_arr)
        )
        if self.page_size <= 0:
            raise SchemaError("page_size must be positive")
        if len(self.qo_indptr) != len(self.kv_indptr):
            raise SchemaError("qo_indptr and kv_indptr must have the same batch size")
        if len(self.kv_len_arr) != self.batch_size:
            raise SchemaError("kv_len_arr must have shape [batch_size]")
        if self.kv_indptr[-1] != len(self.kv_indices):
            raise SchemaError("kv_indices length must equal kv_indptr[-1]")
        if any(index < 0 for index in self.kv_indices):
            raise SchemaError("kv_indices cannot contain negative values")
        for request, (pages, kv_len) in enumerate(
            zip(self.page_counts, self.kv_len_arr)
        ):
            if pages == 0 and kv_len != 0:
                raise SchemaError("empty request %d must have kv_len=0" % request)
            if pages > 0:
                minimum = (pages - 1) * self.page_size + 1
                maximum = pages * self.page_size
                if not minimum <= kv_len <= maximum:
                    raise SchemaError(
                        "request %d kv_len must fit allocated pages" % request
                    )

    @property
    def batch_size(self) -> int:
        return len(self.qo_indptr) - 1

    @property
    def qo_lengths(self) -> Tuple[int, ...]:
        return _segment_lengths(self.qo_indptr)

    @property
    def page_counts(self) -> Tuple[int, ...]:
        return _segment_lengths(self.kv_indptr)

    @property
    def total_qo_tokens(self) -> int:
        return self.qo_indptr[-1]

    @property
    def max_page_index(self) -> int:
        return max(self.kv_indices, default=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "mixed_paged",
            "qo_indptr": list(self.qo_indptr),
            "kv_indptr": list(self.kv_indptr),
            "kv_indices": list(self.kv_indices),
            "kv_len_arr": list(self.kv_len_arr),
            "page_size": self.page_size,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


AttentionMetadata = Union[
    SingleAttentionMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVMetadata,
    MixedPagedKVMetadata,
]


def attention_metadata_from_dict(value: Mapping[str, Any]) -> AttentionMetadata:
    """Restore a versioned metadata object from its canonical dictionary."""

    data = dict(value)
    kind = data.pop("kind", None)
    metadata_types = {
        "single": SingleAttentionMetadata,
        "paged": PagedKVMetadata,
        "ragged": RaggedKVMetadata,
        "mixed_paged": MixedPagedKVMetadata,
    }
    if kind == "paged_prefill":
        paged_value = data.get("paged_kv")
        if not isinstance(paged_value, Mapping):
            raise SchemaError("paged_prefill metadata requires paged_kv")
        data["paged_kv"] = attention_metadata_from_dict(paged_value)
        if not isinstance(data["paged_kv"], PagedKVMetadata):
            raise SchemaError("paged_prefill paged_kv must be paged metadata")
        return _strict_construct(
            PagedPrefillMetadata, data, "PagedPrefillMetadata"
        )
    metadata_type = metadata_types.get(kind)
    if metadata_type is None:
        raise SchemaError("unsupported Attention metadata kind: %r" % kind)
    return _strict_construct(metadata_type, data, metadata_type.__name__)


@dataclass(frozen=True)
class PagedKVCacheSpec:
    """Logical paged KV cache storage contract.

    ``structure`` is either ``"packed"`` for a combined K/V allocation or
    ``"separate"`` for a ``(k_cache, v_cache)`` pair.
    """

    num_pages: int
    page_size: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    dtype: str
    layout: KVLayout = KVLayout.NHD
    structure: str = "packed"
    device: str = "npu"
    quant_spec: Optional[QuantSpec] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", KVLayout(self.layout))
        if self.num_pages < 0:
            raise SchemaError("num_pages cannot be negative")
        for name in ("page_size", "num_kv_heads", "head_dim_qk", "head_dim_vo"):
            if getattr(self, name) <= 0:
                raise SchemaError("%s must be positive" % name)
        if self.structure not in {"packed", "separate"}:
            raise SchemaError("KV cache structure must be packed or separate")
        if self.structure == "packed" and self.head_dim_qk != self.head_dim_vo:
            raise SchemaError(
                "packed KV cache requires equal key and value head dimensions"
            )
        if not self.dtype or not self.device:
            raise SchemaError("KV cache dtype and device must be non-empty")
        if (
            self.quant_spec is not None
            and self.quant_spec.storage_dtype != self.dtype
        ):
            raise SchemaError("KV dtype must match quantized storage dtype")

    @property
    def expected_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        if self.layout == KVLayout.NHD:
            key_shape = (
                self.num_pages,
                self.page_size,
                self.num_kv_heads,
                self.head_dim_qk,
            )
            value_shape = (
                self.num_pages,
                self.page_size,
                self.num_kv_heads,
                self.head_dim_vo,
            )
            packed_shape = (
                self.num_pages,
                2,
                self.page_size,
                self.num_kv_heads,
                self.head_dim_qk,
            )
        else:
            key_shape = (
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim_qk,
            )
            value_shape = (
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim_vo,
            )
            packed_shape = (
                self.num_pages,
                2,
                self.num_kv_heads,
                self.page_size,
                self.head_dim_qk,
            )
        return (packed_shape,) if self.structure == "packed" else (key_shape, value_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "paged",
            "num_pages": self.num_pages,
            "page_size": self.page_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_vo": self.head_dim_vo,
            "dtype": self.dtype,
            "layout": self.layout.value,
            "structure": self.structure,
            "device": self.device,
            "quant_spec": (
                self.quant_spec.to_dict() if self.quant_spec is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PagedKVCacheSpec":
        data = dict(value)
        if data.pop("kind", None) != "paged":
            raise SchemaError("paged KV cache dictionary requires kind='paged'")
        quant_value = data.get("quant_spec")
        if quant_value is not None:
            if not isinstance(quant_value, Mapping):
                raise SchemaError("quant_spec must be a dictionary")
            data["quant_spec"] = QuantSpec.from_dict(quant_value)
        return _strict_construct(cls, data, "PagedKVCacheSpec")


@dataclass(frozen=True)
class RaggedKVCacheSpec:
    total_kv_tokens: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    dtype: str
    layout: KVLayout = KVLayout.NHD
    device: str = "npu"
    quant_spec: Optional[QuantSpec] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", KVLayout(self.layout))
        if self.total_kv_tokens < 0:
            raise SchemaError("total_kv_tokens cannot be negative")
        for name in ("num_kv_heads", "head_dim_qk", "head_dim_vo"):
            if getattr(self, name) <= 0:
                raise SchemaError("%s must be positive" % name)
        if (
            self.quant_spec is not None
            and self.quant_spec.storage_dtype != self.dtype
        ):
            raise SchemaError("KV dtype must match quantized storage dtype")

    @property
    def expected_shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self.layout == KVLayout.NHD:
            return (
                (self.total_kv_tokens, self.num_kv_heads, self.head_dim_qk),
                (self.total_kv_tokens, self.num_kv_heads, self.head_dim_vo),
            )
        return (
            (self.num_kv_heads, self.total_kv_tokens, self.head_dim_qk),
            (self.num_kv_heads, self.total_kv_tokens, self.head_dim_vo),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "ragged",
            "total_kv_tokens": self.total_kv_tokens,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_vo": self.head_dim_vo,
            "dtype": self.dtype,
            "layout": self.layout.value,
            "device": self.device,
            "quant_spec": (
                self.quant_spec.to_dict() if self.quant_spec is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RaggedKVCacheSpec":
        data = dict(value)
        if data.pop("kind", None) != "ragged":
            raise SchemaError("ragged KV cache dictionary requires kind='ragged'")
        quant_value = data.get("quant_spec")
        if quant_value is not None:
            if not isinstance(quant_value, Mapping):
                raise SchemaError("quant_spec must be a dictionary")
            data["quant_spec"] = QuantSpec.from_dict(quant_value)
        return _strict_construct(cls, data, "RaggedKVCacheSpec")


KVCacheSpec = Union[PagedKVCacheSpec, RaggedKVCacheSpec]


@dataclass(frozen=True)
class AttentionPlanSpec:
    mode: AttentionMode
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: Optional[int] = None
    kv_layout: KVLayout = KVLayout.NHD
    causal: bool = False
    pos_encoding_mode: PosEncodingMode = PosEncodingMode.NONE
    q_dtype: str = "float16"
    kv_dtype: Optional[str] = None
    o_dtype: Optional[str] = None
    sm_scale: Optional[float] = None
    logits_soft_cap: Optional[float] = None
    window_left: int = -1
    window_right: int = 0
    q_len_per_req: int = 1
    rope_scale: Optional[float] = None
    rope_theta: Optional[float] = None
    use_fp16_qk_reduction: bool = False
    use_profiler: bool = False
    custom_mask: Optional[CustomMaskSpec] = None
    kv_quant_spec: Optional[QuantSpec] = None
    schema_version: int = ATTENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", AttentionMode(self.mode))
        object.__setattr__(self, "kv_layout", KVLayout(self.kv_layout))
        object.__setattr__(
            self, "pos_encoding_mode", PosEncodingMode(self.pos_encoding_mode)
        )
        if self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise SchemaError("unsupported AttentionPlanSpec schema version")
        if self.num_qo_heads <= 0 or self.num_kv_heads <= 0:
            raise SchemaError("attention head counts must be positive")
        if self.num_qo_heads % self.num_kv_heads != 0:
            raise SchemaError("num_kv_heads must divide num_qo_heads")
        if self.head_dim_qk <= 0:
            raise SchemaError("head_dim_qk must be positive")
        if (
            self.pos_encoding_mode == PosEncodingMode.ROPE_LLAMA
            and self.head_dim_qk % 2 != 0
        ):
            raise SchemaError("ROPE_LLAMA requires an even head_dim_qk")
        if self.head_dim_vo is None:
            object.__setattr__(self, "head_dim_vo", self.head_dim_qk)
        if self.head_dim_vo is None or self.head_dim_vo <= 0:
            raise SchemaError("head_dim_vo must be positive")
        if not self.q_dtype:
            raise SchemaError("q_dtype must be non-empty")
        if self.kv_dtype is None:
            object.__setattr__(self, "kv_dtype", self.q_dtype)
        if (
            self.kv_quant_spec is not None
            and self.kv_quant_spec.storage_dtype != self.kv_dtype
        ):
            raise SchemaError("kv_dtype must match quantized storage dtype")
        if self.o_dtype is None:
            object.__setattr__(self, "o_dtype", self.q_dtype)
        if self.window_left < -1:
            raise SchemaError("window_left must be -1 or non-negative")
        if self.window_right < -1:
            raise SchemaError("window_right must be -1 or non-negative")
        if self.q_len_per_req < 1:
            raise SchemaError("q_len_per_req must be at least one")
        if (
            self.mode != AttentionMode.BATCH_DECODE_PAGED
            and self.q_len_per_req != 1
        ):
            raise SchemaError(
                "q_len_per_req is only defined for batch paged decode"
            )
        if (
            self.mode == AttentionMode.BATCH_DECODE_PAGED
            and self.q_len_per_req > 1
        ):
            object.__setattr__(self, "causal", True)
        if self.sm_scale is None:
            object.__setattr__(self, "sm_scale", 1.0 / math.sqrt(self.head_dim_qk))
        elif not math.isfinite(self.sm_scale) or self.sm_scale <= 0:
            raise SchemaError("sm_scale must be finite and positive")
        if self.logits_soft_cap is None:
            object.__setattr__(self, "logits_soft_cap", 0.0)
        elif not math.isfinite(self.logits_soft_cap) or self.logits_soft_cap < 0:
            raise SchemaError("logits_soft_cap must be finite and non-negative")
        if self.rope_scale is None:
            object.__setattr__(self, "rope_scale", 1.0)
        elif not math.isfinite(self.rope_scale) or self.rope_scale <= 0:
            raise SchemaError("rope_scale must be finite and positive")
        if self.rope_theta is None:
            object.__setattr__(self, "rope_theta", 1e4)
        elif not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise SchemaError("rope_theta must be finite and positive")
        if (
            self.mode == AttentionMode.BATCH_MIXED_PAGED
            and self.pos_encoding_mode != PosEncodingMode.NONE
        ):
            raise SchemaError("BatchAttention currently supports pos encoding NONE only")

    def validate_metadata(self, metadata: AttentionMetadata) -> None:
        expected_type = {
            AttentionMode.SINGLE_PREFILL: SingleAttentionMetadata,
            AttentionMode.SINGLE_DECODE: SingleAttentionMetadata,
            AttentionMode.BATCH_PREFILL_PAGED: PagedPrefillMetadata,
            AttentionMode.BATCH_DECODE_PAGED: PagedKVMetadata,
            AttentionMode.BATCH_PREFILL_RAGGED: RaggedKVMetadata,
            AttentionMode.BATCH_MIXED_PAGED: MixedPagedKVMetadata,
        }.get(self.mode)
        if expected_type is None:
            raise SchemaError("unsupported attention mode")
        if not isinstance(metadata, expected_type):
            raise SchemaError(
                "%s requires %s" % (self.mode.value, expected_type.__name__)
            )
        self._validate_custom_mask(metadata)
        if self.effective_causal:
            for request, (qo_len, kv_len) in enumerate(
                self._query_kv_lengths(metadata)
            ):
                if kv_len < qo_len:
                    raise SchemaError(
                        "causal request %d requires kv_len >= qo_len" % request
                    )

    @property
    def effective_causal(self) -> bool:
        return self.causal and self.custom_mask is None

    def _query_kv_lengths(
        self, metadata: AttentionMetadata
    ) -> Tuple[Tuple[int, int], ...]:
        if isinstance(metadata, SingleAttentionMetadata):
            return ((metadata.qo_len, metadata.kv_len),)
        if isinstance(metadata, PagedPrefillMetadata):
            return tuple(
                zip(metadata.qo_lengths, metadata.paged_kv.sequence_lengths)
            )
        if isinstance(metadata, RaggedKVMetadata):
            return tuple(zip(metadata.qo_lengths, metadata.kv_lengths))
        if isinstance(metadata, PagedKVMetadata):
            return tuple(
                (self.q_len_per_req, kv_len)
                for kv_len in metadata.sequence_lengths
            )
        return tuple(zip(metadata.qo_lengths, metadata.kv_len_arr))

    def _mask_segment_sizes(
        self, metadata: AttentionMetadata
    ) -> Tuple[int, ...]:
        if isinstance(metadata, SingleAttentionMetadata):
            return (metadata.qo_len * metadata.kv_len,)
        if isinstance(metadata, PagedPrefillMetadata):
            return tuple(
                qo_len * kv_len
                for qo_len, kv_len in zip(
                    metadata.qo_lengths, metadata.paged_kv.sequence_lengths
                )
            )
        if isinstance(metadata, RaggedKVMetadata):
            return tuple(
                qo_len * kv_len
                for qo_len, kv_len in zip(metadata.qo_lengths, metadata.kv_lengths)
            )
        raise SchemaError("custom masks are supported by prefill modes only")

    def _validate_custom_mask(self, metadata: AttentionMetadata) -> None:
        if self.custom_mask is None:
            return
        if self.mode not in {
            AttentionMode.SINGLE_PREFILL,
            AttentionMode.BATCH_PREFILL_PAGED,
            AttentionMode.BATCH_PREFILL_RAGGED,
        }:
            raise SchemaError("custom masks are supported by prefill modes only")
        segment_sizes = self._mask_segment_sizes(metadata)
        if self.custom_mask.packed:
            expected_numel = sum((size + 7) // 8 for size in segment_sizes)
        else:
            expected_numel = sum(segment_sizes)
        if self.custom_mask.numel != expected_numel:
            raise SchemaError(
                "custom mask numel must be %d, got %d"
                % (expected_numel, self.custom_mask.numel)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_vo": self.head_dim_vo,
            "kv_layout": self.kv_layout.value,
            "causal": self.causal,
            "pos_encoding_mode": self.pos_encoding_mode.value,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "o_dtype": self.o_dtype,
            "sm_scale": self.sm_scale,
            "logits_soft_cap": self.logits_soft_cap,
            "window_left": self.window_left,
            "window_right": self.window_right,
            "q_len_per_req": self.q_len_per_req,
            "rope_scale": self.rope_scale,
            "rope_theta": self.rope_theta,
            "use_fp16_qk_reduction": self.use_fp16_qk_reduction,
            "use_profiler": self.use_profiler,
            "custom_mask": (
                self.custom_mask.to_dict() if self.custom_mask is not None else None
            ),
            "kv_quant_spec": (
                self.kv_quant_spec.to_dict()
                if self.kv_quant_spec is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionPlanSpec":
        data = dict(value)
        mask_value = data.get("custom_mask")
        if mask_value is not None:
            if not isinstance(mask_value, Mapping):
                raise SchemaError("custom_mask must be a dictionary")
            data["custom_mask"] = CustomMaskSpec.from_dict(mask_value)
        quant_value = data.get("kv_quant_spec")
        if quant_value is not None:
            if not isinstance(quant_value, Mapping):
                raise SchemaError("kv_quant_spec must be a dictionary")
            data["kv_quant_spec"] = QuantSpec.from_dict(quant_value)
        return _strict_construct(cls, data, "AttentionPlanSpec")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_workload_spec(self, metadata: AttentionMetadata) -> WorkloadSpec:
        self.validate_metadata(metadata)
        if isinstance(metadata, SingleAttentionMetadata):
            if self.mode == AttentionMode.SINGLE_DECODE and metadata.qo_len != 1:
                raise SchemaError("single decode requires qo_len=1")
            total_qo = metadata.qo_len
            total_kv = metadata.kv_len
            page_size = 0
            batch_size = 1
        elif isinstance(metadata, PagedKVMetadata):
            total_qo = metadata.batch_size * self.q_len_per_req
            total_kv = sum(metadata.sequence_lengths)
            page_size = metadata.page_size
            batch_size = metadata.batch_size
        elif isinstance(metadata, PagedPrefillMetadata):
            total_qo = metadata.total_qo_tokens
            total_kv = sum(metadata.paged_kv.sequence_lengths)
            page_size = metadata.paged_kv.page_size
            batch_size = metadata.batch_size
        elif isinstance(metadata, RaggedKVMetadata):
            total_qo = metadata.total_qo_tokens
            total_kv = metadata.total_kv_tokens
            page_size = 0
            batch_size = metadata.batch_size
        else:
            total_qo = metadata.total_qo_tokens
            total_kv = sum(metadata.kv_len_arr)
            page_size = metadata.page_size
            batch_size = metadata.batch_size
        quant_specs = (self.kv_quant_spec,) if self.kv_quant_spec is not None else ()
        return WorkloadSpec(
            op="attention.%s" % self.mode.value,
            dtypes=(self.q_dtype, self.kv_dtype or self.q_dtype, self.o_dtype or self.q_dtype),
            layouts=(self.kv_layout.value,),
            static_dims=(
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim_qk,
                int(self.head_dim_vo),
                page_size,
            ),
            dynamic_bounds=(batch_size, total_qo, total_kv),
            quant_specs=quant_specs,
            causal=self.effective_causal,
            pos_encoding=self.pos_encoding_mode.value,
            attributes=(
                ("metadata_fingerprint", metadata.fingerprint),
                ("window_left", str(self.window_left)),
                ("window_right", str(self.window_right)),
                ("q_len_per_req", str(self.q_len_per_req)),
                ("logits_soft_cap", str(self.logits_soft_cap)),
                ("rope_scale", str(self.rope_scale)),
                ("rope_theta", str(self.rope_theta)),
                ("use_fp16_qk_reduction", str(self.use_fp16_qk_reduction)),
                (
                    "custom_mask",
                    "none"
                    if self.custom_mask is None
                    else (
                        "packed_little"
                        if self.custom_mask.packed
                        else "unpacked"
                    ),
                ),
            ),
        )
