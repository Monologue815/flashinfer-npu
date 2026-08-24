"""Canonical mode-specific Attention plan metadata wire format."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import CFieldABI, CPrimitive, CStructABI, SchemaError

from .launch_contract import (
    ATTENTION_PLAN_METADATA_HEADER_C_ABI,
    AttentionDTypeCode,
    AttentionKVLayoutCode,
    AttentionModeCode,
    AttentionPlanFlags,
    AttentionPositionEncodingCode,
    attention_dtype_code,
    attention_kernel_binary_abi,
)
from .planner import AttentionFrameworkPlan
from .resource_limits import measure_attention_resources
from .schema import (
    AttentionMetadata,
    AttentionMode,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    RaggedKVMetadata,
    SingleAttentionMetadata,
)


ATTENTION_PLAN_METADATA_WIRE_VERSION = 1


class AttentionMetadataSectionKind(IntEnum):
    QO_INDPTR = 1
    KV_INDPTR = 2
    KV_INDICES = 3
    LAST_PAGE_LEN = 4
    KV_LEN = 5
    MASK_INDPTR = 6


class AttentionMetadataElementType(IntEnum):
    INT32 = 1

    @property
    def itemsize(self) -> int:
        return 4


ATTENTION_PLAN_CONFIG_C_ABI = CStructABI(
    name="FlashInferNpuAttentionPlanConfigV1",
    fields=(
        CFieldABI("section_count", CPrimitive.U32),
        CFieldABI("directory_entry_nbytes", CPrimitive.U32),
        CFieldABI("batch_size", CPrimitive.U32),
        CFieldABI("num_qo_heads", CPrimitive.U32),
        CFieldABI("num_kv_heads", CPrimitive.U32),
        CFieldABI("head_dim_qk", CPrimitive.U32),
        CFieldABI("head_dim_vo", CPrimitive.U32),
        CFieldABI("page_size", CPrimitive.U32),
        CFieldABI("q_len_per_req", CPrimitive.U32),
        CFieldABI("pos_encoding_code", CPrimitive.U16),
        CFieldABI("kv_layout_code", CPrimitive.U16),
        CFieldABI("q_dtype_code", CPrimitive.U16),
        CFieldABI("kv_dtype_code", CPrimitive.U16),
        CFieldABI("o_dtype_code", CPrimitive.U16),
        CFieldABI("index_dtype_code", CPrimitive.U16),
        CFieldABI("window_left", CPrimitive.I32),
        CFieldABI("window_right", CPrimitive.I32),
        CFieldABI("sm_scale", CPrimitive.F64),
        CFieldABI("logits_soft_cap", CPrimitive.F64),
        CFieldABI("rope_scale", CPrimitive.F64),
        CFieldABI("rope_theta", CPrimitive.F64),
        CFieldABI("total_qo_tokens", CPrimitive.U64),
        CFieldABI("total_kv_tokens", CPrimitive.U64),
        CFieldABI("custom_mask_numel", CPrimitive.U64),
        CFieldABI("reserved", CPrimitive.U8, 16, reserved=True),
    ),
)


ATTENTION_METADATA_SECTION_C_ABI = CStructABI(
    name="FlashInferNpuAttentionMetadataSectionV1",
    fields=(
        CFieldABI("kind_code", CPrimitive.U16),
        CFieldABI("element_type_code", CPrimitive.U16),
        CFieldABI("flags", CPrimitive.U32),
        CFieldABI("offset_bytes", CPrimitive.U64),
        CFieldABI("element_count", CPrimitive.U64),
        CFieldABI("nbytes", CPrimitive.U64),
    ),
)


_MODE_CODES = {
    AttentionMode.SINGLE_PREFILL: AttentionModeCode.SINGLE_PREFILL,
    AttentionMode.SINGLE_DECODE: AttentionModeCode.SINGLE_DECODE,
    AttentionMode.BATCH_PREFILL_PAGED: AttentionModeCode.BATCH_PREFILL_PAGED,
    AttentionMode.BATCH_PREFILL_RAGGED: AttentionModeCode.BATCH_PREFILL_RAGGED,
    AttentionMode.BATCH_DECODE_PAGED: AttentionModeCode.BATCH_DECODE_PAGED,
    AttentionMode.BATCH_MIXED_PAGED: AttentionModeCode.BATCH_MIXED_PAGED,
}
_LAYOUT_CODES = {KVLayout.NHD: AttentionKVLayoutCode.NHD, KVLayout.HND: AttentionKVLayoutCode.HND}
_POSITION_CODES = {
    PosEncodingMode.NONE: AttentionPositionEncodingCode.NONE,
    PosEncodingMode.ROPE_LLAMA: AttentionPositionEncodingCode.ROPE_LLAMA,
    PosEncodingMode.ALIBI: AttentionPositionEncodingCode.ALIBI,
}
_BASE_SECTIONS = {
    AttentionModeCode.SINGLE_PREFILL: (),
    AttentionModeCode.SINGLE_DECODE: (),
    AttentionModeCode.BATCH_PREFILL_PAGED: (
        AttentionMetadataSectionKind.QO_INDPTR,
        AttentionMetadataSectionKind.KV_INDPTR,
        AttentionMetadataSectionKind.KV_INDICES,
        AttentionMetadataSectionKind.LAST_PAGE_LEN,
    ),
    AttentionModeCode.BATCH_PREFILL_RAGGED: (
        AttentionMetadataSectionKind.QO_INDPTR,
        AttentionMetadataSectionKind.KV_INDPTR,
    ),
    AttentionModeCode.BATCH_DECODE_PAGED: (
        AttentionMetadataSectionKind.KV_INDPTR,
        AttentionMetadataSectionKind.KV_INDICES,
        AttentionMetadataSectionKind.LAST_PAGE_LEN,
    ),
    AttentionModeCode.BATCH_MIXED_PAGED: (
        AttentionMetadataSectionKind.QO_INDPTR,
        AttentionMetadataSectionKind.KV_INDPTR,
        AttentionMetadataSectionKind.KV_INDICES,
        AttentionMetadataSectionKind.KV_LEN,
    ),
}
_KNOWN_PLAN_FLAGS = sum(int(item) for item in AttentionPlanFlags)


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _hash_bytes(value: str) -> Tuple[int, ...]:
    _require_hash("fingerprint", value)
    return tuple(bytes.fromhex(value))


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _checked_i32(values: Sequence[int], name: str) -> Tuple[int, ...]:
    result = tuple(int(item) for item in values)
    if any(item < -(1 << 31) or item >= 1 << 31 for item in result):
        raise SchemaError("%s contains a value outside int32" % name)
    return result


@dataclass(frozen=True)
class AttentionPlanMetadataDecodeLimits:
    max_total_nbytes: int = 64 * 1024 * 1024
    max_section_count: int = 16
    max_section_elements: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SchemaError("metadata wire decode limits must be positive integers")


DEFAULT_ATTENTION_PLAN_METADATA_DECODE_LIMITS = AttentionPlanMetadataDecodeLimits()


@dataclass(frozen=True)
class AttentionMetadataSection:
    kind: AttentionMetadataSectionKind
    values: Tuple[int, ...]
    element_type: AttentionMetadataElementType = AttentionMetadataElementType.INT32
    flags: int = 0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", AttentionMetadataSectionKind(self.kind))
            object.__setattr__(
                self, "element_type", AttentionMetadataElementType(self.element_type)
            )
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention metadata section enum is invalid") from error
        object.__setattr__(self, "values", _checked_i32(self.values, self.kind.name))
        if self.flags != 0:
            raise SchemaError("Attention metadata section v1 flags must be zero")

    @property
    def payload(self) -> bytes:
        try:
            return struct.pack("<%di" % len(self.values), *self.values)
        except struct.error as error:  # pragma: no cover - guarded by _checked_i32
            raise SchemaError("Attention metadata section cannot be packed") from error


@dataclass(frozen=True)
class AttentionPlanWireConfig:
    batch_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    q_len_per_req: int
    pos_encoding_code: AttentionPositionEncodingCode
    kv_layout_code: AttentionKVLayoutCode
    q_dtype_code: AttentionDTypeCode
    kv_dtype_code: AttentionDTypeCode
    o_dtype_code: AttentionDTypeCode
    index_dtype_code: AttentionDTypeCode
    window_left: int
    window_right: int
    sm_scale: float
    logits_soft_cap: float
    rope_scale: float
    rope_theta: float
    total_qo_tokens: int
    total_kv_tokens: int
    custom_mask_numel: int

    def __post_init__(self) -> None:
        for name in (
            "batch_size", "num_qo_heads", "num_kv_heads", "head_dim_qk",
            "head_dim_vo", "q_len_per_req",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SchemaError("Attention plan wire dimensions must be positive")
        for name in ("page_size", "total_qo_tokens", "total_kv_tokens", "custom_mask_numel"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SchemaError("Attention plan wire counts cannot be negative")
        for name in ("window_left", "window_right"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SchemaError("Attention plan window must be an integer")
            if value < -(1 << 31) or value >= 1 << 31:
                raise SchemaError("Attention plan window is outside int32")
        for name in ("sm_scale", "logits_soft_cap", "rope_scale", "rope_theta"):
            if not math.isfinite(float(getattr(self, name))):
                raise SchemaError("Attention plan wire scalars must be finite")
        try:
            object.__setattr__(self, "pos_encoding_code", AttentionPositionEncodingCode(self.pos_encoding_code))
            object.__setattr__(self, "kv_layout_code", AttentionKVLayoutCode(self.kv_layout_code))
            for name in ("q_dtype_code", "kv_dtype_code", "o_dtype_code", "index_dtype_code"):
                object.__setattr__(self, name, AttentionDTypeCode(getattr(self, name)))
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention plan wire enum is invalid") from error
        if self.index_dtype_code != AttentionDTypeCode.INT32:
            raise SchemaError("Attention metadata wire v1 requires int32 indices")

    def pod_values(self, section_count: int) -> Dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name in ("pos_encoding_code", "kv_layout_code", "q_dtype_code", "kv_dtype_code", "o_dtype_code", "index_dtype_code"):
            result[name] = int(result[name])
        result["section_count"] = section_count
        result["directory_entry_nbytes"] = ATTENTION_METADATA_SECTION_C_ABI.size_bytes
        return result

    @classmethod
    def from_pod(cls, value: Mapping[str, object]) -> "AttentionPlanWireConfig":
        data = dict(value)
        data.pop("reserved", None)
        data.pop("section_count", None)
        data.pop("directory_entry_nbytes", None)
        return cls(**data)


@dataclass(frozen=True)
class AttentionPlanMetadataWire:
    mode_code: AttentionModeCode
    flags: int
    plan_fingerprint: str
    admission_fingerprint: str
    dispatch_fingerprint: str
    binary_abi_fingerprint: str
    config: AttentionPlanWireConfig
    sections: Tuple[AttentionMetadataSection, ...]
    schema_version: int = ATTENTION_PLAN_METADATA_WIRE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_PLAN_METADATA_WIRE_VERSION:
            raise SchemaError("unsupported Attention plan metadata wire version")
        try:
            object.__setattr__(self, "mode_code", AttentionModeCode(self.mode_code))
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention plan mode code is invalid") from error
        for name in ("plan_fingerprint", "admission_fingerprint", "dispatch_fingerprint", "binary_abi_fingerprint"):
            _require_hash(name, getattr(self, name))
        if not isinstance(self.flags, int) or isinstance(self.flags, bool) or self.flags < 0 or self.flags > 0xFFFF:
            raise SchemaError("Attention plan flags must fit uint16")
        if self.flags & ~_KNOWN_PLAN_FLAGS:
            raise SchemaError("Attention plan flags contain unknown bits")
        values = tuple(self.sections)
        kinds = tuple(item.kind for item in values)
        if kinds != tuple(sorted(kinds, key=int)) or len(kinds) != len(set(kinds)):
            raise SchemaError("Attention metadata sections must be unique and sorted")
        expected = list(_BASE_SECTIONS[self.mode_code])
        has_mask = bool(self.flags & int(AttentionPlanFlags.CUSTOM_MASK))
        packed_mask = bool(self.flags & int(AttentionPlanFlags.CUSTOM_MASK_PACKED))
        if packed_mask and not has_mask:
            raise SchemaError("packed-mask flag requires custom-mask flag")
        if has_mask:
            if self.mode_code not in {
                AttentionModeCode.SINGLE_PREFILL,
                AttentionModeCode.BATCH_PREFILL_PAGED,
                AttentionModeCode.BATCH_PREFILL_RAGGED,
            }:
                raise SchemaError("custom mask sections are only valid for prefill")
            expected.append(AttentionMetadataSectionKind.MASK_INDPTR)
        if kinds != tuple(sorted(expected, key=int)):
            raise SchemaError("Attention metadata sections do not match mode/flags")
        object.__setattr__(self, "sections", values)
        self._validate_structure()

    @property
    def section_map(self) -> Dict[AttentionMetadataSectionKind, Tuple[int, ...]]:
        return {item.kind: item.values for item in self.sections}

    def _metadata(self) -> AttentionMetadata:
        sections = self.section_map
        page_size = self.config.page_size
        if self.mode_code in {AttentionModeCode.SINGLE_PREFILL, AttentionModeCode.SINGLE_DECODE}:
            return SingleAttentionMetadata(self.config.total_qo_tokens, self.config.total_kv_tokens)
        if self.mode_code == AttentionModeCode.BATCH_DECODE_PAGED:
            return PagedKVMetadata(
                sections[AttentionMetadataSectionKind.KV_INDPTR],
                sections[AttentionMetadataSectionKind.KV_INDICES],
                sections[AttentionMetadataSectionKind.LAST_PAGE_LEN],
                page_size,
            )
        if self.mode_code == AttentionModeCode.BATCH_PREFILL_PAGED:
            paged = PagedKVMetadata(
                sections[AttentionMetadataSectionKind.KV_INDPTR],
                sections[AttentionMetadataSectionKind.KV_INDICES],
                sections[AttentionMetadataSectionKind.LAST_PAGE_LEN],
                page_size,
            )
            return PagedPrefillMetadata(
                sections[AttentionMetadataSectionKind.QO_INDPTR], paged
            )
        if self.mode_code == AttentionModeCode.BATCH_PREFILL_RAGGED:
            return RaggedKVMetadata(
                sections[AttentionMetadataSectionKind.QO_INDPTR],
                sections[AttentionMetadataSectionKind.KV_INDPTR],
            )
        return MixedPagedKVMetadata(
            sections[AttentionMetadataSectionKind.QO_INDPTR],
            sections[AttentionMetadataSectionKind.KV_INDPTR],
            sections[AttentionMetadataSectionKind.KV_INDICES],
            sections[AttentionMetadataSectionKind.KV_LEN],
            page_size,
        )

    def _validate_structure(self) -> None:
        metadata = self._metadata()
        if metadata.batch_size != self.config.batch_size:
            raise SchemaError("metadata wire batch_size does not match sections")
        if isinstance(metadata, SingleAttentionMetadata):
            total_qo, total_kv = metadata.qo_len, metadata.kv_len
            qo_lengths, kv_lengths = (metadata.qo_len,), (metadata.kv_len,)
        elif isinstance(metadata, PagedKVMetadata):
            total_qo = metadata.batch_size * self.config.q_len_per_req
            total_kv = sum(metadata.sequence_lengths)
            qo_lengths = (self.config.q_len_per_req,) * metadata.batch_size
            kv_lengths = metadata.sequence_lengths
        elif isinstance(metadata, PagedPrefillMetadata):
            total_qo = metadata.total_qo_tokens
            total_kv = sum(metadata.paged_kv.sequence_lengths)
            qo_lengths, kv_lengths = metadata.qo_lengths, metadata.paged_kv.sequence_lengths
        elif isinstance(metadata, RaggedKVMetadata):
            total_qo, total_kv = metadata.total_qo_tokens, metadata.total_kv_tokens
            qo_lengths, kv_lengths = metadata.qo_lengths, metadata.kv_lengths
        else:
            total_qo, total_kv = metadata.total_qo_tokens, sum(metadata.kv_len_arr)
            qo_lengths, kv_lengths = metadata.qo_lengths, metadata.kv_len_arr
        if (total_qo, total_kv) != (self.config.total_qo_tokens, self.config.total_kv_tokens):
            raise SchemaError("metadata wire token totals do not match sections")
        mask = self.section_map.get(AttentionMetadataSectionKind.MASK_INDPTR)
        if mask is not None:
            if len(mask) != self.config.batch_size + 1 or mask[0] != 0 or any(a > b for a, b in zip(mask, mask[1:])):
                raise SchemaError("mask_indptr is invalid")
            packed = bool(self.flags & int(AttentionPlanFlags.CUSTOM_MASK_PACKED))
            expected = tuple(
                ((q * k + 7) // 8 if packed else q * k)
                for q, k in zip(qo_lengths, kv_lengths)
            )
            offsets = [0]
            for size in expected:
                offsets.append(offsets[-1] + size)
            if mask != tuple(offsets) or mask[-1] != self.config.custom_mask_numel:
                raise SchemaError("mask_indptr does not match request mask segments")
        elif self.config.custom_mask_numel != 0:
            raise SchemaError("custom_mask_numel requires mask metadata")

    @property
    def payload(self) -> bytes:
        section_count = len(self.sections)
        prefix_size = ATTENTION_PLAN_CONFIG_C_ABI.size_bytes + section_count * ATTENTION_METADATA_SECTION_C_ABI.size_bytes
        cursor = prefix_size
        entries = []
        bodies = []
        for section in self.sections:
            body = section.payload
            entries.append(
                ATTENTION_METADATA_SECTION_C_ABI.pack(
                    {
                        "kind_code": int(section.kind),
                        "element_type_code": int(section.element_type),
                        "flags": section.flags,
                        "offset_bytes": cursor,
                        "element_count": len(section.values),
                        "nbytes": len(body),
                    }
                )
            )
            padded = body + bytes(_align8(len(body)) - len(body))
            bodies.append(padded)
            cursor += len(padded)
        config = ATTENTION_PLAN_CONFIG_C_ABI.pack(self.config.pod_values(section_count))
        return config + b"".join(entries) + b"".join(bodies)

    def to_bytes(self) -> bytes:
        payload = self.payload
        header = ATTENTION_PLAN_METADATA_HEADER_C_ABI.pack(
            {
                "schema_version": self.schema_version,
                "mode_code": int(self.mode_code),
                "flags": self.flags,
                "payload_nbytes": len(payload),
                "plan_fingerprint": _hash_bytes(self.plan_fingerprint),
                "admission_fingerprint": _hash_bytes(self.admission_fingerprint),
                "dispatch_fingerprint": _hash_bytes(self.dispatch_fingerprint),
                "binary_abi_fingerprint": _hash_bytes(self.binary_abi_fingerprint),
            }
        )
        return header + payload

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def validate_plan(self, plan: AttentionFrameworkPlan) -> None:
        expected = materialize_attention_plan_metadata(
            plan,
            dispatch_fingerprint=self.dispatch_fingerprint,
            binary_abi_fingerprint=self.binary_abi_fingerprint,
        )
        if self.to_bytes() != expected.to_bytes():
            raise SchemaError("Attention plan metadata wire is stale or non-canonical")

    @classmethod
    def from_bytes(
        cls,
        value: bytes,
        *,
        limits: AttentionPlanMetadataDecodeLimits = DEFAULT_ATTENTION_PLAN_METADATA_DECODE_LIMITS,
    ) -> "AttentionPlanMetadataWire":
        if not isinstance(value, bytes):
            raise SchemaError("Attention plan metadata must be bytes")
        if not isinstance(limits, AttentionPlanMetadataDecodeLimits):
            raise TypeError("limits must be AttentionPlanMetadataDecodeLimits")
        if len(value) > limits.max_total_nbytes:
            raise SchemaError("Attention plan metadata exceeds decode byte limit")
        header_size = ATTENTION_PLAN_METADATA_HEADER_C_ABI.size_bytes
        if len(value) < header_size + ATTENTION_PLAN_CONFIG_C_ABI.size_bytes:
            raise SchemaError("Attention plan metadata is truncated")
        header = ATTENTION_PLAN_METADATA_HEADER_C_ABI.unpack(value[:header_size])
        if header["schema_version"] != ATTENTION_PLAN_METADATA_WIRE_VERSION:
            raise SchemaError("unsupported Attention plan metadata wire version")
        if len(value) != header_size + header["payload_nbytes"]:
            raise SchemaError("Attention plan metadata byte length mismatch")
        payload = value[header_size:]
        config_raw = ATTENTION_PLAN_CONFIG_C_ABI.unpack(payload[:ATTENTION_PLAN_CONFIG_C_ABI.size_bytes])
        section_count = config_raw["section_count"]
        if section_count > limits.max_section_count:
            raise SchemaError("Attention plan metadata has too many sections")
        if config_raw["directory_entry_nbytes"] != ATTENTION_METADATA_SECTION_C_ABI.size_bytes:
            raise SchemaError("Attention metadata directory entry size mismatch")
        directory_end = ATTENTION_PLAN_CONFIG_C_ABI.size_bytes + section_count * ATTENTION_METADATA_SECTION_C_ABI.size_bytes
        if directory_end > len(payload):
            raise SchemaError("Attention metadata directory is truncated")
        sections = []
        cursor = directory_end
        for index in range(section_count):
            start = ATTENTION_PLAN_CONFIG_C_ABI.size_bytes + index * ATTENTION_METADATA_SECTION_C_ABI.size_bytes
            entry = ATTENTION_METADATA_SECTION_C_ABI.unpack(payload[start:start + ATTENTION_METADATA_SECTION_C_ABI.size_bytes])
            try:
                kind = AttentionMetadataSectionKind(entry["kind_code"])
                element_type = AttentionMetadataElementType(entry["element_type_code"])
            except ValueError as error:
                raise SchemaError("Attention metadata directory contains unknown enum") from error
            count = entry["element_count"]
            nbytes = entry["nbytes"]
            if count > limits.max_section_elements:
                raise SchemaError("Attention metadata section exceeds element limit")
            if nbytes != count * element_type.itemsize:
                raise SchemaError("Attention metadata section byte count mismatch")
            if entry["offset_bytes"] != cursor or cursor + nbytes > len(payload):
                raise SchemaError("Attention metadata section offset is non-canonical")
            raw = payload[cursor:cursor + nbytes]
            values = struct.unpack("<%di" % count, raw) if count else ()
            padded_end = cursor + _align8(nbytes)
            if padded_end > len(payload) or any(payload[cursor + nbytes:padded_end]):
                raise SchemaError("Attention metadata section padding must be zero")
            sections.append(AttentionMetadataSection(kind, values, element_type, entry["flags"]))
            cursor = padded_end
        if cursor != len(payload):
            raise SchemaError("Attention plan metadata contains trailing bytes")
        return cls(
            mode_code=header["mode_code"],
            flags=header["flags"],
            plan_fingerprint=bytes(header["plan_fingerprint"]).hex(),
            admission_fingerprint=bytes(header["admission_fingerprint"]).hex(),
            dispatch_fingerprint=bytes(header["dispatch_fingerprint"]).hex(),
            binary_abi_fingerprint=bytes(header["binary_abi_fingerprint"]).hex(),
            config=AttentionPlanWireConfig.from_pod(config_raw),
            sections=tuple(sections),
            schema_version=header["schema_version"],
        )


def _cumulative(values: Sequence[int]) -> Tuple[int, ...]:
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return tuple(result)


def _sections_for(plan: AttentionFrameworkPlan) -> Tuple[AttentionMetadataSection, ...]:
    metadata = plan.metadata
    values = []
    if isinstance(metadata, PagedKVMetadata):
        values.extend(
            (
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDPTR, metadata.indptr),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDICES, metadata.indices),
                AttentionMetadataSection(AttentionMetadataSectionKind.LAST_PAGE_LEN, metadata.last_page_len),
            )
        )
    elif isinstance(metadata, PagedPrefillMetadata):
        values.extend(
            (
                AttentionMetadataSection(AttentionMetadataSectionKind.QO_INDPTR, metadata.qo_indptr),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDPTR, metadata.paged_kv.indptr),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDICES, metadata.paged_kv.indices),
                AttentionMetadataSection(AttentionMetadataSectionKind.LAST_PAGE_LEN, metadata.paged_kv.last_page_len),
            )
        )
    elif isinstance(metadata, RaggedKVMetadata):
        values.extend(
            (
                AttentionMetadataSection(AttentionMetadataSectionKind.QO_INDPTR, metadata.qo_indptr),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDPTR, metadata.kv_indptr),
            )
        )
    elif isinstance(metadata, MixedPagedKVMetadata):
        values.extend(
            (
                AttentionMetadataSection(AttentionMetadataSectionKind.QO_INDPTR, metadata.qo_indptr),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDPTR, metadata.kv_indptr),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_INDICES, metadata.kv_indices),
                AttentionMetadataSection(AttentionMetadataSectionKind.KV_LEN, metadata.kv_len_arr),
            )
        )
    if plan.spec.custom_mask is not None:
        pairs = plan.spec._query_kv_lengths(metadata)
        segment_sizes = tuple(q * k for q, k in pairs)
        if plan.spec.custom_mask.packed:
            segment_sizes = tuple((item + 7) // 8 for item in segment_sizes)
        values.append(
            AttentionMetadataSection(
                AttentionMetadataSectionKind.MASK_INDPTR,
                _cumulative(segment_sizes),
            )
        )
    return tuple(sorted(values, key=lambda item: int(item.kind)))


def materialize_attention_plan_metadata(
    plan: AttentionFrameworkPlan,
    *,
    dispatch_fingerprint: str,
    binary_abi_fingerprint: Optional[str] = None,
) -> AttentionPlanMetadataWire:
    if not isinstance(plan, AttentionFrameworkPlan):
        raise TypeError("plan must be AttentionFrameworkPlan")
    _require_hash("dispatch_fingerprint", dispatch_fingerprint)
    canonical_binary = attention_kernel_binary_abi().fingerprint
    binary = canonical_binary if binary_abi_fingerprint is None else binary_abi_fingerprint
    _require_hash("binary_abi_fingerprint", binary)
    if binary != canonical_binary:
        raise SchemaError("Attention plan metadata requires canonical binary ABI")
    usage = measure_attention_resources(plan.spec, plan.metadata)
    flags = 0
    if plan.spec.effective_causal:
        flags |= int(AttentionPlanFlags.CAUSAL)
    if plan.spec.custom_mask is not None:
        flags |= int(AttentionPlanFlags.CUSTOM_MASK)
        if plan.spec.custom_mask.packed:
            flags |= int(AttentionPlanFlags.CUSTOM_MASK_PACKED)
    if plan.spec.use_fp16_qk_reduction:
        flags |= int(AttentionPlanFlags.FP16_QK_REDUCTION)
    if plan.spec.use_profiler:
        flags |= int(AttentionPlanFlags.PROFILER)
    if plan.spec.kv_quant_spec is not None:
        flags |= int(AttentionPlanFlags.QUANTIZED_KV)
    config = AttentionPlanWireConfig(
        batch_size=usage.batch_size,
        num_qo_heads=plan.spec.num_qo_heads,
        num_kv_heads=plan.spec.num_kv_heads,
        head_dim_qk=plan.spec.head_dim_qk,
        head_dim_vo=int(plan.spec.head_dim_vo),
        page_size=usage.page_size,
        q_len_per_req=plan.spec.q_len_per_req,
        pos_encoding_code=_POSITION_CODES[plan.spec.pos_encoding_mode],
        kv_layout_code=_LAYOUT_CODES[plan.spec.kv_layout],
        q_dtype_code=attention_dtype_code(plan.spec.q_dtype),
        kv_dtype_code=attention_dtype_code(plan.spec.kv_dtype or plan.spec.q_dtype),
        o_dtype_code=attention_dtype_code(plan.spec.o_dtype or plan.spec.q_dtype),
        index_dtype_code=AttentionDTypeCode.INT32,
        window_left=plan.spec.window_left,
        window_right=plan.spec.window_right,
        sm_scale=float(plan.spec.sm_scale),
        logits_soft_cap=float(plan.spec.logits_soft_cap),
        rope_scale=float(plan.spec.rope_scale),
        rope_theta=float(plan.spec.rope_theta),
        total_qo_tokens=usage.total_qo_tokens,
        total_kv_tokens=usage.total_kv_tokens,
        custom_mask_numel=(plan.spec.custom_mask.numel if plan.spec.custom_mask is not None else 0),
    )
    return AttentionPlanMetadataWire(
        mode_code=_MODE_CODES[plan.spec.mode],
        flags=flags,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        dispatch_fingerprint=dispatch_fingerprint,
        binary_abi_fingerprint=binary,
        config=config,
        sections=_sections_for(plan),
    )


__all__ = [
    "ATTENTION_METADATA_SECTION_C_ABI",
    "ATTENTION_PLAN_CONFIG_C_ABI",
    "ATTENTION_PLAN_METADATA_WIRE_VERSION",
    "DEFAULT_ATTENTION_PLAN_METADATA_DECODE_LIMITS",
    "AttentionMetadataElementType",
    "AttentionMetadataSection",
    "AttentionMetadataSectionKind",
    "AttentionPlanMetadataDecodeLimits",
    "AttentionPlanMetadataWire",
    "AttentionPlanWireConfig",
    "materialize_attention_plan_metadata",
]
