"""Stable Host-described C ABI contracts for future Attention launchers."""

from __future__ import annotations

from enum import IntEnum, IntFlag

from flashinfer_npu.runtime import (
    CFieldABI,
    CPrimitive,
    CStructABI,
    KernelArgumentABI,
    KernelArgumentDirection,
    KernelArgumentPassing,
    KernelBinaryABI,
    KernelErrorABI,
    KernelErrorCodeABI,
    SchemaError,
)


ATTENTION_LAUNCH_CONTRACT_VERSION = 1
ATTENTION_MAX_TENSOR_RANK = 8


class AttentionDTypeCode(IntEnum):
    BOOL = 1
    INT8 = 2
    UINT8 = 3
    INT32 = 4
    INT64 = 5
    FLOAT16 = 6
    BFLOAT16 = 7
    FLOAT32 = 8
    FLOAT64 = 9
    FLOAT8_E4M3FN = 10
    FLOAT8_E5M2 = 11
    INT16 = 12
    UINT16 = 13
    UINT32 = 14
    UINT64 = 15
    INT4 = 16
    INT4_PACKED = 17
    UINT4_PACKED = 18


class AttentionTensorFlags(IntFlag):
    CONTIGUOUS = 1 << 0
    WRITABLE = 1 << 1
    EMPTY = 1 << 2


class AttentionKVFlags(IntFlag):
    PACKED = 1 << 0
    QUANTIZED = 1 << 1
    PHYSICAL_LAYOUT = 1 << 2


class AttentionKVPhysicalLayoutAccessCode(IntEnum):
    LOGICAL = 1
    KERNEL_NATIVE = 2


class AttentionKVLayoutCode(IntEnum):
    NHD = 1
    HND = 2


class AttentionPositionEncodingCode(IntEnum):
    NONE = 1
    ROPE_LLAMA = 2
    ALIBI = 3


class AttentionModeCode(IntEnum):
    SINGLE_PREFILL = 1
    SINGLE_DECODE = 2
    BATCH_PREFILL_PAGED = 3
    BATCH_PREFILL_RAGGED = 4
    BATCH_DECODE_PAGED = 5
    BATCH_MIXED_PAGED = 6


class AttentionPlanFlags(IntFlag):
    CAUSAL = 1 << 0
    CUSTOM_MASK = 1 << 1
    CUSTOM_MASK_PACKED = 1 << 2
    FP16_QK_REDUCTION = 1 << 3
    PROFILER = 1 << 4
    QUANTIZED_KV = 1 << 5


class AttentionTensorRole(IntEnum):
    Q = 1
    KV_KEY_STORAGE = 2
    KV_KEY_SCALE = 3
    KV_KEY_ZERO_POINT = 4
    KV_VALUE_STORAGE = 5
    KV_VALUE_SCALE = 6
    KV_VALUE_ZERO_POINT = 7
    OUT = 8
    LSE = 9
    KV_PACKED_STORAGE = 10


class AttentionAuxiliaryRole(IntEnum):
    CUSTOM_MASK = 1
    Q_SCALE = 2
    K_SCALE = 3
    V_SCALE = 4
    PROFILER = 5
    ALIBI_SLOPES = 6


ATTENTION_TENSOR_VIEW_C_ABI = CStructABI(
    name="FlashInferNpuTensorViewV1",
    fields=(
        CFieldABI("data_ptr", CPrimitive.U64),
        CFieldABI("storage_nbytes", CPrimitive.U64),
        CFieldABI("storage_offset_elements", CPrimitive.U64),
        CFieldABI("shape", CPrimitive.I64, ATTENTION_MAX_TENSOR_RANK),
        CFieldABI("strides", CPrimitive.I64, ATTENTION_MAX_TENSOR_RANK),
        CFieldABI("ndim", CPrimitive.U32),
        CFieldABI("dtype_code", CPrimitive.U16),
        CFieldABI("role_code", CPrimitive.U16),
        CFieldABI("flags", CPrimitive.U32),
        CFieldABI("device_index", CPrimitive.I32),
        CFieldABI("reserved", CPrimitive.U32, reserved=True),
    ),
)


ATTENTION_KV_CACHE_VIEW_C_ABI = CStructABI(
    name="FlashInferNpuKVCacheViewV1",
    fields=(
        CFieldABI("components_ptr", CPrimitive.U64),
        CFieldABI("component_count", CPrimitive.U32),
        CFieldABI("layout_code", CPrimitive.U16),
        CFieldABI("flags", CPrimitive.U16),
        CFieldABI("quant_spec_fingerprint", CPrimitive.U8, 32),
        CFieldABI("reserved", CPrimitive.U8, 16, reserved=True),
    ),
)


# V1 remains frozen for existing kernels. V2 adds enough identity to admit an
# explicitly registered non-logical quantized KV layout without putting
# variable-length layout strings into the device-facing ABI.
ATTENTION_KV_CACHE_VIEW_V2_C_ABI = CStructABI(
    name="FlashInferNpuKVCacheViewV2",
    fields=(
        CFieldABI("components_ptr", CPrimitive.U64),
        CFieldABI("component_count", CPrimitive.U32),
        CFieldABI("layout_code", CPrimitive.U16),
        CFieldABI("flags", CPrimitive.U16),
        CFieldABI("physical_layout_access_code", CPrimitive.U16),
        CFieldABI("reserved_header", CPrimitive.U8, 6, reserved=True),
        CFieldABI("quant_spec_fingerprint", CPrimitive.U8, 32),
        CFieldABI("physical_layout_descriptor_fingerprint", CPrimitive.U8, 32),
        CFieldABI("physical_layout_catalog_fingerprint", CPrimitive.U8, 32),
        CFieldABI("physical_layout_binding_fingerprint", CPrimitive.U8, 32),
        CFieldABI("dispatch_receipt_fingerprint", CPrimitive.U8, 32),
        CFieldABI("reserved", CPrimitive.U8, 8, reserved=True),
    ),
)


ATTENTION_AUXILIARY_VIEW_C_ABI = CStructABI(
    name="FlashInferNpuAttentionAuxiliaryViewV1",
    fields=(
        CFieldABI("components_ptr", CPrimitive.U64),
        CFieldABI("component_count", CPrimitive.U32),
        CFieldABI("flags", CPrimitive.U32),
        CFieldABI("reserved", CPrimitive.U8, 16, reserved=True),
    ),
)


ATTENTION_RUN_OPTIONS_C_ABI = CStructABI(
    name="FlashInferNpuAttentionRunOptionsV1",
    fields=(
        CFieldABI("q_scale", CPrimitive.F64),
        CFieldABI("k_scale", CPrimitive.F64),
        CFieldABI("v_scale", CPrimitive.F64),
        CFieldABI("logits_soft_cap", CPrimitive.F64),
        CFieldABI("flags", CPrimitive.U32),
        CFieldABI("reserved", CPrimitive.U8, 28, reserved=True),
    ),
)


ATTENTION_PLAN_METADATA_HEADER_C_ABI = CStructABI(
    name="FlashInferNpuAttentionPlanHeaderV1",
    fields=(
        CFieldABI("schema_version", CPrimitive.U32),
        CFieldABI("mode_code", CPrimitive.U16),
        CFieldABI("flags", CPrimitive.U16),
        CFieldABI("payload_nbytes", CPrimitive.U64),
        CFieldABI("plan_fingerprint", CPrimitive.U8, 32),
        CFieldABI("admission_fingerprint", CPrimitive.U8, 32),
        CFieldABI("dispatch_fingerprint", CPrimitive.U8, 32),
        CFieldABI("binary_abi_fingerprint", CPrimitive.U8, 32),
        CFieldABI("reserved", CPrimitive.U8, 16, reserved=True),
    ),
)


ATTENTION_KERNEL_ERROR_ABI = KernelErrorABI(
    name="flashinfer_npu.kernel_error.v1",
    codes=(
        KernelErrorCodeABI("success", 0),
        KernelErrorCodeABI("invalid_argument", 1),
        KernelErrorCodeABI("unsupported", 2),
        KernelErrorCodeABI("workspace_too_small", 3),
        KernelErrorCodeABI("artifact_mismatch", 4),
        KernelErrorCodeABI("abi_mismatch", 5),
        KernelErrorCodeABI("launch_failure", 6),
        KernelErrorCodeABI("resource_busy", 7, retryable=True),
        KernelErrorCodeABI("async_failure", 8, asynchronous=True),
    ),
)


ATTENTION_LAUNCH_ARGUMENT_NAMES = (
    "q",
    "kv",
    "aux",
    "run_options",
    "out",
    "lse",
    "plan_metadata",
    "plan_metadata_nbytes",
    "float_workspace",
    "float_workspace_nbytes",
    "int_workspace",
    "int_workspace_nbytes",
    "stream",
)


def _pointer(
    name,
    direction,
    *,
    nullable=False,
    alignment=8,
    pointee=None,
):
    return KernelArgumentABI(
        name=name,
        passing=KernelArgumentPassing.POINTER,
        direction=direction,
        nullable=nullable,
        required_alignment=alignment,
        pointee_abi_name=(pointee.name if pointee is not None else None),
        pointee_abi_fingerprint=(
            pointee.fingerprint if pointee is not None else None
        ),
    )


def attention_kernel_binary_abi() -> KernelBinaryABI:
    return KernelBinaryABI(
        abi_name="flashinfer_npu.attention.binary.v1",
        arguments=(
            _pointer(
                "q",
                KernelArgumentDirection.INPUT,
                pointee=ATTENTION_TENSOR_VIEW_C_ABI,
            ),
            _pointer(
                "kv",
                KernelArgumentDirection.INPUT,
                pointee=ATTENTION_KV_CACHE_VIEW_C_ABI,
            ),
            _pointer(
                "aux",
                KernelArgumentDirection.INOUT,
                pointee=ATTENTION_AUXILIARY_VIEW_C_ABI,
            ),
            _pointer(
                "run_options",
                KernelArgumentDirection.INPUT,
                pointee=ATTENTION_RUN_OPTIONS_C_ABI,
            ),
            _pointer(
                "out",
                KernelArgumentDirection.OUTPUT,
                pointee=ATTENTION_TENSOR_VIEW_C_ABI,
            ),
            _pointer(
                "lse",
                KernelArgumentDirection.OUTPUT,
                nullable=True,
                pointee=ATTENTION_TENSOR_VIEW_C_ABI,
            ),
            _pointer(
                "plan_metadata",
                KernelArgumentDirection.INPUT,
                pointee=ATTENTION_PLAN_METADATA_HEADER_C_ABI,
            ),
            KernelArgumentABI(
                "plan_metadata_nbytes",
                KernelArgumentPassing.VALUE,
                KernelArgumentDirection.INPUT,
            ),
            _pointer(
                "float_workspace",
                KernelArgumentDirection.INOUT,
                nullable=True,
                alignment=32,
            ),
            KernelArgumentABI(
                "float_workspace_nbytes",
                KernelArgumentPassing.VALUE,
                KernelArgumentDirection.INPUT,
            ),
            _pointer(
                "int_workspace",
                KernelArgumentDirection.INOUT,
                nullable=True,
                alignment=32,
            ),
            KernelArgumentABI(
                "int_workspace_nbytes",
                KernelArgumentPassing.VALUE,
                KernelArgumentDirection.INPUT,
            ),
            KernelArgumentABI(
                "stream",
                KernelArgumentPassing.OPAQUE_HANDLE,
                KernelArgumentDirection.INPUT,
            ),
        ),
        error_abi=ATTENTION_KERNEL_ERROR_ABI,
    )


def attention_kernel_binary_abi_v2() -> KernelBinaryABI:
    """Binary ABI for kernels that can consume KV descriptor v2.

    The thirteen logical arguments and their mutation directions are unchanged;
    only the pointee ABI of ``kv`` is versioned.
    """

    v1 = attention_kernel_binary_abi()
    arguments = tuple(
        _pointer(
            "kv",
            KernelArgumentDirection.INPUT,
            pointee=ATTENTION_KV_CACHE_VIEW_V2_C_ABI,
        )
        if item.name == "kv"
        else item
        for item in v1.arguments
    )
    return KernelBinaryABI(
        abi_name="flashinfer_npu.attention.binary.v2",
        arguments=arguments,
        error_abi=ATTENTION_KERNEL_ERROR_ABI,
    )


_DTYPE_CODES = {
    "bool": AttentionDTypeCode.BOOL,
    "int8": AttentionDTypeCode.INT8,
    "uint8": AttentionDTypeCode.UINT8,
    "int16": AttentionDTypeCode.INT16,
    "uint16": AttentionDTypeCode.UINT16,
    "int32": AttentionDTypeCode.INT32,
    "uint32": AttentionDTypeCode.UINT32,
    "int64": AttentionDTypeCode.INT64,
    "uint64": AttentionDTypeCode.UINT64,
    "float16": AttentionDTypeCode.FLOAT16,
    "bfloat16": AttentionDTypeCode.BFLOAT16,
    "float32": AttentionDTypeCode.FLOAT32,
    "float64": AttentionDTypeCode.FLOAT64,
    "float8_e4m3fn": AttentionDTypeCode.FLOAT8_E4M3FN,
    "float8_e5m2": AttentionDTypeCode.FLOAT8_E5M2,
    "int4": AttentionDTypeCode.INT4,
    "int4_packed": AttentionDTypeCode.INT4_PACKED,
    "uint4_packed": AttentionDTypeCode.UINT4_PACKED,
}


def attention_dtype_code(dtype: str) -> AttentionDTypeCode:
    try:
        return _DTYPE_CODES[str(dtype)]
    except KeyError as error:
        raise SchemaError("dtype is not representable by Attention ABI v1") from error


__all__ = [
    "ATTENTION_AUXILIARY_VIEW_C_ABI",
    "ATTENTION_KERNEL_ERROR_ABI",
    "ATTENTION_KV_CACHE_VIEW_C_ABI",
    "ATTENTION_KV_CACHE_VIEW_V2_C_ABI",
    "ATTENTION_LAUNCH_ARGUMENT_NAMES",
    "ATTENTION_LAUNCH_CONTRACT_VERSION",
    "ATTENTION_MAX_TENSOR_RANK",
    "ATTENTION_PLAN_METADATA_HEADER_C_ABI",
    "ATTENTION_RUN_OPTIONS_C_ABI",
    "ATTENTION_TENSOR_VIEW_C_ABI",
    "AttentionAuxiliaryRole",
    "AttentionDTypeCode",
    "AttentionKVLayoutCode",
    "AttentionKVFlags",
    "AttentionKVPhysicalLayoutAccessCode",
    "AttentionModeCode",
    "AttentionPlanFlags",
    "AttentionPositionEncodingCode",
    "AttentionTensorRole",
    "AttentionTensorFlags",
    "attention_dtype_code",
    "attention_kernel_binary_abi",
    "attention_kernel_binary_abi_v2",
]
