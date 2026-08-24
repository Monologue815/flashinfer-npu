import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    ATTENTION_AUXILIARY_VIEW_C_ABI,
    ATTENTION_KERNEL_ERROR_ABI,
    ATTENTION_KV_CACHE_VIEW_C_ABI,
    ATTENTION_LAUNCH_ARGUMENT_NAMES,
    ATTENTION_PLAN_METADATA_HEADER_C_ABI,
    ATTENTION_RUN_OPTIONS_C_ABI,
    ATTENTION_TENSOR_VIEW_C_ABI,
    attention_kernel_binary_abi,
)
from flashinfer_npu.runtime import (
    CFieldABI,
    CPrimitive,
    CStructABI,
    KernelArgumentDirection,
    KernelBinaryABI,
    KernelErrorABI,
    KernelErrorCodeABI,
    KernelLaunchABI,
    SchemaError,
)


def logical_attention_abi():
    return KernelLaunchABI(
        abi_name="flashinfer_npu.attention.v1",
        entry_point="attention_entry",
        argument_names=ATTENTION_LAUNCH_ARGUMENT_NAMES,
        mutable_arguments=(
            "aux",
            "out",
            "lse",
            "float_workspace",
            "int_workspace",
        ),
        stream_argument="stream",
    )


class CStructABITests(unittest.TestCase):
    def test_attention_pod_sizes_offsets_and_fingerprints_are_frozen(self):
        self.assertEqual(ATTENTION_TENSOR_VIEW_C_ABI.size_bytes, 176)
        self.assertEqual(ATTENTION_KV_CACHE_VIEW_C_ABI.size_bytes, 64)
        self.assertEqual(ATTENTION_AUXILIARY_VIEW_C_ABI.size_bytes, 32)
        self.assertEqual(ATTENTION_RUN_OPTIONS_C_ABI.size_bytes, 64)
        self.assertEqual(ATTENTION_PLAN_METADATA_HEADER_C_ABI.size_bytes, 160)
        self.assertEqual(
            ATTENTION_TENSOR_VIEW_C_ABI.fingerprint,
            "16aa51eecb63ed2e4bdb592f0ed7915958198b544d9859f1d0c39c8fd78b519b",
        )
        self.assertEqual(
            ATTENTION_KV_CACHE_VIEW_C_ABI.fingerprint,
            "cb56b9c848b4d623d96faff3125803baea110a61452309c5763b7e06701f621e",
        )
        self.assertEqual(
            ATTENTION_PLAN_METADATA_HEADER_C_ABI.fingerprint,
            "e8bc8ff35debc2b8c9004614394ccb99178f4bc0602241ea0283298072341cbf",
        )
        self.assertEqual(
            ATTENTION_AUXILIARY_VIEW_C_ABI.fingerprint,
            "8cdd7385ac2a4c04096b2db4cc5a9de28c0956f4161427a5c7d2e49630553c52",
        )
        self.assertEqual(
            ATTENTION_RUN_OPTIONS_C_ABI.fingerprint,
            "9daa9d966ecc02d1c23b7b19060492f5d87bc49e4a1f815dd83b752a7b2763de",
        )
        offsets = dict(ATTENTION_TENSOR_VIEW_C_ABI.field_offsets)
        self.assertEqual(
            offsets,
            {
                "data_ptr": 0,
                "storage_nbytes": 8,
                "storage_offset_elements": 16,
                "shape": 24,
                "strides": 88,
                "ndim": 152,
                "dtype_code": 156,
                "role_code": 158,
                "flags": 160,
                "device_index": 164,
                "reserved": 168,
            },
        )
        restored = CStructABI.from_dict(ATTENTION_TENSOR_VIEW_C_ABI.to_dict())
        self.assertEqual(restored, ATTENTION_TENSOR_VIEW_C_ABI)
        self.assertEqual(restored.fingerprint, ATTENTION_TENSOR_VIEW_C_ABI.fingerprint)

    def test_pack_unpack_uses_explicit_little_endian_padding_and_zero_reserved(self):
        values = {
            "data_ptr": 0x1000,
            "storage_nbytes": 4096,
            "storage_offset_elements": 2,
            "shape": (2, 3, 4, 0, 0, 0, 0, 0),
            "strides": (12, 4, 1, 0, 0, 0, 0, 0),
            "ndim": 3,
            "dtype_code": 8,
            "role_code": 1,
            "flags": 3,
            "device_index": 0,
        }
        payload = ATTENTION_TENSOR_VIEW_C_ABI.pack(values)
        self.assertEqual(len(payload), 176)
        restored = ATTENTION_TENSOR_VIEW_C_ABI.unpack(payload)
        for name, value in values.items():
            self.assertEqual(restored[name], value)
        self.assertEqual(restored["reserved"], 0)

        corrupted = bytearray(payload)
        corrupted[dict(ATTENTION_TENSOR_VIEW_C_ABI.field_offsets)["reserved"]] = 1
        with self.assertRaisesRegex(SchemaError, "reserved field"):
            ATTENTION_TENSOR_VIEW_C_ABI.unpack(bytes(corrupted))

    def test_layout_rejects_implicit_or_ambiguous_shapes(self):
        with self.assertRaisesRegex(SchemaError, "unique"):
            CStructABI(
                "Duplicate",
                (
                    CFieldABI("x", CPrimitive.U32),
                    CFieldABI("x", CPrimitive.U32),
                ),
            )
        with self.assertRaisesRegex(SchemaError, "smaller"):
            CStructABI(
                "UnderAligned",
                (CFieldABI("x", CPrimitive.U64),),
                alignment=4,
            )
        with self.assertRaisesRegex(SchemaError, "little-endian"):
            replace(ATTENTION_TENSOR_VIEW_C_ABI, endianness="native")
        with self.assertRaisesRegex(SchemaError, "requires 8 values"):
            ATTENTION_TENSOR_VIEW_C_ABI.pack(
                {
                    "data_ptr": 0,
                    "storage_nbytes": 0,
                    "storage_offset_elements": 0,
                    "shape": (1,),
                    "strides": (1,) * 8,
                    "ndim": 1,
                    "dtype_code": 1,
                    "role_code": 1,
                    "flags": 0,
                    "device_index": 0,
                }
            )


class KernelBinaryABITests(unittest.TestCase):
    def test_attention_binary_abi_matches_logical_order_mutation_and_stream(self):
        binary = attention_kernel_binary_abi()
        logical = logical_attention_abi()
        binary.validate_logical(logical)
        self.assertEqual(
            tuple(item.name for item in binary.arguments),
            ATTENTION_LAUNCH_ARGUMENT_NAMES,
        )
        arguments = {item.name: item for item in binary.arguments}
        self.assertEqual(
            arguments["aux"].pointee_abi_fingerprint,
            ATTENTION_AUXILIARY_VIEW_C_ABI.fingerprint,
        )
        self.assertEqual(
            arguments["run_options"].pointee_abi_fingerprint,
            ATTENTION_RUN_OPTIONS_C_ABI.fingerprint,
        )
        self.assertEqual(
            arguments["aux"].direction,
            KernelArgumentDirection.INOUT,
        )
        self.assertEqual(
            KernelBinaryABI.from_dict(binary.to_dict()),
            binary,
        )
        self.assertEqual(binary.return_primitive, CPrimitive.I32)
        self.assertEqual(
            binary.fingerprint,
            "cb5285127b5926a81fcd28d2e27997ce5fc556d6c1a4a797f5d24a0060821b7c",
        )
        self.assertEqual(
            ATTENTION_KERNEL_ERROR_ABI.fingerprint,
            "b74b754fb5990aa9234ed91049cb2b2ec67eebeae1409db813e620a3d235f456",
        )
        async_code = next(
            item for item in ATTENTION_KERNEL_ERROR_ABI.codes
            if item.name == "async_failure"
        )
        self.assertTrue(async_code.asynchronous)

    def test_binary_logical_mutation_and_error_code_drift_are_rejected(self):
        binary = attention_kernel_binary_abi()
        changed_arguments = tuple(
            replace(item, direction=KernelArgumentDirection.INPUT)
            if item.name == "out"
            else item
            for item in binary.arguments
        )
        with self.assertRaisesRegex(SchemaError, "mutation set"):
            replace(binary, arguments=changed_arguments).validate_logical(
                logical_attention_abi()
            )
        with self.assertRaisesRegex(SchemaError, "success=0"):
            KernelErrorABI(
                "invalid",
                (KernelErrorCodeABI("ok", 0),),
            )
        with self.assertRaisesRegex(SchemaError, "unique"):
            KernelErrorABI(
                "duplicate",
                (
                    KernelErrorCodeABI("success", 0),
                    KernelErrorCodeABI("other", 0),
                ),
            )
        with self.assertRaisesRegex(SchemaError, "error return must be i32"):
            replace(binary, return_primitive=CPrimitive.U64)
        with self.assertRaisesRegex(SchemaError, "only pointer"):
            replace(
                next(
                    item
                    for item in binary.arguments
                    if item.name == "plan_metadata_nbytes"
                ),
                nullable=True,
            )
        with self.assertRaisesRegex(SchemaError, "input-only"):
            replace(
                next(
                    item
                    for item in binary.arguments
                    if item.name == "plan_metadata_nbytes"
                ),
                direction=KernelArgumentDirection.OUTPUT,
            )


if __name__ == "__main__":
    unittest.main()
