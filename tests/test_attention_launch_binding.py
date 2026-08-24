import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    ATTENTION_AUXILIARY_VIEW_C_ABI,
    ATTENTION_RUN_OPTIONS_C_ABI,
    ATTENTION_TENSOR_VIEW_C_ABI,
    AttentionAddressBinding,
    AttentionAuxiliaryContract,
    AttentionAuxiliaryRole,
    AttentionAuxiliaryTensor,
    AttentionAuxiliaryViewPOD,
    AttentionFrameworkSession,
    AttentionHostBufferLease,
    AttentionKVCacheViewPOD,
    AttentionMode,
    AttentionPlanSpec,
    AttentionRunOptions,
    AttentionRunTensorContract,
    AttentionStorageLease,
    AttentionStorageLifetime,
    AttentionTensorAccessPolicy,
    AttentionTensorRole,
    AttentionTensorViewPOD,
    CustomMaskSpec,
    KVCacheView,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    StreamContext,
    TensorView,
    contiguous_strides,
    dtype_itemsize,
    materialize_attention_auxiliary_view,
    materialize_attention_kv_cache_view,
    materialize_attention_tensor_view,
)
from flashinfer_npu.runtime import SchemaError


def view(
    shape,
    storage_id,
    *,
    dtype="float32",
    writable=False,
    device="npu:0",
):
    shape = tuple(shape)
    return TensorView(
        shape,
        contiguous_strides(shape),
        dtype,
        device,
        storage_id,
        max(1, dtype_itemsize(dtype)) * _numel(shape),
        data_ptr_alignment=64,
        writable=writable,
    )


def _numel(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result


def lease_for(value, address, *, writable=None):
    return AttentionStorageLease(
        "lease:" + value.storage_id,
        value.storage_id,
        value.device,
        address,
        value.storage_nbytes,
        64,
        "allocator:npu0",
        1,
        AttentionStorageLifetime.RUN,
        value.writable if writable is None else writable,
    )


def bind(role, value, address):
    return AttentionAddressBinding(role, value, lease_for(value, address))


def table_lease(name, address, size, *, device="npu:0"):
    return AttentionHostBufferLease(
        "lease:" + name,
        address,
        size,
        64,
        "host-arena",
        1,
        AttentionStorageLifetime.RUN,
        True,
    )


def masked_plan(*, profiler=False, alibi=False):
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=2,
        q_dtype="float32",
        kv_dtype="float32",
        custom_mask=CustomMaskSpec(1, packed=True),
        use_profiler=profiler,
        pos_encoding_mode=(PosEncodingMode.ALIBI if alibi else PosEncodingMode.NONE),
        logits_soft_cap=5.0,
    )
    metadata = PagedPrefillMetadata(
        (0, 1), PagedKVMetadata((0, 1), (0,), (2,), 2)
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


class AttentionAuxiliaryContractTests(unittest.TestCase):
    def test_run_options_are_exact_64_byte_pod_and_plan_bound(self):
        options = AttentionRunOptions(2.0, 3.0, 4.0, 1.5)
        payload = options.pack()
        self.assertEqual(len(payload), ATTENTION_RUN_OPTIONS_C_ABI.size_bytes)
        self.assertEqual(len(payload), 64)
        self.assertEqual(AttentionRunOptions.from_bytes(payload), options)
        self.assertNotEqual(options.fingerprint, replace(options, q_scale=2.5).fingerprint)
        capped = masked_plan()
        uncapped = replace(capped, spec=replace(capped.spec, logits_soft_cap=None))
        with self.assertRaisesRegex(SchemaError, "capped plan"):
            options.validate_plan(uncapped)

    def test_auxiliary_roles_validate_shape_dtype_access_and_plan_presence(self):
        plan = masked_plan(profiler=True, alibi=True)
        auxiliary = AttentionAuxiliaryContract(
            (
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.CUSTOM_MASK,
                    view((1,), "mask", dtype="uint8"),
                ),
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.Q_SCALE,
                    view((2,), "q-scale"),
                ),
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.PROFILER,
                    view((8,), "profiler", dtype="uint64", writable=True),
                ),
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.ALIBI_SLOPES,
                    view((2,), "slopes"),
                ),
            )
        )
        auxiliary.validate_plan(plan, "npu:0")
        self.assertEqual(AttentionAuxiliaryContract.from_dict(auxiliary.to_dict()), auxiliary)
        rebound = AttentionAuxiliaryContract(
            tuple(
                replace(item, view=replace(item.view, storage_id="new:" + item.view.storage_id))
                for item in auxiliary.components
            )
        )
        self.assertEqual(rebound.fingerprint, auxiliary.fingerprint)
        with self.assertRaisesRegex(SchemaError, "canonical order"):
            AttentionAuxiliaryContract(tuple(reversed(auxiliary.components)))
        with self.assertRaisesRegex(SchemaError, "read-only"):
            AttentionAuxiliaryTensor(
                AttentionAuxiliaryRole.Q_SCALE,
                view((2,), "bad", writable=True),
            )
        with self.assertRaisesRegex(SchemaError, "custom-mask"):
            AttentionAuxiliaryContract(
                tuple(item for item in auxiliary.components if item.role != AttentionAuxiliaryRole.CUSTOM_MASK)
            ).validate_plan(plan, "npu:0")


class AttentionPODMaterializationTests(unittest.TestCase):
    def test_tensor_view_materializes_from_lease_and_strictly_round_trips(self):
        value = view((2, 3), "q")
        binding = bind("q", value, 0x1000)
        pod = materialize_attention_tensor_view(
            binding, AttentionTensorRole.Q, device_index=0
        )
        payload = pod.pack()
        self.assertEqual(len(payload), ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        self.assertEqual(AttentionTensorViewPOD.from_bytes(payload), pod)
        self.assertEqual(pod.data_ptr, 0x1000)
        values = ATTENTION_TENSOR_VIEW_C_ABI.unpack(payload)
        values.pop("reserved")
        values["shape"] = tuple(values["shape"][:2]) + (1,) + tuple(values["shape"][3:])
        with self.assertRaisesRegex(SchemaError, "trailing"):
            AttentionTensorViewPOD.from_bytes(ATTENTION_TENSOR_VIEW_C_ABI.pack(values))

    def test_dense_kv_descriptor_binds_canonical_component_table(self):
        spec = PagedKVCacheSpec(
            1, 2, 1, 2, 2, "float32", structure="separate", device="npu:0"
        )
        key = view((1, 2, 1, 2), "key")
        value = view((1, 2, 1, 2), "value")
        kv = KVCacheView(spec, key, value)
        bindings = (
            bind("kv.key_storage", key, 0x1000),
            bind("kv.value_storage", value, 0x2000),
        )
        table = table_lease("kv-table", 0x3000, 2 * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        pod = materialize_attention_kv_cache_view(kv, bindings, table, 0)
        self.assertEqual(
            tuple(item.role_code for item in pod.components),
            (
                int(AttentionTensorRole.KV_KEY_STORAGE),
                int(AttentionTensorRole.KV_VALUE_STORAGE),
            ),
        )
        self.assertEqual(
            AttentionKVCacheViewPOD.from_bytes(pod.pack(), pod.component_blob), pod
        )
        with self.assertRaisesRegex(SchemaError, "missing"):
            materialize_attention_kv_cache_view(kv, bindings[:1], table, 0)

    def test_auxiliary_descriptor_handles_empty_and_nonempty_tables(self):
        empty_lease = table_lease("empty", 0, 0)
        empty = materialize_attention_auxiliary_view(
            AttentionAuxiliaryContract(), (), empty_lease, 0
        )
        self.assertEqual(empty.components_ptr, 0)
        self.assertEqual(
            AttentionAuxiliaryViewPOD.from_bytes(empty.pack(), b""), empty
        )

        mask = view((2,), "mask", dtype="uint8")
        auxiliary = AttentionAuxiliaryContract(
            (AttentionAuxiliaryTensor(AttentionAuxiliaryRole.CUSTOM_MASK, mask),)
        )
        table = table_lease("aux-table", 0x5000, ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        pod = materialize_attention_auxiliary_view(
            auxiliary, (bind("aux.custom_mask", mask, 0x4000),), table, 0
        )
        self.assertEqual(len(pod.pack()), ATTENTION_AUXILIARY_VIEW_C_ABI.size_bytes)
        self.assertEqual(
            AttentionAuxiliaryViewPOD.from_bytes(pod.pack(), pod.component_blob), pod
        )

    def test_runtime_contract_detects_auxiliary_output_alias(self):
        plan = masked_plan(profiler=True)
        q = view((1, 2, 2), "q")
        spec = PagedKVCacheSpec(
            1, 2, 1, 2, 2, "float32", structure="separate", device="npu:0"
        )
        kv = KVCacheView(
            spec,
            view((1, 2, 1, 2), "key"),
            view((1, 2, 1, 2), "value"),
        )
        profiler = view((2,), "q", dtype="uint64", writable=True)
        auxiliary = AttentionAuxiliaryContract(
            (
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.CUSTOM_MASK,
                    view((1,), "mask", dtype="uint8"),
                ),
                AttentionAuxiliaryTensor(AttentionAuxiliaryRole.PROFILER, profiler),
            )
        )
        contract = AttentionRunTensorContract(
            q, kv, StreamContext("npu:0", "stream"), auxiliary=auxiliary
        )
        with self.assertRaisesRegex(SchemaError, "output cannot alias"):
            contract.validate(AttentionTensorAccessPolicy(), plan=plan)


if __name__ == "__main__":
    unittest.main()
