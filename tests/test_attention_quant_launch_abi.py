import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    ATTENTION_KV_CACHE_VIEW_V2_C_ABI,
    ATTENTION_TENSOR_VIEW_C_ABI,
    AttentionAddressBinding,
    AttentionHostBufferLease,
    AttentionKVCacheViewPODV2,
    AttentionKVFlags,
    AttentionKVPhysicalLayoutAccessCode,
    AttentionStorageLease,
    AttentionStorageLifetime,
    KVCacheView,
    PagedKVCacheSpec,
    QuantPhysicalLayoutCatalog,
    QuantizedTensorView,
    attention_kernel_binary_abi,
    attention_kernel_binary_abi_v2,
    bind_attention_kv_physical_layout,
    materialize_attention_kv_cache_view,
    materialize_attention_kv_cache_view_v2,
    plan_quant_layout_conversion,
    select_attention_dispatch,
    validate_attention_kernel_bindings,
)
from flashinfer_npu.runtime import KernelConstraints, SchemaError
from tests.test_attention_capability import (
    attention_launch_abi,
    bound_kernel,
    functional_profile,
    group_plan,
)
from tests.test_attention_quant_physical_layout import (
    synthetic_layout,
    uint8_spec,
)
from tests.test_attention_tensor_contract import view


def _physical_kv(descriptor):
    logical_shape = (2, 3, 4, 5)
    spec = uint8_spec(descriptor.layout_id)
    shapes = descriptor.physical_shapes(logical_shape, spec)

    def quantized(prefix):
        return QuantizedTensorView(
            logical_shape,
            view(shapes.storage, dtype="uint8", storage_id=prefix + ".storage"),
            view(shapes.scale, storage_id=prefix + ".scale"),
            spec,
            view(
                shapes.zero_point,
                dtype="int32",
                storage_id=prefix + ".zero",
            ),
            descriptor,
        )

    cache = PagedKVCacheSpec(
        2,
        3,
        4,
        5,
        5,
        "uint8",
        structure="separate",
        device="cpu",
        quant_spec=spec,
    )
    return KVCacheView(cache, quantized("key"), quantized("value"))


def _device_bindings(kv):
    result = []
    for index, (role, value) in enumerate(kv.named_component_views):
        lease = AttentionStorageLease(
            "lease:" + role,
            value.storage_id,
            value.device,
            0x100000 + index * 0x10000,
            value.storage_nbytes,
            64,
            "allocator:" + role,
            1,
            AttentionStorageLifetime.RUN,
            value.writable,
        )
        result.append(AttentionAddressBinding(role, value, lease))
    return tuple(result)


def _table_lease(component_count):
    return AttentionHostBufferLease(
        "lease:kv-v2-table",
        0x900000,
        component_count * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes,
        64,
        "host-arena",
        1,
        AttentionStorageLifetime.RUN,
        True,
    )


def _native_fixture():
    descriptor = synthetic_layout()
    catalog = QuantPhysicalLayoutCatalog((descriptor,))
    kv = _physical_kv(descriptor)

    base_profile = functional_profile()
    environment = replace(
        base_profile.environment,
        features=tuple(
            sorted(base_profile.environment.features + descriptor.required_features)
        ),
    )
    base_rule = base_profile.rules[0]
    rule = replace(
        base_rule,
        dtype_signatures=(("float32", "uint8", "float32"),),
        quant_specs=(kv.spec.quant_spec,),
        required_features=tuple(
            sorted(base_rule.required_features + descriptor.required_features)
        ),
    )
    profile = replace(base_profile, environment=environment, rules=(rule,))
    launch_v2 = replace(
        attention_launch_abi(), abi_name="flashinfer_npu.attention.v2"
    )
    constraints = KernelConstraints(
        supported_socs=(environment.soc_version,),
        dtype_signatures=rule.dtype_signatures,
        layout_signatures=((rule.kv_layouts[0].value,),),
        required_features=rule.required_features,
        quant_storage_dtypes=(kv.spec.quant_spec.storage_dtype,),
    )
    kernel = bound_kernel(
        profile,
        launch_abi=launch_v2,
        binary_abi=attention_kernel_binary_abi_v2(),
        constraints=constraints,
    )

    original_profile = functional_profile()
    original_kernel = bound_kernel(original_profile)
    original_receipt = select_attention_dispatch(
        group_plan(),
        (original_profile,),
        (original_kernel,),
        original_profile.environment,
    )
    evidence = profile.evidence[0]
    receipt = replace(
        original_receipt,
        profile_id=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        rule_id=rule.rule_id,
        environment_fingerprint=environment.fingerprint,
        evidence_id=evidence.evidence_id,
        evidence_result_digest=evidence.result_digest,
        kernel_id=kernel.kernel_id,
        kernel_fingerprint=kernel.fingerprint,
        artifact_fingerprint=kernel.artifact.fingerprint,
        launch_abi_fingerprint=kernel.launch_abi.fingerprint,
        binary_abi_fingerprint=kernel.binary_abi.fingerprint,
    )
    binding = bind_attention_kv_physical_layout(
        kv, catalog, profile, kernel, environment, receipt
    )
    return kv, catalog, profile, kernel, receipt, binding


class AttentionQuantizedKVLaunchABITests(unittest.TestCase):
    def test_v1_is_frozen_and_v2_has_an_independent_binary_identity(self):
        self.assertEqual(ATTENTION_KV_CACHE_VIEW_V2_C_ABI.size_bytes, 192)
        self.assertEqual(
            ATTENTION_KV_CACHE_VIEW_V2_C_ABI.fingerprint,
            "b91698a86a8c0c141a0a6959b4c8c16970270d49444ab7afc995cb17923f196c",
        )
        self.assertEqual(
            dict(ATTENTION_KV_CACHE_VIEW_V2_C_ABI.field_offsets),
            {
                "components_ptr": 0,
                "component_count": 8,
                "layout_code": 12,
                "flags": 14,
                "physical_layout_access_code": 16,
                "reserved_header": 18,
                "quant_spec_fingerprint": 24,
                "physical_layout_descriptor_fingerprint": 56,
                "physical_layout_catalog_fingerprint": 88,
                "physical_layout_binding_fingerprint": 120,
                "dispatch_receipt_fingerprint": 152,
                "reserved": 184,
            },
        )
        self.assertEqual(
            attention_kernel_binary_abi_v2().abi_name,
            "flashinfer_npu.attention.binary.v2",
        )
        self.assertEqual(
            attention_kernel_binary_abi_v2().fingerprint,
            "3087d1b9867361fbb9beac7a116ca60d15964be575943e0258481656b4f721ac",
        )
        self.assertNotEqual(
            attention_kernel_binary_abi_v2().fingerprint,
            attention_kernel_binary_abi().fingerprint,
        )
        v1_kv = next(
            item for item in attention_kernel_binary_abi().arguments if item.name == "kv"
        )
        v2_kv = next(
            item for item in attention_kernel_binary_abi_v2().arguments if item.name == "kv"
        )
        self.assertNotEqual(
            v1_kv.pointee_abi_fingerprint, v2_kv.pointee_abi_fingerprint
        )

    def test_native_layout_binding_and_pod_v2_round_trip(self):
        kv, catalog, _, _, receipt, binding = _native_fixture()
        bindings = _device_bindings(kv)
        table = _table_lease(len(bindings))
        pod = materialize_attention_kv_cache_view_v2(
            kv,
            bindings,
            table,
            0,
            catalog=catalog,
            physical_layout_binding=binding,
            dispatch_receipt=receipt,
        )
        self.assertTrue(pod.flags & AttentionKVFlags.PHYSICAL_LAYOUT)
        self.assertEqual(
            pod.physical_layout_access_code,
            AttentionKVPhysicalLayoutAccessCode.KERNEL_NATIVE,
        )
        self.assertEqual(
            pod.physical_layout_descriptor_fingerprint,
            catalog.descriptors[0].fingerprint,
        )
        self.assertEqual(pod.physical_layout_catalog_fingerprint, catalog.fingerprint)
        self.assertEqual(pod.physical_layout_binding_fingerprint, binding.fingerprint)
        self.assertEqual(pod.dispatch_receipt_fingerprint, receipt.fingerprint)
        payload = pod.pack()
        self.assertEqual(len(payload), 192)
        restored = AttentionKVCacheViewPODV2.from_bytes(payload, pod.component_blob)
        self.assertEqual(restored, pod)
        self.assertEqual(restored.fingerprint, pod.fingerprint)
        self.assertEqual(
            type(binding).from_dict(binding.to_dict()), binding
        )

    def test_v1_and_incomplete_v2_reject_nonlogical_layout(self):
        kv, catalog, _, _, receipt, binding = _native_fixture()
        bindings = _device_bindings(kv)
        table = _table_lease(len(bindings))
        with self.assertRaisesRegex(SchemaError, "POD v1"):
            materialize_attention_kv_cache_view(kv, bindings, table, 0)
        with self.assertRaisesRegex(SchemaError, "requires catalog"):
            materialize_attention_kv_cache_view_v2(kv, bindings, table, 0)
        with self.assertRaisesRegex(SchemaError, "stale"):
            materialize_attention_kv_cache_view_v2(
                kv,
                bindings,
                table,
                0,
                catalog=catalog,
                physical_layout_binding=replace(
                    binding, descriptor_fingerprint="1" * 64
                ),
                dispatch_receipt=receipt,
            )

    def test_conversion_plan_is_not_launch_authorization(self):
        kv, catalog, _, _, receipt, _ = _native_fixture()
        plan = plan_quant_layout_conversion(
            kv.key.logical_shape,
            uint8_spec(),
            kv.spec.quant_spec,
            catalog,
        )
        with self.assertRaisesRegex(TypeError, "physical_layout_binding"):
            materialize_attention_kv_cache_view_v2(
                kv,
                _device_bindings(kv),
                _table_lease(len(kv.named_component_views)),
                0,
                catalog=catalog,
                physical_layout_binding=plan,
                dispatch_receipt=receipt,
            )

    def test_capability_requires_a_matched_v2_launch_and_binary_abi(self):
        _, _, profile, kernel, _, _ = _native_fixture()
        self.assertEqual(validate_attention_kernel_bindings((profile,), (kernel,)), 1)
        v1 = replace(
            kernel,
            launch_abi=attention_launch_abi(),
            binary_abi=attention_kernel_binary_abi(),
        )
        with self.assertRaisesRegex(SchemaError, "requires KV POD v2"):
            validate_attention_kernel_bindings((profile,), (v1,))
        mixed = replace(kernel, launch_abi=attention_launch_abi())
        with self.assertRaisesRegex(SchemaError, "launch ABI is incompatible"):
            validate_attention_kernel_bindings((profile,), (mixed,))

    def test_v2_reserved_bytes_and_physical_evidence_are_canonical(self):
        kv, catalog, _, _, receipt, binding = _native_fixture()
        pod = materialize_attention_kv_cache_view_v2(
            kv,
            _device_bindings(kv),
            _table_lease(len(kv.named_component_views)),
            0,
            catalog=catalog,
            physical_layout_binding=binding,
            dispatch_receipt=receipt,
        )
        corrupted = bytearray(pod.pack())
        corrupted[-1] = 1
        with self.assertRaisesRegex(SchemaError, "reserved field"):
            AttentionKVCacheViewPODV2.from_bytes(
                bytes(corrupted), pod.component_blob
            )
        with self.assertRaisesRegex(SchemaError, "evidence is incomplete"):
            replace(pod, physical_layout_binding_fingerprint="0" * 64)
        with self.assertRaisesRegex(SchemaError, "canonical zero"):
            replace(
                pod,
                flags=pod.flags & ~AttentionKVFlags.PHYSICAL_LAYOUT,
                physical_layout_access_code=AttentionKVPhysicalLayoutAccessCode.LOGICAL,
            )


if __name__ == "__main__":
    unittest.main()
