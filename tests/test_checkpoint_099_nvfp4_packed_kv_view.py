import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionNvfp4PackedLayoutDescriptor,
    AttentionPlanSpec,
    KVLayout,
    SingleAttentionMetadata,
    attention_nvfp4_kv_quant_spec,
    infer_quant_scale_shape,
    inspect_attention_nvfp4_packed_kv_input,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_096_nvfp4_scale_factor_contract import (
    FakeInspector,
    FakeTensor,
    paged_plan,
)


NVFP4_QUANT_SPEC = attention_nvfp4_kv_quant_spec(
    physical_layout="test_nvfp4_linear_pair_v1",
    packing_order="test_low_nibble_first",
)


def descriptor(*, storage_alignment=16, scale_alignment=16):
    return AttentionNvfp4PackedLayoutDescriptor(
        physical_layout=NVFP4_QUANT_SPEC.physical_layout,
        packing_order=NVFP4_QUANT_SPEC.packing_order,
        storage_required_alignment=storage_alignment,
        scale_required_alignment=scale_alignment,
    )


def tensor(name, shape, *, dtype, device="npu:0", strides=None, alignment=64):
    result = FakeTensor(
        name, shape, dtype=dtype, device=device, strides=strides
    )
    result.tensor_view = replace(
        result.tensor_view, data_ptr_alignment=alignment
    )
    return result


def storage(name, shape, **kwargs):
    return tensor(name, shape, dtype="uint8", **kwargs)


def scale(name, shape, **kwargs):
    return tensor(name, shape, dtype="float8_e4m3fn", **kwargs)


class Nvfp4PackedKvViewCheckpoint(unittest.TestCase):
    """Packed bytes and block scales close one plan-bound NVFP4 input."""

    def test_layout_descriptor_round_trip_preserves_audit_identity(self):
        value = descriptor(storage_alignment=32, scale_alignment=16)
        restored = AttentionNvfp4PackedLayoutDescriptor.from_dict(value.to_dict())

        self.assertEqual(restored, value)
        self.assertEqual(restored.fingerprint, value.fingerprint)
        invalid = value.to_dict()
        invalid["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            AttentionNvfp4PackedLayoutDescriptor.from_dict(invalid)

    def test_paged_nhd_separate_storage_and_scales_form_one_view(self):
        plan = paged_plan(quant_spec=NVFP4_QUANT_SPEC)
        result = inspect_attention_nvfp4_packed_kv_input(
            plan,
            (
                storage("key", (6, 16, 2, 64)),
                storage("value", (6, 16, 2, 64)),
            ),
            (
                scale("key-sf", (6, 16, 2, 8)),
                scale("value-sf", (6, 16, 2, 8)),
            ),
            FakeInspector(),
            "npu:0",
            descriptor(),
        )

        self.assertEqual(result.structure, "separate")
        self.assertEqual(result.key_logical_shape, (6, 16, 2, 128))
        self.assertEqual(result.value_logical_shape, (6, 16, 2, 128))
        self.assertEqual(
            tuple(name for name, _ in result.named_views),
            (
                "kv.key.storage",
                "kv.value.storage",
                "kv_cache_sf.key",
                "kv_cache_sf.value",
            ),
        )
        self.assertEqual(len(result.fingerprint), 64)
        self.assertEqual(
            infer_quant_scale_shape(result.key_logical_shape, NVFP4_QUANT_SPEC),
            (8,),
        )

    def test_paged_hnd_combined_storage_and_scales_share_kv_axis(self):
        plan = paged_plan(KVLayout.HND, quant_spec=NVFP4_QUANT_SPEC)
        result = inspect_attention_nvfp4_packed_kv_input(
            plan,
            storage("combined", (6, 2, 2, 16, 64)),
            scale("combined-sf", (6, 2, 2, 16, 8)),
            FakeInspector(),
            "npu:0",
            descriptor(),
        )

        self.assertEqual(result.structure, "combined")
        self.assertEqual(result.key_logical_shape, (6, 2, 16, 128))
        self.assertEqual(
            tuple(name for name, _ in result.named_views),
            ("kv.packed_storage", "kv_cache_sf"),
        )

    def test_single_unequal_kv_dimensions_require_separate_components(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_PREFILL,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            head_dim_vo=64,
            kv_layout=KVLayout.NHD,
            q_dtype="bfloat16",
            kv_dtype=NVFP4_QUANT_SPEC.storage_dtype,
            kv_quant_spec=NVFP4_QUANT_SPEC,
            o_dtype="bfloat16",
        )
        plan = AttentionFrameworkSession(spec.mode).plan(
            spec, SingleAttentionMetadata(qo_len=3, kv_len=7)
        )
        result = inspect_attention_nvfp4_packed_kv_input(
            plan,
            (
                storage("key", (7, 2, 64)),
                storage("value", (7, 2, 32)),
            ),
            (
                scale("key-sf", (7, 2, 8)),
                scale("value-sf", (7, 2, 4)),
            ),
            FakeInspector(),
            "npu:0",
            descriptor(),
        )

        self.assertEqual(result.key_logical_shape, (7, 2, 128))
        self.assertEqual(result.value_logical_shape, (7, 2, 64))

    def test_descriptor_drift_fails_before_tensor_observation(self):
        plan = paged_plan(quant_spec=NVFP4_QUANT_SPEC)
        inspector = FakeInspector()
        wrong = AttentionNvfp4PackedLayoutDescriptor(
            physical_layout="another_nvfp4_layout",
            packing_order=NVFP4_QUANT_SPEC.packing_order,
        )

        with self.assertRaisesRegex(SchemaError, "differs from QuantSpec"):
            inspect_attention_nvfp4_packed_kv_input(
                plan,
                storage("combined", (6, 2, 16, 2, 64)),
                scale("combined-sf", (6, 2, 16, 2, 8)),
                inspector,
                "npu:0",
                wrong,
            )

        self.assertEqual(inspector.calls, [])

    def test_shape_dtype_contiguity_alignment_and_structure_fail_closed(self):
        plan = paged_plan(quant_spec=NVFP4_QUANT_SPEC)
        good_scale = scale("combined-sf", (6, 2, 16, 2, 8))
        cases = (
            (storage("shape", (6, 2, 16, 2, 63)), descriptor(), "shape"),
            (
                tensor("dtype", (6, 2, 16, 2, 64), dtype="int8"),
                descriptor(),
                "dtype",
            ),
            (
                storage(
                    "stride",
                    (6, 2, 16, 2, 64),
                    strides=(5000, 2048, 128, 64, 1),
                ),
                descriptor(),
                "contiguous",
            ),
            (
                storage("alignment", (6, 2, 16, 2, 64), alignment=16),
                descriptor(storage_alignment=32),
                "32-byte aligned",
            ),
        )
        for packed, layout, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    inspect_attention_nvfp4_packed_kv_input(
                        plan,
                        packed,
                        good_scale,
                        FakeInspector(),
                        "npu:0",
                        layout,
                    )

        with self.assertRaisesRegex(SchemaError, "structures differ"):
            inspect_attention_nvfp4_packed_kv_input(
                plan,
                (
                    storage("key", (6, 16, 2, 64)),
                    storage("value", (6, 16, 2, 64)),
                ),
                good_scale,
                FakeInspector(),
                "npu:0",
                descriptor(),
            )

    def test_storage_scale_alias_is_rejected(self):
        plan = paged_plan(quant_spec=NVFP4_QUANT_SPEC)
        with self.assertRaisesRegex(SchemaError, "cannot alias"):
            inspect_attention_nvfp4_packed_kv_input(
                plan,
                storage("shared", (6, 2, 16, 2, 64)),
                scale("shared", (6, 2, 16, 2, 8)),
                FakeInspector(),
                "npu:0",
                descriptor(),
            )


if __name__ == "__main__":
    unittest.main()
