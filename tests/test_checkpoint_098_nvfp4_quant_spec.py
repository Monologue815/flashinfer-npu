import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorNvfp4ScaleFactorBinding,
    AttentionOperatorNvfp4ScaleFactorRunAdapter,
    AttentionTensorAccessPolicy,
    attention_nvfp4_kv_quant_spec,
    infer_attention_nvfp4_packed_storage_shape,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_096_nvfp4_scale_factor_contract import (
    FakeInspector,
    FakeTensor,
)
from tests.test_checkpoint_097_nvfp4_operation_binding import (
    BaseAdapter,
    NVFP4_QUANT_SPEC,
    OPERATION_ID,
    active_plan,
    operation,
    request,
)


class Nvfp4QuantSpecCheckpoint(unittest.TestCase):
    """NVFP4 lowering is inseparable from one exact packed QuantSpec."""

    def test_canonical_builder_keeps_provider_layout_and_packing_explicit(self):
        spec = attention_nvfp4_kv_quant_spec(
            physical_layout="ascend_nvfp4_fractal_v1",
            packing_order="ascend_e2m1_pair_v1",
            compute_dtype="float16",
            accumulator_dtype="float32",
        )

        self.assertEqual(spec.scheme, "symmetric")
        self.assertEqual(spec.storage_dtype, "uint8")
        self.assertEqual(spec.scale_dtype, "float8_e4m3fn")
        self.assertEqual(spec.granularity, "block")
        self.assertEqual(spec.group_size, (16,))
        self.assertEqual(spec.axis, (-1,))
        self.assertEqual(spec.physical_layout, "ascend_nvfp4_fractal_v1")
        self.assertEqual(spec.packing_order, "ascend_e2m1_pair_v1")

        with self.assertRaisesRegex(SchemaError, "non-logical"):
            attention_nvfp4_kv_quant_spec(
                physical_layout="logical", packing_order="e2m1-pair"
            )
        with self.assertRaisesRegex(SchemaError, "packing order"):
            attention_nvfp4_kv_quant_spec(
                physical_layout="ascend_nvfp4_v1", packing_order=""
            )

    def test_packed_storage_shape_contains_two_values_per_byte(self):
        self.assertEqual(
            infer_attention_nvfp4_packed_storage_shape((6, 16, 2, 128)),
            (6, 16, 2, 64),
        )
        self.assertEqual(
            infer_attention_nvfp4_packed_storage_shape((2, 7, 64)),
            (2, 7, 32),
        )
        with self.assertRaisesRegex(SchemaError, "must be even"):
            infer_attention_nvfp4_packed_storage_shape((2, 7, 63))

    def test_binding_rejects_nearby_non_nvfp4_quant_specs(self):
        invalid_specs = (
            replace(NVFP4_QUANT_SPEC, scale_dtype="float16"),
            replace(NVFP4_QUANT_SPEC, granularity="tensor", group_size=None, axis=None),
            replace(NVFP4_QUANT_SPEC, physical_layout="logical"),
            replace(NVFP4_QUANT_SPEC, packing_order=None),
        )
        for spec in invalid_specs:
            with self.subTest(spec=spec.to_dict()):
                with self.assertRaisesRegex(SchemaError, "not canonical"):
                    AttentionOperatorNvfp4ScaleFactorBinding(
                        provider_id="cann",
                        operation_id=OPERATION_ID,
                        quant_spec=spec,
                        combined_argument="kv_cache_sf",
                    )

    def test_plan_quant_spec_drift_fails_before_metadata_or_base_lowering(self):
        binding = AttentionOperatorNvfp4ScaleFactorBinding(
            provider_id="cann",
            operation_id=OPERATION_ID,
            quant_spec=NVFP4_QUANT_SPEC,
            combined_argument="kv_cache_sf",
        )
        inspector = FakeInspector()
        base = BaseAdapter()
        adapter = AttentionOperatorNvfp4ScaleFactorRunAdapter(
            base,
            operation(),
            binding,
            inspector,
            AttentionTensorAccessPolicy(required_alignment=16),
            "npu:0",
        )
        other_spec = attention_nvfp4_kv_quant_spec(
            physical_layout="other_nvfp4_layout_v1",
            packing_order="test_low_nibble_first",
        )
        active = active_plan(other_spec)
        scale = FakeTensor("combined", (6, 2, 16, 2, 8))

        with self.assertRaisesRegex(SchemaError, "bound QuantSpec"):
            adapter.lower(active, request(active, scale))

        self.assertEqual(inspector.calls, [])
        self.assertEqual(base.calls, 0)

    def test_binding_fingerprint_changes_with_layout_or_packing(self):
        first = AttentionOperatorNvfp4ScaleFactorBinding(
            provider_id="cann", operation_id=OPERATION_ID,
            quant_spec=NVFP4_QUANT_SPEC, combined_argument="kv_cache_sf",
        )
        second_spec = attention_nvfp4_kv_quant_spec(
            physical_layout=NVFP4_QUANT_SPEC.physical_layout,
            packing_order="test_high_nibble_first",
        )
        second = replace(first, quant_spec=second_spec)

        self.assertNotEqual(first.quant_spec.fingerprint, second.quant_spec.fingerprint)
        self.assertNotEqual(first.fingerprint, second.fingerprint)


if __name__ == "__main__":
    unittest.main()
