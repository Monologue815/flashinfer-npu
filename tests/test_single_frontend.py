import inspect
import math
import unittest

import flashinfer_npu
from flashinfer_npu.attention.reference import ReferenceTensor
from flashinfer_npu.decode import single_decode_with_kv_cache
from flashinfer_npu.prefill import single_prefill_with_kv_cache
from flashinfer_npu.runtime import DispatchError, SchemaError


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


class SingleFrontendSignatureTests(unittest.TestCase):
    def test_single_prefill_signature_matches_flashinfer_order(self):
        signature = inspect.signature(single_prefill_with_kv_cache)
        self.assertEqual(
            list(signature.parameters),
            [
                "q", "k", "v", "scale_q", "scale_k", "scale_v", "o_dtype",
                "custom_mask", "packed_custom_mask", "causal", "kv_layout",
                "pos_encoding_mode", "use_fp16_qk_reduction", "sm_scale",
                "window_left", "logits_soft_cap", "rope_scale", "rope_theta",
                "backend", "return_lse", "kv_cache_sf", "k_scale", "v_scale",
            ],
        )
        self.assertEqual(signature.parameters["backend"].default, "auto")
        self.assertFalse(signature.parameters["return_lse"].default)

    def test_single_decode_signature_matches_flashinfer_order(self):
        signature = inspect.signature(single_decode_with_kv_cache)
        self.assertEqual(
            list(signature.parameters),
            [
                "q", "k", "v", "kv_layout", "pos_encoding_mode",
                "use_tensor_cores", "q_scale", "k_scale", "v_scale",
                "window_left", "logits_soft_cap", "sm_scale", "rope_scale",
                "rope_theta", "return_lse",
            ],
        )
        self.assertEqual(signature.parameters["kv_layout"].default, "NHD")
        self.assertFalse(signature.parameters["return_lse"].default)

    def test_package_root_reexports_single_attention_functions(self):
        self.assertIs(
            flashinfer_npu.single_prefill_with_kv_cache,
            single_prefill_with_kv_cache,
        )
        self.assertIs(
            flashinfer_npu.single_decode_with_kv_cache,
            single_decode_with_kv_cache,
        )


class SinglePrefillFrontendTests(unittest.TestCase):
    def setUp(self):
        self.q = tensor([[[0.0]]])
        self.k = tensor([[[1.0]], [[2.0]]])
        self.v = tensor([[[4.0]], [[8.0]]])

    def test_auto_backend_never_silently_selects_reference(self):
        with self.assertRaisesRegex(DispatchError, "explicitly"):
            single_prefill_with_kv_cache(self.q, self.k, self.v)

    def test_basic_output_lse_and_output_dtype(self):
        output, lse = single_prefill_with_kv_cache(
            self.q,
            self.k,
            self.v,
            backend="reference",
            o_dtype="float16",
            return_lse=True,
        )
        self.assertEqual(output.shape, (1, 1, 1))
        self.assertEqual(output.dtype, "float16")
        self.assertAlmostEqual(output.data[0], 6.0)
        self.assertEqual(lse.shape, (1, 1))
        self.assertAlmostEqual(lse.data[0], math.log(2.0))

    def test_hnd_layout_supports_distinct_value_dimension_and_gqa(self):
        q = tensor([[[0.0], [0.0]]])
        k = tensor([[[1.0], [2.0]]])
        v = tensor([[[2.0, 4.0], [6.0, 8.0]]])
        output = single_prefill_with_kv_cache(
            q, k, v, kv_layout="HND", backend="reference"
        )
        self.assertEqual(output.shape, (1, 2, 2))
        self.assertEqual(output.data, (4.0, 6.0, 4.0, 6.0))

    def test_packed_mask_takes_precedence_and_uses_little_endian_bits(self):
        ignored_invalid_custom_mask = tensor([True], dtype="bool")
        packed_mask = tensor([2], dtype="uint8")
        output = single_prefill_with_kv_cache(
            self.q,
            self.k,
            tensor([[[10.0]], [[20.0]]]),
            custom_mask=ignored_invalid_custom_mask,
            packed_custom_mask=packed_mask,
            backend="reference",
        )
        self.assertEqual(output.data, (20.0,))

    def test_unpacked_mask_and_reference_mask_value_validation(self):
        output = single_prefill_with_kv_cache(
            self.q,
            self.k,
            tensor([[[10.0]], [[20.0]]]),
            custom_mask=tensor([[True, False]], dtype="bool"),
            backend="reference",
        )
        self.assertEqual(output.data, (10.0,))
        with self.assertRaisesRegex(SchemaError, "boolean"):
            single_prefill_with_kv_cache(
                self.q,
                self.k,
                self.v,
                custom_mask=tensor([[0, 2]], dtype="bool"),
                backend="reference",
            )
        with self.assertRaisesRegex(SchemaError, "integers"):
            single_prefill_with_kv_cache(
                self.q,
                self.k,
                self.v,
                packed_custom_mask=tensor([1.5], dtype="uint8"),
                backend="reference",
            )

    def test_per_head_and_legacy_scales_compose(self):
        q = tensor([[[1.0], [1.0]]])
        k = tensor([[[1.0]], [[0.0]]])
        v = tensor([[[10.0]], [[0.0]]])
        output = single_prefill_with_kv_cache(
            q,
            k,
            v,
            scale_q=tensor([1.0, 2.0]),
            scale_k=tensor([0.5]),
            scale_v=tensor([0.5]),
            sm_scale=1.0,
            k_scale=2.0,
            v_scale=3.0,
            backend="reference",
        )
        expected_head_0 = 15.0 * math.exp(1.0) / (math.exp(1.0) + 1.0)
        expected_head_1 = 15.0 * math.exp(2.0) / (math.exp(2.0) + 1.0)
        self.assertAlmostEqual(output.data[0], expected_head_0)
        self.assertAlmostEqual(output.data[1], expected_head_1)

    def test_nvfp4_scale_factor_is_an_explicit_gap(self):
        with self.assertRaisesRegex(NotImplementedError, "NVFP4"):
            single_prefill_with_kv_cache(
                self.q,
                self.k,
                self.v,
                kv_cache_sf=tensor([1.0]),
                backend="reference",
            )


class SingleDecodeFrontendTests(unittest.TestCase):
    def test_nhd_and_hnd_layouts_return_flashinfer_shapes(self):
        q = tensor([[0.0], [0.0]])
        for layout, k, v in (
            ("NHD", tensor([[[1.0]], [[2.0]]]), tensor([[[2.0]], [[6.0]]])),
            ("HND", tensor([[[1.0], [2.0]]]), tensor([[[2.0], [6.0]]])),
        ):
            with self.subTest(layout=layout):
                output, lse = single_decode_with_kv_cache(
                    q, k, v, kv_layout=layout, return_lse=True
                )
                self.assertEqual(output.shape, (2, 1))
                self.assertEqual(output.data, (4.0, 4.0))
                self.assertEqual(lse.shape, (2,))
                self.assertAlmostEqual(lse.data[0], math.log(2.0))

    def test_decode_scalar_scales_compose(self):
        output = single_decode_with_kv_cache(
            tensor([[1.0]]),
            tensor([[[1.0]], [[0.0]]]),
            tensor([[[10.0]], [[0.0]]]),
            q_scale=2.0,
            k_scale=0.5,
            v_scale=3.0,
            sm_scale=1.0,
            use_tensor_cores=True,
        )
        expected = 30.0 * math.exp(1.0) / (math.exp(1.0) + 1.0)
        self.assertAlmostEqual(output.data[0], expected)

    def test_decode_rejects_unsupported_value_dimension(self):
        with self.assertRaisesRegex(SchemaError, "equal"):
            single_decode_with_kv_cache(
                tensor([[0.0]]),
                tensor([[[1.0]]]),
                tensor([[[1.0, 2.0]]]),
            )

    def test_invalid_layout_position_mode_and_scale_are_schema_errors(self):
        q = tensor([[0.0]])
        k = tensor([[[1.0]]])
        v = tensor([[[1.0]]])
        with self.assertRaisesRegex(SchemaError, "NHD or HND"):
            single_decode_with_kv_cache(q, k, v, kv_layout="BAD")
        with self.assertRaisesRegex(SchemaError, "ROPE_LLAMA"):
            single_decode_with_kv_cache(q, k, v, pos_encoding_mode="BAD")
        with self.assertRaisesRegex(SchemaError, "scalar"):
            single_decode_with_kv_cache(q, k, v, sm_scale="BAD")


if __name__ == "__main__":
    unittest.main()
