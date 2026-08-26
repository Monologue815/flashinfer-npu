import math
import unittest

from flashinfer_npu.attention import (
    BatchAttention,
    KVLayout,
    PagedKVCacheSpec,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
)
from flashinfer_npu.decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    single_decode_with_kv_cache,
)
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
)
from flashinfer_npu.runtime import QuantSpec, SchemaError
from flashinfer_npu.attention.frontend import single_fp8_per_head_quant_spec


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def workspace(size=256):
    return ReferenceTensor.zeros((size,), dtype="uint8", device="cpu")


def symmetric_spec(storage_dtype="int8", packing_order=None):
    return QuantSpec(
        scheme="symmetric",
        storage_dtype=storage_dtype,
        compute_dtype="float32",
        accumulator_dtype="float32",
        scale_dtype="float32",
        granularity="tensor",
        physical_layout="logical",
        packing_order=packing_order,
    )


def quantized(logical_shape, storage, scale, spec, zero_point=None):
    return ReferenceQuantizedTensor(
        logical_shape=logical_shape,
        storage=storage,
        scale=scale,
        quant_spec=spec,
        zero_point=zero_point,
    )


class QuantizedTensorContractTests(unittest.TestCase):
    def test_fp8_per_head_reference_uses_decoded_storage_values(self):
        spec = single_fp8_per_head_quant_spec("float8_e4m3fn", "NHD")
        value = quantized(
            (2, 2, 1),
            tensor(
                [[[1.0], [2.0]], [[-0.5], [4.0]]],
                dtype="float8_e4m3fn",
            ),
            tensor([0.5, 0.25]),
            spec,
        )

        self.assertEqual(value.scale_shape, (2,))
        self.assertEqual(value.dequantize().data, (0.5, 0.5, -0.25, 1.0))

    def test_int8_per_tensor_dequantizes_without_materializing_kv(self):
        value = quantized(
            (2, 1, 2),
            tensor([[[2, -4]], [[6, 8]]], dtype="int8"),
            tensor(0.5),
            symmetric_spec(),
        )
        self.assertEqual(value.quantized_at(0, 0, 1), -4)
        self.assertEqual(value.dequantized_at(1, 0, 0), 3.0)
        self.assertEqual(value.dequantize().data, (1.0, -2.0, 3.0, 4.0))

    def test_uint8_asymmetric_per_token_uses_explicit_zero_point(self):
        spec = QuantSpec(
            scheme="asymmetric",
            storage_dtype="uint8",
            compute_dtype="float32",
            accumulator_dtype="float32",
            granularity="token",
            axis=(0,),
            has_zero_point=True,
        )
        value = quantized(
            (2, 1, 1),
            tensor([[[3]], [[6]]], dtype="uint8"),
            tensor([0.5, 2.0]),
            spec,
            tensor([1, 2], dtype="int32"),
        )
        self.assertEqual(value.dequantize().data, (1.0, 8.0))

    def test_group_scales_index_logical_axes(self):
        spec = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="float32",
            accumulator_dtype="float32",
            granularity="group",
            axis=(0, 1, 2),
            group_size=(1, 1, 2),
        )
        value = quantized(
            (2, 1, 4),
            tensor([[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype="int8"),
            tensor([[[1, 2]], [[3, 4]]]),
            spec,
        )
        self.assertEqual(value.scale_shape, (2, 1, 2))
        self.assertEqual(value.dequantize().data, (1, 1, 2, 2, 3, 3, 4, 4))

    def test_packed_int4_decodes_both_nibble_orders(self):
        low = quantized(
            (1, 1, 3),
            tensor([[[0xF1, 0x02]]], dtype="uint8"),
            tensor(1.0),
            symmetric_spec("int4_packed", "low_nibble_first"),
        )
        high = quantized(
            (1, 1, 2),
            tensor([[[0x1F]]], dtype="uint8"),
            tensor(1.0),
            symmetric_spec("int4_packed", "high_nibble_first"),
        )
        self.assertEqual(low.dequantize().data, (1.0, -1.0, 2.0))
        self.assertEqual(high.dequantize().data, (1.0, -1.0))

    def test_scale_and_packed_padding_validation_are_explicit(self):
        spec = symmetric_spec()
        with self.assertRaisesRegex(SchemaError, "scale shape"):
            quantized(
                (1, 1, 1),
                tensor([[[1]]], dtype="int8"),
                tensor([1.0]),
                spec,
            )
        with self.assertRaisesRegex(SchemaError, "finite and positive"):
            quantized(
                (1, 1, 1),
                tensor([[[1]]], dtype="int8"),
                tensor(0.0),
                spec,
            )
        with self.assertRaisesRegex(SchemaError, "padding nibble"):
            quantized(
                (1, 1, 3),
                tensor([[[0x11, 0xF2]]], dtype="uint8"),
                tensor(1.0),
                symmetric_spec("int4_packed", "low_nibble_first"),
            )


class QuantizedKVAttentionTests(unittest.TestCase):
    def test_single_prefill_consumes_quantized_kv(self):
        spec = symmetric_spec()
        key = quantized(
            (2, 1, 1),
            tensor([[[0]], [[0]]], dtype="int8"),
            tensor(0.5),
            spec,
        )
        value = quantized(
            (2, 1, 1),
            tensor([[[2]], [[6]]], dtype="int8"),
            tensor(2.0),
            spec,
        )
        output = single_prefill_with_kv_cache(
            tensor([[[0.0]]]), key, value, backend="reference"
        )
        self.assertEqual(output.data, (8.0,))

    def test_single_decode_int8_uses_independent_k_and_v_scales(self):
        spec = symmetric_spec()
        key = quantized(
            (2, 1, 1),
            tensor([[[2]], [[0]]], dtype="int8"),
            tensor(0.5),
            spec,
        )
        value = quantized(
            (2, 1, 1),
            tensor([[[10]], [[0]]], dtype="int8"),
            tensor(1.0),
            spec,
        )
        output = single_decode_with_kv_cache(tensor([[1.0]]), key, value)
        self.assertAlmostEqual(output.data[0], 10.0 * math.e / (math.e + 1.0))

    def test_single_decode_packed_int4_is_end_to_end_executable(self):
        spec = symmetric_spec("int4_packed", "low_nibble_first")
        key = quantized(
            (2, 1, 2),
            tensor([[[0x01]], [[0x10]]], dtype="uint8"),
            tensor(1.0),
            spec,
        )
        value = quantized(
            (2, 1, 2),
            tensor([[[0x42]], [[0x76]]], dtype="uint8"),
            tensor(1.0),
            spec,
        )
        output = single_decode_with_kv_cache(tensor([[1.0, 1.0]]), key, value)
        self.assertEqual(output.data, (4.0, 5.5))

    def test_batch_decode_quant_spec_is_in_plan_workload_and_run_contract(self):
        spec = symmetric_spec()
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            tensor([0, 1], dtype="int32"),
            tensor([0], dtype="int32"),
            tensor([2], dtype="int32"),
            1,
            1,
            1,
            2,
            q_data_type="float32",
            kv_data_type=spec,
        )
        cache_spec = PagedKVCacheSpec(
            num_pages=1,
            page_size=2,
            num_kv_heads=1,
            head_dim_qk=1,
            head_dim_vo=1,
            dtype="int8",
            layout=KVLayout.NHD,
            structure="separate",
            device="cpu",
            quant_spec=spec,
        )
        key = quantized(
            (1, 2, 1, 1),
            tensor([[[[2]], [[0]]]], dtype="int8"),
            tensor(0.5),
            spec,
        )
        value = quantized(
            (1, 2, 1, 1),
            tensor([[[[10]], [[0]]]], dtype="int8"),
            tensor(1.0),
            spec,
        )
        output = wrapper.run(
            tensor([[[1.0]]]),
            ReferenceQuantizedKVData(cache_spec, key, value),
        )
        self.assertAlmostEqual(output.data[0], 10.0 * math.e / (math.e + 1.0))
        self.assertEqual(wrapper.plan_state.workload.quant_specs, (spec,))

        ordinary_key = tensor([[[[1]], [[0]]]], dtype="int8")
        ordinary_value = tensor([[[[10]], [[0]]]], dtype="int8")
        with self.assertRaisesRegex(SchemaError, "quantized plan requires"):
            wrapper.run(tensor([[[1.0]]]), (ordinary_key, ordinary_value))

    def test_ragged_prefill_consumes_quantized_kv(self):
        spec = symmetric_spec()
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            tensor([0, 1], dtype="int32"),
            tensor([0, 2], dtype="int32"),
            1,
            1,
            1,
            q_data_type="float32",
            kv_data_type=spec,
        )
        key = quantized(
            (2, 1, 1),
            tensor([[[0]], [[0]]], dtype="int8"),
            tensor(0.25),
            spec,
        )
        value = quantized(
            (2, 1, 1),
            tensor([[[2]], [[6]]], dtype="int8"),
            tensor(2.0),
            spec,
        )
        output = wrapper.run(tensor([[[0.0]]]), key, value)
        self.assertEqual(output.data, (8.0,))

    def test_paged_prefill_consumes_quantized_kv(self):
        spec = symmetric_spec()
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            tensor([0, 1], dtype="int32"),
            tensor([0, 1], dtype="int32"),
            tensor([0], dtype="int32"),
            tensor([2], dtype="int32"),
            1,
            1,
            1,
            2,
            q_data_type="float32",
            kv_data_type=spec,
        )
        cache_spec = PagedKVCacheSpec(
            num_pages=1,
            page_size=2,
            num_kv_heads=1,
            head_dim_qk=1,
            head_dim_vo=1,
            dtype="int8",
            structure="separate",
            device="cpu",
            quant_spec=spec,
        )
        key = quantized(
            (1, 2, 1, 1),
            tensor([[[[0]], [[0]]]], dtype="int8"),
            tensor(1.0),
            spec,
        )
        value = quantized(
            (1, 2, 1, 1),
            tensor([[[[2]], [[6]]]], dtype="int8"),
            tensor(2.0),
            spec,
        )
        output = wrapper.run(
            tensor([[[0.0]]]), ReferenceQuantizedKVData(cache_spec, key, value)
        )
        self.assertEqual(output.data, (8.0,))

    def test_mixed_attention_consumes_quantized_paged_kv(self):
        spec = symmetric_spec()
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            tensor([0, 1], dtype="int32"),
            tensor([0, 1], dtype="int32"),
            tensor([0], dtype="int32"),
            tensor([2], dtype="int32"),
            1,
            1,
            1,
            1,
            2,
            q_data_type="float32",
            kv_data_type=spec,
        )
        cache_spec = PagedKVCacheSpec(
            num_pages=1,
            page_size=2,
            num_kv_heads=1,
            head_dim_qk=1,
            head_dim_vo=1,
            dtype="int8",
            structure="separate",
            device="cpu",
            quant_spec=spec,
        )
        key = quantized(
            (1, 2, 1, 1),
            tensor([[[[0]], [[0]]]], dtype="int8"),
            tensor(1.0),
            spec,
        )
        value = quantized(
            (1, 2, 1, 1),
            tensor([[[[2]], [[6]]]], dtype="int8"),
            tensor(2.0),
            spec,
        )
        output, lse = wrapper.run(
            tensor([[[0.0]]]), ReferenceQuantizedKVData(cache_spec, key, value)
        )
        self.assertEqual(output.data, (8.0,))
        self.assertAlmostEqual(lse.data[0], math.log(2.0))

    def test_mixed_attention_consumes_packed_int4_with_odd_value_dimension(self):
        spec = symmetric_spec("int4_packed", "low_nibble_first")
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            tensor([0, 1], dtype="int32"),
            tensor([0, 1], dtype="int32"),
            tensor([0], dtype="int32"),
            tensor([2], dtype="int32"),
            1,
            1,
            3,
            3,
            2,
            q_data_type="float32",
            kv_data_type=spec,
        )
        cache_spec = PagedKVCacheSpec(
            num_pages=1,
            page_size=2,
            num_kv_heads=1,
            head_dim_qk=3,
            head_dim_vo=3,
            dtype="int4_packed",
            structure="separate",
            device="cpu",
            quant_spec=spec,
        )
        key = quantized(
            (1, 2, 1, 3),
            tensor([[[[0x00, 0x00]], [[0x00, 0x00]]]], dtype="uint8"),
            tensor(1.0),
            spec,
        )
        value = quantized(
            (1, 2, 1, 3),
            tensor([[[[0xF1, 0x02]], [[0x43, 0x0E]]]], dtype="uint8"),
            tensor(1.0),
            spec,
        )
        output, lse = wrapper.run(
            tensor([[[0.0, 0.0, 0.0]]]),
            ReferenceQuantizedKVData(cache_spec, key, value),
        )
        self.assertEqual(output.data, (2.0, 1.5, 0.0))
        self.assertAlmostEqual(lse.data[0], math.log(2.0))

    def test_mixed_attention_asymmetric_per_head_and_runtime_scales(self):
        spec = QuantSpec(
            scheme="asymmetric",
            storage_dtype="uint8",
            compute_dtype="float32",
            accumulator_dtype="float32",
            granularity="channel",
            axis=(2,),
            has_zero_point=True,
        )
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            tensor([0, 1], dtype="int32"),
            tensor([0, 1], dtype="int32"),
            tensor([0], dtype="int32"),
            tensor([2], dtype="int32"),
            4,
            2,
            1,
            1,
            2,
            logits_soft_cap=2.0,
            q_data_type="float32",
            kv_data_type=spec,
        )
        cache_spec = PagedKVCacheSpec(
            num_pages=1,
            page_size=2,
            num_kv_heads=2,
            head_dim_qk=1,
            head_dim_vo=1,
            dtype="uint8",
            structure="separate",
            device="cpu",
            quant_spec=spec,
        )
        key = quantized(
            (1, 2, 2, 1),
            tensor([[[[1], [2]], [[1], [2]]]], dtype="uint8"),
            tensor([0.5, 2.0]),
            spec,
            tensor([1, 2], dtype="int32"),
        )
        value = quantized(
            (1, 2, 2, 1),
            tensor([[[[3], [4]], [[5], [1]]]], dtype="uint8"),
            tensor([0.5, 2.0]),
            spec,
            tensor([1, 2], dtype="int32"),
        )
        output, lse = wrapper.run(
            tensor([[[0.0], [0.0], [0.0], [0.0]]]),
            ReferenceQuantizedKVData(cache_spec, key, value),
            k_scale=tensor([0.75, 1.25]),
            v_scale=tensor([2.0, 0.5]),
            logits_soft_cap=1.0,
        )
        self.assertEqual(output.data, (3.0, 3.0, 0.5, 0.5))
        for item in lse.data:
            self.assertAlmostEqual(item, math.log(2.0))

    def test_quantized_and_unquantized_plans_have_different_fingerprints(self):
        metadata = (
            tensor([0, 1], dtype="int32"),
            tensor([0], dtype="int32"),
            tensor([1], dtype="int32"),
        )
        quantized_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        quantized_wrapper.plan(
            *metadata,
            1,
            1,
            1,
            1,
            q_data_type="float32",
            kv_data_type=symmetric_spec(),
        )
        ordinary_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        ordinary_wrapper.plan(
            *metadata,
            1,
            1,
            1,
            1,
            q_data_type="float32",
            kv_data_type="int8",
        )
        self.assertNotEqual(
            quantized_wrapper.plan_state.fingerprint,
            ordinary_wrapper.plan_state.fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
