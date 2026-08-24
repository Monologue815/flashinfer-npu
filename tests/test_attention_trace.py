import json
import math
import random
import unittest
from dataclasses import replace
from functools import reduce
from operator import mul

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionPlanSpec,
    AttentionTrace,
    AttentionTraceMismatchError,
    CustomMaskSpec,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    SingleAttentionMetadata,
    attention_metadata_from_dict,
)
from flashinfer_npu.prefill import single_prefill_with_kv_cache
from flashinfer_npu.runtime import QuantSpec, SchemaError


def tensor(value, dtype="float32"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device="cpu")


def flat_tensor(shape, values, dtype="float32"):
    return ReferenceTensor(tuple(shape), tuple(values), dtype=dtype, device="cpu")


def numel(shape):
    return reduce(mul, shape, 1)


def int8_spec():
    return QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
    )


def quantized(shape, values, scale, spec):
    storage_dtype = (
        "uint8"
        if spec.storage_dtype in {"int4_packed", "uint4_packed"}
        else spec.storage_dtype
    )
    storage_shape = (
        tuple(shape[:-1]) + ((shape[-1] + 1) // 2,)
        if spec.storage_dtype in {"int4_packed", "uint4_packed"}
        else tuple(shape)
    )
    return ReferenceQuantizedTensor(
        logical_shape=tuple(shape),
        storage=flat_tensor(storage_shape, values, storage_dtype),
        scale=tensor(scale),
        quant_spec=spec,
    )


def pack_int4(shape, values, order):
    width = shape[-1]
    rows = numel(shape[:-1])
    packed = []
    for row in range(rows):
        row_values = values[row * width : (row + 1) * width]
        for column in range(0, width, 2):
            first = row_values[column] & 0xF
            second = row_values[column + 1] & 0xF if column + 1 < width else 0
            if order == "low_nibble_first":
                packed.append(first | (second << 4))
            else:
                packed.append((first << 4) | second)
    return packed


def assert_reference_close(test, left, right):
    test.assertEqual(left.shape, right.shape)
    test.assertEqual(left.dtype, right.dtype)
    for observed, expected in zip(left.data, right.data):
        test.assertAlmostEqual(observed, expected, places=12)


class AttentionTraceSchemaTests(unittest.TestCase):
    def test_all_metadata_kinds_round_trip(self):
        paged = PagedKVMetadata((0, 1), (0,), (2,), 2)
        values = (
            SingleAttentionMetadata(1, 2),
            paged,
            RaggedKVMetadata((0, 1), (0, 2)),
            PagedPrefillMetadata((0, 1), paged),
            MixedPagedKVMetadata((0, 1), (0, 1), (0,), (2,), 2),
        )
        for metadata in values:
            with self.subTest(kind=metadata.to_dict()["kind"]):
                self.assertEqual(
                    attention_metadata_from_dict(metadata.to_dict()), metadata
                )

    def test_dense_all_mask_trace_round_trips_negative_infinity(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_PREFILL,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="float32",
            o_dtype="float32",
            custom_mask=CustomMaskSpec(2),
        )
        cache_spec = RaggedKVCacheSpec(
            total_kv_tokens=2,
            num_kv_heads=1,
            head_dim_qk=1,
            head_dim_vo=1,
            dtype="float32",
            device="cpu",
        )
        trace = AttentionTrace.capture(
            spec=spec,
            metadata=SingleAttentionMetadata(1, 2),
            q=tensor([[[1.0]]]),
            kv_data=ReferenceKVData(
                cache_spec,
                (tensor([[[1.0]], [[2.0]]]), tensor([[[3.0]], [[4.0]]])),
            ),
            custom_mask_data=(False, False),
        )
        self.assertEqual(trace.expected_output.data, (0.0,))
        self.assertEqual(trace.expected_lse.data, (float("-inf"),))

        encoded = trace.to_json()
        json.loads(encoded)  # Strict standard JSON; no bare Infinity token.
        self.assertIn('"nonfinite":"-inf"', encoded)
        restored = AttentionTrace.from_json(encoded)
        self.assertEqual(restored.fingerprint, trace.fingerprint)
        self.assertEqual(restored.input_fingerprint, trace.input_fingerprint)
        restored.replay()

    def test_quantized_paged_trace_round_trip_and_replay(self):
        quant_spec = int8_spec()
        plan_spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="int8",
            o_dtype="float32",
            kv_quant_spec=quant_spec,
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
            quant_spec=quant_spec,
        )
        key = quantized((1, 2, 1, 1), (2, 0), 0.5, quant_spec)
        value = quantized((1, 2, 1, 1), (10, 0), 1.0, quant_spec)
        trace = AttentionTrace.capture(
            spec=plan_spec,
            metadata=PagedKVMetadata((0, 1), (0,), (2,), 2),
            q=tensor([[[1.0]]]),
            kv_data=ReferenceQuantizedKVData(cache_spec, key, value),
        )
        restored = AttentionTrace.from_json(trace.to_json(indent=2))
        self.assertIsInstance(restored.kv_data, ReferenceQuantizedKVData)
        self.assertEqual(restored.spec, trace.spec)
        self.assertEqual(restored.metadata, trace.metadata)
        self.assertAlmostEqual(
            restored.replay().output.data[0], 10.0 * math.e / (math.e + 1.0)
        )

    def test_empty_paged_request_replays_zero_output_and_negative_infinity_lse(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="float32",
        )
        cache_spec = PagedKVCacheSpec(
            num_pages=0,
            page_size=2,
            num_kv_heads=1,
            head_dim_qk=1,
            head_dim_vo=1,
            dtype="float32",
            structure="separate",
            device="cpu",
        )
        empty = flat_tensor((0, 2, 1, 1), ())
        trace = AttentionTrace.capture(
            spec=spec,
            metadata=PagedKVMetadata((0, 0), (), (0,), 2),
            q=tensor([[[1.0]]]),
            kv_data=ReferenceKVData(cache_spec, (empty, empty)),
        )
        self.assertEqual(trace.expected_output.data, (0.0,))
        self.assertEqual(trace.expected_lse.data, (float("-inf"),))
        AttentionTrace.from_json(trace.to_json()).replay()

    def test_dense_storage_cannot_claim_a_quantized_cache_contract(self):
        quant_spec = int8_spec()
        cache_spec = RaggedKVCacheSpec(
            1, 1, 1, 1, "int8", device="cpu", quant_spec=quant_spec
        )
        with self.assertRaisesRegex(SchemaError, "dense reference KV"):
            ReferenceKVData(
                cache_spec,
                (tensor([[[1]]], "int8"), tensor([[[2]]], "int8")),
            )

    def test_replay_detects_oracle_drift_without_changing_input_identity(self):
        quant_spec = int8_spec()
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_DECODE,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="int8",
            kv_quant_spec=quant_spec,
        )
        cache_spec = RaggedKVCacheSpec(
            1, 1, 1, 1, "int8", device="cpu", quant_spec=quant_spec
        )
        key = quantized((1, 1, 1), (1,), 1.0, quant_spec)
        value = quantized((1, 1, 1), (2,), 1.0, quant_spec)
        trace = AttentionTrace.capture(
            spec=spec,
            metadata=SingleAttentionMetadata(1, 1),
            q=tensor([[1.0]]),
            kv_data=ReferenceQuantizedKVData(cache_spec, key, value),
        )
        changed = replace(
            trace,
            expected_output=ReferenceTensor(
                trace.expected_output.shape,
                (trace.expected_output.data[0] + 1.0,),
                trace.expected_output.dtype,
                trace.expected_output.device,
            ),
        )
        self.assertEqual(changed.input_fingerprint, trace.input_fingerprint)
        self.assertNotEqual(changed.fingerprint, trace.fingerprint)
        with self.assertRaisesRegex(AttentionTraceMismatchError, "flat index 0"):
            changed.replay()

    def test_trace_schema_rejects_unknown_version_and_fields(self):
        quant_spec = int8_spec()
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_DECODE,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="int8",
            kv_quant_spec=quant_spec,
        )
        cache_spec = RaggedKVCacheSpec(
            1, 1, 1, 1, "int8", device="cpu", quant_spec=quant_spec
        )
        trace = AttentionTrace(
            spec=spec,
            metadata=SingleAttentionMetadata(1, 1),
            q=tensor([[1.0]]),
            kv_data=ReferenceQuantizedKVData(
                cache_spec,
                quantized((1, 1, 1), (1,), 1.0, quant_spec),
                quantized((1, 1, 1), (2,), 1.0, quant_spec),
            ),
        )
        invalid_version = trace.to_dict()
        invalid_version["schema_version"] = 99
        with self.assertRaisesRegex(SchemaError, "schema version"):
            AttentionTrace.from_dict(invalid_version)
        invalid_fields = trace.to_dict()
        invalid_fields["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "fields"):
            AttentionTrace.from_dict(invalid_fields)


class QuantizedKVPropertyTests(unittest.TestCase):
    def test_random_int8_kv_matches_explicit_dequantization(self):
        rng = random.Random(20260805)
        spec = int8_spec()
        for layout in ("NHD", "HND"):
            for case in range(12):
                kv_len = rng.randint(1, 5)
                qo_len = rng.randint(1, kv_len)
                kv_heads = rng.randint(1, 2)
                qo_heads = kv_heads * rng.randint(1, 2)
                qk_dim = rng.randint(1, 5)
                vo_dim = rng.randint(1, 5)
                k_shape = (
                    (kv_len, kv_heads, qk_dim)
                    if layout == "NHD"
                    else (kv_heads, kv_len, qk_dim)
                )
                v_shape = (
                    (kv_len, kv_heads, vo_dim)
                    if layout == "NHD"
                    else (kv_heads, kv_len, vo_dim)
                )
                key = quantized(
                    k_shape,
                    tuple(rng.randint(-8, 7) for _ in range(numel(k_shape))),
                    0.25,
                    spec,
                )
                value = quantized(
                    v_shape,
                    tuple(rng.randint(-8, 7) for _ in range(numel(v_shape))),
                    0.5,
                    spec,
                )
                q = flat_tensor(
                    (qo_len, qo_heads, qk_dim),
                    tuple(
                        rng.randint(-4, 4) / 4.0
                        for _ in range(qo_len * qo_heads * qk_dim)
                    ),
                )
                quant_output, quant_lse = single_prefill_with_kv_cache(
                    q,
                    key,
                    value,
                    causal=bool(case % 2),
                    kv_layout=layout,
                    backend="reference",
                    return_lse=True,
                )
                dense_output, dense_lse = single_prefill_with_kv_cache(
                    q,
                    key.dequantize(),
                    value.dequantize(),
                    causal=bool(case % 2),
                    kv_layout=layout,
                    backend="reference",
                    return_lse=True,
                )
                assert_reference_close(self, quant_output, dense_output)
                assert_reference_close(self, quant_lse, dense_lse)

    def test_random_packed_int4_kv_matches_explicit_dequantization(self):
        rng = random.Random(4)
        for order in ("low_nibble_first", "high_nibble_first"):
            spec = QuantSpec(
                scheme="symmetric",
                storage_dtype="int4_packed",
                compute_dtype="float32",
                accumulator_dtype="float32",
                packing_order=order,
            )
            for layout in ("NHD", "HND"):
                for dim in range(1, 6):
                    kv_len = 3
                    heads = 2
                    shape = (
                        (kv_len, heads, dim)
                        if layout == "NHD"
                        else (heads, kv_len, dim)
                    )
                    key_values = [rng.randint(-8, 7) for _ in range(numel(shape))]
                    value_values = [rng.randint(-8, 7) for _ in range(numel(shape))]
                    key = quantized(
                        shape, pack_int4(shape, key_values, order), 0.25, spec
                    )
                    value = quantized(
                        shape, pack_int4(shape, value_values, order), 0.5, spec
                    )
                    q = flat_tensor(
                        (2, heads, dim),
                        tuple(
                            rng.randint(-2, 2) / 2.0
                            for _ in range(2 * heads * dim)
                        ),
                    )
                    quant_output, quant_lse = single_prefill_with_kv_cache(
                        q,
                        key,
                        value,
                        causal=True,
                        kv_layout=layout,
                        backend="reference",
                        return_lse=True,
                    )
                    dense_output, dense_lse = single_prefill_with_kv_cache(
                        q,
                        key.dequantize(),
                        value.dequantize(),
                        causal=True,
                        kv_layout=layout,
                        backend="reference",
                        return_lse=True,
                    )
                    assert_reference_close(self, quant_output, dense_output)
                    assert_reference_close(self, quant_lse, dense_lse)


if __name__ == "__main__":
    unittest.main()
