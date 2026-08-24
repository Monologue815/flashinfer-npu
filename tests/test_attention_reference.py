import math
import unittest

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionPlanSpec,
    AttentionStateError,
    AttentionWrapper,
    CustomMaskSpec,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    ReferenceAttentionExecutor,
    ReferenceKVData,
    ReferenceTensor,
    SingleAttentionMetadata,
)
from flashinfer_npu.runtime import SchemaError


def _plan(mode, metadata, **overrides):
    values = {
        "mode": mode,
        "num_qo_heads": 1,
        "num_kv_heads": 1,
        "head_dim_qk": 1,
        "q_dtype": "float32",
        "sm_scale": 1.0,
    }
    values.update(overrides)
    return AttentionFrameworkSession(mode).plan(AttentionPlanSpec(**values), metadata)


def _ragged_data(keys, values, *, num_heads=1, qk_dim=1, vo_dim=1):
    spec = RaggedKVCacheSpec(
        total_kv_tokens=len(keys),
        num_kv_heads=num_heads,
        head_dim_qk=qk_dim,
        head_dim_vo=vo_dim,
        dtype="float32",
        device="cpu",
    )
    return ReferenceKVData(
        spec,
        (
            ReferenceTensor.from_nested(keys),
            ReferenceTensor.from_nested(values),
        ),
    )


def _paged_data(keys, values, *, layout=KVLayout.NHD):
    num_pages = len(keys)
    if layout == KVLayout.NHD:
        page_size = len(keys[0])
    else:
        page_size = len(keys[0][0])
    spec = PagedKVCacheSpec(
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        dtype="float32",
        layout=layout,
        structure="separate",
        device="cpu",
    )
    return ReferenceKVData(
        spec,
        (
            ReferenceTensor.from_nested(keys),
            ReferenceTensor.from_nested(values),
        ),
    )


class ReferenceTensorTests(unittest.TestCase):
    def test_nested_tensor_is_rectangular_and_indexable(self):
        tensor = ReferenceTensor.from_nested([[1, 2], [3, 4]])
        self.assertEqual(tensor.shape, (2, 2))
        self.assertEqual(tensor.at(1, 0), 3.0)
        with self.assertRaisesRegex(SchemaError, "rectangular"):
            ReferenceTensor.from_nested([[1], [2, 3]])

    def test_kv_values_must_match_declared_storage_shape(self):
        spec = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
        with self.assertRaisesRegex(SchemaError, "shape"):
            ReferenceKVData(
                spec,
                (
                    ReferenceTensor.from_nested([[[0.0]]]),
                    ReferenceTensor.from_nested([[[1.0]]]),
                ),
            )


class ReferenceAttentionTests(unittest.TestCase):
    def setUp(self):
        self.executor = ReferenceAttentionExecutor()

    def test_single_prefill_computes_output_and_lse(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=2)
        plan = _plan(AttentionMode.SINGLE_PREFILL, metadata)
        q = ReferenceTensor.from_nested([[[1.0]]])
        kv = _ragged_data([[[1.0]], [[2.0]]], [[[10.0]], [[20.0]]])

        result = self.executor.execute(plan, q, kv, return_lse=True)

        expected = (10.0 * math.exp(1.0) + 20.0 * math.exp(2.0)) / (
            math.exp(1.0) + math.exp(2.0)
        )
        self.assertAlmostEqual(result.output.data[0], expected)
        self.assertAlmostEqual(result.lse.data[0], math.log(math.exp(1) + math.exp(2)))
        self.assertEqual(result.output.shape, (1, 1, 1))
        self.assertEqual(result.lse.shape, (1, 1))

    def test_prefill_causal_mask_is_bottom_right_aligned(self):
        metadata = SingleAttentionMetadata(qo_len=2, kv_len=3)
        plan = _plan(AttentionMode.SINGLE_PREFILL, metadata, causal=True)
        q = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
        kv = _ragged_data(
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[10.0]], [[20.0]], [[30.0]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (15.0, 20.0))

    def test_custom_mask_overrides_causal(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=3)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            causal=True,
            custom_mask=CustomMaskSpec(numel=3),
        )
        q = ReferenceTensor.from_nested([[[0.0]]])
        kv = _ragged_data(
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[10.0]], [[20.0]], [[30.0]]],
        )

        result = self.executor.execute(
            plan, q, kv, custom_mask_data=(False, True, False)
        )

        self.assertEqual(result.output.data, (20.0,))

    def test_gqa_maps_query_heads_to_shared_kv_head(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=2)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            num_qo_heads=2,
            num_kv_heads=1,
        )
        q = ReferenceTensor.from_nested([[[0.0], [0.0]]])
        kv = _ragged_data([[[0.0]], [[0.0]]], [[[2.0]], [[4.0]]])

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.shape, (1, 2, 1))
        self.assertEqual(result.output.data, (3.0, 3.0))

    def test_paged_decode_follows_logical_to_physical_page_table(self):
        metadata = PagedKVMetadata((0, 2), (1, 0), (2,), 2)
        plan = _plan(AttentionMode.BATCH_DECODE_PAGED, metadata)
        q = ReferenceTensor.from_nested([[[0.0]]])
        kv = _paged_data(
            [[[[0.0]], [[0.0]]], [[[0.0]], [[0.0]]]],
            [[[[30.0]], [[40.0]]], [[[10.0]], [[20.0]]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (25.0,))

    def test_paged_prefill_combines_page_table_and_causal_alignment(self):
        paged = PagedKVMetadata((0, 2), (1, 0), (1,), 2)
        metadata = PagedPrefillMetadata((0, 2), paged)
        plan = _plan(AttentionMode.BATCH_PREFILL_PAGED, metadata, causal=True)
        q = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
        kv = _paged_data(
            [[[[0.0]], [[99.0]]], [[[0.0]], [[0.0]]]],
            [[[[30.0]], [[999.0]]], [[[10.0]], [[20.0]]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (15.0, 20.0))

    def test_sliding_window_is_relative_to_query_position(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=4)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            window_left=1,
            window_right=0,
        )
        q = ReferenceTensor.from_nested([[[0.0]]])
        kv = _ragged_data(
            [[[0.0]], [[0.0]], [[0.0]], [[0.0]]],
            [[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (35.0,))

    def test_ragged_prefill_keeps_request_segments_separate(self):
        metadata = RaggedKVMetadata((0, 1, 2), (0, 1, 3))
        plan = _plan(AttentionMode.BATCH_PREFILL_RAGGED, metadata)
        q = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
        kv = _ragged_data(
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[5.0]], [[10.0]], [[20.0]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (5.0, 15.0))

    def test_multi_token_batch_decode_is_causal(self):
        metadata = PagedKVMetadata((0, 2), (0, 1), (1,), 2)
        plan = _plan(
            AttentionMode.BATCH_DECODE_PAGED,
            metadata,
            q_len_per_req=2,
        )
        q = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
        kv = _paged_data(
            [[[[0.0]], [[0.0]]], [[[0.0]], [[99.0]]]],
            [[[[10.0]], [[20.0]]], [[[30.0]], [[999.0]]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (15.0, 20.0))

    def test_mixed_attention_executes_prefill_and_decode_requests(self):
        metadata = MixedPagedKVMetadata(
            qo_indptr=(0, 1, 3),
            kv_indptr=(0, 1, 2),
            kv_indices=(1, 0),
            kv_len_arr=(1, 2),
            page_size=2,
        )
        plan = _plan(AttentionMode.BATCH_MIXED_PAGED, metadata)
        q = ReferenceTensor.from_nested([[[0.0]], [[0.0]], [[0.0]]])
        kv = _paged_data(
            [[[[0.0]], [[0.0]]], [[[0.0]], [[99.0]]]],
            [[[[10.0]], [[20.0]]], [[[5.0]], [[999.0]]]],
        )

        result = self.executor.execute(plan, q, kv)

        self.assertEqual(result.output.data, (5.0, 15.0, 15.0))
        self.assertEqual(result.lse.shape, (3, 1))
        self.assertAlmostEqual(result.lse.data[0], 0.0)
        self.assertAlmostEqual(result.lse.data[1], math.log(2.0))

    def test_alibi_and_scale_soft_cap_are_executable_plan_features(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=2)
        alibi_plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            pos_encoding_mode=PosEncodingMode.ALIBI,
        )
        q = ReferenceTensor.from_nested([[[0.0]]])
        kv = _ragged_data([[[0.0]], [[0.0]]], [[[0.0]], [[3.0]]])
        alibi = self.executor.execute(
            alibi_plan, q, kv, alibi_slopes=(math.log(2.0),)
        )
        self.assertAlmostEqual(alibi.output.data[0], 2.0)

        capped_plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            logits_soft_cap=1.0,
        )
        q = ReferenceTensor.from_nested([[[1.0]]])
        kv = _ragged_data([[[0.0]], [[2.0]]], [[[0.0]], [[1.0]]])
        capped = self.executor.execute(
            capped_plan,
            q,
            kv,
            q_scale=2.0,
            k_scale=0.5,
            v_scale=3.0,
            logits_soft_cap=1.0,
        )
        expected = 3.0 / (1.0 + math.exp(-math.tanh(2.0)))
        self.assertAlmostEqual(capped.output.data[0], expected)

    def test_per_query_head_scale_is_applied_before_gqa_reduction(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=2)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            num_qo_heads=2,
            num_kv_heads=1,
        )
        q = ReferenceTensor.from_nested([[[1.0], [1.0]]])
        kv = _ragged_data([[[0.0]], [[1.0]]], [[[0.0]], [[1.0]]])

        result = self.executor.execute(plan, q, kv, q_scale=(1.0, 2.0))

        self.assertAlmostEqual(result.output.data[0], 1.0 / (1.0 + math.exp(-1.0)))
        self.assertAlmostEqual(result.output.data[1], 1.0 / (1.0 + math.exp(-2.0)))

    def test_per_kv_head_value_scale_uses_gqa_head_mapping(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=1)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            num_qo_heads=2,
            num_kv_heads=2,
        )
        q = ReferenceTensor.from_nested([[[0.0], [0.0]]])
        kv = _ragged_data(
            [[[0.0], [0.0]]],
            [[[2.0], [3.0]]],
            num_heads=2,
        )

        result = self.executor.execute(plan, q, kv, v_scale=(10.0, 100.0))

        self.assertEqual(result.output.data, (20.0, 300.0))

    def test_per_head_scale_shape_is_validated(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=1)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            num_qo_heads=2,
            num_kv_heads=1,
        )
        q = ReferenceTensor.from_nested([[[0.0], [0.0]]])
        kv = _ragged_data([[[0.0]]], [[[1.0]]])

        with self.assertRaisesRegex(SchemaError, "one value per head"):
            self.executor.execute(plan, q, kv, q_scale=(1.0,))

    def test_packed_hnd_cache_is_addressed_correctly(self):
        metadata = PagedKVMetadata((0, 1), (0,), (2,), 2)
        plan = _plan(
            AttentionMode.BATCH_DECODE_PAGED,
            metadata,
            kv_layout=KVLayout.HND,
        )
        q = ReferenceTensor.from_nested([[[0.0]]])
        cache_spec = PagedKVCacheSpec(
            1,
            2,
            1,
            1,
            1,
            "float32",
            layout=KVLayout.HND,
            structure="packed",
            device="cpu",
        )
        packed = ReferenceTensor.from_nested(
            [[[[[0.0], [0.0]]], [[[2.0], [4.0]]]]]
        )

        result = self.executor.execute(plan, q, ReferenceKVData(cache_spec, (packed,)))

        self.assertEqual(result.output.data, (3.0,))

    def test_rope_llama_uses_bottom_right_positions_and_scale(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=2)
        rope_plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            pos_encoding_mode=PosEncodingMode.ROPE_LLAMA,
            head_dim_qk=2,
            head_dim_vo=1,
            rope_scale=2.0,
        )
        q = ReferenceTensor.from_nested([[[1.0, 0.0]]])
        kv = _ragged_data(
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[0.0]], [[1.0]]],
            qk_dim=2,
            vo_dim=1,
        )

        result = self.executor.execute(rope_plan, q, kv)

        expected = 1.0 / (1.0 + math.exp(-(1.0 - math.cos(0.5))))
        self.assertAlmostEqual(result.output.data[0], expected)

    def test_rope_llama_frequency_uses_half_split_dimension_and_theta(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=2)
        plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            pos_encoding_mode=PosEncodingMode.ROPE_LLAMA,
            head_dim_qk=4,
            head_dim_vo=1,
            rope_theta=4.0,
        )
        q = ReferenceTensor.from_nested([[[0.0, 1.0, 0.0, 0.0]]])
        kv = _ragged_data(
            [[[0.0, 1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]],
            [[[0.0]], [[1.0]]],
            qk_dim=4,
            vo_dim=1,
        )

        result = self.executor.execute(plan, q, kv)

        expected = 1.0 / (1.0 + math.exp(-(1.0 - math.cos(0.5))))
        self.assertAlmostEqual(result.output.data[0], expected)

    def test_packed_custom_mask_uses_little_endian_bits(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=3)
        packed_mask_plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            custom_mask=CustomMaskSpec(numel=1, packed=True),
        )
        q = ReferenceTensor.from_nested([[[0.0]]])
        kv = _ragged_data(
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[10.0]], [[20.0]], [[30.0]]],
        )

        result = self.executor.execute(
            packed_mask_plan, q, kv, custom_mask_data=(0b00000101,)
        )

        self.assertEqual(result.output.data, (20.0,))

    def test_packed_mask_restarts_at_each_ragged_request(self):
        metadata = RaggedKVMetadata((0, 1, 2), (0, 3, 6))
        plan = _plan(
            AttentionMode.BATCH_PREFILL_RAGGED,
            metadata,
            custom_mask=CustomMaskSpec(numel=2, packed=True),
        )
        q = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
        kv = _ragged_data(
            [[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]],
            [[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]]],
        )

        result = self.executor.execute(
            plan, q, kv, custom_mask_data=(0b00000010, 0b00000100)
        )

        self.assertEqual(result.output.data, (20.0, 60.0))

    def test_packed_mask_rejects_values_outside_uint8(self):
        metadata = SingleAttentionMetadata(qo_len=1, kv_len=1)
        packed_mask_plan = _plan(
            AttentionMode.SINGLE_PREFILL,
            metadata,
            custom_mask=CustomMaskSpec(numel=1, packed=True),
        )
        q = ReferenceTensor.from_nested([[[0.0]]])
        kv = _ragged_data([[[0.0]]], [[[1.0]]])

        with self.assertRaisesRegex(SchemaError, "uint8"):
            self.executor.execute(
                packed_mask_plan, q, kv, custom_mask_data=(256,)
            )


class AttentionWrapperTests(unittest.TestCase):
    def test_wrapper_enforces_plan_then_delegates_to_executor(self):
        wrapper = AttentionWrapper(
            AttentionMode.SINGLE_DECODE, ReferenceAttentionExecutor()
        )
        q = ReferenceTensor.from_nested([[0.0]])
        kv = _ragged_data([[[0.0]]], [[[7.0]]])
        with self.assertRaisesRegex(AttentionStateError, "not been planned"):
            wrapper.run(q, kv)

        metadata = SingleAttentionMetadata(qo_len=1, kv_len=1)
        wrapper.plan(
            AttentionPlanSpec(
                mode=AttentionMode.SINGLE_DECODE,
                num_qo_heads=1,
                num_kv_heads=1,
                head_dim_qk=1,
                q_dtype="float32",
            ),
            metadata,
        )
        result = wrapper.run(q, kv)
        self.assertEqual(result.output.shape, (1, 1))
        self.assertEqual(result.output.data, (7.0,))


if __name__ == "__main__":
    unittest.main()
