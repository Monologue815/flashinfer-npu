import math
import unittest

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionPlanSpec,
    CustomMaskSpec,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    SingleAttentionMetadata,
)
from flashinfer_npu.runtime import SchemaError


class PagedMetadataTests(unittest.TestCase):
    def test_sequence_lengths_follow_flashinfer_page_contract(self):
        metadata = PagedKVMetadata(
            indptr=(0, 2, 3),
            indices=(4, 1, 7),
            last_page_len=(3, 8),
            page_size=8,
        )
        self.assertEqual(metadata.batch_size, 2)
        self.assertEqual(metadata.page_counts, (2, 1))
        self.assertEqual(metadata.sequence_lengths, (11, 8))
        self.assertEqual(metadata.max_page_index, 7)

    def test_empty_request_uses_zero_last_page_length(self):
        metadata = PagedKVMetadata(
            indptr=(0, 0, 1),
            indices=(2,),
            last_page_len=(0, 4),
            page_size=8,
        )
        self.assertEqual(metadata.sequence_lengths, (0, 4))

    def test_invalid_last_page_length_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "last_page_len"):
            PagedKVMetadata(
                indptr=(0, 1),
                indices=(0,),
                last_page_len=(0,),
                page_size=16,
            )

    def test_indices_must_match_indptr(self):
        with self.assertRaisesRegex(SchemaError, "indices length"):
            PagedKVMetadata(
                indptr=(0, 2),
                indices=(0,),
                last_page_len=(1,),
                page_size=16,
            )


class AttentionSchemaTests(unittest.TestCase):
    def test_ragged_offsets_produce_lengths(self):
        metadata = RaggedKVMetadata(
            qo_indptr=(0, 2, 5), kv_indptr=(0, 7, 11)
        )
        self.assertEqual(metadata.qo_lengths, (2, 3))
        self.assertEqual(metadata.kv_lengths, (7, 4))

    def test_mixed_metadata_checks_kv_length_capacity(self):
        with self.assertRaisesRegex(SchemaError, "fit allocated pages"):
            MixedPagedKVMetadata(
                qo_indptr=(0, 1),
                kv_indptr=(0, 2),
                kv_indices=(0, 1),
                kv_len_arr=(8,),
                page_size=8,
            )

    def test_gqa_requires_divisible_head_counts(self):
        with self.assertRaisesRegex(SchemaError, "divide"):
            AttentionPlanSpec(
                mode=AttentionMode.BATCH_DECODE_PAGED,
                num_qo_heads=10,
                num_kv_heads=3,
                head_dim_qk=128,
            )

    def test_default_softmax_scale_matches_head_dimension(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=16,
            num_kv_heads=4,
            head_dim_qk=64,
        )
        self.assertAlmostEqual(spec.sm_scale, 1.0 / math.sqrt(64))

    def test_paged_prefill_becomes_a_stable_workload_key(self):
        paged = PagedKVMetadata(
            indptr=(0, 1, 3),
            indices=(0, 2, 3),
            last_page_len=(8, 2),
            page_size=8,
        )
        metadata = PagedPrefillMetadata(qo_indptr=(0, 3, 5), paged_kv=paged)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_PAGED,
            num_qo_heads=16,
            num_kv_heads=4,
            head_dim_qk=64,
            head_dim_vo=32,
            causal=True,
            kv_layout=KVLayout.HND,
        )
        workload = spec.to_workload_spec(metadata)
        self.assertEqual(workload.op, "attention.batch_prefill_paged")
        self.assertEqual(workload.dynamic_bounds, (2, 5, 18))
        self.assertEqual(workload.layouts, ("HND",))
        self.assertTrue(workload.causal)

    def test_single_decode_requires_one_query_token(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_DECODE,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
        )
        with self.assertRaisesRegex(SchemaError, "qo_len=1"):
            spec.to_workload_spec(SingleAttentionMetadata(qo_len=2, kv_len=32))

    def test_causal_attention_requires_kv_to_cover_query_positions(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_PREFILL,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=2,
            causal=True,
        )
        with self.assertRaisesRegex(SchemaError, "kv_len >= qo_len"):
            spec.to_workload_spec(SingleAttentionMetadata(qo_len=3, kv_len=2))

    def test_multi_token_decode_requires_existing_kv_for_every_query(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=2,
            q_len_per_req=3,
        )
        metadata = PagedKVMetadata((0, 1), (0,), (2,), 2)
        with self.assertRaisesRegex(SchemaError, "kv_len >= qo_len"):
            spec.to_workload_spec(metadata)

    def test_mixed_attention_rejects_unsupported_pos_encoding(self):
        with self.assertRaisesRegex(SchemaError, "pos encoding NONE"):
            AttentionPlanSpec(
                mode=AttentionMode.BATCH_MIXED_PAGED,
                num_qo_heads=8,
                num_kv_heads=2,
                head_dim_qk=64,
                pos_encoding_mode=PosEncodingMode.ROPE_LLAMA,
            )

    def test_segment_packed_mask_counts_each_request_independently(self):
        metadata = RaggedKVMetadata(
            qo_indptr=(0, 2, 3),
            kv_indptr=(0, 3, 12),
        )
        # Segment bit counts are 6 and 9, which require 1 + 2 bytes.
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            causal=True,
            custom_mask=CustomMaskSpec(numel=3, packed=True),
        )
        workload = spec.to_workload_spec(metadata)
        self.assertFalse(workload.causal)
        self.assertEqual(spec.custom_mask.bit_order, "little")
        self.assertIn(("custom_mask", "packed_little"), workload.attributes)

    def test_attention_packed_mask_rejects_non_little_bit_order(self):
        with self.assertRaisesRegex(SchemaError, "little bit order"):
            CustomMaskSpec(numel=1, packed=True, bit_order="big")

    def test_rope_requires_even_query_key_dimension(self):
        with self.assertRaisesRegex(SchemaError, "even head_dim_qk"):
            AttentionPlanSpec(
                mode=AttentionMode.SINGLE_PREFILL,
                num_qo_heads=1,
                num_kv_heads=1,
                head_dim_qk=3,
                pos_encoding_mode=PosEncodingMode.ROPE_LLAMA,
            )

    def test_custom_mask_size_is_checked(self):
        metadata = RaggedKVMetadata(qo_indptr=(0, 2), kv_indptr=(0, 3))
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            custom_mask=CustomMaskSpec(numel=5),
        )
        with self.assertRaisesRegex(SchemaError, "custom mask numel must be 6"):
            spec.to_workload_spec(metadata)

    def test_decode_rejects_custom_mask(self):
        metadata = PagedKVMetadata((0, 1), (0,), (1,), 8)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            custom_mask=CustomMaskSpec(numel=1),
        )
        with self.assertRaisesRegex(SchemaError, "prefill modes only"):
            spec.to_workload_spec(metadata)

    def test_rope_parameters_must_be_positive(self):
        with self.assertRaisesRegex(SchemaError, "rope_theta"):
            AttentionPlanSpec(
                mode=AttentionMode.SINGLE_PREFILL,
                num_qo_heads=8,
                num_kv_heads=2,
                head_dim_qk=64,
                rope_theta=0,
            )

    def test_paged_cache_shapes_are_layout_explicit(self):
        cache = PagedKVCacheSpec(
            num_pages=5,
            page_size=16,
            num_kv_heads=4,
            head_dim_qk=128,
            head_dim_vo=128,
            dtype="float16",
            layout=KVLayout.HND,
        )
        self.assertEqual(cache.expected_shapes, ((5, 2, 4, 16, 128),))

    def test_ragged_cache_shapes_are_layout_explicit(self):
        cache = RaggedKVCacheSpec(
            total_kv_tokens=11,
            num_kv_heads=2,
            head_dim_qk=64,
            head_dim_vo=32,
            dtype="float16",
            layout=KVLayout.NHD,
        )
        self.assertEqual(cache.expected_shapes, ((11, 2, 64), (11, 2, 32)))


if __name__ == "__main__":
    unittest.main()
