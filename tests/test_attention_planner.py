import unittest

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionPlanSpec,
    AttentionStateError,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    SingleAttentionMetadata,
    TensorSpec,
)
from flashinfer_npu.runtime import SchemaError


class AttentionPlannerTests(unittest.TestCase):
    def test_run_before_plan_is_rejected(self):
        session = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED)
        with self.assertRaisesRegex(AttentionStateError, "not been planned"):
            session.infer_run(
                TensorSpec((1, 8, 64), "float16"),
                PagedKVCacheSpec(1, 16, 2, 64, 64, "float16"),
            )

    def test_batch_decode_infers_output_and_lse(self):
        metadata = PagedKVMetadata(
            indptr=(0, 1, 3),
            indices=(0, 1, 2),
            last_page_len=(16, 4),
            page_size=16,
        )
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED)
        plan = session.plan(spec, metadata)
        result = session.infer_run(
            TensorSpec((2, 8, 64), "float16"),
            PagedKVCacheSpec(3, 16, 2, 64, 64, "float16"),
            return_lse=True,
        )
        self.assertEqual(plan.batch_size, 2)
        self.assertEqual(result.output.shape, (2, 8, 64))
        self.assertEqual(result.lse.shape, (2, 8))
        self.assertEqual(result.lse.dtype, "float32")

    def test_plan_fingerprint_is_stable_across_replanning(self):
        metadata = PagedKVMetadata((0, 1), (0,), (4,), 8)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED)
        first = session.plan(spec, metadata)
        second = session.plan(spec, metadata)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual((first.generation, second.generation), (1, 2))

    def test_paged_prefill_supports_different_qk_and_vo_dimensions(self):
        paged = PagedKVMetadata(
            indptr=(0, 1, 2),
            indices=(0, 1),
            last_page_len=(8, 4),
            page_size=8,
        )
        metadata = PagedPrefillMetadata(qo_indptr=(0, 3, 5), paged_kv=paged)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            head_dim_vo=32,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_PREFILL_PAGED)
        session.plan(spec, metadata)
        result = session.infer_run(
            TensorSpec((5, 8, 64), "float16"),
            PagedKVCacheSpec(
                2, 8, 2, 64, 32, "float16", structure="separate"
            ),
        )
        self.assertEqual(result.output.shape, (5, 8, 32))
        self.assertIsNone(result.lse)

    def test_referenced_page_must_exist(self):
        metadata = PagedKVMetadata(
            indptr=(0, 1),
            indices=(4,),
            last_page_len=(3,),
            page_size=8,
        )
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED)
        session.plan(spec, metadata)
        with self.assertRaisesRegex(SchemaError, "referenced page"):
            session.infer_run(
                TensorSpec((1, 8, 64), "float16"),
                PagedKVCacheSpec(4, 8, 2, 64, 64, "float16"),
            )

    def test_graph_enabled_session_locks_batch_size(self):
        first = PagedKVMetadata((0, 1, 2), (0, 1), (1, 1), 8)
        second = PagedKVMetadata((0, 1, 2, 3), (0, 1, 2), (1, 1, 1), 8)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
        )
        session = AttentionFrameworkSession(
            AttentionMode.BATCH_DECODE_PAGED, graph_enabled=True
        )
        session.plan(spec, first)
        with self.assertRaisesRegex(AttentionStateError, "fixed batch size"):
            session.plan(spec, second)

    def test_graph_enabled_session_locks_q_len_per_request(self):
        metadata = PagedKVMetadata((0, 1, 2), (0, 1), (4, 4), 8)
        one_token = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            q_len_per_req=1,
        )
        two_tokens = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            q_len_per_req=2,
        )
        session = AttentionFrameworkSession(
            AttentionMode.BATCH_DECODE_PAGED, graph_enabled=True
        )
        session.plan(one_token, metadata)
        with self.assertRaisesRegex(AttentionStateError, "fixed q_len_per_req"):
            session.plan(two_tokens, metadata)

    def test_multi_token_batch_decode_uses_flattened_query_rows(self):
        metadata = PagedKVMetadata((0, 1, 2), (0, 1), (4, 4), 8)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            q_len_per_req=3,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED)
        plan = session.plan(spec, metadata)
        result = session.infer_run(
            TensorSpec((6, 8, 64), "float16"),
            PagedKVCacheSpec(2, 8, 2, 64, 64, "float16"),
            return_lse=True,
        )
        self.assertTrue(plan.spec.causal)
        self.assertEqual(plan.total_qo_tokens, 6)
        self.assertEqual(result.output.shape, (6, 8, 64))
        self.assertEqual(result.lse.shape, (6, 8))

    def test_runtime_soft_cap_requires_matching_plan_feature(self):
        metadata = PagedKVMetadata((0, 1), (0,), (4,), 8)
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            logits_soft_cap=0.0,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED)
        session.plan(spec, metadata)
        with self.assertRaisesRegex(SchemaError, "requires a capped plan"):
            session.infer_run(
                TensorSpec((1, 8, 64), "float16"),
                PagedKVCacheSpec(1, 8, 2, 64, 64, "float16"),
                logits_soft_cap=30.0,
            )

    def test_profiler_buffer_is_a_plan_time_contract(self):
        metadata = MixedPagedKVMetadata(
            qo_indptr=(0, 1),
            kv_indptr=(0, 1),
            kv_indices=(0,),
            kv_len_arr=(4,),
            page_size=8,
        )
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_MIXED_PAGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            use_profiler=True,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_MIXED_PAGED)
        session.plan(spec, metadata)
        q = TensorSpec((1, 8, 64), "float16")
        cache = PagedKVCacheSpec(1, 8, 2, 64, 64, "float16")
        with self.assertRaisesRegex(SchemaError, "profiler_buffer"):
            session.infer_run(q, cache)
        result = session.infer_run(
            q, cache, profiler_buffer=TensorSpec((1024,), "uint64")
        )
        self.assertIsNotNone(result.lse)

    def test_ragged_metadata_requires_ragged_cache(self):
        metadata = RaggedKVMetadata(qo_indptr=(0, 2), kv_indptr=(0, 5))
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
            kv_layout=KVLayout.NHD,
        )
        session = AttentionFrameworkSession(AttentionMode.BATCH_PREFILL_RAGGED)
        session.plan(spec, metadata)
        with self.assertRaisesRegex(SchemaError, "ragged KV cache"):
            session.infer_run(
                TensorSpec((2, 8, 64), "float16"),
                PagedKVCacheSpec(1, 8, 2, 64, 64, "float16"),
            )

    def test_single_decode_matches_flashinfer_query_shape(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.SINGLE_DECODE,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=64,
        )
        session = AttentionFrameworkSession(AttentionMode.SINGLE_DECODE)
        session.plan(spec, SingleAttentionMetadata(qo_len=1, kv_len=17))
        result = session.infer_run(
            TensorSpec((8, 64), "float16"),
            RaggedKVCacheSpec(17, 2, 64, 64, "float16"),
            return_lse=True,
        )
        self.assertEqual(result.output.shape, (8, 64))
        self.assertEqual(result.lse.shape, (8,))


if __name__ == "__main__":
    unittest.main()
