import inspect
import math
import unittest

import flashinfer_npu
from flashinfer_npu.attention import (
    AttentionStateError,
    ReferenceBuffer,
    ReferenceTensor,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer_npu.runtime import DispatchError, SchemaError


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def index(values, device="cpu"):
    return tensor(values, dtype="int32", device=device)


def workspace(size=256, device="cpu"):
    return ReferenceTensor.zeros((size,), dtype="uint8", device=device)


class BatchFrontendSignatureTests(unittest.TestCase):
    def test_paged_prefill_constructor_plan_and_run_signatures(self):
        self.assertEqual(
            list(inspect.signature(BatchPrefillWithPagedKVCacheWrapper).parameters),
            [
                "float_workspace_buffer", "kv_layout", "use_cuda_graph",
                "qo_indptr_buf", "paged_kv_indptr_buf", "paged_kv_indices_buf",
                "paged_kv_last_page_len_buf", "custom_mask_buf",
                "mask_indptr_buf", "backend", "jit_args", "jit_kwargs",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithPagedKVCacheWrapper.plan
                ).parameters
            ),
            [
                "self", "qo_indptr", "paged_kv_indptr", "paged_kv_indices",
                "paged_kv_last_page_len", "num_qo_heads", "num_kv_heads",
                "head_dim_qk", "page_size", "head_dim_vo", "custom_mask",
                "packed_custom_mask", "causal", "pos_encoding_mode",
                "use_fp16_qk_reduction", "sm_scale", "window_left",
                "logits_soft_cap", "rope_scale", "rope_theta", "q_data_type",
                "kv_data_type", "o_data_type", "non_blocking", "prefix_len_ptr",
                "token_pos_in_items_ptr", "token_pos_in_items_len",
                "max_item_len_ptr", "seq_lens", "seq_lens_q", "block_tables",
                "max_token_per_sequence", "max_sequence_kv", "fixed_split_size",
                "disable_split_kv",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithPagedKVCacheWrapper.run
                ).parameters
            ),
            [
                "self", "q", "paged_kv_cache", "args", "q_scale", "k_scale",
                "v_scale", "out", "lse", "return_lse", "enable_pdl",
                "window_left", "sinks", "kv_cache_sf",
                "skip_softmax_threshold_scale_factor",
                "use_fp16_softmax", "uses_spcompress",
            ],
        )
        run_signature = inspect.signature(
            BatchPrefillWithPagedKVCacheWrapper.run
        )
        for name in ("use_fp16_softmax", "uses_spcompress"):
            self.assertIsNone(run_signature.parameters[name].default)
            self.assertEqual(
                run_signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

    def test_ragged_and_decode_signatures_track_upstream_shape(self):
        self.assertEqual(
            list(inspect.signature(BatchPrefillWithRaggedKVCacheWrapper).parameters),
            [
                "float_workspace_buffer", "kv_layout", "use_cuda_graph",
                "qo_indptr_buf", "kv_indptr_buf", "custom_mask_buf",
                "mask_indptr_buf", "backend", "jit_args", "jit_kwargs",
            ],
        )
        self.assertEqual(
            list(inspect.signature(BatchDecodeWithPagedKVCacheWrapper).parameters),
            [
                "float_workspace_buffer", "kv_layout", "use_cuda_graph",
                "use_tensor_cores", "paged_kv_indptr_buffer",
                "paged_kv_indices_buffer", "paged_kv_last_page_len_buffer",
                "backend", "jit_args",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithRaggedKVCacheWrapper.plan
                ).parameters
            ),
            [
                "self", "qo_indptr", "kv_indptr", "num_qo_heads",
                "num_kv_heads", "head_dim_qk", "head_dim_vo", "custom_mask",
                "packed_custom_mask", "causal", "pos_encoding_mode",
                "use_fp16_qk_reduction", "window_left", "logits_soft_cap",
                "sm_scale", "rope_scale", "rope_theta", "q_data_type",
                "kv_data_type", "o_data_type", "non_blocking", "prefix_len_ptr",
                "token_pos_in_items_ptr", "token_pos_in_items_len",
                "max_item_len_ptr", "fixed_split_size", "disable_split_kv",
                "seq_lens", "seq_lens_q", "max_token_per_sequence",
                "max_sequence_kv", "v_indptr", "o_indptr",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithRaggedKVCacheWrapper.run
                ).parameters
            ),
            [
                "self", "q", "k", "v", "args", "q_scale", "k_scale",
                "v_scale", "o_scale", "out", "lse", "return_lse",
                "enable_pdl", "kv_cache_sf",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan).parameters
            ),
            [
                "self", "indptr", "indices", "last_page_len", "num_qo_heads",
                "num_kv_heads", "head_dim", "page_size",
                "deprecated_positional_args", "kwargs",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(BatchDecodeWithPagedKVCacheWrapper.run).parameters
            ),
            [
                "self", "q", "paged_kv_cache", "args", "q_scale", "k_scale",
                "v_scale", "out", "lse", "return_lse", "enable_pdl",
                "window_left", "sinks", "q_len_per_req",
                "skip_softmax_threshold_scale_factor", "kv_cache_sf",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithPagedKVCacheWrapper.forward
                ).parameters
            ),
            [
                "self", "q", "paged_kv_cache", "causal",
                "pos_encoding_mode", "use_fp16_qk_reduction", "k_scale",
                "v_scale", "window_left", "logits_soft_cap", "sm_scale",
                "rope_scale", "rope_theta",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithRaggedKVCacheWrapper.forward
                ).parameters
            ),
            [
                "self", "q", "k", "v", "causal", "pos_encoding_mode",
                "use_fp16_qk_reduction", "window_left", "logits_soft_cap",
                "sm_scale", "rope_scale", "rope_theta",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(BatchDecodeWithPagedKVCacheWrapper.forward).parameters
            ),
            [
                "self", "q", "paged_kv_cache", "pos_encoding_mode",
                "q_scale", "k_scale", "v_scale", "window_left",
                "logits_soft_cap", "sm_scale", "rope_scale", "rope_theta",
            ],
        )

    def test_package_root_reexports_batch_wrappers(self):
        self.assertIs(
            flashinfer_npu.BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )
        self.assertIs(
            flashinfer_npu.BatchPrefillWithRaggedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
        )
        self.assertIs(
            flashinfer_npu.BatchDecodeWithPagedKVCacheWrapper,
            BatchDecodeWithPagedKVCacheWrapper,
        )


class PagedPrefillFrontendTests(unittest.TestCase):
    def test_auto_backend_does_not_select_host_reference(self):
        with self.assertRaisesRegex(DispatchError, "explicitly"):
            BatchPrefillWithPagedKVCacheWrapper(workspace())

    def test_paged_plan_run_follows_physical_page_table(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1, 2]),
            index([0, 1, 2]),
            index([1, 0]),
            index([1, 2]),
            1,
            1,
            1,
            2,
            q_data_type="float32",
        )
        q = tensor([[[0.0]], [[0.0]]])
        k = tensor(
            [
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[99.0]]],
            ]
        )
        v = tensor(
            [
                [[[10.0]], [[20.0]]],
                [[[5.0]], [[999.0]]],
            ]
        )
        output, lse = wrapper.run(q, (k, v), return_lse=True)
        self.assertEqual(output.shape, (2, 1, 1))
        self.assertEqual(output.data, (5.0, 15.0))
        self.assertAlmostEqual(lse.data[0], 0.0)
        self.assertAlmostEqual(lse.data[1], math.log(2.0))

    def test_segment_packed_mask_and_caller_owned_buffers(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1, 2]),
            index([0, 1, 2]),
            index([0, 1]),
            index([2, 2]),
            1,
            1,
            1,
            2,
            packed_custom_mask=tensor([1, 2], dtype="uint8"),
            q_data_type="float32",
            o_data_type="float16",
        )
        q = tensor([[[0.0]], [[0.0]]])
        k = tensor([[[[0.0]], [[0.0]]], [[[0.0]], [[0.0]]]])
        v = tensor([[[[10.0]], [[20.0]]], [[[30.0]], [[40.0]]]])
        out = ReferenceBuffer.zeros((2, 1, 1), dtype="float16")
        lse = ReferenceBuffer.zeros((2, 1), dtype="float32")
        output, lse_output = wrapper.run(
            q, (k, v), out=out, lse=lse, return_lse=True
        )
        self.assertIs(output, out)
        self.assertIs(lse_output, lse)
        self.assertEqual(out.data, (10.0, 40.0))
        self.assertEqual(lse.data, (0.0, 0.0))

    def test_runtime_window_must_match_plan_and_extensions_fail_explicitly(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        with self.assertRaisesRegex(NotImplementedError, "prefix_len_ptr"):
            wrapper.plan(
                index([0, 1]), index([0, 1]), index([0]), index([1]),
                1, 1, 1, 1, prefix_len_ptr=index([0]),
            )
        wrapper.plan(
            index([0, 1]), index([0, 1]), index([0]), index([1]),
            1, 1, 1, 1, q_data_type="float32", window_left=0,
        )
        q = tensor([[[0.0]]])
        kv = (tensor([[[[0.0]]]]), tensor([[[[1.0]]]]))
        with self.assertRaisesRegex(SchemaError, "match"):
            wrapper.run(q, kv, window_left=-1)
        with self.assertRaisesRegex(NotImplementedError, "enable_pdl"):
            wrapper.run(q, kv, enable_pdl=True)
        with self.assertRaisesRegex(NotImplementedError, "NVFP4"):
            wrapper.run(q, kv, kv_cache_sf=tensor([1.0]))
        self.assertEqual(
            wrapper.run(
                q,
                kv,
                use_fp16_softmax=False,
                uses_spcompress=False,
            ).data,
            (1.0,),
        )
        with self.assertRaisesRegex(NotImplementedError, "FP16 softmax"):
            wrapper.run(q, kv, use_fp16_softmax=True)
        with self.assertRaisesRegex(NotImplementedError, "SP-compressed"):
            wrapper.run(q, kv, uses_spcompress=True)

    def test_deprecated_forward_executes_matching_plan_and_rejects_drift(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1]), index([0, 1]), index([0]), index([1]),
            1, 1, 1, 1, q_data_type="float32", causal=True,
        )
        q = tensor([[[0.0]]])
        kv = (tensor([[[[0.0]]]]), tensor([[[[3.0]]]]))
        with self.assertWarns(DeprecationWarning):
            output = wrapper.forward(q, kv, causal=True)
        self.assertEqual(output.data, (3.0,))
        with self.assertWarns(DeprecationWarning):
            with self.assertRaisesRegex(SchemaError, "causal"):
                wrapper.forward(q, kv)


class RaggedPrefillFrontendTests(unittest.TestCase):
    def test_ragged_plan_run_keeps_request_segments_separate(self):
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1, 2]),
            index([0, 1, 3]),
            1,
            1,
            1,
            q_data_type="float32",
        )
        q = tensor([[[0.0]], [[0.0]]])
        k = tensor([[[0.0]], [[0.0]], [[0.0]]])
        v = tensor([[[5.0]], [[10.0]], [[20.0]]])
        output = wrapper.run(q, k, v)
        self.assertEqual(output.data, (5.0, 15.0))

    def test_ragged_hnd_gqa_and_per_head_scales(self):
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace(), kv_layout="HND", backend="reference"
        )
        wrapper.plan(
            index([0, 1]),
            index([0, 2]),
            2,
            1,
            1,
            q_data_type="float32",
            sm_scale=1.0,
        )
        q = tensor([[[1.0], [1.0]]])
        k = tensor([[[1.0], [0.0]]])
        v = tensor([[[10.0], [0.0]]])
        output = wrapper.run(q, k, v, q_scale=tensor([1.0, 2.0]))
        self.assertAlmostEqual(
            output.data[0], 10.0 * math.exp(1.0) / (math.exp(1.0) + 1.0)
        )
        self.assertAlmostEqual(
            output.data[1], 10.0 * math.exp(2.0) / (math.exp(2.0) + 1.0)
        )

    def test_deprecated_forward_executes_ragged_reference(self):
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1]), index([0, 2]), 1, 1, 1,
            q_data_type="float32",
        )
        with self.assertWarns(DeprecationWarning):
            output = wrapper.forward(
                tensor([[[0.0]]]),
                tensor([[[0.0]], [[0.0]]]),
                tensor([[[2.0]], [[4.0]]]),
            )
        self.assertEqual(output.data, (3.0,))


class BatchDecodeFrontendTests(unittest.TestCase):
    def test_packed_cache_run_and_lse(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1]),
            index([0]),
            index([2]),
            1,
            1,
            1,
            2,
            q_data_type="float32",
        )
        packed = tensor(
            [
                [
                    [[[0.0]], [[0.0]]],
                    [[[10.0]], [[20.0]]],
                ]
            ]
        )
        output, lse = wrapper.run(
            tensor([[[0.0]]]), packed, return_lse=True
        )
        self.assertEqual(output.data, (15.0,))
        self.assertAlmostEqual(lse.data[0], math.log(2.0))

    def test_multi_token_decode_is_causal(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), use_tensor_cores=True, backend="reference"
        )
        wrapper.plan(
            index([0, 2]),
            index([0, 1]),
            index([1]),
            1,
            1,
            1,
            2,
            q_data_type="float32",
            q_len_per_req=2,
        )
        k = tensor([[[[0.0]], [[0.0]]], [[[0.0]], [[99.0]]]])
        v = tensor([[[[10.0]], [[20.0]]], [[[30.0]], [[999.0]]]])
        output = wrapper.run(tensor([[[0.0]], [[0.0]]]), (k, v))
        self.assertEqual(output.data, (15.0, 20.0))

    def test_multi_token_requires_public_matrix_core_preference(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        with self.assertRaisesRegex(ValueError, "use_tensor_cores"):
            wrapper.plan(
                index([0, 1]), index([0]), index([2]),
                1, 1, 1, 2, q_len_per_req=2,
            )

    def test_graph_mode_freezes_batch_and_checks_capacity(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(),
            use_cuda_graph=True,
            paged_kv_indptr_buffer=index([0, 0, 0]),
            paged_kv_indices_buffer=index([0, 0, 0, 0]),
            paged_kv_last_page_len_buffer=index([0, 0]),
            backend="reference",
        )
        wrapper.plan(
            index([0, 1, 2]), index([0, 1]), index([1, 1]),
            1, 1, 1, 1,
        )
        self.assertTrue(wrapper.is_cuda_graph_enabled)
        with self.assertRaisesRegex(AttentionStateError, "fixed batch size"):
            wrapper.plan(
                index([0, 1]), index([0]), index([1]),
                1, 1, 1, 1,
            )

    def test_reset_workspace_and_legacy_plan_positionals(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.reset_workspace_buffer(workspace(64), workspace(32))
        with self.assertWarns(DeprecationWarning):
            wrapper.plan(
                index([0, 1]), index([0]), index([1]),
                1, 1, 1, 1, "NONE", -1, 0.0, "float32",
            )
        k = tensor([[[[0.0]]]])
        v = tensor([[[[7.0]]]])
        self.assertEqual(wrapper.run(tensor([[[0.0]]]), (k, v)).data, (7.0,))

    def test_deprecated_forward_executes_decode_and_rejects_plan_drift(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1]), index([0]), index([2]),
            1, 1, 1, 2, q_data_type="float32", window_left=0,
        )
        q = tensor([[[0.0]]])
        kv = (
            tensor([[[[0.0]], [[0.0]]]]),
            tensor([[[[2.0]], [[6.0]]]]),
        )
        with self.assertWarns(DeprecationWarning):
            output = wrapper.forward(q, kv, window_left=0)
        self.assertEqual(output.data, (6.0,))
        with self.assertWarns(DeprecationWarning):
            with self.assertRaisesRegex(SchemaError, "window_left"):
                wrapper.forward(q, kv)


if __name__ == "__main__":
    unittest.main()
