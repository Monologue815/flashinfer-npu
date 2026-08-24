import inspect
import math
import unittest

import flashinfer_npu
from flashinfer_npu.attention import (
    BatchAttention,
    ReferenceBuffer,
    ReferenceTensor,
)
from flashinfer_npu.runtime import DispatchError, SchemaError


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def index(values):
    return tensor(values, dtype="int32")


class BatchAttentionFrontendTests(unittest.TestCase):
    def test_signatures_match_holistic_attention_surface(self):
        self.assertEqual(
            list(inspect.signature(BatchAttention).parameters),
            ["kv_layout", "device"],
        )
        self.assertEqual(
            list(inspect.signature(BatchAttention.plan).parameters),
            [
                "self", "qo_indptr", "kv_indptr", "kv_indices", "kv_len_arr",
                "num_qo_heads", "num_kv_heads", "head_dim_qk", "head_dim_vo",
                "page_size", "causal", "sm_scale", "logits_soft_cap",
                "q_data_type", "kv_data_type", "use_profiler",
            ],
        )
        self.assertEqual(
            list(inspect.signature(BatchAttention.run).parameters),
            [
                "self", "q", "kv_cache", "out", "lse", "k_scale",
                "v_scale", "logits_soft_cap", "profiler_buffer", "kv_cache_sf",
            ],
        )

    def test_host_execution_requires_explicit_cpu_device(self):
        with self.assertRaisesRegex(DispatchError, "device='cpu'"):
            BatchAttention()
        self.assertIs(flashinfer_npu.BatchAttention, BatchAttention)

    def test_mixed_prefill_decode_run_always_returns_output_and_lse(self):
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            index([0, 1, 3]),
            index([0, 1, 2]),
            index([1, 0]),
            index([1, 2]),
            1,
            1,
            1,
            1,
            2,
            q_data_type="float32",
            kv_data_type="float32",
        )
        q = tensor([[[0.0]], [[0.0]], [[0.0]]])
        k = tensor([[[[0.0]], [[0.0]]], [[[0.0]], [[99.0]]]])
        v = tensor([[[[10.0]], [[20.0]]], [[[5.0]], [[999.0]]]])
        output, lse = wrapper.run(q, (k, v))
        self.assertEqual(output.data, (5.0, 15.0, 15.0))
        self.assertEqual(lse.shape, (3, 1))
        self.assertAlmostEqual(lse.data[0], 0.0)
        self.assertAlmostEqual(lse.data[1], math.log(2.0))

    def test_mixed_run_writes_caller_buffers_and_applies_value_scale(self):
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            index([0, 1]), index([0, 1]), index([0]), index([2]),
            1, 1, 1, 1, 2,
            q_data_type="float32", kv_data_type="float32",
        )
        out = ReferenceBuffer.zeros((1, 1, 1))
        lse = ReferenceBuffer.zeros((1, 1))
        output, lse_output = wrapper.run(
            tensor([[[0.0]]]),
            (tensor([[[[0.0]], [[0.0]]]]), tensor([[[[10.0]], [[20.0]]]])),
            out=out,
            lse=lse,
            v_scale=tensor([2.0]),
        )
        self.assertIs(output, out)
        self.assertIs(lse_output, lse)
        self.assertEqual(out.data, (30.0,))

    def test_all_empty_mixed_batch_returns_empty_output_and_lse(self):
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            index([0, 0, 0]),
            index([0, 0, 0]),
            index([]),
            index([0, 0]),
            1,
            1,
            1,
            1,
            2,
            q_data_type="float32",
            kv_data_type="float32",
        )
        q = ReferenceTensor.zeros((0, 1, 1), dtype="float32", device="cpu")
        k = tensor([[[[0.0]], [[0.0]]]])
        v = tensor([[[[0.0]], [[0.0]]]])
        output, lse = wrapper.run(q, (k, v))
        self.assertEqual(output.shape, (0, 1, 1))
        self.assertEqual(output.data, ())
        self.assertEqual(lse.shape, (0, 1))
        self.assertEqual(lse.data, ())

    def test_runtime_feature_gaps_and_soft_cap_contract_are_explicit(self):
        wrapper = BatchAttention(device="cpu")
        wrapper.plan(
            index([0, 1]), index([0, 1]), index([0]), index([1]),
            1, 1, 1, 1, 1,
            q_data_type="float32", kv_data_type="float32",
        )
        q = tensor([[[0.0]]])
        kv = (tensor([[[[0.0]]]]), tensor([[[[1.0]]]]))
        with self.assertRaisesRegex(SchemaError, "capped plan"):
            wrapper.run(q, kv, logits_soft_cap=1.0)
        with self.assertRaisesRegex(NotImplementedError, "NVFP4"):
            wrapper.run(q, kv, kv_cache_sf=tensor([1.0]))

        profiled = BatchAttention(device="cpu")
        profiled.plan(
            index([0, 1]), index([0, 1]), index([0]), index([1]),
            1, 1, 1, 1, 1,
            q_data_type="float32", kv_data_type="float32", use_profiler=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "profiler"):
            profiled.run(q, kv, profiler_buffer=tensor([0], dtype="uint64"))


if __name__ == "__main__":
    unittest.main()
