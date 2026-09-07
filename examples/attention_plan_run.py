"""Run from the checkout with: python3 -m examples.attention_plan_run.

Small CPU reference example for the public mixed Attention lifecycle.
Reference tensor containers are for framework validation, not NPU inputs.
"""

import json
import math

from flashinfer_npu.attention import (
    BatchAttention,
    PagedKVCacheSpec,
    ReferenceBuffer,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
)
from flashinfer_npu.runtime import QuantSpec


def tensor(data, dtype="float32"):
    return ReferenceTensor.from_nested(data, dtype=dtype, device="cpu")


def make_int8_cache():
    quant = QuantSpec(
        scheme="symmetric", storage_dtype="int8", compute_dtype="float32",
        accumulator_dtype="float32", scale_dtype="float32", granularity="tensor",
    )
    cache_spec = PagedKVCacheSpec(
        num_pages=2, page_size=2, num_kv_heads=1,
        head_dim_qk=1, head_dim_vo=1, dtype="int8",
        structure="separate", device="cpu", quant_spec=quant,
    )
    key = ReferenceQuantizedTensor(
        logical_shape=(2, 2, 1, 1),
        storage=tensor([[[[0]], [[0]]], [[[0]], [[0]]]], "int8"),
        scale=tensor(1.0), quant_spec=quant,
    )
    value = ReferenceQuantizedTensor(
        logical_shape=(2, 2, 1, 1),
        storage=tensor([[[[20]], [[40]]], [[[10]], [[0]]]], "int8"),
        scale=tensor(0.5), quant_spec=quant,
    )
    return quant, ReferenceQuantizedKVData(cache_spec, key, value)


def run_example(kv_dtype, cache):
    attention = BatchAttention(kv_layout="NHD", device="cpu")
    attention.plan(
        # Request 0: one decode token. Request 1: two prefill tokens.
        qo_indptr=tensor([0, 1, 3], "int32"),
        kv_indptr=tensor([0, 1, 2], "int32"),
        kv_indices=tensor([1, 0], "int32"),
        kv_len_arr=tensor([1, 2], "int32"),
        num_qo_heads=1, num_kv_heads=1,
        head_dim_qk=1, head_dim_vo=1, page_size=2,
        causal=True, q_data_type="float32", kv_data_type=kv_dtype,
    )
    planned = attention.plan_state  # Optional read-only diagnostics.
    q = tensor([[[0.0]], [[0.0]], [[0.0]]])
    out = ReferenceBuffer.zeros((3, 1, 1))
    lse = ReferenceBuffer.zeros((3, 1))

    output, logsumexp = attention.run(q, cache, out=out, lse=lse)
    first_output = tuple(output.data)
    if first_output != (5.0, 10.0, 15.0):
        raise AssertionError("mixed causal output differs from the analytic values")
    if not all(math.isclose(a, b, abs_tol=1e-7)
               for a, b in zip(logsumexp.data, (0.0, 0.0, math.log(2.0)))):
        raise AssertionError("LSE differs from the visible token counts")

    # A new layer can reuse the same metadata and caller-owned output buffers.
    output, logsumexp = attention.run(q, cache, out=out, lse=lse, v_scale=2.0)
    if tuple(output.data) != (10.0, 20.0, 30.0):
        raise AssertionError("run-time value scale was not applied exactly once")
    if output is not out or logsumexp is not lse:
        raise AssertionError("caller-owned buffers were not reused")
    if attention.plan_state is not planned:
        raise AssertionError("run unexpectedly replaced the active plan")
    return {
        "route": attention.plan_selection.route,
        "first_output": first_output,
        "scaled_output": tuple(output.data),
        "plan_reused": True,
        "output_buffers_reused": True,
    }


def main():
    dense = (
        tensor([[[[0.0]], [[0.0]]], [[[0.0]], [[0.0]]]]),
        tensor([[[[10.0]], [[20.0]]], [[[5.0]], [[0.0]]]]),
    )
    quant, int8_cache = make_int8_cache()
    print(json.dumps({
        "execution": "CPU reference; no provider operator is installed by this example",
        "dense": run_example("float32", dense),
        "int8": run_example(quant, int8_cache),
    }, indent=2))


if __name__ == "__main__":
    main()
