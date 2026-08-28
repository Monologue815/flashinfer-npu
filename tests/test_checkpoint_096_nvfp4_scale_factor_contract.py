import unittest

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionPlanSpec,
    KVLayout,
    PagedKVMetadata,
    PagedPrefillMetadata,
    SingleAttentionMetadata,
    TensorView,
    contiguous_strides,
    inspect_attention_nvfp4_kv_scale_factors,
)
from flashinfer_npu.runtime import SchemaError


class FakeTensor:
    def __init__(self, name, shape, *, dtype="float8_e4m3fn", device="npu:0", strides=None):
        shape = tuple(shape)
        self.tensor_view = TensorView(
            shape=shape,
            strides=contiguous_strides(shape) if strides is None else tuple(strides),
            dtype=dtype,
            device=device,
            storage_id=name,
            storage_nbytes=max(1, _numel(shape) * 8),
            data_ptr_alignment=64,
        )


class FakeInspector:
    def __init__(self):
        self.calls = []

    def to_view(self, tensor, *, name, writable=False):
        self.calls.append((name, writable))
        return tensor.tensor_view


def _numel(shape):
    result = 1
    for item in shape:
        result *= item
    return result


def paged_plan(layout=KVLayout.NHD, *, head_dim_qk=128, head_dim_vo=128):
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        kv_layout=layout,
        q_dtype="bfloat16",
        kv_dtype="uint8",
        o_dtype="bfloat16",
    )
    metadata = PagedPrefillMetadata(
        qo_indptr=(0, 2, 3),
        paged_kv=PagedKVMetadata(
            indptr=(0, 2, 3),
            indices=(5, 1, 3),
            last_page_len=(8, 16),
            page_size=16,
        ),
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def single_plan(layout=KVLayout.NHD, *, head_dim_qk=128, head_dim_vo=64):
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        kv_layout=layout,
        q_dtype="bfloat16",
        kv_dtype="uint8",
        o_dtype="bfloat16",
    )
    return AttentionFrameworkSession(spec.mode).plan(
        spec, SingleAttentionMetadata(qo_len=3, kv_len=7)
    )


class Nvfp4ScaleFactorContractCheckpoint(unittest.TestCase):
    """NVFP4 scale factors have a provider-independent metadata contract."""

    def test_paged_nhd_separate_shapes_follow_flashinfer_blocks(self):
        inspector = FakeInspector()
        value = (
            FakeTensor("k_sf", (6, 16, 2, 8)),
            FakeTensor("v_sf", (6, 16, 2, 8)),
        )

        result = inspect_attention_nvfp4_kv_scale_factors(
            paged_plan(), value, inspector, "npu:0"
        )

        self.assertEqual(result.structure, "separate")
        self.assertEqual(result.key_shape, (6, 16, 2, 8))
        self.assertEqual(result.value_shape, (6, 16, 2, 8))
        self.assertEqual(tuple(name for name, _ in result.named_views), (
            "kv_cache_sf.key", "kv_cache_sf.value"
        ))

    def test_paged_hnd_combined_shape_is_validated(self):
        result = inspect_attention_nvfp4_kv_scale_factors(
            paged_plan(KVLayout.HND),
            FakeTensor("combined_sf", (6, 2, 2, 16, 8)),
            FakeInspector(),
            "npu:0",
        )

        self.assertEqual(result.structure, "combined")
        self.assertEqual(result.combined.shape, (6, 2, 2, 16, 8))

    def test_single_prefill_requires_separate_linear_k_v_shapes(self):
        result = inspect_attention_nvfp4_kv_scale_factors(
            single_plan(KVLayout.HND),
            (
                FakeTensor("k_sf", (2, 7, 8)),
                FakeTensor("v_sf", (2, 7, 4)),
            ),
            FakeInspector(),
            "npu:0",
        )

        self.assertEqual(result.key_shape, (2, 7, 8))
        self.assertEqual(result.value_shape, (2, 7, 4))
        with self.assertRaisesRegex(SchemaError, r"must be a \(K, V\) tuple"):
            inspect_attention_nvfp4_kv_scale_factors(
                single_plan(), FakeTensor("combined", (7, 2, 2, 8)),
                FakeInspector(), "npu:0"
            )

    def test_dtype_device_layout_and_shape_fail_closed(self):
        plan = paged_plan()
        valid_v = FakeTensor("v_sf", (6, 16, 2, 8))
        cases = (
            (FakeTensor("k_dtype", (6, 16, 2, 8), dtype="float16"), "dtype"),
            (FakeTensor("k_device", (6, 16, 2, 8), device="npu:1"), "device"),
            (FakeTensor("k_shape", (6, 16, 2, 7)), "shape"),
            (FakeTensor("k_stride", (6, 16, 2, 8), strides=(300, 16, 8, 1)), "contiguous"),
        )
        for key, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    inspect_attention_nvfp4_kv_scale_factors(
                        plan, (key, valid_v), FakeInspector(), "npu:0"
                    )

    def test_page_capacity_head_block_and_combined_dimension_fail_closed(self):
        with self.assertRaisesRegex(SchemaError, "referenced KV page"):
            inspect_attention_nvfp4_kv_scale_factors(
                paged_plan(),
                (FakeTensor("k", (5, 16, 2, 8)), FakeTensor("v", (5, 16, 2, 8))),
                FakeInspector(), "npu:0"
            )
        with self.assertRaisesRegex(SchemaError, "divisible by 16"):
            inspect_attention_nvfp4_kv_scale_factors(
                paged_plan(head_dim_qk=120),
                (FakeTensor("k", (6, 16, 2, 8)), FakeTensor("v", (6, 16, 2, 8))),
                FakeInspector(), "npu:0"
            )
        with self.assertRaisesRegex(SchemaError, "equal K/V"):
            inspect_attention_nvfp4_kv_scale_factors(
                paged_plan(head_dim_vo=64),
                FakeTensor("combined", (6, 2, 16, 2, 8)),
                FakeInspector(), "npu:0"
            )


if __name__ == "__main__":
    unittest.main()
