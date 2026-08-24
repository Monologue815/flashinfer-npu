import unittest

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionPlanSpec,
    AttentionRunTensorContract,
    AttentionTensorAccessPolicy,
    KVCacheView,
    PagedKVCacheSpec,
    PagedKVMetadata,
    QuantizedTensorView,
    StreamContext,
    TensorView,
    contiguous_strides,
    dtype_itemsize,
)
from flashinfer_npu.runtime import QuantSpec, SchemaError


def view(
    shape,
    *,
    strides=None,
    dtype="float32",
    storage_id="storage",
    storage_nbytes=None,
    storage_offset=0,
    alignment=64,
    writable=False,
    device="cpu",
):
    shape = tuple(shape)
    strides = contiguous_strides(shape) if strides is None else tuple(strides)
    if storage_nbytes is None:
        max_offset = storage_offset + sum(
            (dim - 1) * stride for dim, stride in zip(shape, strides)
        )
        elements = storage_offset if 0 in shape else max_offset + 1
        storage_nbytes = elements * dtype_itemsize(dtype)
    return TensorView(
        shape=shape,
        strides=strides,
        dtype=dtype,
        device=device,
        storage_id=storage_id,
        storage_nbytes=storage_nbytes,
        storage_offset=storage_offset,
        data_ptr_alignment=alignment,
        writable=writable,
    )


def dense_cache_spec(dtype="float32", dim=2):
    return PagedKVCacheSpec(
        num_pages=1,
        page_size=2,
        num_kv_heads=1,
        head_dim_qk=dim,
        head_dim_vo=dim,
        dtype=dtype,
        structure="separate",
        device="cpu",
    )


def dense_kv(dim=2):
    spec = dense_cache_spec(dim=dim)
    shape = (1, 2, 1, dim)
    return KVCacheView(
        spec,
        view(shape, storage_id="key"),
        view(shape, storage_id="value"),
    )


class TensorViewTests(unittest.TestCase):
    def test_contiguous_transposed_and_sliced_views_are_storage_safe(self):
        contiguous = view((2, 3))
        transposed = view((3, 2), strides=(1, 3), storage_id="transpose")
        sliced = view((2, 3), strides=(6, 2), storage_id="slice")
        self.assertTrue(contiguous.is_contiguous)
        self.assertFalse(transposed.is_contiguous)
        self.assertFalse(sliced.is_contiguous)
        self.assertFalse(transposed.has_internal_overlap)
        self.assertFalse(sliced.has_internal_overlap)

    def test_internal_overlap_negative_stride_and_out_of_bounds_fail(self):
        with self.assertRaisesRegex(SchemaError, "internal overlap"):
            view((3, 3), strides=(2, 4), storage_id="overlap")
        with self.assertRaisesRegex(SchemaError, "negative strides"):
            view((2,), strides=(-1,), storage_id="negative")
        with self.assertRaisesRegex(SchemaError, "exceeds storage"):
            view((4,), storage_id="small", storage_nbytes=12)

    def test_byte_offsets_alignment_and_storage_overlap_are_explicit(self):
        left = view(
            (2,),
            storage_id="shared",
            storage_nbytes=32,
            storage_offset=0,
        )
        right = view(
            (2,),
            storage_id="shared",
            storage_nbytes=32,
            storage_offset=1,
            alignment=4,
        )
        separate = view((2,), storage_id="other")
        self.assertTrue(left.overlaps(right))
        self.assertFalse(left.overlaps(separate))
        with self.assertRaisesRegex(SchemaError, "16-byte aligned"):
            right.require_alignment(16, "right")

    def test_tensor_view_round_trip_and_unknown_dtype(self):
        original = view((2, 3), strides=(4, 1), storage_nbytes=28)
        restored = TensorView.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)
        with self.assertRaisesRegex(SchemaError, "unknown tensor dtype"):
            view((1,), dtype="mystery")


class QuantizedTensorViewTests(unittest.TestCase):
    def test_packed_int4_storage_and_group_scale_shapes_are_derived(self):
        spec = QuantSpec(
            scheme="symmetric",
            storage_dtype="int4_packed",
            compute_dtype="float32",
            accumulator_dtype="float32",
            granularity="group",
            axis=(0, 1, 2, 3),
            group_size=(1, 1, 1, 2),
            packing_order="low_nibble_first",
        )
        value = QuantizedTensorView(
            logical_shape=(1, 2, 1, 3),
            storage=view(
                (1, 2, 1, 2), dtype="uint8", storage_id="packed"
            ),
            scale=view((1, 2, 1, 2), storage_id="scale"),
            quant_spec=spec,
        )
        self.assertEqual(value.storage.shape, (1, 2, 1, 2))
        with self.assertRaisesRegex(SchemaError, "scale view shape"):
            QuantizedTensorView(
                logical_shape=(1, 2, 1, 3),
                storage=view(
                    (1, 2, 1, 2), dtype="uint8", storage_id="bad-packed"
                ),
                scale=view((1,), storage_id="bad-scale"),
                quant_spec=spec,
            )

    def test_asymmetric_view_requires_independent_int32_zero_point(self):
        spec = QuantSpec(
            scheme="asymmetric",
            storage_dtype="uint8",
            compute_dtype="float32",
            accumulator_dtype="float32",
            granularity="token",
            axis=(1,),
            has_zero_point=True,
        )
        with self.assertRaisesRegex(SchemaError, "requires zero_point"):
            QuantizedTensorView(
                (1, 2, 1, 1),
                view((1, 2, 1, 1), dtype="uint8", storage_id="u8"),
                view((2,), storage_id="scale"),
                spec,
            )
        with self.assertRaisesRegex(SchemaError, "components cannot alias"):
            shared = view(
                (1, 1, 1, 1),
                dtype="uint8",
                storage_id="shared",
                storage_nbytes=8,
            )
            # Per-tensor keeps component shapes equal, making alias misuse visible.
            scalar_spec = QuantSpec(
                scheme="symmetric",
                storage_dtype="uint8",
                compute_dtype="float32",
                accumulator_dtype="float32",
            )
            QuantizedTensorView(
                (1, 1, 1, 1),
                shared,
                view((), storage_id="shared", storage_nbytes=shared.storage_nbytes),
                scalar_spec,
            )


class AttentionRunTensorContractTests(unittest.TestCase):
    def test_generic_policy_accepts_strided_q_but_backend_can_require_contiguous(self):
        q = view((1, 1, 2), strides=(4, 4, 1), storage_nbytes=8, storage_id="q")
        contract = AttentionRunTensorContract(
            q=q,
            kv=dense_kv(2),
            stream=StreamContext("cpu", "stream-0"),
        )
        contract.validate(AttentionTensorAccessPolicy())
        with self.assertRaisesRegex(SchemaError, "q must be contiguous"):
            contract.validate(
                AttentionTensorAccessPolicy(require_contiguous_q=True)
            )

    def test_output_must_be_writable_and_cannot_alias_input(self):
        q = view((1, 1, 2), storage_id="shared")
        read_only_out = view((1, 1, 2), storage_id="out")
        with self.assertRaisesRegex(SchemaError, "out view must be writable"):
            AttentionRunTensorContract(
                q, dense_kv(2), StreamContext("cpu", "s"), out=read_only_out
            ).validate(AttentionTensorAccessPolicy())
        aliased_out = view(
            (1, 1, 2), storage_id="shared", writable=True
        )
        with self.assertRaisesRegex(SchemaError, "output cannot alias"):
            AttentionRunTensorContract(
                q, dense_kv(2), StreamContext("cpu", "s"), out=aliased_out
            ).validate(AttentionTensorAccessPolicy())

    def test_workspace_pair_writable_and_alias_rules(self):
        q = view((1, 1, 2), storage_id="q")
        float_ws = view(
            (64,), dtype="uint8", storage_id="ws", writable=True
        )
        with self.assertRaisesRegex(SchemaError, "provided together"):
            AttentionRunTensorContract(
                q,
                dense_kv(2),
                StreamContext("cpu", "s"),
                workspace_float=float_ws,
            ).validate(AttentionTensorAccessPolicy())
        int_ws = view((32,), dtype="uint8", storage_id="ws", writable=True)
        with self.assertRaisesRegex(SchemaError, "workspaces cannot alias"):
            AttentionRunTensorContract(
                q,
                dense_kv(2),
                StreamContext("cpu", "s"),
                workspace_float=float_ws,
                workspace_int=int_ws,
            ).validate(AttentionTensorAccessPolicy())

    def test_stream_device_and_ordering_are_required(self):
        contract = AttentionRunTensorContract(
            view((1, 1, 2), storage_id="q"),
            dense_kv(2),
            StreamContext("cpu:1", "wrong"),
        )
        with self.assertRaisesRegex(SchemaError, "stream device"):
            contract.validate(AttentionTensorAccessPolicy())
        unordered = AttentionRunTensorContract(
            view((1, 1, 2), storage_id="q"),
            dense_kv(2),
            StreamContext("cpu", "unordered", ordered=False),
        )
        with self.assertRaisesRegex(SchemaError, "ordered"):
            unordered.validate(AttentionTensorAccessPolicy())

    def test_plan_validation_checks_semantic_shape_dtype_and_output(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=2,
            q_dtype="float32",
            kv_dtype="float32",
        )
        plan = AttentionFrameworkSession(spec.mode).plan(
            spec, PagedKVMetadata((0, 1), (0,), (2,), 2)
        )
        contract = AttentionRunTensorContract(
            view((1, 1, 2), storage_id="q"),
            dense_kv(2),
            StreamContext("cpu", "s"),
            out=view((1, 1, 2), storage_id="out", writable=True),
            lse=view((1, 1), storage_id="lse", writable=True),
        )
        contract.validate(AttentionTensorAccessPolicy(), plan=plan)
        wrong_q = AttentionRunTensorContract(
            view((2, 1, 2), storage_id="wrong-q"),
            dense_kv(2),
            StreamContext("cpu", "s"),
        )
        with self.assertRaisesRegex(SchemaError, "q shape"):
            wrong_q.validate(AttentionTensorAccessPolicy(), plan=plan)


if __name__ == "__main__":
    unittest.main()
