import sys
import unittest
from unittest import mock

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionPlanSpec,
    AttentionTensorAccessPolicy,
    PagedKVCacheSpec,
    PagedKVMetadata,
    TorchAdapterUnavailableError,
    TorchQuantizedTensorInput,
    TorchTensorViewAdapter,
)
from flashinfer_npu.runtime import QuantSpec, SchemaError


ITEMSIZE = {
    "torch.bool": 1,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
}


def contiguous_strides(shape):
    stride = 1
    values = []
    for dim in reversed(shape):
        values.append(stride)
        stride *= max(dim, 1)
    return tuple(reversed(values))


class FakeUntypedStorage:
    def __init__(self, nbytes, pointer, device="cpu"):
        self._nbytes = nbytes
        self._pointer = pointer
        self.device = device

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._pointer


class FakeTorchTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="torch.float32",
        device="cpu",
        strides=None,
        storage=None,
        storage_offset=0,
        data_ptr=None,
        element_size=None,
        layout="torch.strided",
        requires_grad=False,
        is_conj=False,
        is_neg=False,
        is_sparse=False,
        is_nested=False,
        is_quantized=False,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.requires_grad = requires_grad
        self.is_sparse = is_sparse
        self.is_nested = is_nested
        self.is_quantized = is_quantized
        self._strides = (
            contiguous_strides(self.shape) if strides is None else tuple(strides)
        )
        self._element_size = (
            ITEMSIZE.get(dtype, 4) if element_size is None else element_size
        )
        required = self._element_size
        if self.shape and all(dim > 0 for dim in self.shape):
            required = (
                storage_offset
                + sum(
                    (dim - 1) * stride
                    for dim, stride in zip(self.shape, self._strides)
                )
                + 1
            ) * self._element_size
        elif 0 in self.shape:
            required = storage_offset * self._element_size
        self._storage = storage or FakeUntypedStorage(
            required, (id(self) + 0xF) & ~0xF, device
        )
        self._storage_offset = storage_offset
        self._data_ptr = (
            self._storage.data_ptr() + storage_offset * self._element_size
            if data_ptr is None
            else data_ptr
        )
        self._is_conj = is_conj
        self._is_neg = is_neg
        self.contiguous_called = False

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self._storage

    def data_ptr(self):
        return self._data_ptr

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self._storage_offset

    def is_conj(self):
        return self._is_conj

    def is_neg(self):
        return self._is_neg

    def contiguous(self):
        self.contiguous_called = True
        raise AssertionError("adapter must never call contiguous()")


def adapter(stream_resolver=None):
    return TorchTensorViewAdapter(
        tensor_type=FakeTorchTensor, stream_resolver=stream_resolver
    )


def paged_spec(dtype="float32", dim=1, quant_spec=None, structure="separate"):
    return PagedKVCacheSpec(
        num_pages=1,
        page_size=2,
        num_kv_heads=1,
        head_dim_qk=dim,
        head_dim_vo=dim,
        dtype=dtype,
        structure=structure,
        device="cpu",
        quant_spec=quant_spec,
    )


class TorchAdapterBoundaryTests(unittest.TestCase):
    def test_default_adapter_fails_explicitly_when_torch_is_unavailable(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(TorchAdapterUnavailableError, "not installed"):
                TorchTensorViewAdapter()

    def test_non_tensor_and_unsupported_tensor_states_fail(self):
        value = FakeTorchTensor((1,))
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            adapter().to_view(object(), name="q")
        cases = (
            ("requires_grad", {"requires_grad": True}),
            ("is_conj", {"is_conj": True}),
            ("is_neg", {"is_neg": True}),
            ("is_sparse", {"is_sparse": True}),
            ("is_nested", {"is_nested": True}),
            ("native Torch quantized", {"is_quantized": True}),
            ("torch.strided", {"layout": "torch.sparse_coo"}),
        )
        for message, kwargs in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    adapter().to_view(
                        FakeTorchTensor((1,), **kwargs), name="q"
                    )
        # Ensure introspection did not materialize the valid tensor either.
        adapter().to_view(value, name="q")
        self.assertFalse(value.contiguous_called)

    def test_meta_storage_and_dtype_itemsize_mismatch_fail(self):
        with self.assertRaisesRegex(SchemaError, "meta device"):
            adapter().to_view(
                FakeTorchTensor((1,), device="meta"), name="q"
            )
        with self.assertRaisesRegex(SchemaError, "element_size"):
            adapter().to_view(
                FakeTorchTensor(
                    (1,), dtype="torch.float16", element_size=4
                ),
                name="q",
            )


class TorchTensorMappingTests(unittest.TestCase):
    def test_view_preserves_stride_offset_capacity_and_opaque_storage_identity(self):
        storage = FakeUntypedStorage(128, 0x1000)
        first = FakeTorchTensor(
            (3, 2),
            strides=(1, 4),
            storage=storage,
            storage_offset=1,
            # Official PyTorch docs do not guarantee this equals storage ptr + offset.
            data_ptr=0x2010,
        )
        second = FakeTorchTensor(
            (2,), storage=storage, storage_offset=2, data_ptr=0x3008
        )
        first_view = adapter().to_view(first, name="first")
        second_view = adapter().to_view(second, name="second")
        self.assertEqual(first_view.shape, (3, 2))
        self.assertEqual(first_view.strides, (1, 4))
        self.assertEqual(first_view.storage_offset, 1)
        self.assertEqual(first_view.storage_nbytes, 128)
        self.assertEqual(first_view.data_ptr_alignment, 16)
        self.assertEqual(first_view.storage_id, second_view.storage_id)
        self.assertTrue(first_view.overlaps(second_view))
        self.assertNotIn("4096", first_view.storage_id)

    def test_noncontiguous_view_is_preserved_and_policy_decides_support(self):
        tensor = FakeTorchTensor((3, 2), strides=(1, 3))
        result = adapter().to_view(tensor, name="q")
        self.assertFalse(result.is_contiguous)
        self.assertEqual(result.strides, (1, 3))
        self.assertFalse(tensor.contiguous_called)

    def test_storage_bounds_are_checked_against_untyped_storage_not_view_nbytes(self):
        storage = FakeUntypedStorage(8, 0x1000)
        with self.assertRaisesRegex(SchemaError, "exceeds storage"):
            adapter().to_view(
                FakeTorchTensor((3,), storage=storage), name="q"
            )

    def test_cpu_and_accelerator_stream_boundaries_are_explicit(self):
        self.assertEqual(
            adapter().stream_context("cpu").stream_id,
            "torch-cpu-synchronous",
        )
        with self.assertRaisesRegex(SchemaError, "stream resolver"):
            adapter().stream_context("npu:0")
        resolved = adapter(lambda device: "torch_npu-current:%s" % device)
        stream = resolved.stream_context("npu:0")
        self.assertEqual(stream.device, "npu:0")
        self.assertEqual(stream.stream_id, "torch_npu-current:npu:0")


class TorchAttentionContractBuilderTests(unittest.TestCase):
    def test_dense_paged_run_contract_matches_framework_plan(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="float32",
        )
        plan = AttentionFrameworkSession(spec.mode).plan(
            spec, PagedKVMetadata((0, 1), (0,), (2,), 2)
        )
        cache_spec = paged_spec()
        contract = adapter().build_run_contract(
            q=FakeTorchTensor((1, 1, 1)),
            kv_data=(
                FakeTorchTensor((1, 2, 1, 1)),
                FakeTorchTensor((1, 2, 1, 1)),
            ),
            kv_spec=cache_spec,
            plan=plan,
            out=FakeTorchTensor((1, 1, 1)),
            lse=FakeTorchTensor((1, 1)),
            workspace_float=FakeTorchTensor((0,), dtype="torch.uint8"),
            workspace_int=FakeTorchTensor((0,), dtype="torch.uint8"),
        )
        self.assertEqual(contract.stream.stream_id, "torch-cpu-synchronous")
        self.assertTrue(contract.out.writable)
        self.assertTrue(contract.lse.writable)

    def test_shared_storage_output_alias_is_rejected(self):
        spec = AttentionPlanSpec(
            mode=AttentionMode.BATCH_DECODE_PAGED,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=1,
            q_dtype="float32",
            kv_dtype="float32",
        )
        plan = AttentionFrameworkSession(spec.mode).plan(
            spec, PagedKVMetadata((0, 1), (0,), (2,), 2)
        )
        shared = FakeUntypedStorage(16, 0x4000)
        q = FakeTorchTensor((1, 1, 1), storage=shared)
        out = FakeTorchTensor((1, 1, 1), storage=shared)
        with self.assertRaisesRegex(SchemaError, "output cannot alias"):
            adapter().build_run_contract(
                q=q,
                kv_data=(
                    FakeTorchTensor((1, 2, 1, 1)),
                    FakeTorchTensor((1, 2, 1, 1)),
                ),
                kv_spec=paged_spec(),
                plan=plan,
                out=out,
            )

    def test_explicit_quantized_kv_maps_storage_scale_and_spec(self):
        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="float32",
            accumulator_dtype="float32",
        )
        cache_spec = paged_spec(dtype="int8", quant_spec=quant)
        key = TorchQuantizedTensorInput(
            (1, 2, 1, 1),
            FakeTorchTensor((1, 2, 1, 1), dtype="torch.int8"),
            FakeTorchTensor(()),
            quant,
        )
        value = TorchQuantizedTensorInput(
            (1, 2, 1, 1),
            FakeTorchTensor((1, 2, 1, 1), dtype="torch.int8"),
            FakeTorchTensor(()),
            quant,
        )
        result = adapter().to_kv_view((key, value), cache_spec)
        self.assertTrue(result.quantized)
        self.assertEqual(result.key.logical_shape, (1, 2, 1, 1))
        self.assertEqual(result.key.storage.dtype, "int8")
        self.assertEqual(result.key.scale.shape, ())

    def test_packed_dense_kv_uses_one_physical_view(self):
        spec = paged_spec(structure="packed")
        packed = FakeTorchTensor((1, 2, 2, 1, 1))
        result = adapter().to_kv_view(packed, spec)
        self.assertTrue(result.packed)
        self.assertEqual(len(result.component_views), 1)


if __name__ == "__main__":
    unittest.main()
