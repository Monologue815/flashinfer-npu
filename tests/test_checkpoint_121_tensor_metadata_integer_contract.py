import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorQuantizedKVInput, AttentionOperatorQuantizedTensorInput,
    QuantizedTensorView, TensorView, attention_operator_runtime_registry_snapshot,
    contiguous_strides, install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.operator_quantization import AttentionOperatorImplicitUnitScale
from flashinfer_npu.decode import single_decode_with_kv_cache
from flashinfer_npu.runtime import QuantSpec, SchemaError
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_043_provider_workspace_reset import runtime_registry
from tests.test_checkpoint_102_public_nvfp4_canonicalization import FakeNpuTensor
from tests.test_checkpoint_116_metadata_integer_contract import IndexValue, IntOnly


def tensor_view(shape=(2, 2), dtype="float32", storage_id="test"):
    return TensorView(shape, contiguous_strides(shape), dtype, "npu:0", storage_id, 64)


def quant_spec():
    return QuantSpec(scheme="symmetric", storage_dtype="int8", compute_dtype="float32",
                     accumulator_dtype="float32", scale_dtype="float32", granularity="tensor")


class TensorMetadataIntegerContractCheckpoint(unittest.TestCase):
    def test_storage_view_rejects_non_integer_fields(self):
        view = tensor_view()
        fields = dict(shape=(2, 2), strides=(2, 1), storage_nbytes=64,
                      storage_offset=0, data_ptr_alignment=1)
        for name, current in fields.items():
            number = current[0] if isinstance(current, tuple) else current
            for value in (True, float(number), number + 0.5, str(number),
                          float("inf"), float("nan"), IntOnly()):
                with self.subTest(field=name, value=value):
                    value = (value, current[1]) if isinstance(current, tuple) else value
                    with self.assertRaises(SchemaError):
                        replace(view, **{name: value})

    def test_integer_protocol_preserves_storage_bounds_and_fingerprint(self):
        expected = tensor_view()
        view = replace(expected, shape=(IndexValue(2), IndexValue(2)),
                       strides=(IndexValue(2), IndexValue(1)),
                       storage_nbytes=IndexValue(64), storage_offset=IndexValue(0),
                       data_ptr_alignment=IndexValue(1))
        self.assertEqual(view, expected)
        self.assertEqual(view.fingerprint, expected.fingerprint)
        self.assertEqual(contiguous_strides((IndexValue(2), IndexValue(2))), (2, 1))
        with self.assertRaisesRegex(SchemaError, "storage"):
            replace(view, storage_nbytes=IndexValue(1))

    def shape_factories(self):
        quant = quant_spec()
        storage = tensor_view(dtype="int8")
        scale = tensor_view((), storage_id="scale")
        return (
            lambda shape: contiguous_strides(shape),
            lambda shape: AttentionOperatorQuantizedTensorInput(quant, shape, object(), object()),
            lambda shape: AttentionOperatorQuantizedKVInput(
                quant, object(), object(), object(), object(),
                key_logical_shape=shape, value_logical_shape=(2, 2)),
            lambda shape: AttentionOperatorQuantizedKVInput(
                quant, object(), object(), object(), object(),
                key_logical_shape=(2, 2), value_logical_shape=shape),
            lambda shape: QuantizedTensorView(shape, storage, scale, quant),
        )

    def test_quantized_logical_shapes_cannot_be_silently_truncated(self):
        for slot, factory in enumerate(self.shape_factories()):
            for value in (True, 2.0, 2.5, "2", float("inf"), IntOnly()):
                with self.subTest(slot=slot, value=value):
                    with self.assertRaises(SchemaError):
                        factory((value, 2))
            factory((IndexValue(2), IndexValue(2)))

    def test_implicit_scale_shape_uses_same_integer_contract(self):
        for value in (True, 2.0, 2.5, "2", float("inf"), IntOnly()):
            with self.subTest(value=value):
                with self.assertRaises(SchemaError):
                    AttentionOperatorImplicitUnitScale("run.q_head_scale", (value,), "float32", "npu:0")
        value = AttentionOperatorImplicitUnitScale("run.q_head_scale", (IndexValue(2),), "float32", "npu:0")
        self.assertEqual(value.shape, (2,))

    def test_public_single_decode_rejects_bad_shapes_before_provider_resolution(self):
        original = attention_operator_runtime_registry_snapshot()
        components = build_components()
        try:
            install_attention_operator_runtime_resolvers(
                runtime_registry(components), operation_catalog=components["catalog"])
            for slot in range(3):
                for value in (True, 2.5, float("inf"), "2"):
                    with self.subTest(slot=slot, value=value):
                        tensors = [FakeNpuTensor("q", (8, 128), dtype="bfloat16"),
                                   FakeNpuTensor("k", (64, 2, 128), dtype="bfloat16"),
                                   FakeNpuTensor("v", (64, 2, 128), dtype="bfloat16")]
                        shape = tensors[slot].shape
                        tensors[slot].shape = (value,) + shape[1:]
                        with self.assertRaises(SchemaError):
                            single_decode_with_kv_cache(*tensors)
            self.assertEqual(components["events"], [])
        finally:
            install_attention_operator_runtime_resolvers(
                original.registry, operation_catalog=original.operation_catalog)
