import unittest

from flashinfer_npu.attention import (
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import single_decode_with_kv_cache
from flashinfer_npu.prefill import single_prefill_with_kv_cache
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_102_public_nvfp4_canonicalization import (
    FakeNpuTensor,
    nvfp4_runtime,
)


class SingleUint8SemanticsCheckpoint(unittest.TestCase):
    """Raw UINT8 never reaches provider planning without quantization meaning."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        (
            self.values,
            self.quant_spec,
            self.catalog,
            self.registry,
            self.calls,
        ) = nvfp4_runtime()
        install_attention_operator_runtime_resolvers(
            self.registry, operation_catalog=self.catalog
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def assert_provider_was_not_observed(self):
        self.assertEqual(self.values["gate"].calls, 0)
        self.assertEqual(self.values["loader"].version_calls, 0)
        self.assertEqual(self.values["loader"].resolve_calls, 0)
        self.assertEqual(self.calls, [])

    def test_single_prefill_packed_uint8_requires_kv_cache_sf(self):
        q = FakeNpuTensor("q", (3, 8, 128), dtype="bfloat16")
        key = FakeNpuTensor("key", (7, 2, 64), dtype="uint8")
        value = FakeNpuTensor("value", (7, 2, 64), dtype="uint8")

        with self.assertRaisesRegex(SchemaError, "requires kv_cache_sf"):
            single_prefill_with_kv_cache(q, key, value)

        self.assert_provider_was_not_observed()

    def test_single_prefill_full_width_uint8_is_not_treated_as_dense(self):
        q = FakeNpuTensor("q", (3, 8, 128), dtype="bfloat16")
        key = FakeNpuTensor("key", (7, 2, 128), dtype="uint8")
        value = FakeNpuTensor("value", (7, 2, 128), dtype="uint8")

        with self.assertRaisesRegex(SchemaError, "explicit QuantSpec"):
            single_prefill_with_kv_cache(q, key, value)

        self.assert_provider_was_not_observed()

    def test_single_decode_raw_uint8_requires_explicit_quantization(self):
        q = FakeNpuTensor("q", (8, 128), dtype="bfloat16")
        key = FakeNpuTensor("key", (7, 2, 128), dtype="uint8")
        value = FakeNpuTensor("value", (7, 2, 128), dtype="uint8")

        with self.assertRaisesRegex(SchemaError, "explicit QuantSpec"):
            single_decode_with_kv_cache(q, key, value)

        self.assert_provider_was_not_observed()


if __name__ == "__main__":
    unittest.main()
