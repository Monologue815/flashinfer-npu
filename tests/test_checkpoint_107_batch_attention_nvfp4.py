import unittest

from flashinfer_npu.attention import (
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_102_public_nvfp4_canonicalization import (
    FakeNpuTensor,
    explicit_uint8_spec,
    nvfp4_runtime,
)


class BatchAttentionNvfp4Checkpoint(unittest.TestCase):
    """Unified mixed-paged Attention shares the classic NVFP4 contract."""

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

    def wrapper(self):
        return BatchAttention(kv_layout="NHD", device="npu:0")

    def plan(self, wrapper, *, kv_data_type="uint8"):
        return wrapper.plan(
            [0, 1, 3],
            [0, 1, 2],
            [0, 1],
            [16, 16],
            8,
            2,
            128,
            128,
            16,
            causal=True,
            q_data_type="bfloat16",
            kv_data_type=kv_data_type,
        )

    def test_bare_uint8_plan_uses_the_public_nvfp4_contract(self):
        wrapper = self.wrapper()

        self.assertIsNone(self.plan(wrapper))

        self.assertEqual(wrapper.plan_state.spec.kv_dtype, "uint8")
        self.assertEqual(wrapper.plan_state.spec.kv_quant_spec, self.quant_spec)
        self.assertEqual(wrapper.plan_selection.route, "provider")
        self.assertEqual(self.values["loader"].resolve_calls, 1)

    def test_one_mixed_plan_accepts_separate_then_combined_nvfp4(self):
        wrapper = self.wrapper()
        self.plan(wrapper)
        plan_fingerprint = wrapper.plan_state.fingerprint
        key = FakeNpuTensor("key", (2, 16, 2, 64), dtype="uint8")
        value = FakeNpuTensor("value", (2, 16, 2, 64), dtype="uint8")
        k_sf = FakeNpuTensor(
            "k-sf", (2, 16, 2, 8), dtype="float8_e4m3fn"
        )
        v_sf = FakeNpuTensor(
            "v-sf", (2, 16, 2, 8), dtype="float8_e4m3fn"
        )

        separate = wrapper.run(
            "q-separate", (key, value), kv_cache_sf=(k_sf, v_sf)
        )
        combined_kv = FakeNpuTensor(
            "combined", (2, 2, 16, 2, 64), dtype="uint8"
        )
        combined_sf = FakeNpuTensor(
            "combined-sf", (2, 2, 16, 2, 8), dtype="float8_e4m3fn"
        )
        combined = wrapper.run(
            "q-combined", combined_kv, kv_cache_sf=combined_sf
        )

        self.assertEqual(
            separate, ("nvfp4-output:q-separate", "nvfp4-lse")
        )
        self.assertEqual(
            combined, ("nvfp4-output:q-combined", "nvfp4-lse")
        )
        self.assertEqual(wrapper.plan_state.fingerprint, plan_fingerprint)
        self.assertIs(self.calls[0][3]["k_sf"], k_sf)
        self.assertIs(self.calls[0][3]["v_sf"], v_sf)
        self.assertIs(self.calls[1][1], combined_kv)
        self.assertIsNone(self.calls[1][2])
        self.assertIs(self.calls[1][3]["kv_cache_sf"], combined_sf)
        self.assertEqual(self.values["loader"].resolve_calls, 1)
        self.assertEqual(self.values["authority"].calls, 1)

    def test_missing_scales_fails_without_dense_retry(self):
        wrapper = self.wrapper()
        self.plan(wrapper)
        combined_kv = FakeNpuTensor(
            "combined", (2, 2, 16, 2, 64), dtype="uint8"
        )

        with self.assertRaisesRegex(SchemaError, "requires kv_cache_sf"):
            wrapper.run("q", combined_kv)

        self.assertEqual(self.calls, [])
        self.assertEqual(self.values["loader"].resolve_calls, 1)

    def test_explicit_uint8_quantspec_is_not_reclassified(self):
        wrapper = self.wrapper()
        explicit = explicit_uint8_spec()

        self.plan(wrapper, kv_data_type=explicit)

        self.assertIs(wrapper.plan_state.spec.kv_quant_spec, explicit)
        self.assertNotEqual(explicit.fingerprint, self.quant_spec.fingerprint)


if __name__ == "__main__":
    unittest.main()
