import unittest

from flashinfer_npu.attention import (
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from tests.test_checkpoint_102_public_nvfp4_canonicalization import (
    FakeNpuTensor,
    FakeNpuWorkspace,
    nvfp4_runtime,
)


class PublicCombinedNvfp4RunCheckpoint(unittest.TestCase):
    """One public plan accepts both provider-authorized paged KV structures."""

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

    def test_decode_reuses_one_plan_for_separate_then_combined_nvfp4(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="NHD", backend="auto"
        )
        wrapper.plan(
            [0, 1],
            [0],
            [16],
            8,
            2,
            128,
            16,
            q_data_type="bfloat16",
            kv_data_type="uint8",
            o_data_type="bfloat16",
        )
        plan_fingerprint = wrapper.plan_state.fingerprint
        key = FakeNpuTensor("key", (1, 16, 2, 64), dtype="uint8")
        value = FakeNpuTensor("value", (1, 16, 2, 64), dtype="uint8")
        k_sf = FakeNpuTensor(
            "k-sf", (1, 16, 2, 8), dtype="float8_e4m3fn"
        )
        v_sf = FakeNpuTensor(
            "v-sf", (1, 16, 2, 8), dtype="float8_e4m3fn"
        )

        separate_output = wrapper.run(
            "q-separate", (key, value), kv_cache_sf=(k_sf, v_sf)
        )
        combined = FakeNpuTensor(
            "combined", (1, 2, 16, 2, 64), dtype="uint8"
        )
        combined_sf = FakeNpuTensor(
            "combined-sf", (1, 2, 16, 2, 8), dtype="float8_e4m3fn"
        )
        combined_output = wrapper.run(
            "q-combined", combined, kv_cache_sf=combined_sf
        )

        self.assertEqual(separate_output, "nvfp4-output:q-separate")
        self.assertEqual(combined_output, "nvfp4-output:q-combined")
        self.assertEqual(wrapper.plan_state.fingerprint, plan_fingerprint)
        self.assertIs(self.calls[0][3]["k_sf"], k_sf)
        self.assertIs(self.calls[0][3]["v_sf"], v_sf)
        self.assertIs(self.calls[1][1], combined)
        self.assertIsNone(self.calls[1][2])
        self.assertIs(self.calls[1][3]["kv_cache_sf"], combined_sf)
        self.assertEqual(self.values["loader"].resolve_calls, 1)
        self.assertEqual(self.values["authority"].calls, 1)

    def test_paged_prefill_combined_hnd_reaches_the_same_runtime_route(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )
        wrapper.plan(
            [0, 2],
            [0, 1],
            [0],
            [16],
            8,
            2,
            128,
            16,
            q_data_type="bfloat16",
            kv_data_type="uint8",
            o_data_type="bfloat16",
        )
        combined = FakeNpuTensor(
            "combined-hnd", (1, 2, 2, 16, 64), dtype="uint8"
        )
        combined_sf = FakeNpuTensor(
            "combined-hnd-sf", (1, 2, 2, 16, 8), dtype="float8_e4m3fn"
        )

        output = wrapper.run("q", combined, kv_cache_sf=combined_sf)

        self.assertEqual(output, "nvfp4-output:q")
        self.assertIs(self.calls[0][1], combined)
        self.assertIsNone(self.calls[0][2])
        self.assertIs(self.calls[0][3]["kv_cache_sf"], combined_sf)
        self.assertIsNone(self.calls[0][3]["k_sf"])
        self.assertIsNone(self.calls[0][3]["v_sf"])


if __name__ == "__main__":
    unittest.main()
