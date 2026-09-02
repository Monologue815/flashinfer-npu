import unittest

from flashinfer_npu.attention import (
    AttentionStateError,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from tests.test_checkpoint_102_public_nvfp4_canonicalization import (
    FakeNpuTensor,
    FakeNpuWorkspace,
    explicit_uint8_spec,
    nvfp4_runtime,
)


class Nvfp4WorkspaceQueryCheckpoint(unittest.TestCase):
    """NVFP4 size queries use a canonical, non-publishing provider plan."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        (
            self.values,
            self.quant_spec,
            self.catalog,
            self.registry,
            self.calls,
        ) = nvfp4_runtime()
        self.observed_plans = []
        original_rejection_reasons = self.values["gate"].rejection_reasons

        def record_plan(plan, device):
            self.observed_plans.append((plan, device))
            return original_rejection_reasons(plan, device)

        self.values["gate"].rejection_reasons = record_plan
        install_attention_operator_runtime_resolvers(
            self.registry, operation_catalog=self.catalog
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def decode_wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="NHD", backend="auto"
        )

    def query_decode(self, wrapper, *, kv_data_type="uint8", head_dim=128):
        return wrapper.workspace_size(
            [0, 1],
            [0],
            [16],
            8,
            2,
            head_dim,
            16,
            q_data_type="bfloat16",
            kv_data_type=kv_data_type,
            o_data_type="bfloat16",
        )

    def assert_one_observed_plan_identity(self):
        self.assertGreaterEqual(len(self.observed_plans), 1)
        fingerprints = {
            plan.fingerprint for plan, _device in self.observed_plans
        }
        self.assertEqual(len(fingerprints), 1)
        return self.observed_plans[0]

    def test_decode_query_canonicalizes_uint8_without_publishing_plan(self):
        wrapper = self.decode_wrapper()
        initial_workspace = wrapper.workspace_contract

        size = self.query_decode(wrapper)

        self.assertEqual(size, (0, 0))
        observed, device = self.assert_one_observed_plan_identity()
        self.assertEqual(device, "npu:0")
        self.assertEqual(observed.spec.kv_dtype, "uint8")
        self.assertEqual(observed.spec.kv_quant_spec, self.quant_spec)
        self.assertIs(wrapper.workspace_contract, initial_workspace)
        self.assertFalse(wrapper._operator_runtime.is_planned)
        with self.assertRaisesRegex(AttentionStateError, "plan"):
            _ = wrapper.plan_state
        self.assertEqual(self.calls, [])

    def test_decode_query_preserves_an_existing_nvfp4_plan_and_executor(self):
        wrapper = self.decode_wrapper()
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
        original_plan = wrapper.plan_state
        original_selection = wrapper.plan_selection
        original_workspace = wrapper.workspace_contract
        original_session = wrapper._operator_runtime.operator_session
        original_executor = wrapper._operator_runtime._executor
        self.observed_plans.clear()

        self.assertEqual(self.query_decode(wrapper, head_dim=256), (0, 0))

        observed, _device = self.assert_one_observed_plan_identity()
        self.assertEqual(
            observed.spec.kv_quant_spec, self.quant_spec
        )
        self.assertIs(wrapper.plan_state, original_plan)
        self.assertEqual(wrapper.plan_selection, original_selection)
        self.assertIs(wrapper.workspace_contract, original_workspace)
        self.assertIs(wrapper._operator_runtime.operator_session, original_session)
        self.assertIs(wrapper._operator_runtime._executor, original_executor)

        combined = FakeNpuTensor(
            "combined", (1, 2, 16, 2, 64), dtype="uint8"
        )
        combined_sf = FakeNpuTensor(
            "combined-sf", (1, 2, 16, 2, 8), dtype="float8_e4m3fn"
        )
        self.assertEqual(
            wrapper.run("q", combined, kv_cache_sf=combined_sf),
            "nvfp4-output:q",
        )

    def test_paged_prefill_query_uses_the_same_nvfp4_contract(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

        size = wrapper.workspace_size(
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

        self.assertEqual(size, (0, 0))
        observed, _device = self.assert_one_observed_plan_identity()
        self.assertEqual(
            observed.spec.kv_quant_spec, self.quant_spec
        )
        self.assertFalse(wrapper._operator_runtime.is_planned)
        self.assertEqual(self.calls, [])

    def test_explicit_uint8_quantspec_is_not_reclassified_by_query(self):
        wrapper = self.decode_wrapper()
        explicit = explicit_uint8_spec()

        self.assertEqual(
            self.query_decode(wrapper, kv_data_type=explicit), (0, 0)
        )

        observed, _device = self.assert_one_observed_plan_identity()
        self.assertIs(observed.spec.kv_quant_spec, explicit)
        self.assertNotEqual(explicit.fingerprint, self.quant_spec.fingerprint)
        self.assertFalse(wrapper._operator_runtime.is_planned)


if __name__ == "__main__":
    unittest.main()
