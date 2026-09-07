"""Keep unsupported mask payloads out of the provider path until binding exists."""

import unittest

from flashinfer_npu.attention import (
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
)
from tests.test_checkpoint_019_package_runtime_integration import build_components, package_attention
from tests.test_checkpoint_043_provider_workspace_reset import FakeNpuWorkspace, runtime_registry
from tests.test_checkpoint_102_public_nvfp4_canonicalization import FakeNpuTensor


class UnreadableMask:
    def __getattr__(self, name):
        raise AssertionError("unsupported mask payload must not be inspected")


class CustomMaskProviderBoundaryCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(self.values), operation_catalog=self.values["catalog"])
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry, operation_catalog=self.original.operation_catalog)

    def test_masked_replan_preserves_paged_and_ragged_provider_state(self):
        for wrapper_type, args in (
            (BatchPrefillWithPagedKVCacheWrapper,
             ([0, 1], [0, 1], [0], [64], 8, 2, 128, 128)),
            (BatchPrefillWithRaggedKVCacheWrapper, ([0, 1], [0, 64], 8, 2, 128)),
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = wrapper_type(FakeNpuWorkspace(), kv_layout="NHD")
                wrapper.plan(*args, q_data_type="bfloat16")
                active = wrapper.plan_state
                workspace = wrapper.workspace_contract
                selection = wrapper.plan_selection
                events = list(self.values["events"])
                for keywords in ({"custom_mask": UnreadableMask()},
                                 {"packed_custom_mask": UnreadableMask()},
                                 {"custom_mask": UnreadableMask(),
                                  "packed_custom_mask": UnreadableMask()}):
                    with self.assertRaisesRegex(NotImplementedError, "custom-mask.*not implemented"):
                        wrapper.plan(*args, q_data_type="bfloat16", **keywords)
                    self.assertIs(wrapper.plan_state, active)
                    self.assertIs(wrapper.workspace_contract, workspace)
                    self.assertEqual(wrapper.plan_selection, selection)
                    self.assertEqual(self.values["events"], events)
                if wrapper_type is BatchPrefillWithPagedKVCacheWrapper:
                    result = wrapper.run("q", ("k", "v"))
                else:
                    result = wrapper.run("q", "k", "v")
                self.assertEqual(result, "package-output:q")
        self.assertEqual(len(package_attention.calls), 2)

    def test_single_prefill_rejects_masks_before_any_provider_probe(self):
        q = FakeNpuTensor("q", (1, 8, 128), dtype="bfloat16")
        k = FakeNpuTensor("k", (64, 2, 128), dtype="bfloat16")
        v = FakeNpuTensor("v", (64, 2, 128), dtype="bfloat16")
        for name in ("custom_mask", "packed_custom_mask"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(NotImplementedError, "custom-mask.*not implemented"):
                    single_prefill_with_kv_cache(q, k, v, **{name: UnreadableMask()})
        self.assertEqual(self.values["events"], [])
        self.assertEqual(package_attention.calls, [])
