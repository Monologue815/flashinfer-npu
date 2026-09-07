import unittest

from flashinfer_npu.attention import (
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_043_provider_workspace_reset import (
    FakeNpuWorkspace, plan_wrapper, runtime_registry,
)


class IntegerDimension:
    def __int__(self):
        raise AssertionError("integer dimensions must use the index protocol")

    def __index__(self):
        return 16


class LossyDimension:
    def __int__(self):
        raise AssertionError("workspace dimensions must not use lossy int conversion")


class WorkspaceShapeAdmissionCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.components = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(self.components),
            operation_catalog=self.components["catalog"],
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def test_public_constructors_reject_non_integer_dimensions(self):
        for wrapper_type in (BatchDecodeWithPagedKVCacheWrapper,
                             BatchPrefillWithPagedKVCacheWrapper,
                             BatchPrefillWithRaggedKVCacheWrapper):
            for size in (True, 1.5, -0.5, "1", float("inf"), LossyDimension()):
                with self.subTest(wrapper=wrapper_type.__name__, size=size):
                    with self.assertRaises(SchemaError):
                        wrapper_type(FakeNpuWorkspace(size), backend="auto")
        self.assertEqual(self.components["events"], [])

    def test_invalid_reset_preserves_active_plan_and_workspace(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )
        plan_wrapper(wrapper)
        plan = wrapper.plan_state
        contract = wrapper.workspace_contract
        for size in (True, 1.5, -0.5, "1", float("inf"), LossyDimension()):
            for slot in (0, 1):
                with self.subTest(size=size, slot=slot):
                    buffers = [FakeNpuWorkspace(), FakeNpuWorkspace()]
                    buffers[slot] = FakeNpuWorkspace(size)
                    with self.assertRaises(SchemaError):
                        wrapper.reset_workspace_buffer(*buffers)
                    self.assertIs(wrapper.plan_state, plan)
                    self.assertIs(wrapper.workspace_contract, contract)
        self.assertEqual(wrapper.run("q", ("k", "v")), "package-output:q")

    def test_integer_index_dimensions_are_supported(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(IntegerDimension()), backend="auto"
        )
        self.assertEqual(wrapper.workspace_contract.float_capacity_bytes, 16)
        wrapper.reset_workspace_buffer(
            FakeNpuWorkspace(IntegerDimension()), FakeNpuWorkspace(IntegerDimension())
        )
        self.assertEqual(wrapper.workspace_contract.float_capacity_bytes, 16)
        self.assertEqual(wrapper.workspace_contract.int_capacity_bytes, 16)


if __name__ == "__main__":
    unittest.main()
