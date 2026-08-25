import unittest

from flashinfer_npu.attention import (
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)


class FakeNpuWorkspace:
    def __init__(self, size=1024, *, dtype="uint8", device="npu:0"):
        self.shape = (size,)
        self.dtype = dtype
        self.device = device


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def plan_wrapper(wrapper):
    wrapper.plan(
        [0, 1],
        [7],
        [64],
        8,
        2,
        128,
        128,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        o_data_type="bfloat16",
    )


class ProviderWorkspaceResetCheckpointTests(unittest.TestCase):
    """Package-managed providers retain plans across workspace replacement."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.components = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(self.components),
            operation_catalog=self.components["catalog"],
        )
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

    def test_reset_after_plan_updates_binding_and_preserves_active_plan(self):
        wrapper = self.wrapper()
        plan_wrapper(wrapper)
        plan = wrapper.plan_state
        initial = wrapper.workspace_contract

        wrapper.reset_workspace_buffer(
            FakeNpuWorkspace(2048), FakeNpuWorkspace(512)
        )

        rebound = wrapper.workspace_contract
        self.assertIs(wrapper.plan_state, plan)
        self.assertEqual(rebound.binding_generation, initial.binding_generation + 1)
        self.assertEqual(rebound.plan_generation, plan.generation)
        self.assertEqual(rebound.required_sizes, (0, 0))
        self.assertEqual(rebound.float_capacity_bytes, 2048)
        self.assertEqual(rebound.int_capacity_bytes, 512)
        self.assertIs(
            rebound,
            wrapper._operator_runtime.workspace_contract,
        )
        self.assertEqual(
            wrapper.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_reset_before_plan_is_bound_by_the_selected_operation(self):
        wrapper = self.wrapper()
        wrapper.reset_workspace_buffer(
            FakeNpuWorkspace(256), FakeNpuWorkspace(128)
        )
        preplan = wrapper.workspace_contract
        self.assertFalse(preplan.requirements_known)

        plan_wrapper(wrapper)

        planned = wrapper.workspace_contract
        self.assertEqual(planned.binding_generation, preplan.binding_generation)
        self.assertEqual(planned.required_sizes, (0, 0))
        self.assertEqual(planned.float_capacity_bytes, 256)
        self.assertEqual(planned.int_capacity_bytes, 128)

    def test_reset_rejects_alias_dtype_and_device_drift(self):
        wrapper = self.wrapper()
        shared = FakeNpuWorkspace(64)
        with self.assertRaisesRegex(SchemaError, "cannot alias"):
            wrapper.reset_workspace_buffer(shared, shared)
        with self.assertRaisesRegex(SchemaError, "dtype must be uint8"):
            wrapper.reset_workspace_buffer(
                FakeNpuWorkspace(dtype="float16"), FakeNpuWorkspace(32)
            )
        with self.assertRaisesRegex(SchemaError, "same device"):
            wrapper.reset_workspace_buffer(
                FakeNpuWorkspace(device="npu:0"),
                FakeNpuWorkspace(32, device="npu:1"),
            )
        with self.assertRaisesRegex(SchemaError, "device cannot change"):
            wrapper.reset_workspace_buffer(
                FakeNpuWorkspace(device="npu:1"),
                FakeNpuWorkspace(32, device="npu:1"),
            )


if __name__ == "__main__":
    unittest.main()
