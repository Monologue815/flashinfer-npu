import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorResourceBinding,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionWorkspaceContract,
    attention_operator_runtime_registry_snapshot,
    bind_attention_operator_resources,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)


class FakeNpuWorkspace:
    shape = (1024,)
    dtype = "uint8"
    device = "npu:0"


class FakeQuery:
    device = "npu:1"


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def plan_wrapper(wrapper):
    wrapper.plan(
        [0, 2],
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


class ProviderResourceBindingCheckpointTests(unittest.TestCase):
    """Provider workspace and returned tensors are plan-bound resources."""

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
        return BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

    def test_public_plan_binds_package_managed_workspace_to_generation(self):
        wrapper = self.wrapper()
        plan_wrapper(wrapper)

        contract = wrapper.workspace_contract
        self.assertTrue(contract.requirements_known)
        self.assertEqual(contract.required_sizes, (0, 0))
        self.assertEqual(contract.plan_generation, wrapper.plan_state.generation)
        resource = wrapper._operator_runtime.resource_binding
        self.assertIsInstance(resource, AttentionOperatorResourceBinding)
        self.assertEqual(resource.workspace_ownership, "package_managed")
        self.assertEqual(resource.output_binding, "returned")
        self.assertEqual(resource.lse_binding, "returned")

    def test_caller_owned_output_buffers_fail_before_external_execution(self):
        wrapper = self.wrapper()
        plan_wrapper(wrapper)

        with self.assertRaisesRegex(NotImplementedError, "output-buffer"):
            wrapper.run("q", ("k", "v"), out="out-buffer")
        with self.assertRaisesRegex(NotImplementedError, "LSE-buffer"):
            wrapper.run("q", ("k", "v"), lse="lse-buffer")
        self.assertEqual(package_attention.calls, [])

    def test_query_and_bound_workspace_must_share_device(self):
        wrapper = self.wrapper()
        plan_wrapper(wrapper)

        with self.assertRaisesRegex(SchemaError, "same device"):
            wrapper.run(FakeQuery(), ("k", "v"))
        self.assertEqual(package_attention.calls, [])

    def test_caller_managed_requirement_checks_capacity_atomically(self):
        active = self.wrapper()
        plan_wrapper(active)
        returned = active._operator_runtime.resource_binding
        caller_managed = AttentionOperatorResourceBinding(
            active_plan_fingerprint=returned.active_plan_fingerprint,
            operation_fingerprint=returned.operation_fingerprint,
            provider_id=returned.provider_id,
            operation_id=returned.operation_id,
            workspace_ownership="caller_managed",
            required_float_workspace_bytes=2048,
            required_int_workspace_bytes=16,
            output_binding="returned",
            lse_binding="returned",
        )
        contract = AttentionWorkspaceContract(
            backend="auto",
            device="npu:0",
            float_capacity_bytes=1024,
            int_capacity_bytes=0,
        )

        with self.assertRaisesRegex(SchemaError, "capacity"):
            caller_managed.bind_workspace_contract(contract, plan_generation=1)

    def test_workspace_or_mutable_output_arguments_require_explicit_binders(self):
        wrapper = self.wrapper()
        plan_wrapper(wrapper)
        session = wrapper._operator_runtime.operator_session
        operation = self.components["operation"]

        workspace_operation = replace(
            operation,
            keyword_arguments=operation.keyword_arguments + ("workspace",),
        )
        workspace_binding = replace(
            session.operation_binding,
            operation_fingerprint=workspace_operation.fingerprint,
        )
        with self.assertRaisesRegex(SchemaError, "explicit resource binder"):
            bind_attention_operator_resources(
                workspace_operation,
                session.active_plan,
                workspace_binding,
            )

        output_operation = replace(
            operation,
            keyword_arguments=operation.keyword_arguments + ("out",),
            mutable_arguments=("out",),
        )
        output_binding = replace(
            session.operation_binding,
            operation_fingerprint=output_operation.fingerprint,
        )
        with self.assertRaisesRegex(SchemaError, "explicit resource binder"):
            bind_attention_operator_resources(
                output_operation,
                session.active_plan,
                output_binding,
            )


if __name__ == "__main__":
    unittest.main()
