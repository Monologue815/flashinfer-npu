import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.prefill import BatchPrefillWithRaggedKVCacheWrapper
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)


class FakeNpuWorkspace:
    shape = (1024,)
    dtype = "uint8"
    device = "npu:0"


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def plan_public_wrapper(wrapper, **kwargs):
    return wrapper.plan(
        [0, 2],
        [0, 2],
        8,
        2,
        128,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        o_data_type="bfloat16",
        **kwargs,
    )


class PublicRaggedPrefillProviderRuntimeCheckpointTests(unittest.TestCase):
    """The ragged-prefill facade owns provider selection and plan/run state."""

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
        return BatchPrefillWithRaggedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="HND",
            backend="auto",
        )

    def test_public_plan_selects_provider_and_run_preserves_return_shape(self):
        wrapper = self.wrapper()

        self.assertIsNone(plan_public_wrapper(wrapper))
        output = wrapper.run("q", "k", "v")
        output_with_lse = wrapper.run("q-lse", "k", "v", return_lse=True)

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_with_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        self.assertEqual(
            wrapper.plan_state.spec.mode,
            AttentionMode.BATCH_PREFILL_RAGGED,
        )
        self.assertEqual(wrapper.workspace_contract.required_sizes, (0, 0))
        self.assertEqual(
            wrapper.workspace_contract.plan_generation,
            wrapper.plan_state.generation,
        )
        self.assertEqual(self.components["loader"].resolve_calls, 1)
        self.assertEqual(self.components["authority"].calls, 1)
        self.assertEqual(len(package_attention.calls), 2)

    def test_wrapper_freezes_registry_and_catalog_at_construction(self):
        wrapper = self.wrapper()
        install_attention_operator_runtime_resolvers(
            AttentionOperatorRuntimeResolverRegistry()
        )

        plan_public_wrapper(wrapper)
        self.assertEqual(wrapper.run("q", "k", "v"), "package-output:q")

    def test_unbound_options_fail_before_provider_execution(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper)

        with self.assertRaisesRegex(NotImplementedError, "query-scale"):
            wrapper.run("q", "k", "v", q_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "o_scale"):
            wrapper.run("q", "k", "v", o_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            plan_public_wrapper(wrapper, fixed_split_size=1)
        with self.assertRaisesRegex(NotImplementedError, "custom-mask"):
            plan_public_wrapper(wrapper, custom_mask=[True, True, True, True])
        self.assertEqual(package_attention.calls, [])

    def test_provider_graph_resources_fail_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "graph resources"):
            BatchPrefillWithRaggedKVCacheWrapper(
                FakeNpuWorkspace(),
                use_cuda_graph=True,
                backend="auto",
            )

    def test_public_signature_exposes_no_provider_runtime_handle(self):
        for callable_value in (
            BatchPrefillWithRaggedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper.plan,
            BatchPrefillWithRaggedKVCacheWrapper.run,
        ):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("provider", parameters)
            self.assertNotIn("runtime", parameters)
            self.assertNotIn("plan_handle", parameters)


if __name__ == "__main__":
    unittest.main()
