import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.batch import HostBatchReferenceWrapper
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)


class FakeNpuWorkspace:
    shape = (1024,)
    dtype = "uint8"
    device = "npu:0"


class UnconnectedBatchWrapper(HostBatchReferenceWrapper):
    def __init__(self):
        self._init_host_wrapper(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
            float_workspace_buffer=FakeNpuWorkspace(),
            backend="auto",
            graph_enabled=False,
            fixed_batch_size=None,
        )


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def plan_public_wrapper(wrapper, **kwargs):
    return wrapper.plan(
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
        **kwargs,
    )


class PublicPagedDecodeProviderRuntimeCheckpointTests(unittest.TestCase):
    """The paged-decode facade owns provider selection and plan/run state."""

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
            FakeNpuWorkspace(),
            kv_layout="HND",
            backend="auto",
        )

    def test_public_plan_selects_provider_and_run_preserves_return_shape(self):
        wrapper = self.wrapper()

        self.assertIsNone(plan_public_wrapper(wrapper))
        output = wrapper.run("q", ("k-cache", "v-cache"))
        output_with_lse = wrapper.run(
            "q-lse", ("k-cache", "v-cache"), return_lse=True
        )

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_with_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        self.assertEqual(wrapper.plan_state.spec.mode, AttentionMode.BATCH_DECODE_PAGED)
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
        self.assertEqual(
            wrapper.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_unbound_options_fail_before_provider_execution(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper)

        with self.assertRaisesRegex(NotImplementedError, "query-scale"):
            wrapper.run("q", ("k", "v"), q_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "attention-sink"):
            wrapper.run("q", ("k", "v"), sinks="sink")
        with self.assertRaisesRegex(NotImplementedError, "skip-softmax"):
            wrapper.run(
                "q",
                ("k", "v"),
                skip_softmax_threshold_scale_factor=1.0,
            )
        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            plan_public_wrapper(wrapper, fixed_split_size=1)
        self.assertEqual(package_attention.calls, [])

    def test_provider_resource_controls_fail_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "graph resources"):
            BatchDecodeWithPagedKVCacheWrapper(
                FakeNpuWorkspace(),
                use_cuda_graph=True,
                backend="auto",
            )
        with self.assertRaisesRegex(NotImplementedError, "matrix-core"):
            BatchDecodeWithPagedKVCacheWrapper(
                FakeNpuWorkspace(),
                use_tensor_cores=True,
                backend="auto",
            )
        wrapper = self.wrapper()
        with self.assertRaisesRegex(NotImplementedError, "workspace_size"):
            wrapper.workspace_size([0, 1], [7], [64], 8, 2, 128, 128)

    def test_shared_base_never_enables_provider_resolution_implicitly(self):
        with self.assertRaisesRegex(
            NotImplementedError, "batch_prefill_ragged public wrapper"
        ):
            UnconnectedBatchWrapper()
        self.assertEqual(self.components["loader"].resolve_calls, 0)
        self.assertEqual(self.components["authority"].calls, 0)

    def test_public_signature_exposes_no_provider_runtime_handle(self):
        for callable_value in (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchDecodeWithPagedKVCacheWrapper.plan,
            BatchDecodeWithPagedKVCacheWrapper.run,
        ):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("provider", parameters)
            self.assertNotIn("runtime", parameters)
            self.assertNotIn("plan_handle", parameters)


if __name__ == "__main__":
    unittest.main()
