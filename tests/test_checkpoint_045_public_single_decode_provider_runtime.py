import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import single_decode_with_kv_cache
from flashinfer_npu.runtime import DispatchError, SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)


class FakeNpuTensor:
    def __init__(self, name, shape, *, dtype="bfloat16", device="npu:0"):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def __str__(self):
        return self.name


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def inputs():
    return (
        FakeNpuTensor("q", (8, 128)),
        FakeNpuTensor("k", (3, 2, 128)),
        FakeNpuTensor("v", (3, 2, 128)),
    )


class PublicSingleDecodeProviderRuntimeCheckpointTests(unittest.TestCase):
    """NPU tensors select an ephemeral single-decode provider runtime."""

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

    def test_public_call_plans_selects_and_executes_without_provider_handles(self):
        q, k, v = inputs()

        output = single_decode_with_kv_cache(q, k, v)
        output_with_lse = single_decode_with_kv_cache(
            q, k, v, return_lse=True
        )

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_with_lse,
            ("package-output:q", "package-lse:0.25"),
        )
        self.assertFalse(package_attention.calls[0][5])
        self.assertTrue(package_attention.calls[1][5])
        self.assertEqual(self.components["loader"].resolve_calls, 2)
        self.assertEqual(self.components["authority"].calls, 2)

    def test_unbound_semantics_fail_before_package_resolution(self):
        q, k, v = inputs()

        with self.assertRaisesRegex(NotImplementedError, "matrix-core"):
            single_decode_with_kv_cache(q, k, v, use_tensor_cores=True)
        with self.assertRaisesRegex(NotImplementedError, "Q/K/V scale"):
            single_decode_with_kv_cache(q, k, v, q_scale="scale")
        self.assertEqual(self.components["loader"].resolve_calls, 0)
        self.assertEqual(package_attention.calls, [])

    def test_tensor_contract_fails_before_provider_resolution(self):
        q, k, v = inputs()
        wrong_q = FakeNpuTensor("q", (8, 64))
        cpu_inputs = (
            FakeNpuTensor("q", (8, 128), device="cpu"),
            FakeNpuTensor("k", (3, 2, 128), device="cpu"),
            FakeNpuTensor("v", (3, 2, 128), device="cpu"),
        )

        with self.assertRaisesRegex(SchemaError, "head dimensions"):
            single_decode_with_kv_cache(wrong_q, k, v)
        with self.assertRaisesRegex(DispatchError, "npu"):
            single_decode_with_kv_cache(*cpu_inputs)
        self.assertEqual(self.components["loader"].resolve_calls, 0)

    def test_public_signature_exposes_no_runtime_or_plan_handle(self):
        parameters = inspect.signature(single_decode_with_kv_cache).parameters
        self.assertNotIn("provider", parameters)
        self.assertNotIn("runtime", parameters)
        self.assertNotIn("plan_handle", parameters)


if __name__ == "__main__":
    unittest.main()
