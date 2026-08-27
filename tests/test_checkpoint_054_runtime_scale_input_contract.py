import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorQuantArgumentBinding,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_public_quantized_paged_decode import (
    plan_public_wrapper as plan_decode_wrapper,
    public_paged_decode_runtime,
)
from tests.test_public_quantized_paged_prefill import (
    FakeNpuWorkspace,
    metadata_tensor,
    plan_public_wrapper as plan_prefill_wrapper,
    public_prefill_runtime,
    query_tensor,
    quantized_kv_input,
)


class RuntimeScaleBindingSchemaTests(unittest.TestCase):
    """Provider metadata must describe every accepted scale representation."""

    def test_argument_policy_requires_known_input_kinds(self):
        values = bootstrap_components()
        original = values["spec"].quantization_bindings[0]
        arguments = original.argument_bindings + (
            AttentionOperatorQuantArgumentBinding(
                "run.q_scale", "runtime_query_scale"
            ),
        )

        with self.assertRaisesRegex(SchemaError, "requires input kinds"):
            replace(
                original,
                argument_bindings=arguments,
                runtime_q_scale_policy="argument",
            )
        with self.assertRaisesRegex(SchemaError, "input kinds are invalid"):
            replace(
                original,
                argument_bindings=arguments,
                runtime_q_scale_policy="argument",
                runtime_q_scale_input_kinds=("opaque",),
            )

        binding = replace(
            original,
            argument_bindings=arguments,
            runtime_q_scale_policy="argument",
            runtime_q_scale_input_kinds=("head_tensor", "scalar"),
        )
        self.assertEqual(
            binding.runtime_q_scale_input_kinds,
            ("head_tensor", "scalar"),
        )
        self.assertEqual(
            binding.to_dict()["runtime_q_scale_input_kinds"],
            ["head_tensor", "scalar"],
        )

    def test_paged_prefill_provider_must_cover_scalar_and_head_tensor(self):
        with self.assertRaisesRegex(
            SchemaError, "input kinds.*head_tensor"
        ):
            public_prefill_runtime(runtime_q_scale_input_kinds=("scalar",))


class RuntimeScaleLoweringTests(unittest.TestCase):
    """Scale kinds fail closed before an external package callable is entered."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_prefill_runtime()
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=self.values["catalog"]
        )
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def prefill_wrapper(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=self.case.trace.spec.kv_layout.value,
            backend="auto",
        )
        plan_prefill_wrapper(wrapper, self.case)
        return wrapper

    def test_paged_prefill_accepts_scalar_and_exact_per_head_tensor(self):
        wrapper = self.prefill_wrapper()
        kv_input = quantized_kv_input(self.case.trace.spec.kv_quant_spec)
        head_scale = metadata_tensor(
            "checkpoint-054-head-scale",
            (self.case.trace.spec.num_qo_heads,),
            self.case.trace.spec.kv_quant_spec.scale_dtype,
        )

        wrapper.run(
            query_tensor(self.case, "q-scalar"), kv_input, q_scale=2.0
        )
        wrapper.run(
            query_tensor(self.case, "q-tensor"), kv_input, q_scale=head_scale
        )

        self.assertEqual(package_attention.calls[0][10], 2.0)
        self.assertIs(package_attention.calls[1][10], head_scale)

    def test_invalid_per_head_metadata_and_non_scalars_fail_closed(self):
        wrapper = self.prefill_wrapper()
        spec = self.case.trace.spec
        kv_input = quantized_kv_input(spec.kv_quant_spec)
        cases = (
            (float("nan"), "scalar must be finite"),
            (object(), "finite scalar or a per-head tensor"),
            (
                metadata_tensor(
                    "bad-shape", (spec.num_qo_heads + 1,), "float32"
                ),
                "head tensor shape",
            ),
            (
                metadata_tensor(
                    "bad-dtype", (spec.num_qo_heads,), "float16"
                ),
                "head tensor dtype",
            ),
            (
                metadata_tensor(
                    "bad-device",
                    (spec.num_qo_heads,),
                    "float32",
                    device="npu:1",
                ),
                "head tensor device",
            ),
            (
                metadata_tensor(
                    "non-contiguous",
                    (spec.num_qo_heads,),
                    "float32",
                    strides=(2,),
                ),
                "must be contiguous",
            ),
        )
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    wrapper.run(
                        query_tensor(self.case, "q-invalid"),
                        kv_input,
                        q_scale=value,
                    )
        self.assertEqual(package_attention.calls, [])

    def test_decode_public_contract_stays_scalar_even_if_provider_is_broader(self):
        values, case, registry = public_paged_decode_runtime(
            runtime_scale_input_kinds=("scalar", "head_tensor")
        )
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=values["catalog"]
        )
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=case.trace.spec.kv_layout.value,
            backend="auto",
        )
        plan_decode_wrapper(wrapper, case)
        shape = case.trace.kv_data.key_data.logical_shape
        cache = (
            metadata_tensor("decode-key", shape, "float8_e4m3fn"),
            metadata_tensor("decode-value", shape, "float8_e4m3fn"),
        )
        head_scale = metadata_tensor(
            "decode-head-scale",
            (case.trace.spec.num_qo_heads,),
            "float32",
        )

        with self.assertRaisesRegex(
            SchemaError, "not accepted by the public Attention mode"
        ):
            wrapper.run(
                metadata_tensor(
                    "q",
                    (
                        case.trace.metadata.batch_size
                        * case.trace.spec.q_len_per_req,
                        case.trace.spec.num_qo_heads,
                        case.trace.spec.head_dim_qk,
                    ),
                    case.trace.spec.q_dtype,
                ),
                cache,
                q_scale=head_scale,
            )
        self.assertEqual(package_attention.calls, [])


if __name__ == "__main__":
    unittest.main()
