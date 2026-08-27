import unittest
from dataclasses import replace

from flashinfer_npu.attention import AttentionTensorAccessPolicy
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    metadata_tensor,
    quantized_input,
    query_input,
)


class ProviderQueryTensorContractTests(unittest.TestCase):
    """Every package-backed Attention route closes Q metadata before lowering."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_valid_query_is_inspected_and_forwarded_without_copy(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        query = query_input(plan)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        lowered = session.run(query, kv_input)

        self.assertIs(dict(lowered.positional_arguments)["query"], query)
        self.assertEqual(
            values["tensor_metadata_inspector"].calls[0],
            (query, "query", False),
        )
        self.assertEqual(package_attention.calls, [])

    def test_shape_dtype_and_device_must_match_the_active_plan(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        cases = (
            (
                metadata_tensor(
                    "bad-query-shape",
                    plan.expected_query_shape[:-1] + (plan.spec.head_dim_qk + 1,),
                    plan.spec.q_dtype,
                ),
                "query shape",
            ),
            (
                metadata_tensor(
                    "bad-query-dtype",
                    plan.expected_query_shape,
                    "float16",
                ),
                "query dtype",
            ),
            (
                metadata_tensor(
                    "bad-query-device",
                    plan.expected_query_shape,
                    plan.spec.q_dtype,
                    device="npu:1",
                ),
                "query device",
            ),
        )

        for query, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    session.run(query, kv_input)
        self.assertEqual(package_attention.calls, [])

    def test_non_contiguous_query_fails_before_kv_inspection(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        query = query_input(plan)
        non_contiguous = type(query)(
            replace(
                query.tensor_view,
                strides=(12, 6, 1),
                storage_nbytes=128,
            )
        )

        with self.assertRaisesRegex(SchemaError, "query must be contiguous"):
            session.run(
                non_contiguous,
                quantized_input(plan.spec.kv_quant_spec),
            )

        self.assertEqual(
            tuple(name for _, name, _ in values["tensor_metadata_inspector"].calls),
            ("query",),
        )
        self.assertEqual(package_attention.calls, [])

    def test_provider_alignment_policy_is_enforced_for_query(self):
        values = bootstrap_components()
        spec = replace(
            values["spec"],
            tensor_access_policy=AttentionTensorAccessPolicy(
                require_contiguous_q=True,
                required_alignment=16,
            ),
        )
        plan, session = active_session(values, spec=spec)
        query = query_input(plan)
        under_aligned = type(query)(
            replace(query.tensor_view, data_ptr_alignment=8)
        )

        with self.assertRaisesRegex(SchemaError, "query must be 16-byte aligned"):
            session.run(
                under_aligned,
                quantized_input(plan.spec.kv_quant_spec),
            )
        self.assertEqual(package_attention.calls, [])

    def test_bootstrap_requires_query_inspector_without_quantization(self):
        values = bootstrap_components()
        profile = values["spec"].profiles[0]
        unquantized_profile = replace(
            profile,
            rules=tuple(
                replace(rule, quant_specs=()) for rule in profile.rules
            ),
        )

        with self.assertRaisesRegex(
            SchemaError, "provider runtime requires a tensor metadata inspector"
        ):
            replace(
                values["spec"],
                profiles=(unquantized_profile,),
                quantization_bindings=(),
                tensor_metadata_inspector=None,
            )


if __name__ == "__main__":
    unittest.main()
