import unittest
from dataclasses import replace

from flashinfer_npu.attention import AttentionTensorAccessPolicy
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    quantized_input,
    query_input,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import (
    buffer_runtime,
    buffer_tensor,
)
from tests.test_checkpoint_057_provider_dense_kv_contract import (
    dense_kv,
    dense_runtime,
)
from tests.test_checkpoint_059_runtime_scale_tensor_access_contract import (
    lower_with_query_head_scale,
    scale_buffer_runtime,
)
from tests.test_checkpoint_061_provider_result_tensor_contract import (
    completion_fixture,
)
from tests.test_checkpoint_062_runtime_completion_publication import (
    strict_results,
    strict_runtime,
)


def run_with(runtime, query, kv_input):
    return runtime.run(query, kv_input, return_lse=True)


def result_aliasing(runtime, storage_id, suffix):
    plan = runtime.plan_state
    return (
        buffer_tensor(
            "result-alias-" + suffix,
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=storage_id,
        ),
        buffer_tensor(
            "result-lse-" + suffix,
            plan.expected_lse_shape,
            "float32",
        ),
    )


class ProviderResultInputAliasContractTests(unittest.TestCase):
    """Completed provider results cannot overlap validated run inputs."""

    def setUp(self):
        strict_results[:] = []

    def test_dense_lowering_carries_query_key_and_value_views(self):
        _, plan, session = dense_runtime()
        lowered = session.run(query_input(plan), dense_kv(plan))

        self.assertEqual(
            tuple(name for name, _ in lowered.validated_input_views),
            ("query", "kv.key_storage", "kv.value_storage"),
        )

    def test_quantized_lowering_carries_every_physical_kv_component(self):
        _, plan, session = buffer_runtime()
        lowered = session.run(
            query_input(plan), quantized_input(plan.spec.kv_quant_spec)
        )
        names = tuple(name for name, _ in lowered.validated_input_views)

        self.assertEqual(names[0], "query")
        self.assertIn("kv.key_storage", names)
        self.assertIn("kv.value_storage", names)
        self.assertIn("kv.key_scale", names)
        self.assertIn("kv.value_scale", names)

    def test_runtime_scale_tensor_is_part_of_completion_input_evidence(self):
        _, plan, session = scale_buffer_runtime()
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        query_head_scale = buffer_tensor(
            "query-head-scale",
            (plan.spec.num_qo_heads,),
            plan.spec.kv_quant_spec.scale_dtype,
        )

        lowered = lower_with_query_head_scale(
            session, plan, kv_input, query_head_scale
        )

        self.assertIn(
            "run.q_head_scale",
            tuple(name for name, _ in lowered.validated_input_views),
        )

    def test_provider_output_cannot_alias_query_or_quantized_scale(self):
        _, runtime = strict_runtime()
        plan = runtime.plan_state
        query = query_input(plan)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        cases = (
            (query.tensor_view.storage_id, "query", "query"),
            (
                kv_input.key_scale.tensor_view.storage_id,
                "key-scale",
                "kv.key_scale",
            ),
        )
        for storage_id, suffix, expected_name in cases:
            with self.subTest(expected_name=expected_name):
                strict_results.append(
                    result_aliasing(runtime, storage_id, suffix)
                )
                with self.assertRaisesRegex(
                    SchemaError, "output result cannot alias %s" % expected_name
                ):
                    run_with(runtime, query, kv_input)

    def test_explicit_alias_policy_is_shared_by_run_and_completion(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_q=True,
            permit_output_input_alias=True,
        )
        _, runtime = strict_runtime(access_policy=policy)
        plan = runtime.plan_state
        query = query_input(plan)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        expected = result_aliasing(
            runtime, query.tensor_view.storage_id, "permitted"
        )
        strict_results.append(expected)

        self.assertIs(run_with(runtime, query, kv_input), expected)
        self.assertEqual(
            runtime.last_completion_receipt.access_policy_fingerprint,
            policy.fingerprint,
        )

    def test_completion_requires_outer_validated_query_evidence(self):
        _, plan, _, lowered, validator = completion_fixture()
        result = (
            buffer_tensor("output", plan.expected_output_shape, plan.spec.o_dtype),
            buffer_tensor("lse", plan.expected_lse_shape, "float32"),
        )

        with self.assertRaisesRegex(SchemaError, "no validated query"):
            validator.validate(
                replace(lowered, validated_input_views=()),
                result,
            )

    def test_receipt_freezes_input_and_result_view_fingerprints(self):
        _, runtime = strict_runtime()
        plan = runtime.plan_state
        query = query_input(plan)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        strict_results.append(
            (
                buffer_tensor(
                    "output", plan.expected_output_shape, plan.spec.o_dtype
                ),
                buffer_tensor("lse", plan.expected_lse_shape, "float32"),
            )
        )

        run_with(runtime, query, kv_input)
        receipt = runtime.last_completion_receipt
        input_names = tuple(name for name, _ in receipt.input_view_fingerprints)

        self.assertEqual(input_names[0], "query")
        self.assertIn("kv.key_storage", input_names)
        self.assertEqual(len(receipt.result_view_fingerprints), 2)


if __name__ == "__main__":
    unittest.main()
