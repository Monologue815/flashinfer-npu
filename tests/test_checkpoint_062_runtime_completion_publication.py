import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorPackageRuntimeSpec,
    AttentionOperatorCompletionValidatorFactory,
    AttentionOperatorRuntime,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
    build_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import FakePackageLoader
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    quantized_input,
    query_input,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor


strict_results = []
strict_calls = []


def strict_package_attention(
    query,
    key,
    value,
    *,
    table=None,
    scale=1.0,
    return_softmax_lse=False,
    key_scale=None,
    value_scale=None,
    runtime_key_scale=None,
    runtime_value_scale=None,
    runtime_query_scale=None,
    runtime_output_scale=None,
    runtime_query_head_scale=None,
    runtime_key_head_scale=None,
    runtime_value_head_scale=None,
):
    strict_calls.append((query, key, value, return_softmax_lse))
    return strict_results.pop(0)


class StrictResultPackageLoader(FakePackageLoader):
    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append("resolve_callable")
        return strict_package_attention


def strict_runtime():
    values = bootstrap_components()
    values["loader"] = StrictResultPackageLoader(values["events"])
    values["spec"] = replace(
        values["spec"], validate_provider_results=True
    )
    registry = build_attention_operator_runtime_resolvers(
        (values["spec"],),
        operation_catalog=values["catalog"],
        package_loader=values["loader"],
    )
    plan = group_plan()
    runtime = AttentionOperatorRuntime(
        "npu:0",
        registry,
        values["catalog"],
        mode=plan.spec.mode,
    )
    runtime.plan(plan.spec, plan.metadata)
    return values, runtime


def valid_result(runtime, suffix=""):
    plan = runtime.plan_state
    return (
        buffer_tensor(
            "runtime-output" + suffix,
            plan.expected_output_shape,
            plan.spec.o_dtype,
        ),
        buffer_tensor(
            "runtime-lse" + suffix,
            plan.expected_lse_shape,
            "float32",
        ),
    )


def run_runtime(runtime):
    plan = runtime.plan_state
    return runtime.run(
        query_input(plan),
        quantized_input(plan.spec.kv_quant_spec),
        return_lse=True,
    )


class RuntimeCompletionPublicationTests(unittest.TestCase):
    """Completion validation runs after provider execution and before publication."""

    def setUp(self):
        strict_results[:] = []
        strict_calls[:] = []

    def test_bootstrap_default_is_strict_and_binds_active_plan_validator(self):
        field = AttentionOperatorPackageRuntimeSpec.__dataclass_fields__[
            "validate_provider_results"
        ]
        self.assertIs(field.default, True)

        _, runtime = strict_runtime()
        self.assertEqual(
            runtime.completion_validator.provider_id,
            runtime.operator_session.active_plan.provider_selection.provider_id,
        )
        self.assertEqual(
            runtime.completion_validator.operation_id,
            runtime.operator_session.active_plan.prepared_plan.implementation_id,
        )

    def test_resolved_runtime_rejects_completion_operation_fingerprint_drift(self):
        values = bootstrap_components()
        loader = StrictResultPackageLoader(values["events"])
        spec = replace(values["spec"], validate_provider_results=True)
        registry = build_attention_operator_runtime_resolvers(
            (spec,),
            operation_catalog=values["catalog"],
            package_loader=loader,
        )
        resolved = registry.resolve(group_plan(), "npu:0")
        drifted_operation = replace(
            values["operation"],
            source_url="https://example.com/drifted-completion-operation",
        )
        drifted_factory = AttentionOperatorCompletionValidatorFactory(
            drifted_operation,
            values["tensor_metadata_inspector"],
            spec.tensor_access_policy,
        )

        with self.assertRaisesRegex(SchemaError, "validator identities differ"):
            replace(
                resolved,
                completion_validator_factory=drifted_factory,
            )

    def test_validated_result_is_the_exact_public_result_and_records_receipt(self):
        _, runtime = strict_runtime()
        expected = valid_result(runtime)
        strict_results.append(expected)

        result = run_runtime(runtime)

        self.assertIs(result, expected)
        receipt = runtime.last_completion_receipt
        self.assertEqual(
            receipt.active_plan_fingerprint,
            runtime.operator_session.active_plan.fingerprint,
        )
        self.assertEqual(receipt.return_names, ("output", "softmax_lse"))
        self.assertEqual(len(strict_calls), 1)

    def test_invalid_provider_result_never_crosses_public_boundary(self):
        _, runtime = strict_runtime()
        plan = runtime.plan_state
        strict_results.append(
            (
                buffer_tensor(
                    "wrong-shape",
                    plan.expected_output_shape[:-1]
                    + (plan.expected_output_shape[-1] + 1,),
                    plan.spec.o_dtype,
                ),
                buffer_tensor("lse", plan.expected_lse_shape, "float32"),
            )
        )

        with self.assertRaisesRegex(SchemaError, "output result shape"):
            run_runtime(runtime)
        self.assertEqual(len(strict_calls), 1)
        with self.assertRaisesRegex(AttentionStateError, "no successful"):
            _ = runtime.last_completion_receipt
        with self.assertRaisesRegex(AttentionStateError, "has not been run"):
            _ = runtime.last_lowered_call

    def test_failed_second_completion_clears_previous_success_receipt(self):
        _, runtime = strict_runtime()
        strict_results.append(valid_result(runtime, "-first"))
        run_runtime(runtime)
        first = runtime.last_completion_receipt

        strict_results.append((None, None))
        with self.assertRaisesRegex(SchemaError, "returned no value"):
            run_runtime(runtime)
        with self.assertRaisesRegex(AttentionStateError, "no successful"):
            _ = runtime.last_completion_receipt
        self.assertEqual(first.framework_plan_fingerprint, runtime.plan_state.fingerprint)

    def test_replan_rebinds_completion_to_new_active_generation(self):
        _, runtime = strict_runtime()
        first_validator = runtime.completion_validator
        first_active = runtime.operator_session.active_plan.fingerprint
        plan = runtime.plan_state

        runtime.plan(plan.spec, plan.metadata)

        self.assertIsNot(runtime.completion_validator, first_validator)
        self.assertNotEqual(
            runtime.operator_session.active_plan.fingerprint, first_active
        )
        with self.assertRaisesRegex(AttentionStateError, "no successful"):
            _ = runtime.last_completion_receipt
        strict_results.append(valid_result(runtime, "-replan"))
        run_runtime(runtime)
        self.assertEqual(
            runtime.last_completion_receipt.active_plan_fingerprint,
            runtime.operator_session.active_plan.fingerprint,
        )

    def test_explicit_synthetic_bypass_publishes_no_completion_receipt(self):
        values = bootstrap_components()
        self.assertFalse(values["spec"].validate_provider_results)
        registry = AttentionOperatorRuntimeResolverRegistry(
            build_attention_operator_runtime_resolvers(
                (values["spec"],),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            ).resolvers
        )
        plan = group_plan()
        runtime = AttentionOperatorRuntime(
            "npu:0", registry, values["catalog"], mode=plan.spec.mode
        )
        runtime.plan(plan.spec, plan.metadata)
        result = run_runtime(runtime)

        self.assertEqual(result[1], "package-lse:0.25")
        with self.assertRaisesRegex(AttentionStateError, "no result completion"):
            _ = runtime.completion_validator
        with self.assertRaisesRegex(AttentionStateError, "no successful"):
            _ = runtime.last_completion_receipt


if __name__ == "__main__":
    unittest.main()
