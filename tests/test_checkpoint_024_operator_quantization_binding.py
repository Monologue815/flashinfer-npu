import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizationPlanGate,
    build_attention_operator_package_runtime,
    validate_attention_operator_quantization_bindings,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import functional_profile, group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)


def argument(source, name):
    return AttentionOperatorQuantArgumentBinding(source, name)


def binding_for(quant_spec, operation, *, arguments=None, **policies):
    if arguments is None:
        arguments = (
            argument("kv.key.scale", "key_scale"),
            argument("kv.value.scale", "value_scale"),
        )
    return AttentionOperatorQuantizationBinding(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        quant_spec=quant_spec,
        argument_bindings=arguments,
        **policies,
    )


class OperatorQuantizationBindingCheckpoint(unittest.TestCase):
    """Checkpoint 024: capability QuantSpec must close over catalog arguments."""

    def test_exact_symmetric_binding_is_canonical_and_order_independent(self):
        values = bootstrap_components()
        quant_spec = functional_profile().rules[0].quant_specs[0]
        first = binding_for(quant_spec, values["operation"])
        second = binding_for(
            quant_spec,
            values["operation"],
            arguments=tuple(reversed(first.argument_bindings)),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            first.arguments_by_source,
            {"kv.key.scale": "key_scale", "kv.value.scale": "value_scale"},
        )
        self.assertEqual(len(first.fingerprint), 64)

    def test_asymmetric_binding_requires_independent_key_and_value_zero_points(self):
        values = bootstrap_components()
        symmetric = functional_profile().rules[0].quant_specs[0]
        asymmetric = replace(symmetric, scheme="asymmetric", has_zero_point=True)

        with self.assertRaisesRegex(SchemaError, "missing source.*zero_point"):
            binding_for(asymmetric, values["operation"])

        complete = binding_for(
            asymmetric,
            values["operation"],
            arguments=(
                argument("kv.key.scale", "key_scale"),
                argument("kv.value.scale", "value_scale"),
                argument("kv.key.zero_point", "key_zero"),
                argument("kv.value.zero_point", "value_zero"),
            ),
        )
        self.assertIn("kv.key.zero_point", complete.arguments_by_source)
        self.assertIn("kv.value.zero_point", complete.arguments_by_source)

    def test_runtime_scale_policy_must_match_explicit_argument_source(self):
        values = bootstrap_components()
        quant_spec = functional_profile().rules[0].quant_specs[0]
        for component in ("q", "k", "v"):
            source = "run.%s_scale" % component
            argument_name = "runtime_%s_scale" % component
            policy_name = "runtime_%s_scale_policy" % component
            with self.subTest(component=component, missing_source=True):
                with self.assertRaisesRegex(SchemaError, "policy does not match"):
                    binding_for(
                        quant_spec,
                        values["operation"],
                        **{policy_name: "argument"},
                    )
            with self.subTest(component=component, missing_policy=True):
                with self.assertRaisesRegex(SchemaError, "policy does not match"):
                    binding_for(
                        quant_spec,
                        values["operation"],
                        arguments=(
                            argument("kv.key.scale", "key_scale"),
                            argument("kv.value.scale", "value_scale"),
                            argument(source, argument_name),
                        ),
                    )

    def test_profile_and_operation_binding_validate_as_an_exact_set(self):
        values = bootstrap_components()
        validated = validate_attention_operator_quantization_bindings(
            values["operation"],
            values["spec"].profiles,
            values["spec"].quantization_bindings,
        )

        self.assertEqual(validated, values["spec"].quantization_bindings)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_bootstrap_rejects_quant_capability_without_api_binding(self):
        values = bootstrap_components()
        incomplete = replace(values["spec"], quantization_bindings=())

        with self.assertRaisesRegex(
            SchemaError, "capability QuantSpec has no API argument binding"
        ):
            build_attention_operator_package_runtime(
                incomplete,
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["events"], [])

    def test_non_catalog_argument_and_extra_quantspec_are_rejected(self):
        values = bootstrap_components()
        quant_spec = functional_profile().rules[0].quant_specs[0]
        non_catalog = binding_for(
            quant_spec,
            values["operation"],
            arguments=(
                argument("kv.key.scale", "not_in_catalog"),
                argument("kv.value.scale", "value_scale"),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "non-catalog argument"):
            validate_attention_operator_quantization_bindings(
                values["operation"], values["spec"].profiles, (non_catalog,)
            )

        other_spec = replace(
            quant_spec, granularity="tensor", group_size=None, axis=None
        )
        extra = binding_for(other_spec, values["operation"])
        with self.assertRaisesRegex(
            SchemaError, "binding has no capability QuantSpec"
        ):
            validate_attention_operator_quantization_bindings(
                values["operation"],
                values["spec"].profiles,
                values["spec"].quantization_bindings + (extra,),
            )
        foreign = replace(
            values["spec"].quantization_bindings[0],
            provider_id="flash_attention_npu",
        )
        with self.assertRaisesRegex(SchemaError, "binding identity differs"):
            AttentionOperatorQuantizationPlanGate(
                values["gate"], values["operation"], (foreign,)
            )
        with self.assertRaisesRegex(SchemaError, "duplicate QuantSpec"):
            AttentionOperatorQuantizationPlanGate(
                values["gate"],
                values["operation"],
                (
                    values["spec"].quantization_bindings[0],
                    values["spec"].quantization_bindings[0],
                ),
            )

    def test_composed_gate_accepts_only_the_exact_bound_quantspec(self):
        values = bootstrap_components()
        implementation = build_attention_operator_package_runtime(
            values["spec"],
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        exact_plan = group_plan()
        original = exact_plan.spec.kv_quant_spec
        different = replace(
            original, granularity="tensor", group_size=None, axis=None
        )
        different_spec = replace(exact_plan.spec, kv_quant_spec=different)
        different_plan = AttentionFrameworkSession(different_spec.mode).plan(
            different_spec, exact_plan.metadata
        )

        self.assertEqual(
            implementation.rejection_reasons(exact_plan, "npu:0"), ()
        )
        reasons = implementation.rejection_reasons(different_plan, "npu:0")
        self.assertTrue(
            any("no exact KV QuantSpec argument binding" in item for item in reasons)
        )
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_quantization_gate_preserves_base_reasons_deterministically(self):
        values = bootstrap_components(gate_reasons=("layout rejected",))
        gate = AttentionOperatorQuantizationPlanGate(
            values["gate"],
            values["operation"],
            values["spec"].quantization_bindings,
        )
        plan = group_plan()

        self.assertEqual(
            gate.rejection_reasons(plan, "npu"), ("layout rejected",)
        )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
