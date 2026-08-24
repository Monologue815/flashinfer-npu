import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizedKVInput,
    AttentionOperatorWrapperSession,
    build_attention_operator_package_runtime,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)


def active_session(values, *, spec=None, catalog=None):
    spec = values["spec"] if spec is None else spec
    catalog = values["catalog"] if catalog is None else catalog
    implementation = build_attention_operator_package_runtime(
        spec,
        operation_catalog=catalog,
        package_loader=values["loader"],
    )
    plan = group_plan()
    resolved = implementation.resolve(plan, "npu:0")
    session = AttentionOperatorWrapperSession(catalog)
    session.plan(
        resolved.factory,
        resolved.run_adapter,
        plan,
        resolved.receipt,
        resolved.selection,
        resolved.callable_binding,
    )
    return plan, session


def quantized_input(quant_spec, **overrides):
    values = {
        "quant_spec": quant_spec,
        "key_storage": object(),
        "value_storage": object(),
        "key_scale": object(),
        "value_scale": object(),
    }
    values.update(overrides)
    return AttentionOperatorQuantizedKVInput(**values)


class QuantizedProviderRunLoweringCheckpoint(unittest.TestCase):
    """Checkpoint 025: bound quant inputs lower without executing a package."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_symmetric_input_requires_independent_storage_and_scale_objects(self):
        quant_spec = group_plan().spec.kv_quant_spec
        value = quantized_input(quant_spec)

        self.assertIsNot(value.key_storage, value.value_storage)
        self.assertIsNot(value.key_scale, value.value_scale)
        for field_name in (
            "key_storage",
            "value_storage",
            "key_scale",
            "value_scale",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(SchemaError, "requires %s" % field_name):
                    quantized_input(quant_spec, **{field_name: None})

    def test_zero_point_presence_must_match_exact_quant_spec(self):
        symmetric = group_plan().spec.kv_quant_spec
        with self.assertRaisesRegex(SchemaError, "cannot carry zero points"):
            quantized_input(symmetric, key_zero_point=object())

        asymmetric = replace(symmetric, scheme="asymmetric", has_zero_point=True)
        with self.assertRaisesRegex(SchemaError, "requires independent zero points"):
            quantized_input(asymmetric)
        value = quantized_input(
            asymmetric,
            key_zero_point=object(),
            value_zero_point=object(),
        )
        self.assertIsNotNone(value.key_zero_point)
        self.assertIsNotNone(value.value_zero_point)

    def test_bootstrap_lowering_unwraps_storage_and_injects_bound_scales(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        lowered = session.run("query", kv_input)

        positional = dict(lowered.positional_arguments)
        keywords = dict(lowered.keyword_arguments)
        self.assertIs(positional["key"], kv_input.key_storage)
        self.assertIs(positional["value"], kv_input.value_storage)
        self.assertIs(keywords["key_scale"], kv_input.key_scale)
        self.assertIs(keywords["value_scale"], kv_input.value_scale)
        self.assertEqual(
            lowered.consumed_request_fields,
            ("query", "kv_cache", "logits_soft_cap"),
        )
        self.assertEqual(package_attention.calls, [])

    def test_plain_or_different_quantspec_input_is_rejected_before_base_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        with self.assertRaisesRegex(SchemaError, "QuantizedKVInput"):
            session.run("query", ("key", "value"))

        different = replace(
            plan.spec.kv_quant_spec,
            granularity="tensor",
            group_size=None,
            axis=None,
        )
        with self.assertRaisesRegex(SchemaError, "does not match"):
            session.run("query", quantized_input(different))
        self.assertEqual(len(values["materializer"].calls), 1)
        self.assertEqual(package_attention.calls, [])

    def test_default_binding_rejects_public_runtime_scale_multipliers(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        for name in ("k_scale", "v_scale"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(SchemaError, "rejects run-time"):
                    session.run("query", kv_input, **{name: object()})
        self.assertEqual(package_attention.calls, [])

    def test_explicit_runtime_scale_policy_injects_separate_arguments(self):
        values = bootstrap_components()
        original = values["spec"].quantization_bindings[0]
        extended = replace(
            original,
            argument_bindings=original.argument_bindings
            + (
                AttentionOperatorQuantArgumentBinding(
                    "run.k_scale", "runtime_key_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.v_scale", "runtime_value_scale"
                ),
            ),
            runtime_k_scale_policy="argument",
            runtime_v_scale_policy="argument",
        )
        spec = replace(values["spec"], quantization_bindings=(extended,))
        plan, session = active_session(values, spec=spec)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        runtime_k_scale = object()
        runtime_v_scale = object()

        lowered = session.run(
            "query",
            kv_input,
            k_scale=runtime_k_scale,
            v_scale=runtime_v_scale,
        )

        keywords = dict(lowered.keyword_arguments)
        self.assertIs(keywords["runtime_key_scale"], runtime_k_scale)
        self.assertIs(keywords["runtime_value_scale"], runtime_v_scale)
        self.assertEqual(
            lowered.consumed_request_fields,
            ("query", "kv_cache", "k_scale", "v_scale", "logits_soft_cap"),
        )
        self.assertEqual(package_attention.calls, [])

    def test_injected_argument_cannot_overwrite_base_provider_lowering(self):
        values = bootstrap_components()
        operation = replace(
            values["operation"],
            quant_arguments=values["operation"].quant_arguments + ("scale",),
        )
        catalog = AttentionOperatorOperationCatalog(
            name="checkpoint-025-collision", operations=(operation,)
        )
        original = values["spec"].quantization_bindings[0]
        collision = replace(
            original,
            argument_bindings=(
                AttentionOperatorQuantArgumentBinding("kv.key.scale", "scale"),
                AttentionOperatorQuantArgumentBinding(
                    "kv.value.scale", "value_scale"
                ),
            ),
        )
        spec = replace(values["spec"], quantization_bindings=(collision,))
        plan, session = active_session(values, spec=spec, catalog=catalog)

        with self.assertRaisesRegex(SchemaError, "collides.*scale"):
            session.run("query", quantized_input(plan.spec.kv_quant_spec))
        self.assertEqual(package_attention.calls, [])

    def test_lowering_reuses_one_plan_materialization_without_callable_execution(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        first = session.run("query-1", kv_input)
        second = session.run("query-2", kv_input)

        self.assertEqual(len(values["materializer"].calls), 1)
        self.assertIs(
            dict(first.keyword_arguments)["table"],
            dict(second.keyword_arguments)["table"],
        )
        self.assertEqual(package_attention.calls, [])


if __name__ == "__main__":
    unittest.main()
