import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionNvfp4PackedLayoutDescriptor,
    AttentionOperatorNvfp4PackedKVBinding,
    AttentionOperatorNvfp4PackedKVRunAdapterFactory,
    AttentionOperatorNvfp4ScaleFactorBinding,
    AttentionOperatorOperationCatalog,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizationPlanGate,
    AttentionOperatorQuantizationRunAdapterFactory,
    attention_kernel_binary_abi_v2,
    build_attention_operator_package_runtime,
    describe_attention_operator_package_runtime,
    validate_attention_operator_nvfp4_packed_kv_bindings,
    validate_attention_operator_quantization_bindings,
    attention_nvfp4_kv_quant_spec,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import (
    attention_launch_abi,
    bound_kernel,
    group_plan,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components


NVFP4_QUANT_SPEC = attention_nvfp4_kv_quant_spec(
    physical_layout="test.runtime.nvfp4_pair_v1",
    packing_order="test.runtime.low_nibble_first_v1",
)


def runtime_values():
    values = bootstrap_components()
    operation = replace(
        values["operation"],
        keyword_arguments=values["operation"].keyword_arguments
        + ("kv_cache_sf",),
        quant_arguments=values["operation"].quant_arguments
        + ("kv_cache_sf",),
    )
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-101-nvfp4-runtime", operations=(operation,)
    )
    original_profile = values["spec"].profiles[0]
    original_rule = original_profile.rules[0]
    q_dtype, _, o_dtype = original_rule.dtype_signatures[0]
    rule = replace(
        original_rule,
        dtype_signatures=original_rule.dtype_signatures
        + ((q_dtype, "uint8", o_dtype),),
        quant_specs=original_rule.quant_specs + (NVFP4_QUANT_SPEC,),
    )
    profile = replace(original_profile, rules=(rule,))
    first_kernel = bound_kernel(profile)
    constraints = replace(
        first_kernel.constraints,
        dtype_signatures=rule.dtype_signatures,
        quant_storage_dtypes=tuple(
            sorted({item.storage_dtype for item in rule.quant_specs})
        ),
    )
    kernel = replace(
        first_kernel,
        constraints=constraints,
        launch_abi=replace(
            attention_launch_abi(), abi_name="flashinfer_npu.attention.v2"
        ),
        binary_abi=attention_kernel_binary_abi_v2(),
    )
    scale_binding = AttentionOperatorNvfp4ScaleFactorBinding(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        quant_spec=NVFP4_QUANT_SPEC,
        combined_argument="kv_cache_sf",
    )
    packed_binding = AttentionOperatorNvfp4PackedKVBinding(
        scale_binding,
        AttentionNvfp4PackedLayoutDescriptor(
            physical_layout=NVFP4_QUANT_SPEC.physical_layout,
            packing_order=NVFP4_QUANT_SPEC.packing_order,
            storage_required_alignment=16,
            scale_required_alignment=16,
        ),
    )
    spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(kernel,),
        nvfp4_packed_kv_bindings=(packed_binding,),
    )
    values.update(
        operation=operation,
        catalog=catalog,
        profile=profile,
        packed_binding=packed_binding,
        spec=spec,
    )
    return values


class Nvfp4RuntimeRegistrationCheckpoint(unittest.TestCase):
    """A reviewed NVFP4 route is part of runtime admission and audit identity."""

    def test_capability_quantspec_set_is_closed_by_disjoint_binding_families(self):
        values = runtime_values()
        nvfp4 = validate_attention_operator_nvfp4_packed_kv_bindings(
            values["operation"],
            values["spec"].profiles,
            values["spec"].nvfp4_packed_kv_bindings,
        )
        general = validate_attention_operator_quantization_bindings(
            values["operation"],
            values["spec"].profiles,
            values["spec"].quantization_bindings,
            delegated_quant_specs=tuple(item.quant_spec for item in nvfp4),
        )

        self.assertEqual(general, values["spec"].quantization_bindings)
        self.assertEqual(nvfp4, values["spec"].nvfp4_packed_kv_bindings)
        with self.assertRaisesRegex(SchemaError, "general and delegated"):
            validate_attention_operator_quantization_bindings(
                values["operation"],
                values["spec"].profiles,
                values["spec"].quantization_bindings,
                delegated_quant_specs=(
                    values["spec"].quantization_bindings[0].quant_spec,
                ),
            )

    def test_plan_gate_accepts_general_and_delegated_exact_quantspecs(self):
        values = runtime_values()
        gate = AttentionOperatorQuantizationPlanGate(
            values["gate"],
            values["operation"],
            values["spec"].quantization_bindings,
            (NVFP4_QUANT_SPEC,),
        )
        general_plan = group_plan()
        nvfp4_spec = replace(
            general_plan.spec,
            kv_dtype="uint8",
            kv_quant_spec=NVFP4_QUANT_SPEC,
        )
        nvfp4_plan = AttentionFrameworkSession(nvfp4_spec.mode).plan(
            nvfp4_spec, general_plan.metadata
        )

        self.assertEqual(gate.rejection_reasons(general_plan, "npu:0"), ())
        self.assertEqual(gate.rejection_reasons(nvfp4_plan, "npu:0"), ())
        other = replace(
            NVFP4_QUANT_SPEC,
            packing_order="test.runtime.high_nibble_first_v1",
        )
        other_spec = replace(nvfp4_spec, kv_quant_spec=other)
        other_plan = AttentionFrameworkSession(other_spec.mode).plan(
            other_spec, general_plan.metadata
        )
        self.assertIn(
            "no exact KV QuantSpec",
            gate.rejection_reasons(other_plan, "npu:0")[-1],
        )

    def test_bootstrap_composes_general_then_nvfp4_factory_without_package_probe(self):
        values = runtime_values()

        implementation = build_attention_operator_package_runtime(
            values["spec"],
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        factories = implementation._run_adapter_factory._factories
        self.assertIsInstance(
            factories[0], AttentionOperatorQuantizationRunAdapterFactory
        )
        self.assertIsInstance(
            factories[1], AttentionOperatorNvfp4PackedKVRunAdapterFactory
        )
        self.assertEqual(
            factories[0]._delegated_quant_specs, (NVFP4_QUANT_SPEC,)
        )
        self.assertEqual(
            factories[1]._bindings, (values["packed_binding"],)
        )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_runtime_declaration_captures_binding_and_detects_layout_drift(self):
        values = runtime_values()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )

        self.assertEqual(
            declaration.nvfp4_packed_kv_binding_fingerprints,
            (values["packed_binding"].fingerprint,),
        )
        restored = type(declaration).from_dict(declaration.to_dict())
        self.assertEqual(restored, declaration)
        changed_layout = replace(
            values["packed_binding"].layout_descriptor,
            storage_required_alignment=32,
        )
        changed_binding = replace(
            values["packed_binding"], layout_descriptor=changed_layout
        )
        changed_spec = replace(
            values["spec"], nvfp4_packed_kv_bindings=(changed_binding,)
        )
        with self.assertRaisesRegex(SchemaError, "declaration is stale"):
            declaration.validate_runtime_spec(
                changed_spec, operation_catalog=values["catalog"]
            )

    def test_missing_capability_and_double_routing_fail_before_build(self):
        values = runtime_values()
        original_profile = values["spec"].profiles[0]
        rule = replace(
            original_profile.rules[0],
            quant_specs=(values["spec"].quantization_bindings[0].quant_spec,),
            dtype_signatures=(original_profile.rules[0].dtype_signatures[0],),
        )
        profile = replace(original_profile, rules=(rule,))
        missing = replace(values["spec"], profiles=(profile,))
        with self.assertRaisesRegex(SchemaError, "no capability QuantSpec"):
            build_attention_operator_package_runtime(
                missing,
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )

        general_nvfp4 = AttentionOperatorQuantizationBinding(
            provider_id=values["operation"].provider_id,
            operation_id=values["operation"].operation_id,
            quant_spec=NVFP4_QUANT_SPEC,
            argument_bindings=(
                AttentionOperatorQuantArgumentBinding(
                    "kv.key.scale", "key_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "kv.value.scale", "value_scale"
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "two bindings"):
            replace(
                values["spec"],
                quantization_bindings=values["spec"].quantization_bindings
                + (general_nvfp4,),
            )


if __name__ == "__main__":
    unittest.main()
