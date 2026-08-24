import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionDispatchError,
    AttentionOperatorQuantizedKVInput,
    QuantPhysicalAxisTransform,
    QuantPhysicalLayoutCatalog,
    QuantPhysicalLayoutDescriptor,
    attention_kernel_binary_abi,
    attention_kernel_binary_abi_v2,
    inspect_attention_operator_quantized_kv_input,
    validate_attention_operator_quant_physical_layouts,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import (
    attention_launch_abi,
    bound_kernel,
    functional_profile,
    group_plan,
)
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    metadata_tensor,
    quantized_input,
)


def physical_descriptor(*, layout_id="checkpoint027.blocked", page_block=1):
    page_axes = ("o0", "i0") if page_block > 1 else ("o0",)
    return QuantPhysicalLayoutDescriptor(
        layout_id=layout_id,
        storage_dtypes=("int8",),
        storage_transform=QuantPhysicalAxisTransform(
            4,
            (page_block, 1, 1, 2),
            page_axes + ("o1", "o2", "o3", "i3"),
            required_alignment=16,
        ),
        storage_converter_id="checkpoint027.to_blocked",
        storage_inverse_converter_id="checkpoint027.from_blocked",
        required_features=("int8-group-dequant",),
    )


def aligned_tensor(name, shape, dtype):
    value = metadata_tensor(name, shape, dtype)
    return type(value)(replace(value.tensor_view, data_ptr_alignment=16))


def physical_input(quant_spec, descriptor, **overrides):
    key_shape = (2, 2, 1, 3)
    value_shape = (2, 2, 1, 2)
    key_shapes = descriptor.physical_shapes(key_shape, quant_spec)
    value_shapes = descriptor.physical_shapes(value_shape, quant_spec)
    values = {
        "quant_spec": quant_spec,
        "key_storage": aligned_tensor("physical-key", key_shapes.storage, "int8"),
        "value_storage": aligned_tensor(
            "physical-value", value_shapes.storage, "int8"
        ),
        "key_scale": metadata_tensor(
            "physical-key-scale", key_shapes.scale, quant_spec.scale_dtype
        ),
        "value_scale": metadata_tensor(
            "physical-value-scale", value_shapes.scale, quant_spec.scale_dtype
        ),
    }
    values.update(overrides)
    return AttentionOperatorQuantizedKVInput(**values)


def physical_runtime(*, descriptor=None, binary_abi=None):
    descriptor = physical_descriptor() if descriptor is None else descriptor
    values = bootstrap_components()
    base_plan = group_plan()
    quant_spec = replace(
        base_plan.spec.kv_quant_spec, physical_layout=descriptor.layout_id
    )
    plan = AttentionFrameworkSession(base_plan.spec.mode).plan(
        replace(base_plan.spec, kv_quant_spec=quant_spec), base_plan.metadata
    )
    base_profile = functional_profile()
    rule = replace(base_profile.rules[0], quant_specs=(quant_spec,))
    profile = replace(base_profile, rules=(rule,))
    selected_binary_abi = (
        attention_kernel_binary_abi_v2()
        if binary_abi is None
        else binary_abi
    )
    kernel = bound_kernel(
        profile,
        launch_abi=replace(
            attention_launch_abi(),
            abi_name=(
                "flashinfer_npu.attention.v2"
                if selected_binary_abi.fingerprint
                == attention_kernel_binary_abi_v2().fingerprint
                else "flashinfer_npu.attention.v1"
            ),
        ),
        binary_abi=selected_binary_abi,
    )
    binding = replace(
        values["spec"].quantization_bindings[0], quant_spec=quant_spec
    )
    spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(kernel,),
        observed_environment=profile.environment,
        quantization_bindings=(binding,),
        quant_physical_layout_catalog=QuantPhysicalLayoutCatalog((descriptor,)),
    )
    return values, plan, spec, quant_spec, descriptor


class ProviderPhysicalLayoutBindingCheckpoint(unittest.TestCase):
    """Checkpoint 027: non-logical KV needs exact descriptor and evidence."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_catalog_set_must_exactly_cover_nonlogical_bindings(self):
        values, _, spec, _, descriptor = physical_runtime()
        binding = spec.quantization_bindings[0]
        with self.assertRaisesRegex(SchemaError, "no physical descriptor"):
            validate_attention_operator_quant_physical_layouts(
                (binding,), QuantPhysicalLayoutCatalog()
            )
        with self.assertRaisesRegex(SchemaError, "unbound descriptor"):
            validate_attention_operator_quant_physical_layouts(
                values["spec"].quantization_bindings,
                QuantPhysicalLayoutCatalog((descriptor,)),
            )

    def test_catalog_descriptor_must_support_bound_storage_dtype(self):
        _, _, spec, _, descriptor = physical_runtime()
        unsupported = replace(descriptor, storage_dtypes=("uint8",))
        with self.assertRaisesRegex(SchemaError, "bound storage dtype"):
            validate_attention_operator_quant_physical_layouts(
                spec.quantization_bindings,
                QuantPhysicalLayoutCatalog((unsupported,)),
            )

    def test_bootstrap_rejects_nonlogical_binding_without_catalog(self):
        values, _, spec, _, _ = physical_runtime()
        missing = replace(
            spec, quant_physical_layout_catalog=QuantPhysicalLayoutCatalog()
        )
        with self.assertRaisesRegex(SchemaError, "no physical descriptor"):
            active_session(values, spec=missing)

    def test_valid_physical_kv_metadata_is_inspectable_but_not_authorized(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        value = physical_input(quant_spec, descriptor)

        inspected = inspect_attention_operator_quantized_kv_input(
            plan,
            value,
            values["tensor_metadata_inspector"],
            "npu:0",
            descriptor,
        )

        self.assertTrue(inspected.quantized)
        self.assertEqual(len(values["tensor_metadata_inspector"].calls), 4)
        with self.assertRaisesRegex(
            AttentionDispatchError, "capability evidence invalid"
        ):
            active_session(values, spec=spec, plan=plan)
        self.assertEqual(package_attention.calls, [])

    def test_physical_storage_shape_and_alignment_are_enforced(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        value = physical_input(quant_spec, descriptor)
        wrong_shape = aligned_tensor("wrong-physical-shape", (2, 2, 1, 3), "int8")
        with self.assertRaisesRegex(SchemaError, "storage view shape"):
            inspect_attention_operator_quantized_kv_input(
                plan,
                replace(value, key_storage=wrong_shape),
                values["tensor_metadata_inspector"],
                "npu:0",
                descriptor,
            )

        misaligned = type(value.key_storage)(
            replace(value.key_storage.tensor_view, data_ptr_alignment=8)
        )
        with self.assertRaisesRegex(SchemaError, "16-byte aligned"):
            inspect_attention_operator_quantized_kv_input(
                plan,
                replace(value, key_storage=misaligned),
                values["tensor_metadata_inspector"],
                "npu:0",
                descriptor,
            )

    def test_paged_layout_must_preserve_an_exact_unblocked_page_axis(self):
        descriptor = physical_descriptor(
            layout_id="checkpoint027.blocked_page", page_block=2
        )
        values, plan, spec, quant_spec, descriptor = physical_runtime(
            descriptor=descriptor
        )
        with self.assertRaisesRegex(SchemaError, "preserve an exact page axis"):
            inspect_attention_operator_quantized_kv_input(
                plan,
                physical_input(quant_spec, descriptor),
                values["tensor_metadata_inspector"],
                "npu:0",
                descriptor,
            )

    def test_kernel_native_layout_requires_v2_binary_abi_evidence(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime(
            binary_abi=attention_kernel_binary_abi()
        )
        with self.assertRaisesRegex(SchemaError, "requires KV POD v2"):
            active_session(values, spec=spec, plan=plan)
        self.assertEqual(package_attention.calls, [])

    def test_logical_binding_still_uses_empty_catalog_path(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        lowered = session.run("query", quantized_input(plan.spec.kv_quant_spec))
        self.assertEqual(lowered.provider_id, "cann")
        self.assertEqual(package_attention.calls, [])


if __name__ == "__main__":
    unittest.main()
