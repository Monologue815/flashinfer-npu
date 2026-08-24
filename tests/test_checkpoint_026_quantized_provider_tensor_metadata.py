import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorPackageRuntimeSpec,
    AttentionOperatorQuantizedKVInput,
    TensorView,
    inspect_attention_operator_quantized_kv_input,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    metadata_tensor,
    quantized_input,
)


def with_view(tensor, **changes):
    return type(tensor)(replace(tensor.tensor_view, **changes))


class InvalidMetadataInspector:
    def to_view(self, tensor, *, name, writable=False):
        return object()


class QuantizedProviderTensorMetadataCheckpoint(unittest.TestCase):
    """Checkpoint 026: provider KV metadata is proven before lowering."""

    def test_valid_input_is_inspected_before_non_executing_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        lowered = session.run("query", kv_input)

        calls = values["tensor_metadata_inspector"].calls
        self.assertEqual(
            tuple(name for _, name, _ in calls),
            (
                "kv.key.storage",
                "kv.value.storage",
                "kv.key.scale",
                "kv.value.scale",
            ),
        )
        self.assertTrue(all(writable is False for _, _, writable in calls))
        self.assertIs(dict(lowered.positional_arguments)["key"], kv_input.key_storage)

    def test_storage_dtype_and_logical_shape_must_match_plan(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)

        wrong_dtype = with_view(valid.key_storage, dtype="uint8")
        with self.assertRaisesRegex(SchemaError, "storage view dtype"):
            session.run("query", replace(valid, key_storage=wrong_dtype))

        too_few_pages = metadata_tensor(
            "too-few-pages", (1, 2, 1, 3), "int8"
        )
        with self.assertRaisesRegex(SchemaError, "every referenced page"):
            session.run("query", replace(valid, key_storage=too_few_pages))

    def test_scale_shape_and_dtype_must_match_exact_quant_spec(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)

        wrong_shape = metadata_tensor("wrong-scale-shape", (2, 2, 1, 1), "float32")
        with self.assertRaisesRegex(SchemaError, "scale view shape"):
            session.run("query", replace(valid, key_scale=wrong_shape))

        wrong_dtype = with_view(valid.value_scale, dtype="float16", storage_nbytes=8)
        with self.assertRaisesRegex(SchemaError, "scale view dtype"):
            session.run("query", replace(valid, value_scale=wrong_dtype))

    def test_asymmetric_zero_points_require_int32_exact_shapes(self):
        plan = group_plan()
        asymmetric = replace(
            plan.spec.kv_quant_spec, scheme="asymmetric", has_zero_point=True
        )
        asymmetric_plan = replace(
            plan, spec=replace(plan.spec, kv_quant_spec=asymmetric)
        )
        value = quantized_input(
            asymmetric,
            key_zero_point=metadata_tensor(
                "key-zero", (2, 2, 1, 2), "int32"
            ),
            value_zero_point=metadata_tensor(
                "value-zero", (2, 2, 1, 1), "int32"
            ),
        )
        inspector = bootstrap_components()["tensor_metadata_inspector"]

        inspected = inspect_attention_operator_quantized_kv_input(
            asymmetric_plan, value, inspector, "npu:0"
        )
        self.assertTrue(inspected.quantized)

        wrong = with_view(value.key_zero_point, dtype="int16")
        with self.assertRaisesRegex(SchemaError, "zero_point view dtype"):
            inspect_attention_operator_quantized_kv_input(
                asymmetric_plan,
                replace(value, key_zero_point=wrong),
                inspector,
                "npu:0",
            )

    def test_non_contiguous_components_are_rejected_before_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)
        non_contiguous = with_view(
            valid.key_storage,
            strides=(12, 6, 3, 1),
            storage_nbytes=21,
        )

        with self.assertRaisesRegex(SchemaError, "must be contiguous"):
            session.run("query", replace(valid, key_storage=non_contiguous))

    def test_every_component_must_use_the_resolved_device(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)
        foreign = with_view(valid.value_scale, device="npu:1")

        with self.assertRaisesRegex(SchemaError, "share device"):
            session.run("query", replace(valid, value_scale=foreign))

        all_foreign = {
            name: with_view(getattr(valid, name), device="npu:1")
            for name in ("key_storage", "value_storage", "key_scale", "value_scale")
        }
        with self.assertRaisesRegex(SchemaError, "device does not match"):
            session.run("query", replace(valid, **all_foreign))

    def test_component_and_kv_storage_aliases_are_rejected(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)

        aliased_scale = with_view(
            valid.key_scale,
            storage_id=valid.key_storage.tensor_view.storage_id,
        )
        with self.assertRaisesRegex(SchemaError, "components.*alias"):
            session.run("query", replace(valid, key_scale=aliased_scale))

        aliased_value = with_view(
            valid.value_storage,
            storage_id=valid.key_storage.tensor_view.storage_id,
        )
        with self.assertRaisesRegex(SchemaError, "K and V.*alias"):
            session.run("query", replace(valid, value_storage=aliased_value))

    def test_bootstrap_requires_a_real_inspector_and_rejects_invalid_output(self):
        values = bootstrap_components()
        with self.assertRaisesRegex(SchemaError, "require.*metadata inspector"):
            replace(values["spec"], tensor_metadata_inspector=None)

        invalid = replace(
            values["spec"], tensor_metadata_inspector=InvalidMetadataInspector()
        )
        plan, session = active_session(values, spec=invalid)
        with self.assertRaisesRegex(TypeError, "must return TensorView"):
            session.run("query", quantized_input(plan.spec.kv_quant_spec))


if __name__ == "__main__":
    unittest.main()
