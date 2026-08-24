import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionRunTensorContract,
    KVCacheView,
    PagedKVCacheSpec,
    QuantLayoutConversionPlan,
    QuantPhysicalAxisTransform,
    QuantPhysicalLayoutCatalog,
    QuantPhysicalLayoutDescriptor,
    QuantizedTensorView,
    StreamContext,
    attention_run_tensor_signature,
    infer_logical_quant_storage_shape,
    infer_quant_physical_shapes,
    infer_quant_storage_shape,
    materialize_attention_kv_cache_view,
    plan_quant_layout_conversion,
)
from flashinfer_npu.runtime import QuantSpec, SchemaError
from tests.test_attention_tensor_contract import view


def synthetic_layout(layout_id="synthetic.blocked16.v1", block=16):
    storage = QuantPhysicalAxisTransform(
        4,
        (1, 1, 1, block),
        ("o0", "o1", "o2", "o3", "i3"),
        required_alignment=32,
        padding_value=0,
    )
    scale = QuantPhysicalAxisTransform(
        1,
        (4,),
        ("o0", "i0"),
        required_alignment=16,
        padding_value=1.0,
    )
    zero = QuantPhysicalAxisTransform(
        1,
        (4,),
        ("o0", "i0"),
        required_alignment=16,
        padding_value=0,
    )
    return QuantPhysicalLayoutDescriptor(
        layout_id=layout_id,
        storage_dtypes=("uint8", "int4_packed"),
        storage_transform=storage,
        storage_converter_id="synthetic.storage.to_blocked%d" % block,
        storage_inverse_converter_id="synthetic.storage.from_blocked%d" % block,
        scale_transform=scale,
        scale_converter_id="synthetic.scale.to_blocked4",
        scale_inverse_converter_id="synthetic.scale.from_blocked4",
        zero_point_transform=zero,
        zero_point_converter_id="synthetic.zero.to_blocked4",
        zero_point_inverse_converter_id="synthetic.zero.from_blocked4",
        required_features=("synthetic-blocked-layout",),
    )


def int4_spec(layout="logical"):
    return QuantSpec(
        scheme="symmetric",
        storage_dtype="int4_packed",
        compute_dtype="float32",
        accumulator_dtype="float32",
        granularity="channel",
        axis=(2,),
        physical_layout=layout,
        packing_order="low_nibble_first",
    )


def uint8_spec(layout="logical"):
    return QuantSpec(
        scheme="asymmetric",
        storage_dtype="uint8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        granularity="channel",
        axis=(2,),
        has_zero_point=True,
        physical_layout=layout,
    )


class QuantPhysicalAxisTransformTests(unittest.TestCase):
    def test_blocked_coordinate_mapping_is_reversible_and_marks_padding(self):
        transform = synthetic_layout().storage_transform
        input_shape = (2, 3, 4, 17)
        self.assertEqual(transform.physical_shape(input_shape), (2, 3, 4, 2, 16))
        for coordinate in ((0, 0, 0, 0), (1, 2, 3, 16), (0, 1, 2, 7)):
            physical = transform.logical_to_physical(input_shape, coordinate)
            self.assertEqual(
                transform.physical_to_logical(input_shape, physical), coordinate
            )
        self.assertIsNone(
            transform.physical_to_logical(input_shape, (0, 0, 0, 1, 15))
        )

    def test_axis_schema_rejects_missing_duplicate_and_unblocked_inner_axes(self):
        invalid_axes = (
            ("o0",),
            ("o0", "o1", "o1"),
            ("o0", "i0", "o1"),
        )
        for axes in invalid_axes:
            with self.subTest(axes=axes):
                with self.assertRaises(SchemaError):
                    QuantPhysicalAxisTransform(2, (1, 1), axes)
        with self.assertRaisesRegex(SchemaError, "power of two"):
            QuantPhysicalAxisTransform(1, (1,), ("o0",), required_alignment=3)


class QuantPhysicalLayoutDescriptorTests(unittest.TestCase):
    def _physical_kv(self, descriptor):
        logical_shape = (2, 3, 4, 5)
        spec = uint8_spec(descriptor.layout_id)
        shapes = descriptor.physical_shapes(logical_shape, spec)

        def quantized(prefix):
            return QuantizedTensorView(
                logical_shape,
                view(
                    shapes.storage,
                    dtype="uint8",
                    storage_id=prefix + ".storage",
                ),
                view(shapes.scale, storage_id=prefix + ".scale"),
                spec,
                view(
                    shapes.zero_point,
                    dtype="int32",
                    storage_id=prefix + ".zero",
                ),
                descriptor,
            )

        cache = PagedKVCacheSpec(
            2,
            3,
            4,
            5,
            5,
            "uint8",
            structure="separate",
            device="cpu",
            quant_spec=spec,
        )
        return KVCacheView(cache, quantized("key"), quantized("value"))

    def test_nonlogical_quantized_view_requires_exact_descriptor_and_alignment(self):
        logical_shape = (2, 3, 4, 5)
        descriptor = synthetic_layout()
        spec = uint8_spec(descriptor.layout_id)
        shapes = descriptor.physical_shapes(logical_shape, spec)
        value = QuantizedTensorView(
            logical_shape,
            view(shapes.storage, dtype="uint8", storage_id="physical-storage"),
            view(shapes.scale, storage_id="physical-scale"),
            spec,
            view(shapes.zero_point, dtype="int32", storage_id="physical-zero"),
            descriptor,
        )
        self.assertEqual(value.physical_layout_descriptor.fingerprint, descriptor.fingerprint)
        with self.assertRaisesRegex(SchemaError, "requires a physical layout descriptor"):
            QuantizedTensorView(
                logical_shape,
                value.storage,
                value.scale,
                spec,
                value.zero_point,
            )
        with self.assertRaisesRegex(SchemaError, "32-byte aligned"):
            QuantizedTensorView(
                logical_shape,
                view(
                    shapes.storage,
                    dtype="uint8",
                    storage_id="misaligned-storage",
                    alignment=16,
                ),
                value.scale,
                spec,
                value.zero_point,
                descriptor,
            )

    def test_descriptor_identity_enters_tensor_signature_but_pod_v1_rejects_launch(self):
        first = synthetic_layout()
        second = replace(first, storage_converter_id="synthetic.changed.converter")
        first_kv = self._physical_kv(first)
        second_kv = self._physical_kv(second)
        q = view((1, 1, 1), storage_id="q")
        first_contract = AttentionRunTensorContract(
            q, first_kv, StreamContext("cpu", "stream")
        )
        second_contract = AttentionRunTensorContract(
            q, second_kv, StreamContext("cpu", "stream")
        )
        self.assertNotEqual(
            attention_run_tensor_signature(first_contract),
            attention_run_tensor_signature(second_contract),
        )
        with self.assertRaisesRegex(SchemaError, "POD v1"):
            materialize_attention_kv_cache_view(first_kv, (), None, 0)

        mismatched_value = replace(
            first_kv.value, physical_layout_descriptor=second
        )
        with self.assertRaisesRegex(SchemaError, "descriptors must match"):
            KVCacheView(first_kv.spec, first_kv.key, mismatched_value)

    def test_packed_int4_storage_shape_is_packed_before_blocking(self):
        logical_shape = (2, 3, 4, 33)
        descriptor = synthetic_layout()
        catalog = QuantPhysicalLayoutCatalog((descriptor,))
        spec = int4_spec(descriptor.layout_id)
        self.assertEqual(
            infer_logical_quant_storage_shape(logical_shape, spec),
            (2, 3, 4, 17),
        )
        shapes = infer_quant_physical_shapes(logical_shape, spec, catalog)
        self.assertEqual(shapes.storage, (2, 3, 4, 2, 16))
        self.assertEqual(shapes.scale, (1, 4))
        self.assertIsNone(shapes.zero_point)
        self.assertEqual(
            infer_quant_storage_shape(
                logical_shape, spec, layout_catalog=catalog
            ),
            shapes.storage,
        )
        with self.assertRaisesRegex(NotImplementedError, "registered"):
            infer_quant_storage_shape(logical_shape, spec)

    def test_asymmetric_layout_carries_independent_zero_point_shape(self):
        logical_shape = (2, 3, 4, 5)
        descriptor = synthetic_layout()
        shapes = descriptor.physical_shapes(
            logical_shape, uint8_spec(descriptor.layout_id)
        )
        self.assertEqual(shapes.storage, (2, 3, 4, 1, 16))
        self.assertEqual(shapes.scale, (1, 4))
        self.assertEqual(shapes.zero_point, (1, 4))

    def test_catalog_and_descriptor_round_trip_have_stable_identity(self):
        catalog = QuantPhysicalLayoutCatalog((synthetic_layout(),))
        restored = QuantPhysicalLayoutCatalog.from_dict(catalog.to_dict())
        self.assertEqual(restored, catalog)
        self.assertEqual(restored.fingerprint, catalog.fingerprint)
        self.assertEqual(
            restored.descriptors[0].fingerprint,
            catalog.descriptors[0].fingerprint,
        )
        with self.assertRaisesRegex(SchemaError, "ids must be unique"):
            QuantPhysicalLayoutCatalog((synthetic_layout(), synthetic_layout()))
        with self.assertRaisesRegex(SchemaError, "exactly one"):
            QuantPhysicalLayoutCatalog().resolve(int4_spec("unregistered.layout"))

    def test_dtype_rank_and_converter_contract_mismatches_are_explicit(self):
        descriptor = synthetic_layout()
        with self.assertRaisesRegex(SchemaError, "storage dtype"):
            descriptor.physical_shapes(
                (1, 1, 1, 1),
                replace(uint8_spec(descriptor.layout_id), storage_dtype="int8"),
            )
        bad_rank = replace(
            descriptor,
            storage_transform=QuantPhysicalAxisTransform(3, (1, 1, 1), ("o0", "o1", "o2")),
        )
        with self.assertRaisesRegex(SchemaError, "rank"):
            bad_rank.physical_shapes(
                (1, 1, 1, 1), uint8_spec(descriptor.layout_id)
            )
        with self.assertRaisesRegex(SchemaError, "converter ids"):
            replace(descriptor, scale_converter_id=None)


class QuantLayoutConversionPlanTests(unittest.TestCase):
    def test_logical_to_physical_plan_has_explicit_component_steps(self):
        logical_shape = (2, 3, 4, 5)
        descriptor = synthetic_layout()
        catalog = QuantPhysicalLayoutCatalog((descriptor,))
        source = uint8_spec()
        destination = uint8_spec(descriptor.layout_id)
        plan = plan_quant_layout_conversion(
            logical_shape, source, destination, catalog
        )
        self.assertTrue(plan.requires_conversion)
        self.assertEqual(
            tuple(step.component for step in plan.steps),
            ("storage", "scale", "zero_point"),
        )
        self.assertEqual(
            tuple(step.converter_id for step in plan.steps),
            (
                descriptor.storage_converter_id,
                descriptor.scale_converter_id,
                descriptor.zero_point_converter_id,
            ),
        )
        self.assertEqual(plan.steps[0].input_shape, (2, 3, 4, 5))
        self.assertEqual(plan.steps[0].output_shape, (2, 3, 4, 1, 16))
        self.assertEqual(len(plan.fingerprint), 64)
        restored = QuantLayoutConversionPlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)
        self.assertEqual(restored.fingerprint, plan.fingerprint)

    def test_reverse_and_cross_physical_plans_route_via_logical(self):
        logical_shape = (2, 3, 4, 17)
        first = synthetic_layout()
        second = synthetic_layout("synthetic.blocked8.v1", 8)
        catalog = QuantPhysicalLayoutCatalog((first, second))
        logical = uint8_spec()
        blocked16 = uint8_spec(first.layout_id)
        blocked8 = uint8_spec(second.layout_id)
        reverse = plan_quant_layout_conversion(
            logical_shape, blocked16, logical, catalog
        )
        self.assertEqual(len(reverse.steps), 3)
        self.assertTrue(
            all(step.destination_layout == "logical" for step in reverse.steps)
        )
        cross = plan_quant_layout_conversion(
            logical_shape, blocked16, blocked8, catalog
        )
        self.assertEqual(len(cross.steps), 6)
        for component in ("storage", "scale", "zero_point"):
            component_steps = tuple(
                step for step in cross.steps if step.component == component
            )
            self.assertEqual(len(component_steps), 2)
            self.assertEqual(component_steps[0].destination_layout, "logical")
            self.assertEqual(component_steps[1].source_layout, "logical")

    def test_noop_and_semantic_changes_are_not_hidden_conversions(self):
        spec = uint8_spec()
        plan = plan_quant_layout_conversion((1, 1, 4, 3), spec, spec)
        self.assertFalse(plan.requires_conversion)
        changed = replace(spec, scale_dtype="float16")
        with self.assertRaisesRegex(SchemaError, "cannot change quantization semantics"):
            plan_quant_layout_conversion((1, 1, 4, 3), spec, changed)


if __name__ == "__main__":
    unittest.main()
