import gc
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention import TensorView
from flashinfer_npu.attention.operator_mask import (
    AttentionMaskPlanMetadata, AttentionMaskPlanResource,
    inspect_attention_mask_plan_resource,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_124_mask_plan_resources import OpaquePayload, mask_plan


class MaskInspector:
    def __init__(self, view):
        self.view = view
        self.calls = []

    def to_view(self, tensor, *, name, writable=False):
        self.calls.append((tensor, name, writable))
        return self.view


def inspection_inputs(packed=False):
    session, plan = mask_plan(packed=packed)
    metadata = AttentionMaskPlanMetadata.from_plan(plan)
    resource = AttentionMaskPlanResource(metadata, OpaquePayload(), OpaquePayload())
    view = TensorView((metadata.mask_spec.numel,), (1,), metadata.mask_spec.dtype,
                      "npu:0", "opaque-mask-storage", 64, data_ptr_alignment=16)
    return session, plan, resource, MaskInspector(view)


class MaskTensorInspectionCheckpoint(unittest.TestCase):
    def test_packed_and_unpacked_sources_are_inspected_once_without_conversion(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                _, plan, resource, inspector = inspection_inputs(packed)
                inspected = inspect_attention_mask_plan_resource(
                    plan, resource, inspector, "npu:0", required_alignment=16)
                self.assertIs(inspected.resource, resource)
                self.assertIs(inspected.view, inspector.view)
                self.assertEqual(len(inspector.calls), 1)
                self.assertIs(inspector.calls[0][0], resource.payload)
                self.assertEqual(inspector.calls[0][1:], ("custom_mask", False))
                self.assertEqual(inspected.view.dtype, "uint8" if packed else "bool")
                inspected.validate_plan(plan)

    def test_shape_dtype_device_contiguity_and_alignment_are_enforced(self):
        _, plan, resource, inspector = inspection_inputs()
        original = inspector.view
        for changes, message in (
            ({"shape": (11,)}, "element count"),
            ({"shape": (3, 4), "strides": (4, 1)}, "rank-1"),
            ({"dtype": "uint8"}, "dtype"),
            ({"device": "npu:1"}, "device"),
            ({"strides": (2,)}, "contiguous"),
            ({"data_ptr_alignment": 8}, "aligned"),
        ):
            with self.subTest(changes=changes):
                inspector.view = replace(original, **changes)
                with self.assertRaisesRegex(SchemaError, message):
                    inspect_attention_mask_plan_resource(
                        plan, resource, inspector, "npu:0", required_alignment=16)

    def test_packed_sources_do_not_accept_coalesced_segment_bytes(self):
        _, plan, resource, inspector = inspection_inputs(packed=True)
        # 3 + 9 bits need 1 + 2 bytes, not ceil(12 / 8) bytes.
        inspector.view = replace(inspector.view, shape=(2,))
        with self.assertRaisesRegex(SchemaError, "element count"):
            inspect_attention_mask_plan_resource(plan, resource, inspector, "npu:0")

    def test_stale_plan_and_invalid_requirements_fail_before_inspection(self):
        session, plan, resource, inspector = inspection_inputs()
        new_plan = session.plan(plan.spec, plan.metadata)
        with self.assertRaisesRegex(SchemaError, "does not match"):
            inspect_attention_mask_plan_resource(new_plan, resource, inspector, "npu:0")
        for alignment in (0, 3, True, 16.0):
            with self.subTest(alignment=alignment):
                with self.assertRaises(SchemaError):
                    inspect_attention_mask_plan_resource(
                        plan, resource, inspector, "npu:0", required_alignment=alignment)
        with self.assertRaises(SchemaError):
            inspect_attention_mask_plan_resource(plan, resource, inspector, "")
        self.assertEqual(inspector.calls, [])

    def test_invalid_inspector_result_cannot_be_treated_as_validated(self):
        _, plan, resource, inspector = inspection_inputs()
        inspector.view = object()
        with self.assertRaisesRegex(TypeError, "TensorView"):
            inspect_attention_mask_plan_resource(plan, resource, inspector, "npu:0")
        with self.assertRaisesRegex(TypeError, "Inspector"):
            inspect_attention_mask_plan_resource(plan, resource, object(), "npu:0")

    def test_inspected_result_retains_the_borrowed_owner(self):
        _, plan, resource, inspector = inspection_inputs()
        owner = weakref.ref(resource.owner)
        inspected = inspect_attention_mask_plan_resource(plan, resource, inspector, "npu:0")
        del resource
        gc.collect()
        self.assertIsNotNone(owner())
        del inspected
        gc.collect()
        self.assertIsNone(owner())
