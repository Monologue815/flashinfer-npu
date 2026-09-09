import unittest
from dataclasses import replace

from flashinfer_npu.attention.operator_mask import (
    inspect_attention_mask_plan_resource,
    revalidate_attention_mask_plan_resource,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_125_mask_tensor_inspection import inspection_inputs


def prepared_source(packed=False):
    session, plan, resource, inspector = inspection_inputs(packed)
    inspected = inspect_attention_mask_plan_resource(
        plan, resource, inspector, "npu:0", required_alignment=16)
    inspector.calls.clear()
    return session, plan, inspected, inspector


class MaskResourceRevalidationCheckpoint(unittest.TestCase):
    def test_repeated_uses_reinspect_the_same_source_and_keep_the_binding(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                _, plan, inspected, inspector = prepared_source(packed)
                # Equal metadata may arrive in a new descriptor object.
                inspector.view = replace(inspected.view)
                for _ in range(3):
                    self.assertIs(revalidate_attention_mask_plan_resource(
                        plan, inspected, inspector), inspected)
                self.assertEqual(len(inspector.calls), 3)
                for payload, name, writable in inspector.calls:
                    self.assertIs(payload, inspected.resource.payload)
                    self.assertEqual((name, writable), ("custom_mask", False))

    def test_still_compatible_storage_changes_require_replanning(self):
        for packed in (False, True):
            for changes in (
                {"storage_id": "replacement-allocation"},
                {"storage_nbytes": 128},
                {"storage_offset": 16},
                {"data_ptr_alignment": 32},
                {"writable": True},
            ):
                with self.subTest(packed=packed, changes=changes):
                    _, plan, inspected, inspector = prepared_source(packed)
                    original = inspected.view
                    inspector.view = replace(original, **changes)
                    for _ in range(2):
                        with self.assertRaisesRegex(SchemaError, "changed.*replan"):
                            revalidate_attention_mask_plan_resource(plan, inspected, inspector)
                        self.assertIs(inspected.view, original)
                    # Rejection does not poison the original binding either.
                    inspector.view = original
                    self.assertIs(revalidate_attention_mask_plan_resource(
                        plan, inspected, inspector), inspected)

    def test_incompatible_metadata_is_rejected_again_at_use_time(self):
        _, plan, inspected, inspector = prepared_source()
        for changes, message in (
            ({"shape": (11,)}, "element count"),
            ({"dtype": "uint8"}, "dtype"),
            ({"device": "npu:1"}, "device"),
            ({"strides": (2,)}, "contiguous"),
            ({"data_ptr_alignment": 8}, "aligned"),
        ):
            with self.subTest(changes=changes):
                inspector.view = replace(inspected.view, **changes)
                with self.assertRaisesRegex(SchemaError, message):
                    revalidate_attention_mask_plan_resource(plan, inspected, inspector)

    def test_stale_generation_is_rejected_before_touching_the_source(self):
        session, plan, inspected, inspector = prepared_source()
        new_plan = session.plan(plan.spec, plan.metadata)
        with self.assertRaisesRegex(SchemaError, "active plan"):
            revalidate_attention_mask_plan_resource(new_plan, inspected, inspector)
        self.assertEqual(inspector.calls, [])
        self.assertIs(revalidate_attention_mask_plan_resource(
            plan, inspected, inspector), inspected)

    def test_inspector_failure_or_bad_output_does_not_replace_the_snapshot(self):
        _, plan, inspected, inspector = prepared_source()

        class FailingInspector:
            def to_view(self, tensor, *, name, writable=False):
                raise RuntimeError("source metadata unavailable")

        original = inspected.view
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            revalidate_attention_mask_plan_resource(plan, inspected, FailingInspector())
        inspector.view = object()
        with self.assertRaisesRegex(TypeError, "TensorView"):
            revalidate_attention_mask_plan_resource(plan, inspected, inspector)
        self.assertIs(inspected.view, original)
        inspector.view = original
        self.assertIs(revalidate_attention_mask_plan_resource(
            plan, inspected, inspector), inspected)

    def test_uninspected_resources_cannot_bypass_initial_inspection(self):
        _, plan, inspected, inspector = prepared_source()
        for invalid in (None, object(), inspected.resource):
            with self.subTest(kind=type(invalid).__name__):
                with self.assertRaisesRegex(TypeError, "AttentionInspectedMaskPlanResource"):
                    revalidate_attention_mask_plan_resource(plan, invalid, inspector)
        self.assertEqual(inspector.calls, [])
