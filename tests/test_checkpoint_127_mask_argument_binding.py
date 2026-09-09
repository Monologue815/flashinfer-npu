import gc
import json
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention import AttentionMode, AttentionOperatorOperationSpec
from flashinfer_npu.attention.operator_mask_binding import (
    AttentionOperatorMaskArgumentSpec, lower_attention_mask_arguments,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_126_mask_resource_revalidation import prepared_source


def synthetic_operation():
    return AttentionOperatorOperationSpec(
        operation_id="synthetic.mask_attention@v1", provider_id="synthetic",
        package_name="synthetic-mask-package", callable_path="synthetic.attention",
        api_version="v1", candidate_modes=(AttentionMode.BATCH_PREFILL_RAGGED,),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=("mask", "logical_offsets", "byte_offsets"),
        host_sequence_arguments=("logical_offsets", "byte_offsets"),
        return_names=("output",), source_url="https://example.com/synthetic-mask-contract",
    )


def argument_spec(operation, packed=False):
    return AttentionOperatorMaskArgumentSpec(
        operation.fingerprint,
        "packed_allow_little_segments" if packed else "bool_allow_flat",
        "mask", "logical_offsets", "byte_offsets" if packed else None,
    )


class MaskArgumentBindingCheckpoint(unittest.TestCase):
    def test_synthetic_signature_receives_exact_payload_and_typed_offsets(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                _, plan, inspected, inspector = prepared_source(packed)
                operation = synthetic_operation()
                fragment = lower_attention_mask_arguments(
                    plan, inspected, operation, argument_spec(operation, packed), inspector)

                # This callable records arguments only; it computes no attention.
                def record(query, key, value, *, mask, logical_offsets, byte_offsets=None):
                    return mask, logical_offsets, byte_offsets

                mask, logical, byte_offsets = record("q", "k", "v", **dict(fragment.keyword_arguments))
                self.assertIs(mask, inspected.resource.payload)
                self.assertEqual(logical, (0, 3, 12, 12))
                self.assertEqual(byte_offsets, (0, 1, 3, 3) if packed else None)
                self.assertEqual(len(inspector.calls), 1)
                self.assertIs(fragment.inspected, inspected)
                # Formatting diagnostics must not observe the opaque tensor.
                self.assertNotIn("payload", repr(fragment))
                self.assertNotIn("storage", json.dumps(fragment.spec.to_dict()))

    def test_operation_signature_drift_is_rejected_before_inspection(self):
        _, plan, inspected, inspector = prepared_source()
        operation = synthetic_operation()
        spec = argument_spec(operation)
        for changes in ({"api_version": "v2"}, {"provider_id": "different"},
                        {"callable_path": "synthetic.other"}, {"package_name": "other"}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, "exact operation"):
                    lower_attention_mask_arguments(
                        plan, inspected, replace(operation, **changes), spec, inspector)
        self.assertEqual(inspector.calls, [])

    def test_unsupported_formats_and_offset_unit_aliases_fail_closed(self):
        spec = argument_spec(synthetic_operation(), packed=True)
        for changes in (
            {"encoding": "additive_dense"}, {"encoding": "bool_block_flat"},
            {"encoding": "packed_allow_big_segments"},
            {"packed_byte_indptr_argument": None},
            {"packed_byte_indptr_argument": "logical_offsets"},
            {"logical_element_indptr_argument": "mask"},
            {"mask_argument": "bad-name"}, {"operation_fingerprint": "invalid"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(SchemaError):
                    replace(spec, **changes)
        with self.assertRaises(SchemaError):
            replace(argument_spec(synthetic_operation()), packed_byte_indptr_argument="byte_offsets")

    def test_encoding_mismatch_never_packs_or_unpacks_implicitly(self):
        operation = synthetic_operation()
        for packed in (False, True):
            with self.subTest(packed=packed):
                _, plan, inspected, inspector = prepared_source(packed)
                with self.assertRaisesRegex(SchemaError, "explicit transformation"):
                    lower_attention_mask_arguments(
                        plan, inspected, operation, argument_spec(operation, not packed), inspector)
                self.assertEqual(inspector.calls, [])

    def test_incompatible_argument_roles_are_rejected_before_inspection(self):
        _, plan, inspected, inspector = prepared_source()
        original = synthetic_operation()
        for changes, message in (
            ({"mutable_arguments": ("mask",)}, "another operation role"),
            ({"mutable_arguments": ("logical_offsets",)}, "another operation role"),
            ({"quant_arguments": ("mask",)}, "another operation role"),
            ({"paged_table_argument": "logical_offsets"}, "another operation role"),
            ({"lse_control_argument": "mask"}, "another operation role"),
            ({"host_sequence_arguments": ("mask", "logical_offsets", "byte_offsets")}, "payload"),
            ({"host_sequence_arguments": ("byte_offsets",)}, "host sequences"),
            ({"positional_arguments": ("query", "key", "value", "mask"),
              "keyword_arguments": ("logical_offsets", "byte_offsets")}, "keyword"),
        ):
            with self.subTest(changes=changes):
                operation = replace(original, **changes)
                with self.assertRaisesRegex(SchemaError, message):
                    lower_attention_mask_arguments(
                        plan, inspected, operation, argument_spec(operation), inspector)
        self.assertEqual(inspector.calls, [])

    def test_stale_plan_and_wrong_mode_fail_before_source_inspection(self):
        session, plan, inspected, inspector = prepared_source()
        operation = synthetic_operation()
        new_plan = session.plan(plan.spec, plan.metadata)
        with self.assertRaisesRegex(SchemaError, "active plan"):
            lower_attention_mask_arguments(
                new_plan, inspected, operation, argument_spec(operation), inspector)
        operation = replace(operation, candidate_modes=(AttentionMode.SINGLE_PREFILL,))
        with self.assertRaisesRegex(SchemaError, "planned mode"):
            lower_attention_mask_arguments(plan, inspected, operation, argument_spec(operation), inspector)
        self.assertEqual(inspector.calls, [])

    def test_source_is_revalidated_for_each_argument_fragment(self):
        _, plan, inspected, inspector = prepared_source()
        operation = synthetic_operation()
        spec = argument_spec(operation)
        first = lower_attention_mask_arguments(plan, inspected, operation, spec, inspector)
        inspector.view = replace(inspected.view, storage_id="replacement")
        with self.assertRaisesRegex(SchemaError, "changed.*replan"):
            lower_attention_mask_arguments(plan, inspected, operation, spec, inspector)
        self.assertIs(first.inspected, inspected)
        inspector.view = inspected.view
        second = lower_attention_mask_arguments(plan, inspected, operation, spec, inspector)
        self.assertIsNot(first, second)
        self.assertIs(second.inspected, inspected)
        self.assertEqual(len(inspector.calls), 3)

    def test_fragment_retains_separate_owner_without_serializing_payload(self):
        _, plan, inspected, inspector = prepared_source(packed=True)
        operation = synthetic_operation()
        spec = argument_spec(operation, packed=True)
        owner_ref = weakref.ref(inspected.resource.owner)
        fragment = lower_attention_mask_arguments(plan, inspected, operation, spec, inspector)
        del inspected
        gc.collect()
        self.assertIsNotNone(owner_ref())
        self.assertEqual(len(spec.fingerprint), 64)
        arguments = fragment.keyword_arguments
        del fragment
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertEqual(len(arguments), 3)
