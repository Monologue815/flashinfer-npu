import gc
import json
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession, AttentionMetadataLimits, AttentionMode,
    AttentionPlanSpec, CustomMaskSpec, PagedKVMetadata, PagedPrefillMetadata,
    RaggedKVMetadata, SingleAttentionMetadata,
)
from flashinfer_npu.attention.operator_mask import (
    AttentionMaskPlanMetadata, AttentionMaskPlanResource,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_116_metadata_integer_contract import IndexValue


def mask_plan(packed=False, paged=False):
    mode = (AttentionMode.BATCH_PREFILL_PAGED if paged
            else AttentionMode.BATCH_PREFILL_RAGGED)
    spec = AttentionPlanSpec(mode, 1, 1, 1,
                             custom_mask=CustomMaskSpec(3 if packed else 12, packed))
    if paged:
        metadata = PagedPrefillMetadata(
            (0, 1, 2, 2), PagedKVMetadata((0, 1, 4, 4), (0, 1, 2, 3), (3, 3, 0), 3))
    else:
        metadata = RaggedKVMetadata((0, 1, 2, 2), (0, 3, 12, 12))
    session = AttentionFrameworkSession(mode)
    return session, session.plan(spec, metadata)


class OpaquePayload:
    def __getattr__(self, name):
        raise AssertionError("mask payload must not be inspected")

    def __repr__(self):
        raise AssertionError("mask payload must not be represented")

    def __eq__(self, other):
        raise AssertionError("mask payload must not be compared")

    def __bool__(self):
        raise AssertionError("mask payload must not be read")


class MaskPlanResourcesCheckpoint(unittest.TestCase):
    def test_paged_and_ragged_offsets_keep_units_and_per_request_padding(self):
        for packed in (False, True):
            for paged in (False, True):
                with self.subTest(packed=packed, paged=paged):
                    _, plan = mask_plan(packed, paged)
                    metadata = AttentionMaskPlanMetadata.from_plan(plan)
                    self.assertEqual(metadata.logical_element_indptr, (0, 3, 12, 12))
                    self.assertEqual(metadata.packed_byte_indptr, (0, 1, 3, 3))
                    self.assertEqual(metadata.mask_spec.dtype, "uint8" if packed else "bool")
                    metadata.validate_plan(plan)
                    self.assertEqual(len(metadata.fingerprint), 64)

    def test_single_prefill_derives_one_segment(self):
        spec = AttentionPlanSpec(AttentionMode.SINGLE_PREFILL, 1, 1, 1,
                                 custom_mask=CustomMaskSpec(2, packed=True))
        plan = AttentionFrameworkSession(spec.mode).plan(spec, SingleAttentionMetadata(2, 6))
        metadata = AttentionMaskPlanMetadata.from_plan(plan)
        self.assertEqual(metadata.logical_element_indptr, (0, 12))
        self.assertEqual(metadata.packed_byte_indptr, (0, 2))

    def test_replanning_and_admission_drift_reject_old_resources(self):
        session, first = mask_plan()
        metadata = AttentionMaskPlanMetadata.from_plan(first)
        second = session.plan(first.spec, first.metadata)
        self.assertEqual(first.fingerprint, second.fingerprint)
        with self.assertRaisesRegex(SchemaError, "does not match"):
            metadata.validate_plan(second)
        self.assertNotEqual(metadata.fingerprint,
                            AttentionMaskPlanMetadata.from_plan(second).fingerprint)
        changed_limits = replace(first, resource_limits=AttentionMetadataLimits(max_batch_size=3))
        with self.assertRaisesRegex(SchemaError, "does not match"):
            metadata.validate_plan(changed_limits)
        no_mask = session.plan(replace(first.spec, custom_mask=None), first.metadata)
        with self.assertRaisesRegex(SchemaError, "custom-mask plan"):
            metadata.validate_plan(no_mask)

    def test_inconsistent_or_lossy_offsets_are_rejected(self):
        _, plan = mask_plan(packed=True)
        metadata = AttentionMaskPlanMetadata.from_plan(plan)
        for changes in (
            {"packed_byte_indptr": (0, 1, 2, 2)},
            {"logical_element_indptr": (0, 3, 2, 12)},
            {"logical_element_indptr": (0, 3.0, 12, 12)},
            {"logical_element_indptr": (0, 3, 12)},
            {"packed_byte_indptr": (1, 2, 4, 4)},
            {"mask_spec": CustomMaskSpec(2, packed=True)},
            {"plan_generation": True},
            {"plan_generation": 0},
            {"framework_plan_fingerprint": "invalid"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(SchemaError):
                    replace(metadata, **changes)

    def test_borrowed_resource_retains_owner_without_observation_or_serialization(self):
        _, plan = mask_plan()
        metadata = AttentionMaskPlanMetadata.from_plan(plan)
        payload, owner = OpaquePayload(), OpaquePayload()
        owner_ref = weakref.ref(owner)
        resource = AttentionMaskPlanResource(metadata, payload, owner)
        other = AttentionMaskPlanResource(metadata, payload, owner)
        self.assertIsNot(resource, other)
        self.assertNotEqual(resource, other)
        resource.validate_plan(plan)
        self.assertNotIn("payload=", repr(resource))
        serialized = json.dumps(metadata.to_dict())
        self.assertNotIn("owner", serialized)
        self.assertNotIn("payload", serialized)
        del owner, other
        gc.collect()
        self.assertIsNotNone(owner_ref())
        del resource
        gc.collect()
        self.assertIsNone(owner_ref())
        with self.assertRaises(SchemaError):
            AttentionMaskPlanResource(metadata, payload, None)

    def test_custom_mask_spec_requires_integer_numel_and_boolean_packing(self):
        for numel in (True, 3.0, 3.5, "3", float("inf")):
            with self.subTest(numel=numel):
                with self.assertRaises(SchemaError):
                    CustomMaskSpec(numel)
        for packed in (0, 1, "true", None):
            with self.subTest(packed=packed):
                with self.assertRaises(SchemaError):
                    CustomMaskSpec(3, packed=packed)
        self.assertEqual(CustomMaskSpec(IndexValue(3)).numel, 3)
