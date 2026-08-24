import unittest

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMetadataLimits,
    AttentionMode,
    AttentionPlanSpec,
    AttentionResourceLimitError,
    CustomMaskSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    measure_attention_resources,
)
from flashinfer_npu.runtime import SchemaError


def workload():
    paged = PagedKVMetadata(
        indptr=(0, 1, 3),
        indices=(0, 1, 2),
        last_page_len=(3, 1),
        page_size=4,
    )
    metadata = PagedPrefillMetadata((0, 2, 3), paged)
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=4,
        q_dtype="float32",
        custom_mask=CustomMaskSpec(11),
    )
    return spec, metadata


class AttentionResourceMeasurementTests(unittest.TestCase):
    def test_paged_prefill_usage_measures_every_admission_dimension(self):
        spec, metadata = workload()
        usage = measure_attention_resources(spec, metadata)
        self.assertEqual(
            usage.to_dict(),
            {
                "batch_size": 2,
                "total_qo_tokens": 3,
                "total_kv_tokens": 8,
                "total_pages": 3,
                "max_pages_per_request": 2,
                "page_size": 4,
                "max_qo_tokens_per_request": 2,
                "max_kv_tokens_per_request": 5,
                "custom_mask_bytes": 11,
            },
        )

    def test_limits_round_trip_and_reject_invalid_fields(self):
        limits = AttentionMetadataLimits(
            max_batch_size=8, max_total_kv_tokens=4096
        )
        restored = AttentionMetadataLimits.from_dict(limits.to_dict())
        self.assertEqual(restored, limits)
        self.assertEqual(restored.fingerprint, limits.fingerprint)
        with self.assertRaisesRegex(SchemaError, "cannot be negative"):
            AttentionMetadataLimits(max_batch_size=-1)
        with self.assertRaisesRegex(SchemaError, "fields do not match"):
            AttentionMetadataLimits.from_dict({"schema_version": 1})

    def test_each_resource_dimension_has_an_independent_limit(self):
        spec, metadata = workload()
        usage = measure_attention_resources(spec, metadata)
        cases = {
            "max_batch_size": usage.batch_size,
            "max_total_qo_tokens": usage.total_qo_tokens,
            "max_total_kv_tokens": usage.total_kv_tokens,
            "max_total_pages": usage.total_pages,
            "max_pages_per_request": usage.max_pages_per_request,
            "max_page_size": usage.page_size,
            "max_qo_tokens_per_request": usage.max_qo_tokens_per_request,
            "max_kv_tokens_per_request": usage.max_kv_tokens_per_request,
            "max_custom_mask_bytes": usage.custom_mask_bytes,
        }
        for field, actual in cases.items():
            with self.subTest(field=field):
                accepted = AttentionMetadataLimits(**{field: actual})
                self.assertEqual(accepted.validate(spec, metadata), usage)
                rejected = AttentionMetadataLimits(**{field: actual - 1})
                with self.assertRaisesRegex(
                    AttentionResourceLimitError, "exceeds limit"
                ):
                    rejected.validate(spec, metadata)


class AttentionResourcePlanGateTests(unittest.TestCase):
    def test_plan_records_usage_without_changing_semantic_fingerprint(self):
        spec, metadata = workload()
        loose = AttentionMetadataLimits(max_total_kv_tokens=16)
        tight = AttentionMetadataLimits(max_total_kv_tokens=8)
        first = AttentionFrameworkSession(
            spec.mode, metadata_limits=loose
        ).plan(spec, metadata)
        second = AttentionFrameworkSession(
            spec.mode, metadata_limits=tight
        ).plan(spec, metadata)
        self.assertEqual(first.resource_usage.total_kv_tokens, 8)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.admission_fingerprint, second.admission_fingerprint)

    def test_failed_admission_does_not_publish_a_plan(self):
        spec, metadata = workload()
        session = AttentionFrameworkSession(
            spec.mode,
            metadata_limits=AttentionMetadataLimits(max_total_pages=2),
        )
        with self.assertRaisesRegex(AttentionResourceLimitError, "total_pages"):
            session.plan(spec, metadata)
        self.assertFalse(session.is_planned)

    def test_failed_replan_preserves_the_last_committed_generation(self):
        spec, too_large = workload()
        accepted_paged = PagedKVMetadata(
            indptr=(0, 1, 2),
            indices=(0, 1),
            last_page_len=(3, 4),
            page_size=4,
        )
        accepted = PagedPrefillMetadata((0, 2, 3), accepted_paged)
        accepted_spec = AttentionPlanSpec(
            mode=spec.mode,
            num_qo_heads=spec.num_qo_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim_qk=spec.head_dim_qk,
            q_dtype=spec.q_dtype,
            custom_mask=CustomMaskSpec(10),
        )
        session = AttentionFrameworkSession(
            spec.mode,
            metadata_limits=AttentionMetadataLimits(max_total_pages=2),
        )
        committed = session.plan(accepted_spec, accepted)
        with self.assertRaisesRegex(AttentionResourceLimitError, "total_pages"):
            session.plan(spec, too_large)
        self.assertIs(session.plan_state, committed)
        self.assertEqual(session.plan_state.generation, 1)

    def test_wrong_limit_object_type_fails_explicitly(self):
        with self.assertRaisesRegex(TypeError, "AttentionMetadataLimits"):
            AttentionFrameworkSession(
                AttentionMode.SINGLE_PREFILL, metadata_limits={}
            )


if __name__ == "__main__":
    unittest.main()
