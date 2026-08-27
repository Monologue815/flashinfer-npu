import inspect
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionJsonEnvelopeLimits,
    AttentionOperatorEvidenceResultArtifact,
    AttentionOperatorPhysicalEvidenceManifest,
    BatchAttention,
    QuantPhysicalLayoutCatalog,
    load_attention_operator_physical_evidence_manifest,
    verify_attention_operator_physical_evidence_results,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    query_input,
)
from tests.test_checkpoint_027_provider_physical_layout_binding import (
    physical_input,
    physical_runtime,
)
from tests.test_checkpoint_028_physical_layout_evidence import (
    evidence_bundle,
    physical_evidence,
)


RESULT_PAYLOAD = b"synthetic-external-result-record"


def manifest_fixture():
    values, plan, spec, quant_spec, descriptor = physical_runtime()
    evidence = physical_evidence(
        spec, values["operation"], quant_spec, descriptor
    )
    bundle = evidence_bundle(spec, values["operation"], (evidence,))
    return values, plan, spec, quant_spec, descriptor, evidence, bundle


class EvidenceManifestCheckpoint(unittest.TestCase):
    """Checkpoint 029: package evidence is bounded JSON plus verified bytes."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_manifest_json_round_trip_is_canonical_and_bounded(self):
        *_, bundle = manifest_fixture()
        encoded = bundle.manifest.to_json()

        restored, usage = load_attention_operator_physical_evidence_manifest(
            encoded
        )

        self.assertEqual(restored, bundle.manifest)
        self.assertEqual(restored.fingerprint, bundle.manifest.fingerprint)
        self.assertEqual(usage.encoded_bytes, len(encoded.encode("utf-8")))
        self.assertGreater(usage.nodes, 1)

    def test_loader_rejects_duplicate_unknown_and_oversized_json(self):
        *_, bundle = manifest_fixture()
        encoded = bundle.manifest.to_json()
        duplicate = encoded[:-1] + ',"name":"duplicate"}'
        with self.assertRaisesRegex(SchemaError, "duplicate JSON object key"):
            load_attention_operator_physical_evidence_manifest(duplicate)

        unknown = encoded[:-1] + ',"unknown":1}'
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            load_attention_operator_physical_evidence_manifest(unknown)

        limits = AttentionJsonEnvelopeLimits(max_bytes=len(encoded) - 1)
        with self.assertRaisesRegex(SchemaError, "bytes exceed limit"):
            load_attention_operator_physical_evidence_manifest(
                encoded, limits=limits
            )

    def test_result_locator_is_relative_canonical_and_scoped(self):
        _, _, _, _, _, evidence, _ = manifest_fixture()
        invalid = (
            "/evidence/result.json",
            "../evidence/result.json",
            "evidence/../result.json",
            "evidence//result.json",
            "evidence\\result.json",
            "results/result.json",
        )
        for locator in invalid:
            with self.subTest(locator=locator):
                with self.assertRaisesRegex(SchemaError, "locator"):
                    AttentionOperatorEvidenceResultArtifact(
                        evidence.evidence_id,
                        locator,
                        evidence.result_digest,
                        len(RESULT_PAYLOAD),
                    )

    def test_result_verification_requires_exact_set_size_and_digest(self):
        *_, bundle = manifest_fixture()
        manifest = bundle.manifest
        locator = manifest.result_artifacts[0].locator
        with self.assertRaisesRegex(SchemaError, "payload set"):
            verify_attention_operator_physical_evidence_results(manifest, {})
        with self.assertRaisesRegex(SchemaError, "size mismatch"):
            verify_attention_operator_physical_evidence_results(
                manifest, {locator: RESULT_PAYLOAD[:-1]}
            )
        corrupted = bytearray(RESULT_PAYLOAD)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(SchemaError, "digest mismatch"):
            verify_attention_operator_physical_evidence_results(
                manifest, {locator: bytes(corrupted)}
            )

    def test_manifest_runtime_identity_closes_package_adapter_and_catalog(self):
        values, _, spec, _, _, _, bundle = manifest_fixture()
        manifest = bundle.manifest
        cases = (
            replace(manifest, adapter_version="other-adapter"),
            replace(manifest, supported_package_versions=("9.9.9",)),
            replace(manifest, package_name="foreign-package"),
        )
        for candidate in cases:
            with self.subTest(candidate=candidate.fingerprint):
                with self.assertRaisesRegex(SchemaError, "runtime identity is stale"):
                    candidate.validate_runtime_spec(
                        values["operation"],
                        spec.adapter_version,
                        spec.supported_package_versions,
                        spec.quant_physical_layout_catalog,
                    )
        with self.assertRaisesRegex(SchemaError, "runtime identity is stale"):
            manifest.validate_runtime_spec(
                values["operation"],
                spec.adapter_version,
                spec.supported_package_versions,
                QuantPhysicalLayoutCatalog(),
            )

    def test_bootstrap_accepts_only_verified_bundle_not_bare_manifest(self):
        _, _, spec, _, _, _, bundle = manifest_fixture()
        with self.assertRaisesRegex(TypeError, "must be a verified bundle"):
            replace(
                spec,
                physical_layout_evidence_bundle=bundle.manifest,
            )

    def test_loaded_and_verified_bundle_authorizes_without_callable_execution(self):
        values, plan, spec, quant_spec, descriptor, _, bundle = manifest_fixture()
        loaded, _ = load_attention_operator_physical_evidence_manifest(
            bundle.manifest.to_json()
        )
        locator = loaded.result_artifacts[0].locator
        verified = verify_attention_operator_physical_evidence_results(
            loaded, {locator: RESULT_PAYLOAD}
        )
        authorized = replace(
            spec, physical_layout_evidence_bundle=verified
        )
        active_plan, session = active_session(values, spec=authorized, plan=plan)

        lowered = session.run(
            query_input(active_plan), physical_input(quant_spec, descriptor)
        )

        self.assertEqual(
            session.active_plan.dispatch_receipt.evidence_id,
            loaded.evidences[0].evidence_id,
        )
        self.assertEqual(lowered.provider_id, "cann")
        self.assertEqual(package_attention.calls, [])

    def test_manifest_remains_absent_from_public_flashinfer_surface(self):
        for method in (BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("evidence_manifest", parameters)
            self.assertNotIn("evidence_bundle", parameters)


if __name__ == "__main__":
    unittest.main()
