import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorOperationCatalog,
    AttentionOperatorOperationSpec,
    AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageResolutionError,
    AttentionOperatorPackageResolver,
    ImportlibAttentionOperatorPackageLoader,
)
from flashinfer_npu.runtime import SchemaError


def fake_attention(query, key, value, *, scale=1.0, return_softmax_lse=False):
    fake_attention.calls.append(
        (query, key, value, scale, return_softmax_lse)
    )
    return ("output", "lse")


fake_attention.calls = []


def operation(provider_id="cann"):
    return AttentionOperatorOperationSpec(
        operation_id="%s.fake_attention@v1" % provider_id,
        provider_id=provider_id,
        package_name="fake-attention-package",
        callable_path="fake_attention_package.fake_attention",
        api_version="v1",
        candidate_modes=(AttentionMode.BATCH_MIXED_PAGED,),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=("scale", "return_softmax_lse"),
        return_names=("output", "softmax_lse"),
        lse_control_argument="return_softmax_lse",
        source_url="https://example.com/fake-attention-v1",
    )


class FakePackageLoader:
    loader_id = "checkpoint-018.fake-loader.v1"

    def __init__(self, version="1.2.3", callable_object=fake_attention):
        self.version = version
        self.callable_object = callable_object
        self.version_calls = []
        self.resolve_calls = []

    def package_version(self, package_name):
        self.version_calls.append(package_name)
        return self.version

    def resolve_callable(self, callable_path):
        self.resolve_calls.append(callable_path)
        return self.callable_object


def resolver(loader, provider_id="cann"):
    candidate = operation(provider_id)
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-018-catalog", operations=(candidate,)
    )
    compatibility = AttentionOperatorPackageCompatibility(
        provider_id=provider_id,
        operation_id=candidate.operation_id,
        adapter_version="checkpoint-018-adapter-v1",
        supported_package_versions=("1.2.3", "1.2.4"),
    )
    return AttentionOperatorPackageResolver(catalog, compatibility, loader)


class LazyOperatorPackageResolutionCheckpoint(unittest.TestCase):
    """Checkpoint 018: package metadata gates lazy exact callable injection."""

    def setUp(self):
        fake_attention.calls[:] = []

    def test_construction_and_metadata_explain_never_resolve_or_import_callable(self):
        loader = FakePackageLoader()
        with patch("importlib.import_module") as importer:
            package_resolver = resolver(loader)
            self.assertEqual(loader.version_calls, [])
            self.assertEqual(loader.resolve_calls, [])

            report = package_resolver.explain()

        self.assertTrue(report.accepted)
        self.assertEqual(report.stage, "metadata")
        self.assertFalse(report.callable_loaded)
        self.assertEqual(loader.version_calls, ["fake-attention-package"])
        self.assertEqual(loader.resolve_calls, [])
        importer.assert_not_called()

    def test_missing_package_is_structured_and_cannot_resolve_callable(self):
        loader = FakePackageLoader(version=None)
        package_resolver = resolver(loader)

        report = package_resolver.explain()
        self.assertFalse(report.accepted)
        self.assertIn("not installed", report.reasons[0])
        with self.assertRaisesRegex(
            AttentionOperatorPackageResolutionError, "unavailable"
        ) as captured:
            package_resolver.resolve()
        self.assertEqual(captured.exception.report.stage, "metadata")
        self.assertEqual(loader.resolve_calls, [])

    def test_unapproved_package_version_is_rejected_before_callable_resolution(self):
        loader = FakePackageLoader(version="2.0.0")
        package_resolver = resolver(loader)

        with self.assertRaisesRegex(
            AttentionOperatorPackageResolutionError, "incompatible"
        ) as captured:
            package_resolver.resolve()
        self.assertIn("not adapter-authorized", captured.exception.report.reasons[0])
        self.assertEqual(loader.resolve_calls, [])

    def test_exact_callable_resolves_to_an_unbound_nonexecuted_executor(self):
        loader = FakePackageLoader()
        package_resolver = resolver(loader)

        resolved = package_resolver.resolve()

        self.assertEqual(loader.resolve_calls, [operation().callable_path])
        self.assertEqual(fake_attention.calls, [])
        self.assertTrue(resolved.report.accepted)
        self.assertEqual(resolved.report.stage, "callable")
        self.assertTrue(resolved.report.callable_loaded)
        self.assertEqual(
            resolved.report.callable_observation_fingerprint,
            resolved.observation.fingerprint,
        )
        self.assertFalse(resolved.executor.is_runtime_bound)
        self.assertEqual(
            resolved.callable_binding.provider_probe_fingerprint,
            resolved.provider_probe.fingerprint,
        )

    def test_signature_drift_is_reported_and_no_executor_is_published(self):
        def drifted(query, key, *, scale=1.0):
            raise AssertionError("signature inspection must not invoke callable")

        loader = FakePackageLoader(callable_object=drifted)
        package_resolver = resolver(loader)

        with self.assertRaisesRegex(
            AttentionOperatorPackageResolutionError, "callable is incompatible"
        ) as captured:
            package_resolver.resolve()
        report = captured.exception.report
        self.assertEqual(report.stage, "callable")
        self.assertTrue(report.callable_loaded)
        self.assertFalse(report.accepted)
        self.assertIn("positional signature differs", " ".join(report.reasons))

    def test_noncallable_catalog_object_is_rejected(self):
        loader = FakePackageLoader(callable_object=object())
        with self.assertRaisesRegex(
            AttentionOperatorPackageResolutionError, "not callable"
        ) as captured:
            resolver(loader).resolve()
        self.assertEqual(
            captured.exception.report.reasons,
            ("resolved package object is not callable",),
        )

    def test_compatibility_provider_must_match_catalog(self):
        candidate = operation("cann")
        catalog = AttentionOperatorOperationCatalog(
            name="checkpoint-018-mismatch", operations=(candidate,)
        )
        compatibility = AttentionOperatorPackageCompatibility(
            provider_id="flash_attention_npu",
            operation_id=candidate.operation_id,
            adapter_version="checkpoint-018-adapter-v1",
            supported_package_versions=("1.2.3",),
        )
        with self.assertRaisesRegex(SchemaError, "catalog provider"):
            AttentionOperatorPackageResolver(
                catalog, compatibility, FakePackageLoader()
            )

    def test_importlib_loader_only_resolves_explicit_path(self):
        loader = ImportlibAttentionOperatorPackageLoader()
        self.assertTrue(callable(loader.resolve_callable("json.dumps")))
        with self.assertRaisesRegex(SchemaError, "is absent"):
            loader.resolve_callable("json.definitely_absent")


if __name__ == "__main__":
    unittest.main()
