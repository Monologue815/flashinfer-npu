from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION,
    AttentionOperatorProviderIntegrationBundle,
    AttentionOperatorProviderIntegrationBundleBinding,
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import (
    FakePackageLoader,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)
from tests.test_checkpoint_068_declared_runtime_registry import (
    declared_registration,
)
from tests.test_checkpoint_080_scoring_manifest_bootstrap_binding import (
    scoring_manifest,
)
from tests.test_checkpoint_081_declared_scoring_manifest_snapshot import (
    scored_registration,
)


class AlternateFakePackageLoader(FakePackageLoader):
    """Stable second type used to prove loader type is bundle authority."""


class EmptyIdPackageLoader:
    loader_id = ""

    def package_version(self, package_name):
        raise AssertionError("bundle construction must not probe the package")

    def resolve_callable(self, callable_path):
        raise AssertionError("bundle construction must not resolve callables")


def provider_bundle(values, *, bundle_id="checkpoint.084.cann.bundle.v1", loader=None):
    manifest, registration = scored_registration(values)
    return AttentionOperatorProviderIntegrationBundle(
        bundle_id=bundle_id,
        operation_catalog=values["catalog"],
        registrations=(registration,),
        scoring_manifest=manifest,
        package_loader=values["loader"] if loader is None else loader,
    )


class ProviderIntegrationBundleCheckpoint(unittest.TestCase):
    """Checkpoint 084: one strict bundle publishes one provider generation."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_bundle_is_canonical_complete_and_side_effect_free(self):
        values = bootstrap_components()

        bundle = provider_bundle(values)
        binding = bundle.binding

        self.assertEqual(
            bundle.schema_version,
            ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION,
        )
        self.assertIsInstance(
            binding, AttentionOperatorProviderIntegrationBundleBinding
        )
        self.assertEqual(binding.bundle_fingerprint, bundle.fingerprint)
        self.assertEqual(binding.catalog_fingerprint, values["catalog"].fingerprint)
        self.assertEqual(
            binding.scoring_manifest_fingerprint,
            bundle.scoring_manifest.fingerprint,
        )
        self.assertEqual(binding.package_loader_id, values["loader"].loader_id)
        self.assertEqual(
            binding.registration_bindings,
            (
                (
                    bundle.registrations[0].declaration.provider_id,
                    bundle.registrations[0].declaration.operation_id,
                    bundle.registrations[0].declaration.fingerprint,
                ),
            ),
        )
        self.assertNotIn("package_loader", bundle.to_dict())
        self.assertNotIn("runtime_spec", bundle.to_dict())
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_single_install_publishes_all_bundle_authorities_atomically(self):
        values = bootstrap_components()
        bundle = provider_bundle(values)

        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )
        observed = attention_operator_runtime_registry_snapshot()

        self.assertEqual(installed.generation, observed.generation)
        self.assertIs(installed.registry, observed.registry)
        self.assertEqual(observed.operation_catalog, bundle.operation_catalog)
        self.assertEqual(
            observed.runtime_declaration_binding_tuples,
            bundle.binding.registration_bindings,
        )
        self.assertEqual(
            observed.plan_scoring_manifest_binding,
            bundle.scoring_manifest.binding,
        )
        self.assertEqual(
            observed.provider_integration_bundle_binding,
            bundle.binding,
        )
        self.assertEqual(observed.provider_integration_bundle_id, bundle.bundle_id)
        self.assertEqual(
            observed.provider_integration_bundle_fingerprint,
            bundle.fingerprint,
        )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_stale_generation_cannot_publish_a_drifted_bundle(self):
        values = bootstrap_components()
        bundle = provider_bundle(values)
        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )
        drifted = replace(bundle, bundle_id="checkpoint.084.cann.bundle.v2")

        with self.assertRaisesRegex(SchemaError, "generation changed"):
            install_attention_operator_provider_integration_bundle(
                drifted,
                expected_generation=self.original.generation,
            )

        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, installed.generation)
        self.assertEqual(current.provider_integration_bundle_binding, bundle.binding)
        self.assertNotEqual(bundle.fingerprint, drifted.fingerprint)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_wrapper_captures_bundle_generation_and_legacy_install_clears_it(self):
        before = BatchAttention(kv_layout="HND", device="npu:0")
        values = bootstrap_components()
        bundle = provider_bundle(values)
        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )
        bundled = BatchAttention(kv_layout="HND", device="npu:0")
        restored = install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=installed.generation,
        )
        after = BatchAttention(kv_layout="HND", device="npu:0")

        self.assertEqual(
            before._operator_runtime_registry_snapshot.provider_integration_bundle_binding,
            self.original.provider_integration_bundle_binding,
        )
        self.assertEqual(
            bundled._operator_runtime_registry_snapshot.provider_integration_bundle_binding,
            bundle.binding,
        )
        self.assertIsNone(
            after._operator_runtime_registry_snapshot.provider_integration_bundle_binding
        )
        self.assertEqual(
            after._operator_runtime_registry_snapshot.generation,
            restored.generation,
        )

    def test_bundle_requires_exact_catalog_registration_and_policy_sets(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)

        with self.assertRaisesRegex(SchemaError, "catalog identity set differs"):
            AttentionOperatorProviderIntegrationBundle(
                bundle_id="checkpoint.084.foreign.catalog.v1",
                operation_catalog=load_packaged_attention_operator_catalog(),
                registrations=(registration,),
                scoring_manifest=manifest,
                package_loader=values["loader"],
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_bundle_rejects_unbound_or_drifted_scoring_declarations(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])

        with self.assertRaisesRegex(SchemaError, "before declaration"):
            AttentionOperatorProviderIntegrationBundle(
                bundle_id="checkpoint.084.unbound.scoring.v1",
                operation_catalog=values["catalog"],
                registrations=(declared_registration(values),),
                scoring_manifest=manifest,
                package_loader=values["loader"],
            )

        _, registration = scored_registration(values)
        drifted_manifest = scoring_manifest(values["spec"], score=74)
        with self.assertRaisesRegex(SchemaError, "differs from manifest policy"):
            AttentionOperatorProviderIntegrationBundle(
                bundle_id="checkpoint.084.drifted.scoring.v1",
                operation_catalog=values["catalog"],
                registrations=(registration,),
                scoring_manifest=drifted_manifest,
                package_loader=values["loader"],
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_loader_id_and_stable_type_are_fingerprint_authority(self):
        values = bootstrap_components()
        first = provider_bundle(values)
        alternate_loader = AlternateFakePackageLoader(values["events"])
        second = provider_bundle(values, loader=alternate_loader)

        self.assertEqual(first.package_loader_id, second.package_loader_id)
        self.assertNotEqual(first.package_loader_type, second.package_loader_type)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(alternate_loader.version_calls, 0)

        with self.assertRaisesRegex(SchemaError, "loader_id is invalid"):
            provider_bundle(values, loader=EmptyIdPackageLoader())

        class LocalPackageLoader(EmptyIdPackageLoader):
            loader_id = "checkpoint-084-local-loader-v1"

        with self.assertRaisesRegex(SchemaError, "loader type is not stable"):
            provider_bundle(values, loader=LocalPackageLoader())

    def test_loader_identity_drift_fails_before_probe_or_publish(self):
        values = bootstrap_components()
        bundle = provider_bundle(values)
        original_fingerprint = bundle.fingerprint
        values["loader"].loader_id = "checkpoint-084-drifted-loader-v2"

        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            install_attention_operator_provider_integration_bundle(
                bundle,
                expected_generation=self.original.generation,
            )

        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, self.original.generation)
        self.assertEqual(bundle.fingerprint, original_fingerprint)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_installed_registry_revalidates_loader_identity_at_plan_time(self):
        values = bootstrap_components()
        bundle = provider_bundle(values)
        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )
        values["loader"].loader_id = "checkpoint-084-runtime-drift-v2"

        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            installed.registry.resolvers[0][1].explain(
                group_plan(),
                "npu:0",
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_snapshot_rejects_bundle_authority_drift(self):
        values = bootstrap_components()
        bundle = provider_bundle(values)
        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )

        with self.assertRaisesRegex(SchemaError, "differs from operation catalog"):
            replace(
                installed,
                provider_integration_bundle_binding=replace(
                    bundle.binding,
                    catalog_fingerprint="f" * 64,
                ),
            )
        with self.assertRaisesRegex(SchemaError, "differs from scoring manifest"):
            replace(
                installed,
                provider_integration_bundle_binding=replace(
                    bundle.binding,
                    scoring_manifest_fingerprint="e" * 64,
                ),
            )
        provider_id, operation_id, _ = bundle.binding.registration_bindings[0]
        with self.assertRaisesRegex(SchemaError, "differs from runtime declarations"):
            replace(
                installed,
                provider_integration_bundle_binding=replace(
                    bundle.binding,
                    registration_bindings=(
                        (provider_id, operation_id, "d" * 64),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
