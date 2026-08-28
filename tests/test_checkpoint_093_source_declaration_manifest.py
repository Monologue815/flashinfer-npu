from dataclasses import replace
import json
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSourceDeclarationManifest,
    AttentionOperatorProviderContributionSourceDeclarationManifestLimits,
    AttentionOperatorProviderContributionSourceDeclarationRegistry,
    assemble_attention_operator_provider_integration_source_manifest,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
    load_attention_operator_provider_contribution_source_declaration_manifest,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_087_provider_bundle_assembly import (
    two_provider_assembly_inputs,
)
from tests.test_checkpoint_091_provider_contribution_loader import (
    declaration_values,
)


def source_manifest(declared, *, manifest_id=None, reverse=False):
    declarations = declared["declarations"]
    if reverse:
        declarations = tuple(reversed(declarations))
    return AttentionOperatorProviderContributionSourceDeclarationManifest(
        manifest_id=(
            manifest_id
            or "deployment.attention.adapter-source-declarations.v1"
        ),
        declarations=declarations,
    )


def manifest_bundle(values, *, manifest_id=None):
    declared = declaration_values(values)
    manifest = source_manifest(declared, manifest_id=manifest_id)
    declared["source_manifest"] = manifest
    declared["bundle"] = (
        assemble_attention_operator_provider_integration_source_manifest(
            bundle_id="checkpoint.093.source-manifest.bundle.v1",
            catalog_name="checkpoint-093-source-manifest-catalog",
            scoring_manifest_id="checkpoint.093.source-manifest.scoring.v1",
            source_manifest=manifest,
            factory_loader=declared["loader"],
            approval_manifest=declared["manifest"],
        )
    )
    return declared


class SourceDeclarationManifestCheckpoint(unittest.TestCase):
    """Adapter source declarations are bounded data before loader binding."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_manifest_is_canonical_data_only_and_json_round_trips(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        first = source_manifest(declared)
        second = source_manifest(declared, reverse=True)

        loaded, usage = (
            load_attention_operator_provider_contribution_source_declaration_manifest(
                first.to_json()
            )
        )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(loaded, first)
        self.assertEqual(loaded.fingerprint, first.fingerprint)
        self.assertGreater(usage.encoded_bytes, 0)
        encoded = first.to_json()
        self.assertNotIn("factory_loader", encoded)
        self.assertNotIn("build_contribution", encoded)
        self.assertNotIn("package_loader", encoded)
        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_manifest_limits_and_strict_fields_fail_closed(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        manifest = source_manifest(declared)

        with self.assertRaisesRegex(SchemaError, "declarations exceed limit"):
            load_attention_operator_provider_contribution_source_declaration_manifest(
                manifest.to_json(),
                manifest_limits=(
                    AttentionOperatorProviderContributionSourceDeclarationManifestLimits(
                        max_declarations=1,
                    )
                ),
            )
        expanded = replace(
            declared["declarations"][0],
            supported_package_versions=("1.0.0", "1.0.1"),
        )
        version_manifest = AttentionOperatorProviderContributionSourceDeclarationManifest(
            manifest_id="deployment.attention.adapter-versions.v1",
            declarations=(expanded,),
        )
        with self.assertRaisesRegex(SchemaError, "per-declaration limit"):
            load_attention_operator_provider_contribution_source_declaration_manifest(
                version_manifest.to_json(),
                manifest_limits=(
                    AttentionOperatorProviderContributionSourceDeclarationManifestLimits(
                        max_versions_per_declaration=1,
                    )
                ),
            )
        malformed = manifest.to_dict()
        malformed["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            load_attention_operator_provider_contribution_source_declaration_manifest(
                json.dumps(malformed)
            )

    def test_manifest_binds_loader_without_observation_then_reaches_origins(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        manifest = source_manifest(declared)

        registry = (
            AttentionOperatorProviderContributionSourceDeclarationRegistry.from_manifest(
                manifest,
                factory_loader=declared["loader"],
            )
        )

        self.assertEqual(registry.declaration_manifest_id, manifest.manifest_id)
        self.assertEqual(
            registry.declaration_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])

        sources = registry.load_sources(declared["manifest"])
        self.assertEqual(declared["loader"].version_calls, 2)
        self.assertEqual(declared["loader"].resolve_calls, 2)
        self.assertEqual(declared["factory_events"], [])
        for source in sources.sources:
            origin = source.binding.origin_binding
            self.assertEqual(
                origin.declaration_manifest_id,
                manifest.manifest_id,
            )
            self.assertEqual(
                origin.declaration_manifest_fingerprint,
                manifest.fingerprint,
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_manifest_identity_is_transitive_in_final_bundle(self):
        values = two_provider_assembly_inputs()
        first = manifest_bundle(
            values,
            manifest_id="deployment.attention.adapter-declarations.release.v1",
        )
        second = manifest_bundle(
            values,
            manifest_id="deployment.attention.adapter-declarations.release.v2",
        )
        first_bundle, second_bundle = first["bundle"], second["bundle"]

        self.assertEqual(first_bundle.operation_catalog, second_bundle.operation_catalog)
        self.assertEqual(first_bundle.scoring_manifest, second_bundle.scoring_manifest)
        self.assertEqual(
            first_bundle.contribution_bindings,
            second_bundle.contribution_bindings,
        )
        self.assertEqual(
            tuple(item.fingerprint for item in first_bundle.registrations),
            tuple(item.fingerprint for item in second_bundle.registrations),
        )
        self.assertNotEqual(
            first["source_manifest"].fingerprint,
            second["source_manifest"].fingerprint,
        )
        self.assertNotEqual(
            first_bundle.contribution_source_registry_binding.registry_fingerprint,
            second_bundle.contribution_source_registry_binding.registry_fingerprint,
        )
        self.assertNotEqual(first_bundle.fingerprint, second_bundle.fingerprint)

    def test_loaded_manifest_assembles_installs_and_selects_cann(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        source = source_manifest(declared)
        loaded, _ = (
            load_attention_operator_provider_contribution_source_declaration_manifest(
                source.to_json()
            )
        )

        bundle = assemble_attention_operator_provider_integration_source_manifest(
            bundle_id="checkpoint.093.loaded-manifest.bundle.v1",
            catalog_name="checkpoint-093-loaded-manifest-catalog",
            scoring_manifest_id="checkpoint.093.loaded-manifest.scoring.v1",
            source_manifest=loaded,
            factory_loader=declared["loader"],
            approval_manifest=declared["manifest"],
        )

        self.assertEqual(
            declared["factory_events"],
            ["build:cann", "build:flash_attention_npu"],
        )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="NHD",
            backend="auto",
        )
        plan = group_plan()
        wrapper.plan(
            plan.metadata.indptr,
            plan.metadata.indices,
            plan.metadata.last_page_len,
            plan.spec.num_qo_heads,
            plan.spec.num_kv_heads,
            plan.spec.head_dim_vo,
            plan.metadata.page_size,
            q_data_type=plan.spec.q_dtype,
            kv_data_type=plan.spec.kv_quant_spec,
            o_data_type=plan.spec.o_dtype,
        )

        self.assertEqual(wrapper.plan_selection.provider_id, "cann")
        self.assertEqual(wrapper.plan_selection.plan_score, 73)
        self.assertEqual(
            wrapper.plan_selection.provider_integration_bundle_fingerprint,
            bundle.fingerprint,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
