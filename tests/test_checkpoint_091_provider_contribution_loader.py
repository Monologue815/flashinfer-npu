from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSourceDeclaration,
    AttentionOperatorProviderContributionSourceDeclarationRegistry,
    assemble_attention_operator_provider_integration_source_declarations,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_087_provider_bundle_assembly import (
    two_provider_assembly_inputs,
)
from tests.test_checkpoint_089_provider_contribution_manifest import (
    contributions_and_manifest,
)
from tests.test_checkpoint_090_provider_contribution_sources import (
    FakeContributionFactory,
)


def stable_type_name(value):
    value_type = type(value)
    return "%s.%s" % (value_type.__module__, value_type.__qualname__)


class FakeContributionFactoryLoader:
    def __init__(self, versions, factories, events):
        self.loader_id = "checkpoint-091-contribution-factory-loader-v1"
        self.versions = dict(versions)
        self.factories = dict(factories)
        self.events = events
        self.version_calls = 0
        self.resolve_calls = 0
        self.drift_after_version = False
        self.drift_after_resolve = False

    def package_version(self, package_name):
        self.version_calls += 1
        self.events.append("package_version:%s" % package_name)
        if self.drift_after_version:
            self.loader_id = "drifted-after-version"
        return self.versions[package_name]

    def resolve_factory(self, factory_path):
        self.resolve_calls += 1
        self.events.append("resolve_factory:%s" % factory_path)
        if self.drift_after_resolve:
            self.loader_id = "drifted-after-resolve"
        return self.factories[factory_path]


def declaration_values(values, *, reverse=False, source_version="1.0.0"):
    contributions, manifest = contributions_and_manifest(values)
    factory_events = []
    cann_factory = FakeContributionFactory(
        contributions[0],
        factory_events,
        "cann.attention.contribution.factory.v1",
    )
    flash_factory = FakeContributionFactory(
        contributions[1],
        factory_events,
        "flash_attention_npu.attention.contribution.factory.v1",
    )
    cann_package = "checkpoint-091-cann-adapter"
    flash_package = "checkpoint-091-flash-attention-npu-adapter"
    cann_path = "checkpoint_091_cann_adapter.build_attention_contribution"
    flash_path = (
        "checkpoint_091_flash_attention_npu_adapter.build_attention_contribution"
    )
    declarations = (
        AttentionOperatorProviderContributionSourceDeclaration(
            source_id="cann.attention.contribution.source.v1",
            source_version=source_version,
            provider_id="cann",
            contribution_id=contributions[0].contribution_id,
            adapter_package_name=cann_package,
            supported_package_versions=("1.0.0",),
            factory_path=cann_path,
            factory_id=cann_factory.factory_id,
            factory_type=stable_type_name(cann_factory),
        ),
        AttentionOperatorProviderContributionSourceDeclaration(
            source_id="flash_attention_npu.attention.contribution.source.v1",
            source_version=source_version,
            provider_id="flash_attention_npu",
            contribution_id=contributions[1].contribution_id,
            adapter_package_name=flash_package,
            supported_package_versions=("1.0.0",),
            factory_path=flash_path,
            factory_id=flash_factory.factory_id,
            factory_type=stable_type_name(flash_factory),
        ),
    )
    if reverse:
        declarations = tuple(reversed(declarations))
    loader_events = []
    loader = FakeContributionFactoryLoader(
        versions={cann_package: "1.0.0", flash_package: "1.0.0"},
        factories={cann_path: cann_factory, flash_path: flash_factory},
        events=loader_events,
    )
    registry = AttentionOperatorProviderContributionSourceDeclarationRegistry(
        registry_id="deployment.attention.adapter-source-declarations.v1",
        declarations=declarations,
        factory_loader=loader,
    )
    return {
        "contributions": contributions,
        "manifest": manifest,
        "factory_events": factory_events,
        "loader_events": loader_events,
        "cann_factory": cann_factory,
        "flash_factory": flash_factory,
        "cann_package": cann_package,
        "flash_package": flash_package,
        "cann_path": cann_path,
        "flash_path": flash_path,
        "declarations": tuple(sorted(declarations, key=lambda item: item.source_id)),
        "loader": loader,
        "registry": registry,
    }


def declaration_bundle(values, *, source_version="1.0.0"):
    declared = declaration_values(values, source_version=source_version)
    declared["bundle"] = (
        assemble_attention_operator_provider_integration_source_declarations(
            bundle_id="checkpoint.091.declared-sources.bundle.v1",
            catalog_name="checkpoint-091-declared-sources-catalog",
            scoring_manifest_id="checkpoint.091.declared-sources.scoring.v1",
            source_declarations=declared["registry"],
            approval_manifest=declared["manifest"],
        )
    )
    return declared


class ProviderContributionLoaderCheckpoint(unittest.TestCase):
    """Adapter packages load only through exact reviewed declarations."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_declaration_registry_is_canonical_and_observation_free(self):
        values = two_provider_assembly_inputs()
        first = declaration_values(values)
        second = declaration_values(values, reverse=True)

        self.assertEqual(first["registry"].fingerprint, second["registry"].fingerprint)
        self.assertEqual(first["loader_events"], [])
        self.assertEqual(first["factory_events"], [])
        self.assertEqual(first["loader"].version_calls, 0)
        self.assertEqual(first["loader"].resolve_calls, 0)
        encoded = str(first["registry"].to_dict())
        self.assertNotIn("factory_events", encoded)
        self.assertNotIn("package_loader", encoded)
        self.assertNotIn("provider handle", encoded)
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_manifest_mismatch_fails_before_adapter_package_observation(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        incomplete = AttentionOperatorProviderContributionSourceDeclarationRegistry(
            registry_id="deployment.attention.incomplete-declarations.v1",
            declarations=(declared["declarations"][0],),
            factory_loader=declared["loader"],
        )

        with self.assertRaisesRegex(SchemaError, "declarations differ"):
            incomplete.load_sources(declared["manifest"])

        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(declared["loader"].version_calls, 0)
        self.assertEqual(declared["loader"].resolve_calls, 0)

    def test_loading_records_exact_adapter_origin_without_building(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values, reverse=True)

        sources = declared["registry"].load_sources(declared["manifest"])

        self.assertEqual(
            declared["loader_events"],
            [
                "package_version:%s" % declared["cann_package"],
                "resolve_factory:%s" % declared["cann_path"],
                "package_version:%s" % declared["flash_package"],
                "resolve_factory:%s" % declared["flash_path"],
            ],
        )
        self.assertEqual(declared["loader"].version_calls, 2)
        self.assertEqual(declared["loader"].resolve_calls, 2)
        self.assertEqual(declared["factory_events"], [])
        for source, declaration in zip(sources.sources, declared["declarations"]):
            origin = source.binding.origin_binding
            self.assertIsNotNone(origin)
            self.assertEqual(
                origin.observed_package_version,
                "1.0.0",
            )
            self.assertEqual(origin.factory_path, declaration.factory_path)
            self.assertEqual(
                origin.declaration_fingerprint,
                declaration.fingerprint,
            )
            self.assertEqual(
                origin.factory_loader_id,
                declared["loader"].loader_id,
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_unsupported_version_and_loader_drift_fail_before_factory_build(self):
        values = two_provider_assembly_inputs()
        unsupported = declaration_values(values)
        unsupported["loader"].versions[unsupported["cann_package"]] = "2.0.0"
        with self.assertRaisesRegex(SchemaError, "version is unsupported"):
            unsupported["registry"].load_sources(unsupported["manifest"])
        self.assertEqual(unsupported["loader"].resolve_calls, 0)
        self.assertEqual(unsupported["factory_events"], [])

        drifted = declaration_values(values)
        drifted["loader"].drift_after_version = True
        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            drifted["registry"].load_sources(drifted["manifest"])
        self.assertEqual(drifted["loader"].resolve_calls, 0)
        self.assertEqual(drifted["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_resolved_factory_identity_must_match_declaration(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        declared["loader"].factories[declared["cann_path"]] = (
            declared["flash_factory"]
        )

        with self.assertRaisesRegex(SchemaError, "factory identity differs"):
            declared["registry"].load_sources(declared["manifest"])

        self.assertEqual(declared["loader"].version_calls, 1)
        self.assertEqual(declared["loader"].resolve_calls, 1)
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_declaration_identity_is_transitive_in_final_bundle(self):
        values = two_provider_assembly_inputs()
        first = declaration_bundle(values, source_version="1.0.0")
        second = declaration_bundle(values, source_version="1.0.1")
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
            first_bundle.contribution_source_registry_binding.registry_fingerprint,
            second_bundle.contribution_source_registry_binding.registry_fingerprint,
        )
        self.assertNotEqual(first_bundle.fingerprint, second_bundle.fingerprint)

    def test_declared_sources_assemble_install_and_select_cann(self):
        values = two_provider_assembly_inputs()
        declared = declaration_bundle(values)
        bundle = declared["bundle"]

        self.assertEqual(
            declared["factory_events"],
            ["build:cann", "build:flash_attention_npu"],
        )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        for binding in (
            bundle.contribution_source_registry_binding.source_bindings
        ):
            self.assertIsNotNone(binding.origin_binding)

        installed = install_attention_operator_provider_integration_bundle(
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
        self.assertEqual(
            installed.provider_integration_bundle_binding
            .contribution_source_registry_binding,
            bundle.contribution_source_registry_binding,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
