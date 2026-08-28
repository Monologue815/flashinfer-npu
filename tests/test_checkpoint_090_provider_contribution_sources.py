from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSource,
    AttentionOperatorProviderContributionSourceRegistry,
    assemble_attention_operator_provider_integration_sources,
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
from tests.test_checkpoint_088_provider_contributions import (
    provider_contribution,
)
from tests.test_checkpoint_089_provider_contribution_manifest import (
    contributions_and_manifest,
)


class FakeContributionFactory:
    def __init__(self, contribution, events, factory_id):
        self.contribution = contribution
        self.events = events
        self.factory_id = factory_id
        self.build_calls = 0
        self.drift_after_build = False

    def build_contribution(self):
        self.build_calls += 1
        self.events.append("build:%s" % self.contribution.provider_id)
        if self.drift_after_build:
            self.factory_id = "drifted-after-build"
        return self.contribution


def source_for(contribution, factory_events, *, factory=None):
    if factory is None:
        factory = FakeContributionFactory(
            contribution,
            factory_events,
            "%s.attention.contribution.factory.v1" % contribution.provider_id,
        )
    source = AttentionOperatorProviderContributionSource(
        source_id="%s.attention.contribution.source.v1" % contribution.provider_id,
        source_version="1.0.0",
        provider_id=contribution.provider_id,
        contribution_id=contribution.contribution_id,
        factory=factory,
    )
    return source, factory


def sources_registry(values, *, registry_id=None, reverse=False):
    contributions, manifest = contributions_and_manifest(values)
    factory_events = []
    cann_source, cann_factory = source_for(contributions[0], factory_events)
    flash_source, flash_factory = source_for(contributions[1], factory_events)
    sources = (cann_source, flash_source)
    if reverse:
        sources = tuple(reversed(sources))
    registry = AttentionOperatorProviderContributionSourceRegistry(
        registry_id=(
            registry_id or "deployment.attention.adapter-sources.v1"
        ),
        sources=sources,
    )
    return {
        "contributions": contributions,
        "manifest": manifest,
        "registry": registry,
        "factory_events": factory_events,
        "cann_source": cann_source,
        "flash_source": flash_source,
        "cann_factory": cann_factory,
        "flash_factory": flash_factory,
    }


def source_bundle(values, *, registry_id=None, reverse=False):
    source_values = sources_registry(
        values,
        registry_id=registry_id,
        reverse=reverse,
    )
    source_values["bundle"] = (
        assemble_attention_operator_provider_integration_sources(
            bundle_id="checkpoint.090.provider-sources.bundle.v1",
            catalog_name="checkpoint-090-provider-sources-catalog",
            scoring_manifest_id="checkpoint.090.provider-sources.scoring.v1",
            source_registry=source_values["registry"],
            approval_manifest=source_values["manifest"],
        )
    )
    return source_values


class ProviderContributionSourceCheckpoint(unittest.TestCase):
    """Explicit adapter factories materialize only an approved source set."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_registry_is_canonical_non_executable_and_does_not_build(self):
        values = two_provider_assembly_inputs()
        first = sources_registry(values)
        second = sources_registry(values, reverse=True)

        self.assertEqual(
            first["registry"].fingerprint,
            second["registry"].fingerprint,
        )
        self.assertEqual(first["registry"].binding, second["registry"].binding)
        self.assertEqual(first["factory_events"], [])
        self.assertEqual(first["cann_factory"].build_calls, 0)
        self.assertEqual(first["flash_factory"].build_calls, 0)
        encoded = str(first["registry"].binding.to_dict())
        self.assertNotIn("build_contribution", encoded)
        self.assertNotIn("package_loader", encoded)
        self.assertNotIn("callable", encoded)
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_exact_manifest_materializes_each_source_once_in_canonical_order(self):
        values = two_provider_assembly_inputs()
        source_values = sources_registry(values, reverse=True)

        observed = source_values["registry"].materialize(
            source_values["manifest"]
        )

        self.assertEqual(
            tuple(item.binding for item in observed),
            source_values["manifest"].contribution_bindings,
        )
        self.assertEqual(
            source_values["factory_events"],
            ["build:cann", "build:flash_attention_npu"],
        )
        self.assertEqual(source_values["cann_factory"].build_calls, 1)
        self.assertEqual(source_values["flash_factory"].build_calls, 1)
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_source_set_mismatch_rejects_before_any_factory_call(self):
        values = two_provider_assembly_inputs()
        source_values = sources_registry(values)
        incomplete = AttentionOperatorProviderContributionSourceRegistry(
            registry_id="deployment.attention.incomplete-sources.v1",
            sources=(source_values["cann_source"],),
        )

        with self.assertRaisesRegex(SchemaError, "source set differs"):
            incomplete.materialize(source_values["manifest"])

        self.assertEqual(source_values["factory_events"], [])
        self.assertEqual(source_values["cann_factory"].build_calls, 0)
        self.assertEqual(source_values["flash_factory"].build_calls, 0)
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_factory_identity_and_result_drift_fail_closed(self):
        values = two_provider_assembly_inputs()
        source_values = sources_registry(values)

        source_values["cann_factory"].factory_id = "drifted-before-build"
        with self.assertRaisesRegex(SchemaError, "factory identity changed"):
            source_values["registry"].materialize(source_values["manifest"])
        self.assertEqual(source_values["factory_events"], [])

        fresh = sources_registry(values)
        fresh["cann_factory"].drift_after_build = True
        with self.assertRaisesRegex(SchemaError, "factory identity changed"):
            fresh["registry"].materialize(fresh["manifest"])
        self.assertEqual(fresh["factory_events"], ["build:cann"])
        self.assertEqual(fresh["flash_factory"].build_calls, 0)

        wrong_events = []
        wrong_factory = FakeContributionFactory(
            source_values["contributions"][1],
            wrong_events,
            "cann.attention.wrong-result.factory.v1",
        )
        wrong_source, _ = source_for(
            source_values["contributions"][0],
            wrong_events,
            factory=wrong_factory,
        )
        with self.assertRaisesRegex(SchemaError, "result identity differs"):
            wrong_source.materialize()
        self.assertEqual(wrong_factory.build_calls, 1)
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_source_registry_provenance_changes_final_bundle_identity_only(self):
        values = two_provider_assembly_inputs()
        first = source_bundle(
            values,
            registry_id="deployment.attention.adapter-sources.release.v1",
        )
        second = source_bundle(
            values,
            registry_id="deployment.attention.adapter-sources.release.v2",
        )
        first_bundle, second_bundle = first["bundle"], second["bundle"]

        self.assertEqual(
            first_bundle.operation_catalog,
            second_bundle.operation_catalog,
        )
        self.assertEqual(
            first_bundle.scoring_manifest,
            second_bundle.scoring_manifest,
        )
        self.assertEqual(
            first_bundle.contribution_bindings,
            second_bundle.contribution_bindings,
        )
        self.assertEqual(
            tuple(item.fingerprint for item in first_bundle.registrations),
            tuple(item.fingerprint for item in second_bundle.registrations),
        )
        self.assertNotEqual(
            first["registry"].fingerprint,
            second["registry"].fingerprint,
        )
        self.assertNotEqual(first_bundle.fingerprint, second_bundle.fingerprint)

    def test_bundle_and_binding_reject_incomplete_source_provenance(self):
        values = two_provider_assembly_inputs()
        source_values = source_bundle(values)
        bundle = source_values["bundle"]
        incomplete = AttentionOperatorProviderContributionSourceRegistry(
            registry_id="deployment.attention.incomplete-provenance.v1",
            sources=(source_values["cann_source"],),
        ).binding

        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            replace(
                bundle,
                contribution_source_registry_binding=incomplete,
            )
        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            replace(
                bundle.binding,
                contribution_source_registry_binding=incomplete,
            )

    def test_source_assembled_bundle_installs_and_selects_cann(self):
        values = two_provider_assembly_inputs()
        source_values = source_bundle(values)
        bundle = source_values["bundle"]

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
            source_values["registry"].binding,
        )
        self.assertEqual(
            source_values["factory_events"],
            ["build:cann", "build:flash_attention_npu"],
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
