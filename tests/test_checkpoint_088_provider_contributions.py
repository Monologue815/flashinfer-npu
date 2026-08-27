from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderIntegrationContribution,
    AttentionOperatorProviderIntegrationContributionBinding,
    assemble_attention_operator_provider_integration_contributions,
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


def provider_contribution(
    values,
    provider_id,
    *,
    contribution_id=None,
):
    if provider_id == "cann":
        operation = values["cann"]["operation"]
        spec = values["cann"]["spec"]
        policy = values["policies"][0]
        route = values["routes"][0]
    else:
        operation = values["flash_operation"]
        spec = values["flash_spec"]
        policy = values["policies"][1]
        route = values["routes"][1]
    return AttentionOperatorProviderIntegrationContribution(
        contribution_id=(
            contribution_id
            or "%s.attention.provider.contribution.v1" % provider_id
        ),
        operations=(operation,),
        runtime_specs=(spec,),
        scoring_policies=(policy,),
        package_loader_routes=(route,),
    )


def contribution_bundle(values, *, reverse=False):
    contributions = (
        provider_contribution(values, "cann"),
        provider_contribution(values, "flash_attention_npu"),
    )
    if reverse:
        contributions = tuple(reversed(contributions))
    return assemble_attention_operator_provider_integration_contributions(
        bundle_id="checkpoint.088.provider.contributions.bundle.v1",
        catalog_name="checkpoint-088-provider-contributions-catalog",
        scoring_manifest_id="checkpoint.088.provider.contributions.scoring.v1",
        contributions=contributions,
    )


class ProviderContributionCheckpoint(unittest.TestCase):
    """Provider-owned slices compose into one deployment-owned bundle."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_single_contribution_is_complete_bound_and_side_effect_free(self):
        values = two_provider_assembly_inputs()
        source_spec = values["cann"]["spec"]

        contribution = provider_contribution(values, "cann")

        self.assertIsNone(source_spec.plan_scorer)
        self.assertIs(
            contribution.runtime_specs[0].plan_scorer,
            values["policies"][0],
        )
        self.assertEqual(contribution.provider_id, "cann")
        self.assertIsInstance(
            contribution.binding,
            AttentionOperatorProviderIntegrationContributionBinding,
        )
        row = contribution.operation_bindings[0]
        self.assertEqual(row[:2], contribution.identities[0])
        self.assertEqual(row[2], values["cann"]["operation"].fingerprint)
        self.assertEqual(row[4], values["policies"][0].fingerprint)
        self.assertEqual(row[5], values["routes"][0].binding.fingerprint)
        contribution.validate()
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_contribution_rejects_mixed_provider_and_incomplete_sets(self):
        values = two_provider_assembly_inputs()

        with self.assertRaisesRegex(SchemaError, "exactly one provider"):
            AttentionOperatorProviderIntegrationContribution(
                contribution_id="checkpoint.088.mixed.provider.v1",
                operations=values["operations"],
                runtime_specs=values["specs"],
                scoring_policies=values["policies"],
                package_loader_routes=values["routes"],
            )
        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            AttentionOperatorProviderIntegrationContribution(
                contribution_id="checkpoint.088.missing.policy.v1",
                operations=(values["cann"]["operation"],),
                runtime_specs=(values["cann"]["spec"],),
                scoring_policies=(values["policies"][1],),
                package_loader_routes=(values["routes"][0],),
            )
        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            AttentionOperatorProviderIntegrationContribution(
                contribution_id="checkpoint.088.missing.route.v1",
                operations=(values["cann"]["operation"],),
                runtime_specs=(values["cann"]["spec"],),
                scoring_policies=(values["policies"][0],),
                package_loader_routes=(values["routes"][1],),
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_multi_contribution_assembly_is_canonical_and_regenerates_declarations(self):
        values = two_provider_assembly_inputs()
        cann = provider_contribution(values, "cann")
        flash = provider_contribution(values, "flash_attention_npu")

        first = contribution_bundle(values)
        second = contribution_bundle(values, reverse=True)

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            tuple(
                item.contribution_id for item in first.contribution_bindings
            ),
            tuple(sorted((cann.contribution_id, flash.contribution_id))),
        )
        self.assertEqual(
            {
                identity
                for binding in first.contribution_bindings
                for identity in binding.identities
            },
            set(first.binding.identities),
        )
        local_declarations = {
            row[:2]: row[3]
            for contribution in (cann, flash)
            for row in contribution.operation_bindings
        }
        for registration in first.registrations:
            identity = (
                registration.declaration.provider_id,
                registration.declaration.operation_id,
            )
            self.assertEqual(
                registration.declaration.catalog_fingerprint,
                first.operation_catalog.fingerprint,
            )
            self.assertNotEqual(
                registration.declaration.fingerprint,
                local_declarations[identity],
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_contribution_fingerprints_are_part_of_final_bundle_identity(self):
        values = two_provider_assembly_inputs()
        cann_v1 = provider_contribution(
            values,
            "cann",
            contribution_id="cann.attention.contribution.release.v1",
        )
        cann_v2 = provider_contribution(
            values,
            "cann",
            contribution_id="cann.attention.contribution.release.v2",
        )
        flash = provider_contribution(values, "flash_attention_npu")

        def assemble(cann):
            return assemble_attention_operator_provider_integration_contributions(
                bundle_id="checkpoint.088.identity.bundle.v1",
                catalog_name="checkpoint-088-identity-catalog",
                scoring_manifest_id="checkpoint.088.identity.scoring.v1",
                contributions=(cann, flash),
            )

        first, second = assemble(cann_v1), assemble(cann_v2)

        self.assertNotEqual(cann_v1.fingerprint, cann_v2.fingerprint)
        self.assertEqual(first.operation_catalog, second.operation_catalog)
        self.assertEqual(first.scoring_manifest, second.scoring_manifest)
        self.assertEqual(
            tuple(item.fingerprint for item in first.registrations),
            tuple(item.fingerprint for item in second.registrations),
        )
        self.assertEqual(
            first.package_loader.routing_fingerprint,
            second.package_loader.routing_fingerprint,
        )
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.binding, second.binding)

    def test_drift_duplicate_and_overlap_fail_before_package_probe(self):
        values = two_provider_assembly_inputs()
        cann = provider_contribution(values, "cann")
        flash = provider_contribution(values, "flash_attention_npu")

        values["cann"]["loader"].loader_id = "drifted-loader-id"
        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            cann.validate()
        values["cann"]["loader"].loader_id = "checkpoint-019-loader-v1"

        with self.assertRaisesRegex(SchemaError, "ids duplicate"):
            assemble_attention_operator_provider_integration_contributions(
                bundle_id="checkpoint.088.duplicate.id.bundle.v1",
                catalog_name="checkpoint-088-duplicate-id-catalog",
                scoring_manifest_id="checkpoint.088.duplicate.id.scoring.v1",
                contributions=(cann, cann),
            )
        overlapping = provider_contribution(
            values,
            "cann",
            contribution_id="cann.attention.overlapping.contribution.v1",
        )
        with self.assertRaisesRegex(SchemaError, "duplicate operation_id"):
            assemble_attention_operator_provider_integration_contributions(
                bundle_id="checkpoint.088.overlap.bundle.v1",
                catalog_name="checkpoint-088-overlap-catalog",
                scoring_manifest_id="checkpoint.088.overlap.scoring.v1",
                contributions=(cann, overlapping, flash),
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_bundle_rejects_incomplete_contribution_provenance(self):
        values = two_provider_assembly_inputs()
        cann = provider_contribution(values, "cann")
        flash = provider_contribution(values, "flash_attention_npu")
        bundle = contribution_bundle(values)

        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            replace(bundle, contribution_bindings=(cann.binding,))
        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            replace(bundle.binding, contribution_bindings=(flash.binding,))

    def test_installed_contribution_bundle_selects_cann_by_declared_score(self):
        values = two_provider_assembly_inputs()
        bundle = contribution_bundle(values)

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
            installed.provider_integration_bundle_binding.contribution_bindings,
            bundle.contribution_bindings,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
