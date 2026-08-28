from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionManifest,
    AttentionOperatorProviderContributionManifestLimits,
    assemble_attention_operator_provider_integration_contributions,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
    load_attention_operator_provider_contribution_manifest,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_087_provider_bundle_assembly import (
    two_provider_assembly_inputs,
)
from tests.test_checkpoint_088_provider_contributions import (
    contribution_bundle,
    provider_contribution,
)


def contributions_and_manifest(values, *, manifest_id=None, reverse=False):
    contributions = (
        provider_contribution(values, "cann"),
        provider_contribution(values, "flash_attention_npu"),
    )
    bindings = tuple(item.binding for item in contributions)
    if reverse:
        bindings = tuple(reversed(bindings))
    manifest = AttentionOperatorProviderContributionManifest(
        manifest_id=(
            manifest_id or "deployment.attention.contributions.approval.v1"
        ),
        contribution_bindings=bindings,
    )
    return contributions, manifest


def approved_bundle(values, *, manifest_id=None, reverse=False):
    contributions, manifest = contributions_and_manifest(
        values,
        manifest_id=manifest_id,
        reverse=reverse,
    )
    if reverse:
        contributions = tuple(reversed(contributions))
    bundle = assemble_attention_operator_provider_integration_contributions(
        bundle_id="checkpoint.089.approved.bundle.v1",
        catalog_name="checkpoint-089-approved-catalog",
        scoring_manifest_id="checkpoint.089.approved.scoring.v1",
        contributions=contributions,
        approval_manifest=manifest,
    )
    return contributions, manifest, bundle


class ProviderContributionManifestCheckpoint(unittest.TestCase):
    """Deployment approval is exact, bounded and carried by bundle identity."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_manifest_is_canonical_non_executable_and_bounded_json(self):
        values = two_provider_assembly_inputs()
        _, first = contributions_and_manifest(values)
        _, second = contributions_and_manifest(values, reverse=True)

        loaded, usage = load_attention_operator_provider_contribution_manifest(
            first.to_json()
        )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(loaded, first)
        self.assertEqual(loaded.fingerprint, first.fingerprint)
        self.assertGreater(usage.encoded_bytes, 0)
        encoded = first.to_dict()
        self.assertEqual(
            tuple(item["contribution_id"] for item in encoded["contribution_bindings"]),
            tuple(
                sorted(
                    item.contribution_id for item in first.contribution_bindings
                )
            ),
        )
        self.assertNotIn("package_loader", first.to_json())
        self.assertNotIn("callable", first.to_json())
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_manifest_bound_assembly_is_exact_and_order_independent(self):
        values = two_provider_assembly_inputs()

        _, manifest, first = approved_bundle(values)
        _, reversed_manifest, second = approved_bundle(values, reverse=True)

        self.assertEqual(manifest, reversed_manifest)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.contribution_manifest_id, manifest.manifest_id)
        self.assertEqual(
            first.contribution_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertEqual(
            first.binding.contribution_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_missing_orphan_and_drift_fail_before_package_observation(self):
        values = two_provider_assembly_inputs()
        contributions, manifest = contributions_and_manifest(values)

        with self.assertRaisesRegex(SchemaError, "missing=.*flash_attention_npu"):
            assemble_attention_operator_provider_integration_contributions(
                bundle_id="checkpoint.089.missing.bundle.v1",
                catalog_name="checkpoint-089-missing-catalog",
                scoring_manifest_id="checkpoint.089.missing.scoring.v1",
                contributions=(contributions[0],),
                approval_manifest=manifest,
            )
        cann_only = AttentionOperatorProviderContributionManifest(
            manifest_id="deployment.attention.cann-only.approval.v1",
            contribution_bindings=(contributions[0].binding,),
        )
        with self.assertRaisesRegex(SchemaError, "orphan=.*flash_attention_npu"):
            assemble_attention_operator_provider_integration_contributions(
                bundle_id="checkpoint.089.orphan.bundle.v1",
                catalog_name="checkpoint-089-orphan-catalog",
                scoring_manifest_id="checkpoint.089.orphan.scoring.v1",
                contributions=contributions,
                approval_manifest=cann_only,
            )
        drifted_binding = replace(
            contributions[0].binding,
            contribution_fingerprint="0" * 64,
        )
        drifted = AttentionOperatorProviderContributionManifest(
            manifest_id="deployment.attention.drifted.approval.v1",
            contribution_bindings=(drifted_binding, contributions[1].binding),
        )
        with self.assertRaisesRegex(SchemaError, "drifted=.*cann"):
            assemble_attention_operator_provider_integration_contributions(
                bundle_id="checkpoint.089.drifted.bundle.v1",
                catalog_name="checkpoint-089-drifted-catalog",
                scoring_manifest_id="checkpoint.089.drifted.scoring.v1",
                contributions=contributions,
                approval_manifest=drifted,
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_manifest_identity_changes_final_bundle_identity_only(self):
        values = two_provider_assembly_inputs()
        _, first_manifest, first = approved_bundle(
            values,
            manifest_id="deployment.attention.approval.release.v1",
        )
        _, second_manifest, second = approved_bundle(
            values,
            manifest_id="deployment.attention.approval.release.v2",
        )

        self.assertNotEqual(first_manifest.fingerprint, second_manifest.fingerprint)
        self.assertEqual(first.operation_catalog, second.operation_catalog)
        self.assertEqual(first.scoring_manifest, second.scoring_manifest)
        self.assertEqual(first.contribution_bindings, second.contribution_bindings)
        self.assertEqual(
            tuple(item.fingerprint for item in first.registrations),
            tuple(item.fingerprint for item in second.registrations),
        )
        self.assertEqual(
            first.package_loader.routing_fingerprint,
            second.package_loader.routing_fingerprint,
        )
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_manifest_limits_and_bundle_identity_pairs_fail_closed(self):
        values = two_provider_assembly_inputs()
        _, manifest = contributions_and_manifest(values)

        with self.assertRaisesRegex(SchemaError, "contributions exceed limit"):
            load_attention_operator_provider_contribution_manifest(
                manifest.to_json(),
                manifest_limits=AttentionOperatorProviderContributionManifestLimits(
                    max_contributions=1,
                ),
            )
        unapproved = contribution_bundle(values)
        cann_only = AttentionOperatorProviderContributionManifest(
            manifest_id="deployment.attention.incomplete.v1",
            contribution_bindings=(manifest.contribution_bindings[0],),
        )
        with self.assertRaisesRegex(SchemaError, "differs from contribution manifest"):
            replace(
                unapproved,
                contribution_manifest=cann_only,
            )
        with self.assertRaisesRegex(SchemaError, "identity is incomplete"):
            replace(
                unapproved.binding,
                contribution_manifest_fingerprint="f" * 64,
            )

    def test_approved_bundle_installs_and_selects_declared_winner(self):
        values = two_provider_assembly_inputs()
        _, manifest, bundle = approved_bundle(values)

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
        binding = installed.provider_integration_bundle_binding
        self.assertEqual(binding.contribution_manifest_id, manifest.manifest_id)
        self.assertEqual(
            binding.contribution_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
