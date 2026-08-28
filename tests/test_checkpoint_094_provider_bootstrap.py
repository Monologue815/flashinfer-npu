from dataclasses import replace
import json
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderIntegrationBootstrapManifest,
    assemble_attention_operator_provider_integration_bootstrap,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bootstrap,
    install_attention_operator_runtime_resolvers,
    load_attention_operator_provider_integration_bootstrap_manifest,
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
from tests.test_checkpoint_093_source_declaration_manifest import source_manifest


def bootstrap_values(*, bootstrap_id="deployment.attention.bootstrap.release.v1"):
    provider_values = two_provider_assembly_inputs()
    declared = declaration_values(provider_values)
    source = source_manifest(declared)
    bootstrap = AttentionOperatorProviderIntegrationBootstrapManifest.from_inputs(
        bootstrap_id=bootstrap_id,
        bundle_id="deployment.attention.providers.bundle.v1",
        catalog_name="deployment-attention-provider-catalog",
        scoring_manifest_id="deployment.attention.provider-scoring.v1",
        source_manifest=source,
        approval_manifest=declared["manifest"],
        factory_loader=declared["loader"],
    )
    return provider_values, declared, source, bootstrap


class ProviderBootstrapCheckpoint(unittest.TestCase):
    """One data-only authority closes the provider deployment inputs."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_bootstrap_is_canonical_data_only_and_json_round_trips(self):
        values, declared, _, bootstrap = bootstrap_values()

        loaded, usage = (
            load_attention_operator_provider_integration_bootstrap_manifest(
                bootstrap.to_json()
            )
        )

        self.assertEqual(loaded, bootstrap)
        self.assertEqual(loaded.fingerprint, bootstrap.fingerprint)
        self.assertGreater(usage.encoded_bytes, 0)
        encoded = bootstrap.to_json()
        self.assertNotIn("declarations", json.loads(encoded))
        self.assertNotIn("build_contribution", encoded)
        self.assertNotIn("package_loader", encoded)
        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

        malformed = bootstrap.to_dict()
        malformed["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            load_attention_operator_provider_integration_bootstrap_manifest(
                json.dumps(malformed)
            )

    def test_input_drift_fails_before_adapter_package_observation(self):
        _, declared, source, bootstrap = bootstrap_values()
        different_source = replace(
            source,
            manifest_id="deployment.attention.adapter-source-declarations.v2",
        )

        with self.assertRaisesRegex(SchemaError, "source manifest differs"):
            assemble_attention_operator_provider_integration_bootstrap(
                bootstrap_manifest=bootstrap,
                source_manifest=different_source,
                factory_loader=declared["loader"],
                approval_manifest=declared["manifest"],
            )

        different_approval = replace(
            declared["manifest"],
            manifest_id="deployment.attention.contributions.approved.v2",
        )
        with self.assertRaisesRegex(SchemaError, "contribution manifest differs"):
            assemble_attention_operator_provider_integration_bootstrap(
                bootstrap_manifest=bootstrap,
                source_manifest=source,
                factory_loader=declared["loader"],
                approval_manifest=different_approval,
            )

        declared["loader"].loader_id = "drifted-loader-id"
        with self.assertRaisesRegex(SchemaError, "factory loader differs"):
            assemble_attention_operator_provider_integration_bootstrap(
                bootstrap_manifest=bootstrap,
                source_manifest=source,
                factory_loader=declared["loader"],
                approval_manifest=declared["manifest"],
            )

        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(declared["loader"].version_calls, 0)
        self.assertEqual(declared["loader"].resolve_calls, 0)

    def test_assembly_closes_bootstrap_identity_into_bundle_binding(self):
        _, declared, source, bootstrap = bootstrap_values()

        bundle = assemble_attention_operator_provider_integration_bootstrap(
            bootstrap_manifest=bootstrap,
            source_manifest=source,
            factory_loader=declared["loader"],
            approval_manifest=declared["manifest"],
        )

        self.assertEqual(bundle.bootstrap_manifest_id, bootstrap.bootstrap_id)
        self.assertEqual(
            bundle.bootstrap_manifest_fingerprint,
            bootstrap.fingerprint,
        )
        self.assertEqual(bundle.binding.bootstrap_manifest_id, bootstrap.bootstrap_id)
        self.assertEqual(
            bundle.binding.bootstrap_manifest_fingerprint,
            bootstrap.fingerprint,
        )
        for binding in bundle.contribution_source_registry_binding.source_bindings:
            origin = binding.origin_binding
            self.assertEqual(origin.declaration_manifest_id, source.manifest_id)
            self.assertEqual(
                origin.declaration_manifest_fingerprint,
                source.fingerprint,
            )
            self.assertEqual(origin.factory_loader_id, bootstrap.factory_loader_id)
            self.assertEqual(
                origin.factory_loader_type,
                bootstrap.factory_loader_type,
            )

    def test_bootstrap_release_identity_is_transitive_not_operational(self):
        _, first_declared, first_source, first = bootstrap_values(
            bootstrap_id="deployment.attention.bootstrap.release.v1"
        )
        first_bundle = assemble_attention_operator_provider_integration_bootstrap(
            bootstrap_manifest=first,
            source_manifest=first_source,
            factory_loader=first_declared["loader"],
            approval_manifest=first_declared["manifest"],
        )
        _, second_declared, second_source, second = bootstrap_values(
            bootstrap_id="deployment.attention.bootstrap.release.v2"
        )
        second_bundle = assemble_attention_operator_provider_integration_bootstrap(
            bootstrap_manifest=second,
            source_manifest=second_source,
            factory_loader=second_declared["loader"],
            approval_manifest=second_declared["manifest"],
        )

        self.assertEqual(first_bundle.operation_catalog, second_bundle.operation_catalog)
        self.assertEqual(first_bundle.scoring_manifest, second_bundle.scoring_manifest)
        self.assertEqual(
            first_bundle.contribution_bindings,
            second_bundle.contribution_bindings,
        )
        self.assertEqual(
            first_bundle.contribution_source_registry_binding,
            second_bundle.contribution_source_registry_binding,
        )
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first_bundle.fingerprint, second_bundle.fingerprint)

    def test_bundle_rejects_bootstrap_identity_mismatch(self):
        _, declared, source, bootstrap = bootstrap_values()
        bundle = assemble_attention_operator_provider_integration_bootstrap(
            bootstrap_manifest=bootstrap,
            source_manifest=source,
            factory_loader=declared["loader"],
            approval_manifest=declared["manifest"],
        )
        wrong = replace(bootstrap, bundle_id="deployment.attention.other-bundle.v1")

        with self.assertRaisesRegex(SchemaError, "bundle identity differs"):
            replace(bundle, bootstrap_manifest=wrong)

    def test_one_step_install_keeps_private_auto_plan_selection(self):
        values, declared, source, bootstrap = bootstrap_values()

        installed = install_attention_operator_provider_integration_bootstrap(
            bootstrap_manifest=bootstrap,
            source_manifest=source,
            factory_loader=declared["loader"],
            approval_manifest=declared["manifest"],
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

        binding = installed.provider_integration_bundle_binding
        self.assertEqual(binding.bootstrap_manifest_id, bootstrap.bootstrap_id)
        self.assertEqual(
            binding.bootstrap_manifest_fingerprint,
            bootstrap.fingerprint,
        )
        self.assertEqual(wrapper.plan_selection.provider_id, "cann")
        self.assertEqual(wrapper.plan_selection.plan_score, 73)
        self.assertEqual(
            wrapper.plan_selection.provider_integration_bundle_fingerprint,
            binding.bundle_fingerprint,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
