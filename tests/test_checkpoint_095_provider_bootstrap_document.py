from dataclasses import replace
import json
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionManifestLimits,
    AttentionOperatorProviderContributionSourceDeclarationManifestLimits,
    AttentionOperatorProviderIntegrationBootstrapDocument,
    assemble_attention_operator_provider_integration_bootstrap_document,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bootstrap_document,
    install_attention_operator_runtime_resolvers,
    load_attention_operator_provider_integration_bootstrap_document,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_094_provider_bootstrap import bootstrap_values


def document_values():
    values, declared, source, bootstrap = bootstrap_values()
    document = AttentionOperatorProviderIntegrationBootstrapDocument(
        bootstrap_manifest=bootstrap,
        source_manifest=source,
        contribution_manifest=declared["manifest"],
    )
    return values, declared, document


class ProviderBootstrapDocumentCheckpoint(unittest.TestCase):
    """A single bounded document carries the complete data-only deployment."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_document_is_canonical_data_only_and_round_trips(self):
        values, declared, document = document_values()

        loaded, usage = load_attention_operator_provider_integration_bootstrap_document(
            document.to_json()
        )

        self.assertEqual(loaded, document)
        self.assertEqual(loaded.fingerprint, document.fingerprint)
        self.assertGreater(usage.encoded_bytes, 0)
        encoded = document.to_json()
        self.assertNotIn("build_contribution", encoded)
        self.assertNotIn("provider_callable", encoded)
        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_nested_identity_drift_is_rejected_without_loader_observation(self):
        _, declared, document = document_values()
        different_source = replace(
            document.source_manifest,
            manifest_id="deployment.attention.adapter-source-declarations.v2",
        )

        with self.assertRaisesRegex(SchemaError, "source manifest differs"):
            replace(document, source_manifest=different_source)

        different_contributions = replace(
            document.contribution_manifest,
            manifest_id="deployment.attention.contributions.approval.v2",
        )
        with self.assertRaisesRegex(SchemaError, "contribution manifest differs"):
            replace(document, contribution_manifest=different_contributions)

        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])

    def test_nested_semantic_limits_are_enforced_during_json_loading(self):
        _, _, document = document_values()

        with self.assertRaisesRegex(SchemaError, "declarations exceed limit"):
            load_attention_operator_provider_integration_bootstrap_document(
                document.to_json(),
                source_manifest_limits=(
                    AttentionOperatorProviderContributionSourceDeclarationManifestLimits(
                        max_declarations=1
                    )
                ),
            )
        with self.assertRaisesRegex(SchemaError, "contributions exceed limit"):
            load_attention_operator_provider_integration_bootstrap_document(
                document.to_json(),
                contribution_manifest_limits=(
                    AttentionOperatorProviderContributionManifestLimits(
                        max_contributions=1
                    )
                ),
            )
        malformed = document.to_dict()
        malformed["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            load_attention_operator_provider_integration_bootstrap_document(
                json.dumps(malformed)
            )

    def test_loader_drift_fails_before_adapter_package_observation(self):
        _, declared, document = document_values()
        declared["loader"].loader_id = "drifted-loader-id"

        with self.assertRaisesRegex(SchemaError, "factory loader differs"):
            assemble_attention_operator_provider_integration_bootstrap_document(
                bootstrap_document=document,
                factory_loader=declared["loader"],
            )

        self.assertEqual(declared["loader_events"], [])
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(declared["loader"].version_calls, 0)
        self.assertEqual(declared["loader"].resolve_calls, 0)

    def test_document_install_preserves_private_auto_selection(self):
        values, declared, document = document_values()
        loaded, _ = load_attention_operator_provider_integration_bootstrap_document(
            document.to_json()
        )

        installed = install_attention_operator_provider_integration_bootstrap_document(
            bootstrap_document=loaded,
            factory_loader=declared["loader"],
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
            installed.provider_integration_bundle_binding.bootstrap_manifest_id,
            document.bootstrap_manifest.bootstrap_id,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
