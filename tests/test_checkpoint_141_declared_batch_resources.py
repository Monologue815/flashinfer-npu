import json
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog, BatchAttention,
    attention_operator_runtime_registry_snapshot,
    install_declared_attention_operator_runtime_resolvers,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_provider_integration_bootstrap,
    install_attention_operator_provider_integration_bootstrap_document,
)
from flashinfer_npu.attention.holistic import _install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.operator_mask_binding import AttentionBatchMaskIntegration, AttentionOperatorMaskArgumentSpec
from flashinfer_npu.attention.operator_runtime_owner import AttentionBatchRuntimeOwner
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_043_provider_workspace_reset import FakeNpuWorkspace
from tests.test_checkpoint_068_declared_runtime_registry import declared_registration
from tests.test_checkpoint_084_provider_integration_bundle import provider_bundle
from tests.test_checkpoint_094_provider_bootstrap import bootstrap_values
from tests.test_checkpoint_095_provider_bootstrap_document import document_values
from tests.test_checkpoint_137_batch_recorder_bootstrap import Factory


def inputs():
    values = bootstrap_components()
    original = values["catalog"].operations[0]
    operation = replace(original, keyword_arguments=original.keyword_arguments + ("mask", "offsets"),
                        host_sequence_arguments=("offsets",))
    values["catalog"] = AttentionOperatorOperationCatalog("synthetic-declared-mask", (operation,))
    integration = AttentionBatchMaskIntegration(
        (AttentionOperatorMaskArgumentSpec(operation.fingerprint, "bool_allow_flat", "mask", "offsets"),),
        values["tensor_metadata_inspector"])
    return values, integration


class DeclaredBatchResourcesCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.integration = inputs()
        self.factory, self.owner = Factory(), AttentionBatchRuntimeOwner()

    def tearDown(self):
        self.owner.close()
        old = self.original
        _install_attention_operator_runtime_resolvers(
            old.registry, operation_catalog=old.operation_catalog,
            runtime_declarations=old.runtime_declarations,
            plan_scoring_manifest_binding=old.plan_scoring_manifest_binding,
            provider_integration_bundle_binding=old.provider_integration_bundle_binding,
            batch_completion_event_recorder_factory=old.batch_completion_event_recorder_factory,
            batch_runtime_owner=old.batch_runtime_owner, batch_mask_integration=old.batch_mask_integration)

    def options(self):
        return dict(batch_completion_event_recorder_factory=self.factory,
                    batch_runtime_owner=self.owner, batch_mask_integration=self.integration)

    def assert_resources(self, snapshot, mask_expected=True):
        self.assertIs(snapshot.batch_completion_event_recorder_factory, self.factory)
        self.assertIs(snapshot.batch_runtime_owner, self.owner)
        self.assertIs(snapshot.batch_mask_integration, self.integration if mask_expected else None)

    def assert_unobserved(self):
        self.assertEqual(self.values["loader"].version_calls, 0)
        self.assertEqual(self.values["loader"].resolve_calls, 0)
        self.assertEqual(self.factory.calls, [])
        self.assertEqual(self.owner.owned_runtime_ids, ())

    def test_declared_install_captures_resources_with_reviewed_declaration(self):
        registration = declared_registration(self.values)
        snapshot = install_declared_attention_operator_runtime_resolvers(
            (registration,), operation_catalog=self.values["catalog"],
            package_loader=self.values["loader"], **self.options())
        self.assert_resources(snapshot)
        self.assertEqual(snapshot.runtime_declarations, (registration.binding,))
        self.assert_unobserved()
        wrapper = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        self.assert_resources(wrapper._operator_runtime_registry_snapshot)
        self.assertEqual(len(self.owner.owned_runtime_ids), 1)

    def test_bundle_identity_and_resource_configuration_publish_together_without_serialization(self):
        bundle = provider_bundle(self.values)
        encoded = json.dumps(bundle.to_dict(), sort_keys=True)
        fingerprint = bundle.fingerprint
        snapshot = install_attention_operator_provider_integration_bundle(bundle, **self.options())
        self.assert_resources(snapshot)
        self.assertEqual(snapshot.provider_integration_bundle_binding, bundle.binding)
        self.assertEqual(snapshot.plan_scoring_manifest_binding, bundle.scoring_manifest.binding)
        self.assertEqual(bundle.fingerprint, fingerprint)
        self.assertEqual(json.dumps(bundle.to_dict(), sort_keys=True), encoded)
        self.assertNotIn("batch_runtime_owner", encoded)
        self.assertNotIn("batch_mask_integration", encoded)
        self.assert_unobserved()

    def test_stale_generation_keeps_installed_authorities_and_resources(self):
        bundle = provider_bundle(self.values)
        snapshot = install_attention_operator_provider_integration_bundle(bundle, **self.options())
        with self.assertRaisesRegex(SchemaError, "generation changed"):
            install_attention_operator_provider_integration_bundle(
                replace(bundle, bundle_id="synthetic.changed.bundle"), expected_generation=snapshot.generation - 1)
        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, snapshot.generation)
        self.assertEqual(current.provider_integration_bundle_binding, snapshot.provider_integration_bundle_binding)
        self.assert_resources(current)
        self.assert_unobserved()

    def test_mask_dependency_and_catalog_errors_do_not_partially_install_bundle(self):
        baseline = attention_operator_runtime_registry_snapshot()
        with self.assertRaisesRegex(SchemaError, "ownership and completion"):
            install_attention_operator_provider_integration_bundle(
                provider_bundle(self.values), batch_mask_integration=self.integration)
        plain_values = bootstrap_components()
        with self.assertRaisesRegex(SchemaError, "absent.*catalog"):
            install_attention_operator_provider_integration_bundle(provider_bundle(plain_values), **self.options())
        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, baseline.generation)
        self.assertIs(current.registry, baseline.registry)
        self.assert_unobserved()
        self.assertEqual(plain_values["loader"].version_calls, 0)

    def test_bootstrap_manifest_and_document_forward_lifetime_objects_outside_data(self):
        for document_mode in (False, True):
            with self.subTest(document=document_mode):
                if document_mode:
                    values, declared, document = document_values()
                    before = document.to_json()
                    snapshot = install_attention_operator_provider_integration_bootstrap_document(
                        bootstrap_document=document, factory_loader=declared["loader"],
                        batch_completion_event_recorder_factory=self.factory, batch_runtime_owner=self.owner)
                    self.assertEqual(document.to_json(), before)
                else:
                    values, declared, source, bootstrap = bootstrap_values()
                    before = bootstrap.to_json()
                    snapshot = install_attention_operator_provider_integration_bootstrap(
                        bootstrap_manifest=bootstrap, source_manifest=source,
                        factory_loader=declared["loader"], approval_manifest=declared["manifest"],
                        batch_completion_event_recorder_factory=self.factory, batch_runtime_owner=self.owner)
                    self.assertEqual(bootstrap.to_json(), before)
                self.assert_resources(snapshot, mask_expected=False)
                self.assertEqual(values["cann"]["loader"].version_calls, 0)
                self.assertEqual(values["flash_loader"].version_calls, 0)
                wrapper = BatchAttention(device="npu:0")
                self.assert_resources(wrapper._operator_runtime_registry_snapshot, mask_expected=False)
        self.assertEqual(len(self.owner.owned_runtime_ids), 2)

    def test_default_bundle_reinstall_affects_future_wrappers_only(self):
        bundle = provider_bundle(self.values)
        install_attention_operator_provider_integration_bundle(bundle, **self.options())
        old = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        install_attention_operator_provider_integration_bundle(bundle)
        new = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        self.assert_resources(old._operator_runtime_registry_snapshot)
        self.assertIsNone(new._operator_runtime_registry_snapshot.batch_mask_integration)
        self.assertIsNone(new._operator_runtime_registry_snapshot.batch_runtime_owner)
        self.assertEqual(len(self.owner.owned_runtime_ids), 1)
        self.assertEqual(len(self.factory.calls), 1)

    def test_bootstrap_entrypoints_do_not_drop_invalid_mask_configuration(self):
        baseline = attention_operator_runtime_registry_snapshot().generation
        _, declared, document = document_values()
        with self.assertRaisesRegex(SchemaError, "absent.*catalog"):
            install_attention_operator_provider_integration_bootstrap_document(
                bootstrap_document=document, factory_loader=declared["loader"], **self.options())
        _, declared, source, bootstrap = bootstrap_values()
        with self.assertRaisesRegex(SchemaError, "absent.*catalog"):
            install_attention_operator_provider_integration_bootstrap(
                bootstrap_manifest=bootstrap, source_manifest=source,
                factory_loader=declared["loader"], approval_manifest=declared["manifest"], **self.options())
        self.assertEqual(attention_operator_runtime_registry_snapshot().generation, baseline)
        self.assertEqual(self.factory.calls, [])
