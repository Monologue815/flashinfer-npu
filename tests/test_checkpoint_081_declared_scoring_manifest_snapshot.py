from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionDeclaredOperatorPackageRuntimeSpec,
    AttentionOperatorPlanScoringManifest,
    AttentionOperatorPlanScoringManifestBinding,
    AttentionOperatorRuntimeRegistrySnapshot,
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    bind_attention_operator_plan_scoring_manifest,
    build_declared_attention_operator_runtime_resolvers,
    describe_attention_operator_package_runtime,
    install_attention_operator_runtime_resolvers,
    install_declared_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)
from tests.test_checkpoint_068_declared_runtime_registry import (
    declared_registration,
)
from tests.test_checkpoint_080_scoring_manifest_bootstrap_binding import (
    scoring_manifest,
)


def scored_registration(values, *, score=73):
    manifest = scoring_manifest(values["spec"], score=score)
    spec = bind_attention_operator_plan_scoring_manifest(
        (values["spec"],), manifest
    )[0]
    declaration = describe_attention_operator_package_runtime(
        spec,
        operation_catalog=values["catalog"],
    )
    registration = AttentionDeclaredOperatorPackageRuntimeSpec(
        declaration=declaration,
        runtime_spec=spec,
    )
    return manifest, registration


class DeclaredScoringManifestSnapshotCheckpoint(unittest.TestCase):
    """Checkpoint 081: one scored declaration set is published atomically."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_manifest_binding_is_non_executable_and_canonical(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])

        binding = manifest.binding

        self.assertIsInstance(
            binding, AttentionOperatorPlanScoringManifestBinding
        )
        self.assertEqual(binding.manifest_id, manifest.manifest_id)
        self.assertEqual(binding.manifest_fingerprint, manifest.fingerprint)
        self.assertEqual(
            binding.identities,
            ((values["spec"].provider_id, values["spec"].operation_id),),
        )
        self.assertEqual(
            binding.policy_fingerprint(
                values["spec"].provider_id,
                values["spec"].operation_id,
            ),
            manifest.policies[0].fingerprint,
        )
        self.assertNotIn("rules", binding.to_dict())

    def test_declared_builder_requires_binding_before_declaration(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])
        registration = declared_registration(values)

        with self.assertRaisesRegex(
            SchemaError, "bind the scoring manifest before declaration"
        ):
            build_declared_attention_operator_runtime_resolvers(
                (registration,),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
                plan_scoring_manifest=manifest,
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_declared_builder_rejects_manifest_policy_drift(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)
        drifted = scoring_manifest(values["spec"], score=74)

        with self.assertRaisesRegex(SchemaError, "differs from manifest policy"):
            build_declared_attention_operator_runtime_resolvers(
                (registration,),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
                plan_scoring_manifest=drifted,
            )

        self.assertNotEqual(manifest.fingerprint, drifted.fingerprint)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_install_publishes_manifest_with_registry_generation(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)

        installed = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
            expected_generation=self.original.generation,
        )
        observed = attention_operator_runtime_registry_snapshot()

        self.assertEqual(
            installed.plan_scoring_manifest_binding,
            manifest.binding,
        )
        self.assertEqual(
            observed.plan_scoring_manifest_binding,
            manifest.binding,
        )
        self.assertEqual(
            observed.plan_scoring_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertEqual(
            observed.plan_scoring_manifest_id,
            manifest.manifest_id,
        )
        self.assertEqual(observed.generation, installed.generation)
        self.assertIs(observed.registry, installed.registry)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_stale_generation_cannot_publish_drifted_manifest(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)
        installed = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
            expected_generation=self.original.generation,
        )
        drifted, drifted_registration = scored_registration(values, score=74)

        with self.assertRaisesRegex(SchemaError, "generation changed"):
            install_declared_attention_operator_runtime_resolvers(
                (drifted_registration,),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
                plan_scoring_manifest=drifted,
                expected_generation=self.original.generation,
            )

        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, installed.generation)
        self.assertEqual(current.plan_scoring_manifest_binding, manifest.binding)
        self.assertIs(current.registry, installed.registry)

    def test_legacy_install_clears_manifest_binding_for_future_snapshot(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)
        installed = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
            expected_generation=self.original.generation,
        )

        restored = install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=installed.generation,
        )

        self.assertIsNone(restored.plan_scoring_manifest_binding)
        self.assertIsNone(restored.plan_scoring_manifest_id)
        self.assertIsNone(restored.plan_scoring_manifest_fingerprint)

    def test_wrappers_capture_scoring_manifest_generation_immutably(self):
        old_wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        values = bootstrap_components()
        manifest, registration = scored_registration(values)
        installed = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
            expected_generation=self.original.generation,
        )
        scored_wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        restored = install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=installed.generation,
        )
        future_wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        old_snapshot = old_wrapper._operator_runtime_registry_snapshot
        scored_snapshot = scored_wrapper._operator_runtime_registry_snapshot
        future_snapshot = future_wrapper._operator_runtime_registry_snapshot

        self.assertEqual(
            old_snapshot.plan_scoring_manifest_binding,
            self.original.plan_scoring_manifest_binding,
        )
        self.assertEqual(
            scored_snapshot.plan_scoring_manifest_binding,
            manifest.binding,
        )
        self.assertEqual(scored_snapshot.generation, installed.generation)
        self.assertIsNone(future_snapshot.plan_scoring_manifest_binding)
        self.assertEqual(
            future_snapshot.generation,
            restored.generation,
        )

    def test_snapshot_rejects_unpaired_or_mismatched_binding(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)
        installed = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
            expected_generation=self.original.generation,
        )

        with self.assertRaisesRegex(
            SchemaError, "requires runtime declarations"
        ):
            AttentionOperatorRuntimeRegistrySnapshot(
                generation=installed.generation,
                device_types=installed.device_types,
                registry=installed.registry,
                operation_catalog=installed.operation_catalog,
                runtime_declarations=(),
                plan_scoring_manifest_binding=manifest.binding,
            )

        policy = manifest.policies[0]
        foreign_policy = replace(
            policy,
            policy_id="cann.foreign.snapshot.policy.v1",
            operation_id="cann.foreign_snapshot_attention@v1",
        )
        foreign_manifest = AttentionOperatorPlanScoringManifest(
            "checkpoint.081.foreign.snapshot.v1",
            (foreign_policy,),
        )
        with self.assertRaisesRegex(
            SchemaError, "differs from runtime declarations"
        ):
            replace(
                installed,
                plan_scoring_manifest_binding=foreign_manifest.binding,
            )


if __name__ == "__main__":
    unittest.main()
