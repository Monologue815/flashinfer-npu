import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorPlanSession,
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    build_provider_plan_selection,
    build_reference_plan_selection,
    install_attention_operator_runtime_resolvers,
    install_declared_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_006_attention_operator_active_plan import (
    FakePlanFactory,
    dispatch_receipt,
    framework_plan,
    provider_selection,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_068_declared_runtime_registry import declared_registration


class DeclaredRegistrySnapshotCheckpoint(unittest.TestCase):
    """Reviewed declaration identity follows the installed wrapper generation."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def _install(self):
        values = bootstrap_components()
        registration = declared_registration(values)
        snapshot = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            expected_generation=self.original.generation,
        )
        return values, registration, snapshot

    def test_declared_install_publishes_one_atomic_snapshot_binding(self):
        values, registration, installed = self._install()
        observed = attention_operator_runtime_registry_snapshot()

        self.assertEqual(installed.runtime_declarations, (registration.binding,))
        self.assertEqual(observed.runtime_declarations, (registration.binding,))
        self.assertIs(observed.registry, installed.registry)
        self.assertIs(observed.operation_catalog, values["catalog"])
        self.assertEqual(observed.generation, installed.generation)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_wrapper_captures_declaration_generation_immutably(self):
        old_wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        _, registration, installed = self._install()
        declared_wrapper = BatchAttention(kv_layout="HND", device="npu:0")

        restored = install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=installed.generation,
        )
        future_wrapper = BatchAttention(kv_layout="HND", device="npu:0")

        self.assertEqual(
            old_wrapper._operator_runtime_registry_snapshot.runtime_declarations,
            self.original.runtime_declarations,
        )
        self.assertEqual(
            declared_wrapper._operator_runtime_registry_snapshot.runtime_declarations,
            (registration.binding,),
        )
        self.assertEqual(
            declared_wrapper._operator_runtime_registry_snapshot.generation,
            installed.generation,
        )
        self.assertEqual(
            future_wrapper._operator_runtime_registry_snapshot.runtime_declarations,
            (),
        )
        self.assertEqual(
            future_wrapper._operator_runtime_registry_snapshot.generation,
            restored.generation,
        )

    def test_stale_generation_cannot_partially_publish_declarations(self):
        _, registration, installed = self._install()

        with self.assertRaisesRegex(SchemaError, "generation changed"):
            install_declared_attention_operator_runtime_resolvers(
                (registration,),
                operation_catalog=installed.operation_catalog,
                expected_generation=self.original.generation,
            )

        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, installed.generation)
        self.assertEqual(current.runtime_declarations, (registration.binding,))
        self.assertIs(current.registry, installed.registry)

    def test_plan_selection_binds_the_selected_declaration_fingerprint(self):
        values, registration, installed = self._install()
        plan = framework_plan()
        receipt = dispatch_receipt(plan)
        selection = provider_selection(receipt)
        factory = FakePlanFactory()
        factory.operation_id = values["operation"].operation_id
        session = AttentionOperatorPlanSession()
        session.plan(factory, plan, receipt, selection)
        active_plan = session.active_plan
        fingerprint = installed.declaration_fingerprint(
            selection.provider_id,
            active_plan.prepared_plan.implementation_id,
        )

        public = build_provider_plan_selection(
            plan,
            active_plan,
            registry_generation=installed.generation,
            runtime_declaration_fingerprint=fingerprint,
        )

        self.assertEqual(fingerprint, registration.declaration.fingerprint)
        self.assertEqual(
            public.runtime_declaration_fingerprint,
            registration.declaration.fingerprint,
        )
        self.assertEqual(
            public.to_dict()["runtime_declaration_fingerprint"],
            registration.declaration.fingerprint,
        )

    def test_legacy_and_reference_plan_selection_remain_explicitly_unbound(self):
        plan = framework_plan()
        receipt = dispatch_receipt(plan)
        selection = provider_selection(receipt)
        factory = FakePlanFactory()
        session = AttentionOperatorPlanSession()
        session.plan(factory, plan, receipt, selection)

        provider = build_provider_plan_selection(
            plan,
            session.active_plan,
            registry_generation=0,
        )
        reference = build_reference_plan_selection(plan)

        self.assertIsNone(provider.runtime_declaration_fingerprint)
        self.assertIsNone(reference.runtime_declaration_fingerprint)

    def test_model_facing_signatures_remain_provider_free(self):
        self.assertNotIn("registration", inspect.signature(BatchAttention.plan).parameters)
        self.assertNotIn(
            "runtime_declaration", inspect.signature(BatchAttention.run).parameters
        )


if __name__ == "__main__":
    unittest.main()
