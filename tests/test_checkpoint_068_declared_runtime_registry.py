import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDeclaredOperatorPackageRuntimeSpec,
    AttentionOperatorOperationCatalog,
    AttentionOperatorRuntimeResolverRegistry,
    build_declared_attention_operator_runtime_resolvers,
    describe_attention_operator_package_runtime,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components


def declared_registration(values):
    declaration = describe_attention_operator_package_runtime(
        values["spec"], operation_catalog=values["catalog"]
    )
    return AttentionDeclaredOperatorPackageRuntimeSpec(
        declaration=declaration,
        runtime_spec=values["spec"],
    )


class DeclaredRuntimeRegistryCheckpoint(unittest.TestCase):
    """Registry composition fails before package probing on declaration drift."""

    def test_matching_declaration_builds_without_package_probe(self):
        values = bootstrap_components()
        registration = declared_registration(values)

        registry = build_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        self.assertIsInstance(registry, AttentionOperatorRuntimeResolverRegistry)
        self.assertEqual(tuple(item[0] for item in registry.resolvers), ("npu",))
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_spec_drift_fails_before_registry_composition(self):
        values = bootstrap_components()
        registration = declared_registration(values)
        drifted = replace(values["spec"], priority=values["spec"].priority + 1)
        stale = replace(registration, runtime_spec=drifted)

        with self.assertRaisesRegex(SchemaError, "declaration is stale"):
            build_declared_attention_operator_runtime_resolvers(
                (stale,),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_catalog_drift_fails_before_registry_composition(self):
        values = bootstrap_components()
        registration = declared_registration(values)
        changed_catalog = AttentionOperatorOperationCatalog(
            name="changed-catalog-name",
            operations=values["catalog"].operations,
        )

        with self.assertRaisesRegex(SchemaError, "declaration is stale"):
            build_declared_attention_operator_runtime_resolvers(
                (registration,),
                operation_catalog=changed_catalog,
                package_loader=values["loader"],
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_duplicate_registration_set_is_rejected_atomically(self):
        values = bootstrap_components()
        registration = declared_registration(values)

        with self.assertRaisesRegex(SchemaError, "registrations are duplicated"):
            build_declared_attention_operator_runtime_resolvers(
                (registration, registration),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_registration_rejects_an_immediate_identity_mismatch(self):
        values = bootstrap_components()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )
        drifted = replace(values["spec"], adapter_version="adapter-v2")

        with self.assertRaisesRegex(SchemaError, "identity differs"):
            AttentionDeclaredOperatorPackageRuntimeSpec(
                declaration=declaration,
                runtime_spec=drifted,
            )

    def test_empty_declared_set_preserves_empty_default_semantics(self):
        values = bootstrap_components()

        registry = build_declared_attention_operator_runtime_resolvers(
            (),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        self.assertEqual(registry.resolvers, ())
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
