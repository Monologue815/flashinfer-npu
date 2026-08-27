import inspect
import json
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorPackageRuntimeDeclaration,
    BatchAttention,
    describe_attention_operator_package_runtime,
    load_attention_operator_package_runtime_declaration,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components


class RuntimeDeclarationCheckpoint(unittest.TestCase):
    """A provider spec is reviewable without loading or executing its package."""

    def test_description_has_no_package_or_device_side_effects(self):
        values = bootstrap_components()

        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )

        self.assertEqual(declaration.provider_id, "cann")
        self.assertEqual(declaration.operation_id, values["operation"].operation_id)
        self.assertEqual(declaration.package_name, values["operation"].package_name)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_bounded_json_round_trip_is_canonical(self):
        values = bootstrap_components()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )

        restored, usage = load_attention_operator_package_runtime_declaration(
            declaration.to_json()
        )

        self.assertIsInstance(restored, AttentionOperatorPackageRuntimeDeclaration)
        self.assertEqual(restored, declaration)
        self.assertEqual(restored.fingerprint, declaration.fingerprint)
        self.assertEqual(usage.encoded_bytes, len(declaration.to_json().encode()))

    def test_declaration_captures_all_non_executable_authority_identities(self):
        values = bootstrap_components()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )
        roles = {item.role: item for item in declaration.components}

        self.assertEqual(
            set(roles),
            {
                "plan_gate",
                "logical_factory",
                "logical_run_adapter",
                "tensor_materializer",
                "tensor_metadata_inspector",
            },
        )
        self.assertEqual(
            dict(roles["tensor_materializer"].identities)["materializer_id"],
            values["materializer"].materializer_id,
        )
        self.assertEqual(len(declaration.profile_bindings), 1)
        self.assertEqual(len(declaration.descriptor_bindings), 1)
        self.assertEqual(len(declaration.quantization_binding_fingerprints), 1)
        self.assertFalse(declaration.validate_provider_results)

    def test_declaration_detects_runtime_spec_and_catalog_drift(self):
        values = bootstrap_components()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )
        declaration.validate_runtime_spec(
            values["spec"], operation_catalog=values["catalog"]
        )

        changed = replace(values["spec"], adapter_version="adapter-v2")
        with self.assertRaisesRegex(SchemaError, "declaration is stale"):
            declaration.validate_runtime_spec(
                changed, operation_catalog=values["catalog"]
            )

    def test_serialized_declaration_contains_no_executable_objects(self):
        values = bootstrap_components()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )
        encoded = declaration.to_json()
        decoded = json.loads(encoded)

        self.assertNotIn("opaque_state", encoded)
        self.assertNotIn("callable_object", encoded)
        self.assertNotIn("tensor", decoded)
        self.assertNotIn("0x", encoded)
        self.assertTrue(all("type_name" in item for item in decoded["components"]))

    def test_unknown_or_changed_schema_is_rejected(self):
        values = bootstrap_components()
        declaration = describe_attention_operator_package_runtime(
            values["spec"], operation_catalog=values["catalog"]
        )
        data = declaration.to_dict()
        data["unknown"] = True
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            AttentionOperatorPackageRuntimeDeclaration.from_dict(data)

        data = declaration.to_dict()
        data["schema_version"] += 1
        with self.assertRaisesRegex(SchemaError, "unsupported"):
            AttentionOperatorPackageRuntimeDeclaration.from_dict(data)

    def test_model_facing_api_does_not_accept_a_declaration(self):
        for method in (BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("runtime_declaration", parameters)
            self.assertNotIn("provider_declaration", parameters)


if __name__ == "__main__":
    unittest.main()
