import sys
import unittest

from flashinfer_npu.attention import (
    EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
    AttentionOperatorPackageRuntimeSpec,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorRuntimeResolutionError,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionTensorAccessPolicy,
    build_attention_operator_package_runtime,
    build_attention_operator_runtime_resolvers,
    build_default_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import bound_kernel, functional_profile, group_plan
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalFactory,
    FakeLogicalRunAdapter,
    FakePackageLoader,
    FakePlanGate,
    FakeTensorMaterializer,
    fake_operation,
)
from flashinfer_npu.attention import AttentionOperatorOperationCatalog


def bootstrap_components(*, package_version="1.0.0", gate_reasons=()):
    events = []
    operation = fake_operation()
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-022-catalog", operations=(operation,)
    )
    loader = FakePackageLoader(events, version=package_version)
    gate = FakePlanGate(events, gate_reasons)
    profile = functional_profile()
    materializer = FakeTensorMaterializer(events)
    tensor_metadata_inspector = FakeTensorMetadataInspector()
    quantization_binding = AttentionOperatorQuantizationBinding(
        provider_id="cann",
        operation_id=operation.operation_id,
        quant_spec=profile.rules[0].quant_specs[0],
        argument_bindings=(
            AttentionOperatorQuantArgumentBinding(
                "kv.key.scale", "key_scale"
            ),
            AttentionOperatorQuantArgumentBinding(
                "kv.value.scale", "value_scale"
            ),
        ),
    )
    spec = AttentionOperatorPackageRuntimeSpec(
        operation_id=operation.operation_id,
        priority=100,
        adapter_version="checkpoint-022-adapter-v1",
        supported_package_versions=("1.0.0",),
        profiles=(profile,),
        descriptors=(bound_kernel(profile),),
        observed_environment=profile.environment,
        plan_gate=gate,
        logical_factory=FakeLogicalFactory(events),
        logical_run_adapter=FakeLogicalRunAdapter(),
        tensor_materializer=materializer,
        quantization_bindings=(quantization_binding,),
        tensor_metadata_inspector=tensor_metadata_inspector,
        tensor_access_policy=AttentionTensorAccessPolicy(
            require_contiguous_q=True
        ),
    )
    return {
        "events": events,
        "operation": operation,
        "catalog": catalog,
        "loader": loader,
        "gate": gate,
        "materializer": materializer,
        "tensor_metadata_inspector": tensor_metadata_inspector,
        "spec": spec,
    }


class ForeignPlanGate(FakePlanGate):
    provider_id = "flash_attention_npu"


class FakeTensorMetadataInspector:
    """Strict metadata-only inspector used by synthetic package fixtures."""

    def __init__(self):
        self.calls = []

    def to_view(self, tensor, *, name, writable=False):
        self.calls.append((tensor, name, writable))
        view = getattr(tensor, "tensor_view", None)
        if view is None:
            raise SchemaError("synthetic tensor has no TensorView metadata")
        return view


class OperatorRuntimeBootstrapCheckpoint(unittest.TestCase):
    """Checkpoint 022: one declarative root composes the NPU runtime tree."""

    def test_default_bootstrap_is_empty_and_import_side_effect_free(self):
        imported_before = set(sys.modules)

        registry = build_default_attention_operator_runtime_resolvers()

        self.assertIs(registry, EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS)
        self.assertEqual(registry.resolvers, ())
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_build_one_candidate_does_not_probe_or_import_the_package(self):
        values = bootstrap_components()

        implementation = build_attention_operator_package_runtime(
            values["spec"],
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        self.assertEqual(implementation.provider_id, "cann")
        self.assertEqual(implementation.operation_id, values["operation"].operation_id)
        self.assertEqual(implementation.priority, 100)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_registry_routes_npu_and_explain_is_metadata_only(self):
        values = bootstrap_components()
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        self.assertIsInstance(registry, AttentionOperatorRuntimeResolverRegistry)
        self.assertEqual(tuple(item[0] for item in registry.resolvers), ("npu",))
        report = registry.resolvers[0][1].explain(group_plan(), "npu:0")
        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(values["loader"].version_calls, 1)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertNotIn("authorize", values["events"])
        self.assertNotIn("materialize", values["events"])

    def test_resolve_closes_evidence_before_callable_without_materializing(self):
        values = bootstrap_components()
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        plan = group_plan()

        resolved = registry.resolve(plan, "npu:2")

        self.assertEqual(resolved.framework_plan_fingerprint, plan.fingerprint)
        self.assertEqual(resolved.selection.provider_id, "cann")
        self.assertEqual(resolved.receipt.profile_id, functional_profile().profile_id)
        self.assertEqual(resolved.factory.operation_id, values["operation"].operation_id)
        self.assertEqual(values["loader"].resolve_calls, 1)
        self.assertNotIn("prepare", values["events"])
        self.assertNotIn("materialize", values["events"])

    def test_unavailable_package_stops_before_authority_and_callable(self):
        values = bootstrap_components(package_version=None)
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        with self.assertRaisesRegex(
            AttentionOperatorRuntimeResolutionError, "not installed"
        ):
            registry.resolve(group_plan(), "npu:0")
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertNotIn("authorize", values["events"])
        self.assertNotIn("materialize", values["events"])

    def test_catalog_and_component_provider_mismatch_fails_at_build(self):
        values = bootstrap_components()
        spec = values["spec"]
        foreign_gate = ForeignPlanGate(values["events"])
        mismatched = AttentionOperatorPackageRuntimeSpec(
            operation_id=spec.operation_id,
            priority=spec.priority,
            adapter_version=spec.adapter_version,
            supported_package_versions=spec.supported_package_versions,
            profiles=spec.profiles,
            descriptors=spec.descriptors,
            observed_environment=spec.observed_environment,
            plan_gate=foreign_gate,
            logical_factory=spec.logical_factory,
            logical_run_adapter=spec.logical_run_adapter,
            tensor_materializer=spec.tensor_materializer,
            quantization_bindings=spec.quantization_bindings,
            tensor_metadata_inspector=spec.tensor_metadata_inspector,
            tensor_access_policy=spec.tensor_access_policy,
        )

        with self.assertRaisesRegex(SchemaError, "providers differ"):
            build_attention_operator_package_runtime(
                mismatched,
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["events"], [])

    def test_duplicate_operation_specs_are_rejected_without_package_probe(self):
        values = bootstrap_components()

        with self.assertRaisesRegex(SchemaError, "duplicate.*identity"):
            build_attention_operator_runtime_resolvers(
                (values["spec"], values["spec"]),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["events"], [])

    def test_spec_rejects_ambiguous_versions_and_tuning_ids(self):
        values = bootstrap_components()
        spec = values["spec"]
        fields = {
            name: getattr(spec, name)
            for name in spec.__dataclass_fields__
            if name != "schema_version"
        }
        cases = (
            (
                {"supported_package_versions": ("1.0.0", "1.0.0")},
                "package versions must be unique",
            ),
            (
                {"tuned_kernel_ids": ("same", "same")},
                "tuned kernel ids must be unique",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                candidate = dict(fields)
                candidate.update(overrides)
                with self.assertRaisesRegex(SchemaError, message):
                    AttentionOperatorPackageRuntimeSpec(**candidate)


if __name__ == "__main__":
    unittest.main()
