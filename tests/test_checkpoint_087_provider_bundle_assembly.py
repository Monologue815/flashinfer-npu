from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorPackageLoaderRoute,
    AttentionOperatorProviderIntegrationBundle,
    AttentionOperatorRoutedPackageLoader,
    assemble_attention_operator_provider_integration_bundle,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalFactory,
    FakeLogicalRunAdapter,
    FakePackageLoader,
    FakePlanGate,
    FakeTensorMaterializer,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_080_scoring_manifest_bootstrap_binding import (
    CustomPlanScorer,
    scoring_policy,
)


FLASH_OPERATION_ID = "flash_attention_npu.checkpoint_087_attention@v1"


class FlashPlanGate(FakePlanGate):
    provider_id = "flash_attention_npu"
    operation_id = FLASH_OPERATION_ID


class FlashLogicalFactory(FakeLogicalFactory):
    provider_id = "flash_attention_npu"
    operation_id = FLASH_OPERATION_ID


class FlashLogicalRunAdapter(FakeLogicalRunAdapter):
    provider_id = "flash_attention_npu"


class FlashTensorMaterializer(FakeTensorMaterializer):
    provider_id = "flash_attention_npu"
    materializer_id = "checkpoint-087-flash-materializer-v1"


def single_assembly_inputs():
    values = bootstrap_components()
    policy = scoring_policy(values["spec"])
    route = AttentionOperatorPackageLoaderRoute.from_catalog_operation(
        values["operation"],
        values["loader"],
    )
    return values, policy, route


def two_provider_assembly_inputs():
    cann = bootstrap_components()
    events = cann["events"]
    flash_operation = replace(
        cann["operation"],
        operation_id=FLASH_OPERATION_ID,
        provider_id="flash_attention_npu",
        package_name="checkpoint-087-flash-attention-package",
        callable_path="checkpoint_087_flash_attention.attention",
        api_version="checkpoint-087-v1",
        source_url="https://example.invalid/checkpoint-087-flash-attention",
    )
    flash_loader = FakePackageLoader(events, version="1.0.0")
    flash_quantization = replace(
        cann["spec"].quantization_bindings[0],
        provider_id="flash_attention_npu",
        operation_id=FLASH_OPERATION_ID,
    )
    flash_spec = replace(
        cann["spec"],
        operation_id=FLASH_OPERATION_ID,
        adapter_version="checkpoint-087-flash-adapter-v1",
        plan_gate=FlashPlanGate(events),
        logical_factory=FlashLogicalFactory(events),
        logical_run_adapter=FlashLogicalRunAdapter(),
        tensor_materializer=FlashTensorMaterializer(events),
        quantization_bindings=(flash_quantization,),
    )
    cann_policy = scoring_policy(
        cann["spec"],
        score=73,
        policy_id="cann.checkpoint.087.policy.v1",
    )
    flash_policy = scoring_policy(
        flash_spec,
        score=72,
        policy_id="flash_attention_npu.checkpoint.087.policy.v1",
    )
    routes = (
        AttentionOperatorPackageLoaderRoute.from_catalog_operation(
            cann["operation"],
            cann["loader"],
        ),
        AttentionOperatorPackageLoaderRoute.from_catalog_operation(
            flash_operation,
            flash_loader,
        ),
    )
    return {
        "cann": cann,
        "flash_operation": flash_operation,
        "flash_spec": flash_spec,
        "flash_loader": flash_loader,
        "operations": (cann["operation"], flash_operation),
        "specs": (cann["spec"], flash_spec),
        "policies": (cann_policy, flash_policy),
        "routes": routes,
        "events": events,
    }


def assemble_two_provider(values, *, reverse=False):
    def ordered(items):
        return tuple(reversed(items)) if reverse else tuple(items)

    return assemble_attention_operator_provider_integration_bundle(
        bundle_id="checkpoint.087.two-provider.bundle.v1",
        catalog_name="checkpoint-087-two-provider-catalog",
        scoring_manifest_id="checkpoint.087.two-provider.scoring.v1",
        operations=ordered(values["operations"]),
        runtime_specs=ordered(values["specs"]),
        scoring_policies=ordered(values["policies"]),
        package_loader_routes=ordered(values["routes"]),
    )


class ProviderBundleAssemblyCheckpoint(unittest.TestCase):
    """Checkpoint 087: derive the complete reviewed bundle in one ordering."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_single_provider_assembly_binds_before_declaration_without_probe(self):
        values, policy, route = single_assembly_inputs()

        bundle = assemble_attention_operator_provider_integration_bundle(
            bundle_id="checkpoint.087.single.bundle.v1",
            catalog_name="checkpoint-087-single-catalog",
            scoring_manifest_id="checkpoint.087.single.scoring.v1",
            operations=(values["operation"],),
            runtime_specs=(values["spec"],),
            scoring_policies=(policy,),
            package_loader_routes=(route,),
        )

        self.assertIsInstance(bundle, AttentionOperatorProviderIntegrationBundle)
        self.assertIsInstance(
            bundle.package_loader,
            AttentionOperatorRoutedPackageLoader,
        )
        self.assertIsNone(values["spec"].plan_scorer)
        self.assertIs(bundle.registrations[0].runtime_spec.plan_scorer, policy)
        scorer = next(
            item
            for item in bundle.registrations[0].declaration.components
            if item.role == "plan_scorer"
        )
        self.assertIn(("policy_fingerprint", policy.fingerprint), scorer.identities)
        self.assertEqual(
            bundle.registrations[0].declaration.catalog_fingerprint,
            bundle.operation_catalog.fingerprint,
        )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_two_provider_assembly_is_complete_canonical_and_side_effect_free(self):
        values = two_provider_assembly_inputs()

        first = assemble_two_provider(values)
        second = assemble_two_provider(values, reverse=True)

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            first.package_loader.routing_fingerprint,
            second.package_loader.routing_fingerprint,
        )
        self.assertEqual(
            tuple(
                (item.declaration.provider_id, item.declaration.operation_id)
                for item in first.registrations
            ),
            (
                (
                    "cann",
                    values["cann"]["operation"].operation_id,
                ),
                ("flash_attention_npu", FLASH_OPERATION_ID),
            ),
        )
        self.assertEqual(
            set(first.scoring_manifest.binding.identities),
            {
                (item.provider_id, item.operation_id)
                for item in first.package_loader.operation_catalog.operations
            },
        )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_two_provider_assembly_installs_and_plan_selects_unique_score(self):
        values = two_provider_assembly_inputs()
        bundle = assemble_two_provider(values)

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
            installed.operation_catalog.operations,
            bundle.operation_catalog.operations,
        )
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)

    def test_missing_or_orphan_policy_fails_before_loader_observation(self):
        values = two_provider_assembly_inputs()

        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            assemble_attention_operator_provider_integration_bundle(
                bundle_id="checkpoint.087.missing.policy.v1",
                catalog_name="checkpoint-087-missing-policy-catalog",
                scoring_manifest_id="checkpoint.087.missing.policy.scoring.v1",
                operations=values["operations"],
                runtime_specs=values["specs"],
                scoring_policies=(values["policies"][0],),
                package_loader_routes=values["routes"],
            )
        orphan = replace(
            values["policies"][1],
            operation_id="flash_attention_npu.orphan_attention@v1",
        )
        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            assemble_attention_operator_provider_integration_bundle(
                bundle_id="checkpoint.087.orphan.policy.v1",
                catalog_name="checkpoint-087-orphan-policy-catalog",
                scoring_manifest_id="checkpoint.087.orphan.policy.scoring.v1",
                operations=values["operations"],
                runtime_specs=values["specs"],
                scoring_policies=(values["policies"][0], orphan),
                package_loader_routes=values["routes"],
            )

        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["events"], [])

    def test_missing_or_drifted_route_fails_before_declaration_publication(self):
        values = two_provider_assembly_inputs()

        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            assemble_attention_operator_provider_integration_bundle(
                bundle_id="checkpoint.087.missing.route.v1",
                catalog_name="checkpoint-087-missing-route-catalog",
                scoring_manifest_id="checkpoint.087.missing.route.scoring.v1",
                operations=values["operations"],
                runtime_specs=values["specs"],
                scoring_policies=values["policies"],
                package_loader_routes=(values["routes"][0],),
            )
        with self.assertRaisesRegex(SchemaError, "differs from catalog"):
            assemble_attention_operator_provider_integration_bundle(
                bundle_id="checkpoint.087.drifted.route.v1",
                catalog_name="checkpoint-087-drifted-route-catalog",
                scoring_manifest_id="checkpoint.087.drifted.route.scoring.v1",
                operations=values["operations"],
                runtime_specs=values["specs"],
                scoring_policies=values["policies"],
                package_loader_routes=(
                    values["routes"][0],
                    replace(
                        values["routes"][1],
                        callable_path="drifted.flash.attention",
                    ),
                ),
            )

        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_custom_or_different_existing_scorer_is_never_overwritten(self):
        values, policy, route = single_assembly_inputs()
        custom = replace(
            values["spec"],
            plan_scorer=CustomPlanScorer(values["spec"]),
        )

        with self.assertRaisesRegex(SchemaError, "non-manifest plan scorer"):
            assemble_attention_operator_provider_integration_bundle(
                bundle_id="checkpoint.087.custom.scorer.v1",
                catalog_name="checkpoint-087-custom-scorer-catalog",
                scoring_manifest_id="checkpoint.087.custom.scorer.manifest.v1",
                operations=(values["operation"],),
                runtime_specs=(custom,),
                scoring_policies=(policy,),
                package_loader_routes=(route,),
            )

        different = replace(
            values["spec"],
            plan_scorer=replace(policy, default_score=policy.default_score + 1),
        )
        with self.assertRaisesRegex(SchemaError, "differs from manifest policy"):
            assemble_attention_operator_provider_integration_bundle(
                bundle_id="checkpoint.087.different.scorer.v1",
                catalog_name="checkpoint-087-different-scorer-catalog",
                scoring_manifest_id="checkpoint.087.different.scorer.manifest.v1",
                operations=(values["operation"],),
                runtime_specs=(different,),
                scoring_policies=(policy,),
                package_loader_routes=(route,),
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
