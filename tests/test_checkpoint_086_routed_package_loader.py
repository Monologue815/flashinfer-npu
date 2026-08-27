from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorPackageLoader,
    AttentionOperatorPackageLoaderRoute,
    AttentionOperatorRoutedPackageLoader,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_084_provider_integration_bundle import (
    provider_bundle,
)


class RecordingPackageLoader:
    def __init__(self, loader_id, version, resolved_value, events):
        self.loader_id = loader_id
        self.version = version
        self.resolved_value = resolved_value
        self.events = events
        self.version_calls = 0
        self.resolve_calls = 0

    def package_version(self, package_name):
        self.version_calls += 1
        self.events.append((self.loader_id, "version", package_name))
        return self.version

    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append((self.loader_id, "resolve", callable_path))
        return self.resolved_value


def two_provider_routes():
    packaged = load_packaged_attention_operator_catalog()
    cann = packaged.get(
        "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"
    )
    flash = packaged.get(
        "flash_attention_npu.flash_attn_with_kvcache@v3"
    )
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-086-two-provider-catalog",
        operations=(flash, cann),
    )
    events = []
    cann_loader = RecordingPackageLoader(
        "checkpoint-086-cann-loader-v1",
        "7.3.0",
        "cann-callable",
        events,
    )
    flash_loader = RecordingPackageLoader(
        "checkpoint-086-flash-loader-v1",
        "3.0.0",
        "flash-callable",
        events,
    )
    routes = (
        AttentionOperatorPackageLoaderRoute.from_catalog_operation(
            cann, cann_loader
        ),
        AttentionOperatorPackageLoaderRoute.from_catalog_operation(
            flash, flash_loader
        ),
    )
    return {
        "catalog": catalog,
        "cann": cann,
        "flash": flash,
        "cann_loader": cann_loader,
        "flash_loader": flash_loader,
        "events": events,
        "routes": routes,
    }


class RoutedPackageLoaderCheckpoint(unittest.TestCase):
    """Checkpoint 086: one bundle can route exact operations to distinct loaders."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_routing_identity_is_canonical_and_construction_is_side_effect_free(self):
        values = two_provider_routes()

        first = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            values["routes"],
        )
        second = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            tuple(reversed(values["routes"])),
        )

        self.assertIsInstance(first, AttentionOperatorPackageLoader)
        self.assertEqual(first.routing_fingerprint, second.routing_fingerprint)
        self.assertEqual(first.loader_id, second.loader_id)
        self.assertIn(first.routing_fingerprint, first.loader_id)
        self.assertEqual(
            tuple(item.identity for item in first.route_bindings),
            tuple(
                sorted(
                    (
                        (values["cann"].provider_id, values["cann"].operation_id),
                        (
                            values["flash"].provider_id,
                            values["flash"].operation_id,
                        ),
                    )
                )
            ),
        )
        self.assertNotIn("package_loader", first.to_dict())
        self.assertEqual(values["events"], [])

    def test_metadata_and_callable_observations_use_exact_distinct_delegates(self):
        values = two_provider_routes()
        loader = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            values["routes"],
        )

        self.assertEqual(
            loader.package_version(values["cann"].package_name),
            "7.3.0",
        )
        self.assertEqual(
            loader.package_version(values["flash"].package_name),
            "3.0.0",
        )
        self.assertEqual(
            loader.resolve_callable(values["cann"].callable_path),
            "cann-callable",
        )
        self.assertEqual(
            loader.resolve_callable(values["flash"].callable_path),
            "flash-callable",
        )
        self.assertEqual(values["cann_loader"].version_calls, 1)
        self.assertEqual(values["flash_loader"].version_calls, 1)
        self.assertEqual(values["cann_loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 1)

    def test_unknown_package_or_callable_fails_without_delegate_calls(self):
        values = two_provider_routes()
        loader = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            values["routes"],
        )

        with self.assertRaisesRegex(SchemaError, "no exact.*route"):
            loader.package_version("unreviewed-attention-package")
        with self.assertRaisesRegex(SchemaError, "no exact.*route"):
            loader.resolve_callable("unreviewed.attention")

        self.assertEqual(values["events"], [])

    def test_route_set_and_catalog_fields_must_match_exactly(self):
        values = two_provider_routes()

        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            AttentionOperatorRoutedPackageLoader(
                values["catalog"],
                (values["routes"][0],),
            )
        with self.assertRaisesRegex(SchemaError, "routes duplicate"):
            AttentionOperatorRoutedPackageLoader(
                values["catalog"],
                (values["routes"][0], values["routes"][0]),
            )
        with self.assertRaisesRegex(SchemaError, "differs from catalog"):
            AttentionOperatorRoutedPackageLoader(
                values["catalog"],
                (
                    replace(
                        values["routes"][0],
                        callable_path="drifted.attention",
                    ),
                    values["routes"][1],
                ),
            )

        self.assertEqual(values["events"], [])

    def test_shared_package_or_callable_cannot_route_to_different_objects(self):
        values = two_provider_routes()
        shared_package_flash = replace(
            values["flash"],
            package_name=values["cann"].package_name,
        )
        catalog = AttentionOperatorOperationCatalog(
            name="checkpoint-086-ambiguous-package-catalog",
            operations=(values["cann"], shared_package_flash),
        )
        routes = (
            values["routes"][0],
            AttentionOperatorPackageLoaderRoute.from_catalog_operation(
                shared_package_flash,
                values["flash_loader"],
            ),
        )

        with self.assertRaisesRegex(SchemaError, "routes are ambiguous"):
            AttentionOperatorRoutedPackageLoader(catalog, routes)

        self.assertEqual(values["events"], [])

    def test_any_delegate_identity_drift_invalidates_the_composite_loader(self):
        values = two_provider_routes()
        loader = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            values["routes"],
        )
        frozen_loader_id = loader.loader_id
        values["cann_loader"].loader_id = "checkpoint-086-cann-loader-v2"

        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            _ = loader.loader_id
        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            loader.package_version(values["flash"].package_name)
        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            loader.resolve_callable(values["flash"].callable_path)

        self.assertTrue(frozen_loader_id.endswith(loader.routing_fingerprint))
        self.assertEqual(values["events"], [])

    def test_routed_loader_is_one_side_effect_free_bundle_loader(self):
        values = bootstrap_components()
        route = AttentionOperatorPackageLoaderRoute.from_catalog_operation(
            values["operation"],
            values["loader"],
        )
        routed = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            (route,),
        )
        bundle = provider_bundle(values, loader=routed)

        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(
            installed.provider_integration_bundle_binding.package_loader_id,
            routed.loader_id,
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

        self.assertGreater(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 1)
        self.assertEqual(
            wrapper.plan_selection.provider_integration_bundle_fingerprint,
            bundle.fingerprint,
        )

    def test_inner_loader_drift_fails_bundle_install_before_package_probe(self):
        values = bootstrap_components()
        routed = AttentionOperatorRoutedPackageLoader(
            values["catalog"],
            (
                AttentionOperatorPackageLoaderRoute.from_catalog_operation(
                    values["operation"],
                    values["loader"],
                ),
            ),
        )
        bundle = provider_bundle(values, loader=routed)
        values["loader"].loader_id = "checkpoint-086-inner-drift-v2"

        with self.assertRaisesRegex(SchemaError, "loader identity changed"):
            install_attention_operator_provider_integration_bundle(
                bundle,
                expected_generation=self.original.generation,
            )

        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, self.original.generation)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
