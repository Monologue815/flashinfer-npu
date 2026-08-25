import inspect
import sys
import unittest

from flashinfer_npu.attention import (
    EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
    AttentionOperatorRuntimeRegistrySnapshot,
    AttentionOperatorRuntimeResolverRegistry,
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import DispatchError, SchemaError
from tests.test_checkpoint_014_public_batch_attention_provider_runtime import (
    FakeAutoResolver,
    plan_public_wrapper,
)


def npu_registry(resolver=None):
    resolver = FakeAutoResolver() if resolver is None else resolver
    return (
        AttentionOperatorRuntimeResolverRegistry((("npu", resolver),)),
        resolver,
    )


class PublicRuntimeRegistrySnapshotCheckpoint(unittest.TestCase):
    """Checkpoint 023: integration swaps never leak into existing wrappers."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.assertIs(
            self.original.registry, EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS
        )

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        if (
            current.registry is not self.original.registry
            or current.operation_catalog is not self.original.operation_catalog
        ):
            install_attention_operator_runtime_resolvers(
                self.original.registry,
                operation_catalog=self.original.operation_catalog,
                expected_generation=current.generation,
            )

    def test_snapshot_is_side_effect_free_and_matches_default_registry(self):
        imported_before = set(sys.modules)

        snapshot = attention_operator_runtime_registry_snapshot()

        self.assertEqual(snapshot.device_types, ())
        self.assertIs(snapshot.registry, EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS)
        self.assertGreater(len(snapshot.operation_catalog.operations), 0)
        self.assertGreaterEqual(snapshot.generation, 0)
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_install_affects_only_wrappers_constructed_after_commit(self):
        old_wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        registry, resolver = npu_registry()

        installed = install_attention_operator_runtime_resolvers(
            registry, expected_generation=self.original.generation
        )
        new_wrapper = BatchAttention(kv_layout="HND", device="npu:0")

        self.assertEqual(installed.generation, self.original.generation + 1)
        self.assertIs(
            new_wrapper._operator_runtime_registry_snapshot.registry, registry
        )
        self.assertIsNone(
            plan_public_wrapper(new_wrapper, page_size=128, kv_lengths=(192, 128))
        )
        self.assertEqual(
            new_wrapper.run("query", ("key", "value")),
            ("public-output-generation-1", "public-lse-generation-1"),
        )
        self.assertEqual(len(resolver.resolve_calls), 1)
        with self.assertRaisesRegex(DispatchError, "no Attention operator runtime"):
            plan_public_wrapper(old_wrapper, page_size=128, kv_lengths=(192, 128))

    def test_restored_default_only_changes_future_wrappers(self):
        registry, _ = npu_registry()
        installed = install_attention_operator_runtime_resolvers(
            registry, expected_generation=self.original.generation
        )
        installed_wrapper = BatchAttention(kv_layout="HND", device="npu")
        restored = install_attention_operator_runtime_resolvers(
            self.original.registry, expected_generation=installed.generation
        )
        future_wrapper = BatchAttention(kv_layout="HND", device="npu")

        self.assertEqual(restored.generation, installed.generation + 1)
        self.assertIsNone(
            plan_public_wrapper(
                installed_wrapper, page_size=128, kv_lengths=(192, 128)
            )
        )
        with self.assertRaisesRegex(DispatchError, "no Attention operator runtime"):
            plan_public_wrapper(
                future_wrapper, page_size=128, kv_lengths=(192, 128)
            )

    def test_stale_generation_cannot_overwrite_current_installation(self):
        registry, _ = npu_registry()
        installed = install_attention_operator_runtime_resolvers(
            registry, expected_generation=self.original.generation
        )

        with self.assertRaisesRegex(SchemaError, "generation changed"):
            install_attention_operator_runtime_resolvers(
                self.original.registry,
                expected_generation=self.original.generation,
            )

        current = attention_operator_runtime_registry_snapshot()
        self.assertEqual(current.generation, installed.generation)
        self.assertIs(current.registry, registry)

    def test_registry_and_operation_catalog_are_one_atomic_snapshot(self):
        registry, _ = npu_registry()
        catalog = self.original.operation_catalog

        installed = install_attention_operator_runtime_resolvers(
            registry,
            operation_catalog=catalog,
            expected_generation=self.original.generation,
        )
        observed = attention_operator_runtime_registry_snapshot()

        self.assertIs(installed.operation_catalog, catalog)
        self.assertIs(observed.operation_catalog, catalog)
        self.assertIs(observed.registry, registry)

    def test_install_rejects_non_npu_routes_and_invalid_generation(self):
        cpu_registry = AttentionOperatorRuntimeResolverRegistry(
            (("cpu", FakeAutoResolver()),)
        )
        with self.assertRaisesRegex(SchemaError, "only route npu"):
            install_attention_operator_runtime_resolvers(cpu_registry)
        with self.assertRaisesRegex(TypeError, "must be Attention"):
            install_attention_operator_runtime_resolvers(object())
        with self.assertRaisesRegex(TypeError, "operation_catalog"):
            install_attention_operator_runtime_resolvers(
                EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
                operation_catalog=object(),
            )
        for generation in (-1, True, "0"):
            with self.subTest(generation=generation):
                with self.assertRaisesRegex(SchemaError, "non-negative"):
                    install_attention_operator_runtime_resolvers(
                        EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
                        expected_generation=generation,
                    )

    def test_install_and_snapshot_do_not_resolve_or_touch_a_device(self):
        registry, resolver = npu_registry()
        imported_before = set(sys.modules)

        installed = install_attention_operator_runtime_resolvers(
            registry, expected_generation=self.original.generation
        )
        observed = attention_operator_runtime_registry_snapshot()

        self.assertEqual(resolver.resolve_calls, [])
        self.assertEqual(observed.generation, installed.generation)
        self.assertIs(observed.registry, registry)
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_snapshot_schema_rejects_stale_or_invalid_identity(self):
        registry, _ = npu_registry()
        with self.assertRaisesRegex(SchemaError, "snapshot is stale"):
            AttentionOperatorRuntimeRegistrySnapshot(0, (), registry)
        with self.assertRaisesRegex(SchemaError, "cannot be negative"):
            AttentionOperatorRuntimeRegistrySnapshot(-1, ("npu",), registry)
        with self.assertRaisesRegex(TypeError, "must be Attention"):
            AttentionOperatorRuntimeRegistrySnapshot(0, (), object())

    def test_public_flashinfer_style_signatures_remain_provider_free(self):
        self.assertEqual(
            list(inspect.signature(BatchAttention).parameters),
            ["kv_layout", "device"],
        )
        self.assertNotIn("provider", inspect.signature(BatchAttention.plan).parameters)
        self.assertNotIn("plan", inspect.signature(BatchAttention.run).parameters)


if __name__ == "__main__":
    unittest.main()
