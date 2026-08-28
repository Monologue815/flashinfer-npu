import importlib.metadata
import types
import unittest
from unittest.mock import call, patch

from flashinfer_npu.attention import (
    AttentionOperatorProviderContributionSourceDeclarationRegistry,
    ImportlibAttentionOperatorProviderContributionFactoryLoader,
    assemble_attention_operator_provider_integration_source_declarations,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_087_provider_bundle_assembly import (
    two_provider_assembly_inputs,
)
from tests.test_checkpoint_091_provider_contribution_loader import (
    declaration_values,
)


VERSION_TARGET = (
    "flashinfer_npu.attention.provider_contribution_loader."
    "importlib.metadata.version"
)
IMPORT_TARGET = (
    "flashinfer_npu.attention.provider_contribution_loader."
    "importlib.import_module"
)


def concrete_registry(declared):
    return AttentionOperatorProviderContributionSourceDeclarationRegistry(
        registry_id=declared["registry"].registry_id,
        declarations=declared["declarations"],
        factory_loader=(
            ImportlibAttentionOperatorProviderContributionFactoryLoader()
        ),
    )


def adapter_modules(declared):
    result = {}
    for path, factory in (
        (declared["cann_path"], declared["cann_factory"]),
        (declared["flash_path"], declared["flash_factory"]),
    ):
        module_name, _, attribute_name = path.rpartition(".")
        module = types.ModuleType(module_name)
        setattr(module, attribute_name, factory)
        result[module_name] = module
    return result


class ImportlibContributionLoaderCheckpoint(unittest.TestCase):
    """Concrete importlib loading remains explicit and declaration-gated."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_distribution_version_is_exact_and_missing_is_none(self):
        loader = ImportlibAttentionOperatorProviderContributionFactoryLoader()

        with patch(VERSION_TARGET, return_value="1.2.3") as version:
            self.assertEqual(loader.package_version("reviewed-adapter"), "1.2.3")
        version.assert_called_once_with("reviewed-adapter")

        with patch(
            VERSION_TARGET,
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            self.assertIsNone(loader.package_version("missing-adapter"))

    def test_factory_resolution_is_exact_and_errors_are_normalized(self):
        loader = ImportlibAttentionOperatorProviderContributionFactoryLoader()
        module = types.ModuleType("reviewed_adapter.attention")
        factory = object()
        module.build_contribution = factory

        with patch(IMPORT_TARGET, return_value=module) as importer:
            observed = loader.resolve_factory(
                "reviewed_adapter.attention.build_contribution"
            )
        self.assertIs(observed, factory)
        importer.assert_called_once_with("reviewed_adapter.attention")

        with self.assertRaisesRegex(SchemaError, "module and name"):
            loader.resolve_factory("factory_without_module")
        with self.assertRaisesRegex(SchemaError, "path is invalid"):
            loader.resolve_factory("reviewed-adapter.factory")
        with patch(IMPORT_TARGET, side_effect=ModuleNotFoundError("missing")):
            with self.assertRaisesRegex(SchemaError, "module .* unavailable"):
                loader.resolve_factory("missing_adapter.factory")
        empty_module = types.ModuleType("reviewed_adapter.empty")
        with patch(IMPORT_TARGET, return_value=empty_module):
            with self.assertRaisesRegex(SchemaError, "factory .* is absent"):
                loader.resolve_factory("reviewed_adapter.empty.factory")

    def test_manifest_mismatch_prevents_concrete_metadata_and_import_calls(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        registry = AttentionOperatorProviderContributionSourceDeclarationRegistry(
            registry_id="deployment.attention.incomplete-importlib.v1",
            declarations=(declared["declarations"][0],),
            factory_loader=(
                ImportlibAttentionOperatorProviderContributionFactoryLoader()
            ),
        )

        with patch(VERSION_TARGET) as version, patch(IMPORT_TARGET) as importer:
            with self.assertRaisesRegex(SchemaError, "declarations differ"):
                registry.load_sources(declared["manifest"])

        version.assert_not_called()
        importer.assert_not_called()
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_missing_adapter_distribution_fails_before_module_import(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        registry = concrete_registry(declared)

        with patch(
            VERSION_TARGET,
            side_effect=importlib.metadata.PackageNotFoundError,
        ) as version, patch(IMPORT_TARGET) as importer:
            with self.assertRaisesRegex(SchemaError, "version is unsupported"):
                registry.load_sources(declared["manifest"])

        version.assert_called_once_with(declared["cann_package"])
        importer.assert_not_called()
        self.assertEqual(declared["factory_events"], [])
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

    def test_concrete_loader_reaches_bundle_then_plan_selects_cann(self):
        values = two_provider_assembly_inputs()
        declared = declaration_values(values)
        registry = concrete_registry(declared)
        modules = adapter_modules(declared)

        with patch(VERSION_TARGET, return_value="1.0.0") as version, patch(
            IMPORT_TARGET,
            side_effect=lambda module_name: modules[module_name],
        ) as importer:
            bundle = (
                assemble_attention_operator_provider_integration_source_declarations(
                    bundle_id="checkpoint.092.importlib.bundle.v1",
                    catalog_name="checkpoint-092-importlib-catalog",
                    scoring_manifest_id="checkpoint.092.importlib.scoring.v1",
                    source_declarations=registry,
                    approval_manifest=declared["manifest"],
                )
            )

        self.assertEqual(
            version.call_args_list,
            [call(declared["cann_package"]), call(declared["flash_package"])],
        )
        self.assertEqual(
            importer.call_args_list,
            [
                call(declared["cann_path"].rpartition(".")[0]),
                call(declared["flash_path"].rpartition(".")[0]),
            ],
        )
        self.assertEqual(
            declared["factory_events"],
            ["build:cann", "build:flash_attention_npu"],
        )
        for source_binding in (
            bundle.contribution_source_registry_binding.source_bindings
        ):
            origin = source_binding.origin_binding
            self.assertEqual(origin.observed_package_version, "1.0.0")
            self.assertEqual(
                origin.factory_loader_id,
                ImportlibAttentionOperatorProviderContributionFactoryLoader.loader_id,
            )
        self.assertEqual(values["cann"]["loader"].version_calls, 0)
        self.assertEqual(values["flash_loader"].version_calls, 0)

        install_attention_operator_provider_integration_bundle(
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
        self.assertGreater(values["cann"]["loader"].version_calls, 0)
        self.assertGreater(values["flash_loader"].version_calls, 0)
        self.assertEqual(values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(values["flash_loader"].resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
