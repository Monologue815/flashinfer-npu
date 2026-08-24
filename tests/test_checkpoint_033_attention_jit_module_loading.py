import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import flashinfer_npu
from flashinfer_npu.attention import (
    AttentionOperatorIntegrationError,
    AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
    BatchAttention,
)
from flashinfer_npu.attention.schema import AttentionMode
from flashinfer_npu.jit import JitCacheIndex, JitLoadedModule
from flashinfer_npu.jit.attention import (
    ConfiguredAttentionJitModuleResolver,
    attention_jit_entry_points,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_014_public_batch_attention_provider_runtime import (
    plan_public_wrapper,
)
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalRunAdapter,
    build_components,
    framework_plan,
)
from tests.test_checkpoint_030_attention_jit_framework import runtime_environment
from tests.test_checkpoint_031_attention_jit_active_plan import (
    FakeJitAuthorityResolver,
    FakeJitAutoResolver,
    RecordingJitArtifactResolver,
    RecordingJitPlanResolver,
    SyntheticModuleLoader,
    package_jit_implementation,
)
from tests.test_checkpoint_032_attention_jit_artifact_verification import (
    direct_artifact_case,
)


class LyingIdentityLoader(SyntheticModuleLoader):
    def load(self, record, verification, required_entry_points):
        loaded = super().load(record, verification, required_entry_points)
        receipt = replace(loaded.receipt, loader_id="different.loader")
        return JitLoadedModule(receipt, loaded.opaque_module)


class AttentionJitModuleLoadingCheckpointTests(unittest.TestCase):
    """Checkpoint 033: exact loaded-module identity remains wrapper-owned."""

    def wrapper(self, resolver):
        registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            return BatchAttention(kv_layout="NHD", device="npu:0")

    def loaded_case(self, *, symbol_mode="exact", loader=None):
        plan, binding, record, cache, _, _, artifact_resolver = (
            direct_artifact_case()
        )
        artifact_binding = artifact_resolver.resolve(binding)
        if loader is None:
            loader = SyntheticModuleLoader(symbol_mode=symbol_mode)
        resolver = ConfiguredAttentionJitModuleResolver(cache, loader)
        return (
            plan,
            binding,
            artifact_binding,
            record,
            cache,
            loader,
            resolver,
        )

    def test_entry_point_contract_matches_flashinfer_module_shape(self):
        self.assertEqual(
            attention_jit_entry_points(AttentionMode.SINGLE_PREFILL),
            ("run",),
        )
        self.assertEqual(
            attention_jit_entry_points(AttentionMode.SINGLE_DECODE),
            ("run",),
        )
        for mode in (
            AttentionMode.BATCH_PREFILL_PAGED,
            AttentionMode.BATCH_PREFILL_RAGGED,
            AttentionMode.BATCH_DECODE_PAGED,
            AttentionMode.BATCH_MIXED_PAGED,
        ):
            self.assertEqual(attention_jit_entry_points(mode), ("plan", "run"))

    def test_exact_loader_receipt_binds_artifact_and_symbols(self):
        _, binding, artifact, record, _, loader, resolver = self.loaded_case()
        module_binding = resolver.resolve(binding, artifact)
        module_binding.validate_bindings(binding, artifact)
        module_binding.loaded_module.receipt.validate(
            record,
            artifact.verification,
            binding.module_spec.jit_spec.entry_points,
        )
        self.assertEqual(
            tuple(
                item.name
                for item in module_binding.loaded_module.receipt.symbols
            ),
            ("plan", "run"),
        )
        self.assertEqual(len(loader.calls), 1)

    def test_missing_or_extra_symbol_fails_closed(self):
        for mode in ("missing", "extra"):
            _, binding, artifact, _, _, loader, resolver = self.loaded_case(
                symbol_mode=mode
            )
            with self.assertRaisesRegex(SchemaError, "artifact and symbols"):
                resolver.resolve(binding, artifact)
            self.assertEqual(len(loader.calls), 1)

    def test_loader_cannot_change_declared_identity(self):
        loader = LyingIdentityLoader()
        _, binding, artifact, _, _, _, resolver = self.loaded_case(loader=loader)
        with self.assertRaisesRegex(SchemaError, "declared identity"):
            resolver.resolve(binding, artifact)

    def test_cache_record_must_survive_until_load(self):
        _, binding, artifact, _, _, loader, _ = self.loaded_case()
        resolver = ConfiguredAttentionJitModuleResolver(
            JitCacheIndex(), loader
        )
        with self.assertRaisesRegex(SchemaError, "disappeared before load"):
            resolver.resolve(binding, artifact)
        self.assertEqual(loader.calls, [])

    def test_missing_module_resolver_stops_before_jit_and_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(
            components["events"], environment
        )
        artifact_resolver = RecordingJitArtifactResolver(
            components["events"], jit_resolver
        )
        implementation = AttentionOperatorPackageRuntimeImplementation(
            priority=base.priority,
            package_resolver=base._package_resolver,
            plan_gate=components["gate"],
            authority_resolver=FakeJitAuthorityResolver(
                components["events"], environment
            ),
            logical_factory=components["factory"],
            logical_run_adapter=FakeLogicalRunAdapter(),
            tensor_materializer=components["materializer"],
            jit_plan_resolver=jit_resolver,
            jit_artifact_resolver=artifact_resolver,
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT module resolver"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertEqual(artifact_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_module_load_precedes_package_callable_import(self):
        components, implementation, _, _, module_resolver = (
            package_jit_implementation()
        )
        resolved = implementation.resolve(framework_plan(), "npu:0")
        events = components["events"]
        self.assertIsNotNone(resolved.jit_module_binding)
        self.assertLess(events.index("artifact_verify"), events.index("module_load"))
        self.assertLess(events.index("module_load"), events.index("resolve_callable"))
        self.assertEqual(module_resolver.calls, 1)

    def test_public_plan_owns_module_and_run_does_not_reload(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        module_binding = runtime.jit_module_binding
        self.assertEqual(
            runtime.operator_session.active_plan.jit_module_binding_fingerprint,
            module_binding.fingerprint,
        )
        self.assertEqual(resolver.module_events, ["module_load"])
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertEqual(resolver.module_events, ["module_load"])

    def test_failed_module_replan_preserves_old_runtime(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_module = runtime.jit_module_binding
        old_executor = resolver.executors[0]

        resolver.module_symbol_mode = "missing"
        with self.assertRaises(SchemaError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_module_binding, old_module)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_loaded_module_drift_before_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = resolver.executors[0]

        other = self.wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_module_binding = (
            other._operator_runtime.jit_module_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "module active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_public_surface_has_no_module_handle_or_build_control(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("module", parameters)
            self.assertNotIn("jit", parameters)
        self.assertFalse(hasattr(flashinfer_npu.jit, "build_and_load"))
        self.assertFalse(hasattr(flashinfer_npu.jit, "default_module_loader"))


if __name__ == "__main__":
    unittest.main()
