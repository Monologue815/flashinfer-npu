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
from flashinfer_npu.runtime import Backend, SchemaError
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
    RecordingJitModuleResolver,
    RecordingJitPlanResolver,
    SyntheticJitExecutorBinder,
    fake_operation,
    package_jit_implementation,
)


class AttentionJitPlannerBindingCheckpointTests(unittest.TestCase):
    """The loaded JIT plan symbol remains private and wrapper-owned."""

    def wrapper(self, resolver):
        registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            return BatchAttention(kv_layout="NHD", device="npu:0")

    def test_exact_planner_receipt_closes_module_and_plan_factory(self):
        wrapper = self.wrapper(FakeJitAutoResolver())
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        binding = runtime.jit_planner_binding

        binding.validate(runtime.jit_module_binding)
        self.assertEqual(
            binding.operation_id,
            runtime.operator_session.active_plan.prepared_plan.implementation_id,
        )
        self.assertNotIn("factory", binding.to_dict())
        self.assertEqual(len(binding.fingerprint), 64)

    def test_missing_planner_binder_stops_before_jit_or_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(components["events"], environment)
        artifact_resolver = RecordingJitArtifactResolver(
            components["events"], jit_resolver
        )
        module_resolver = RecordingJitModuleResolver(
            components["events"], jit_resolver
        )
        executor_binder = SyntheticJitExecutorBinder(
            "cann", fake_operation().operation_id, components["events"]
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
            jit_module_resolver=module_resolver,
            jit_executor_binder=executor_binder,
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT planner binder"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_planner_binding_precedes_callable_and_executor_binding(self):
        components, implementation, _, _, _ = package_jit_implementation()
        resolved = implementation.resolve(framework_plan(), "npu:0")
        events = components["events"]

        self.assertLess(events.index("module_load"), events.index("planner_bind"))
        self.assertLess(events.index("planner_bind"), events.index("resolve_callable"))
        self.assertLess(events.index("resolve_callable"), events.index("executor_bind"))
        self.assertIs(resolved.factory, resolved.jit_planner_binding.factory)
        self.assertEqual(len(components["jit_planner_binder"].calls), 1)

    def test_stale_planner_receipt_fails_before_plan_publication(self):
        resolver = FakeJitAutoResolver()
        resolver.planner_binding_mode = "stale_module"
        wrapper = self.wrapper(resolver)

        with self.assertRaisesRegex(SchemaError, "planner binding is stale"):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        self.assertFalse(wrapper._operator_runtime.is_planned)
        self.assertEqual(resolver.executors, [])

    def test_public_plan_owns_planner_binding_without_rebinding_on_run(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        binding = runtime.jit_planner_binding

        self.assertEqual(
            runtime.operator_session.active_plan.jit_planner_binding_fingerprint,
            binding.fingerprint,
        )
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertIs(runtime.jit_planner_binding, binding)
        self.assertEqual(len(resolver.planner_bindings), 1)

    def test_failed_planner_replan_preserves_previous_runtime(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_binding = runtime.jit_planner_binding
        old_executor = resolver.executors[0]

        resolver.planner_binding_mode = "stale_module"
        with self.assertRaises(SchemaError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_planner_binding, old_binding)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_planner_binding_drift_before_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = resolver.executors[0]

        other = self.wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_planner_binding = (
            other._operator_runtime.jit_planner_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "planner active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_non_jit_runtime_rejects_planner_binding(self):
        resolver = FakeJitAutoResolver()
        resolved = resolver.resolve(framework_plan(), "npu:0")
        aot_receipt = replace(resolved.receipt, backend=Backend.ASCENDC_AOT)
        aot_selection = replace(
            resolved.selection,
            dispatch_receipt_fingerprint=aot_receipt.fingerprint,
            backend=Backend.ASCENDC_AOT,
        )
        with self.assertRaisesRegex(SchemaError, "non-JIT.*JIT bindings"):
            replace(
                resolved,
                receipt=aot_receipt,
                selection=aot_selection,
                jit_plan_binding=None,
                jit_artifact_binding=None,
                jit_module_binding=None,
                jit_executor_binding=None,
            )

    def test_public_surface_has_no_planner_or_jit_handle(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("planner", parameters)
            self.assertNotIn("jit", parameters)
        self.assertFalse(hasattr(flashinfer_npu.jit, "default_planner_binder"))


if __name__ == "__main__":
    unittest.main()
