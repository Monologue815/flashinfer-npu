import hashlib
import inspect
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from flashinfer_npu.attention import (
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorIntegrationError,
    AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorRuntimeAuthority,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionResolvedOperatorRuntime,
    AttentionStateError,
    BatchAttention,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedRunAdapter,
    bind_attention_operator_callable,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.jit import (
    ConfiguredJitArtifactVerifier,
    JitLoadedModule,
    JitModuleLoadReceipt,
    JitResolvedSymbol,
    JitCacheIndex,
    JitCacheRecord,
    JitCompilationPolicy,
    JitSpecRegistry,
    MissingJITCacheError,
)
from flashinfer_npu.jit.attention import (
    ConfiguredAttentionJitArtifactResolver,
    ConfiguredAttentionJitModuleResolver,
    ConfiguredAttentionJitPlanResolver,
    gen_attention_jit_module_spec,
    resolve_attention_jit_plan,
)
from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    Backend,
    SchemaError,
)
from tests.test_checkpoint_014_public_batch_attention_provider_runtime import (
    FakeExecutor,
    plan_public_wrapper,
)
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalRunAdapter,
    build_components,
    fake_operation,
    framework_plan,
)
from tests.test_checkpoint_030_attention_jit_framework import runtime_environment


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def jit_receipt(plan, environment, provider_id="flash_attention_npu"):
    return AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=digest("checkpoint-031-numerics"),
        profile_id="checkpoint.031.%s.profile" % provider_id,
        profile_fingerprint=digest("checkpoint-031-profile-" + provider_id),
        rule_id="checkpoint_031_rule",
        environment_fingerprint=environment.fingerprint,
        evidence_id="checkpoint-031-evidence",
        evidence_result_digest=digest("checkpoint-031-evidence"),
        kernel_id="checkpoint-031-attention-jit",
        kernel_fingerprint=digest("checkpoint-031-kernel"),
        artifact_fingerprint=digest("checkpoint-031-source"),
        launch_abi_fingerprint=digest("checkpoint-031-launch"),
        binary_abi_fingerprint=digest("checkpoint-031-binary"),
        backend=Backend.ASCENDC_JIT,
        float_workspace_bytes=4096,
        int_workspace_bytes=1024,
        float_workspace_alignment=64,
        int_workspace_alignment=64,
        selection_source="priority",
        requested_backend="auto",
    )


def cache_record(spec):
    payload = cache_payload(spec)
    artifact = ArtifactRef(
        kind=ArtifactKind.FILE,
        format=ArtifactFormat.ASCENDC_OBJECT,
        locator="jit-cache/%s.o" % spec.name,
        digest=hashlib.sha256(payload).hexdigest(),
        target_soc=spec.target_soc,
        build_id="checkpoint-031-synthetic-cache",
        size_bytes=len(payload),
    )
    return JitCacheRecord(
        spec_name=spec.name,
        spec_fingerprint=spec.fingerprint,
        environment_fingerprint=spec.environment_fingerprint,
        artifact=artifact,
        producer_id="checkpoint-031-fake-builder",
        build_metadata_fingerprint=digest("checkpoint-031-build-metadata"),
    )


def cache_payload(spec):
    seed = ("checkpoint-031-object-" + spec.name).encode("ascii")
    return (seed * (256 // len(seed) + 1))[:256]


class SyntheticArtifactReader:
    def __init__(self, payloads, events=None):
        self.payloads = dict(payloads)
        self.events = events
        self.calls = []

    def read(self, artifact):
        self.calls.append(artifact)
        if self.events is not None:
            self.events.append("artifact_verify")
        return self.payloads[artifact.fingerprint]


def verified_artifact_binding(binding, cache, record, events=None, payload=None):
    if payload is None:
        payload = cache_payload(binding.module_spec.jit_spec)
    reader = SyntheticArtifactReader(
        {record.artifact.fingerprint: payload},
        events,
    )
    resolver = ConfiguredAttentionJitArtifactResolver(
        cache,
        ConfiguredJitArtifactVerifier("checkpoint031.synthetic", reader),
    )
    return resolver.resolve(binding)


class SyntheticModuleLoader:
    loader_id = "checkpoint031.synthetic"
    loader_version = "1"

    def __init__(self, events=None, symbol_mode="exact"):
        self.events = events
        self.symbol_mode = symbol_mode
        self.calls = []

    def load(self, record, verification, required_entry_points):
        self.calls.append((record, verification, required_entry_points))
        if self.events is not None:
            self.events.append("module_load")
        names = tuple(required_entry_points)
        if self.symbol_mode == "missing":
            names = names[:-1]
        elif self.symbol_mode == "extra":
            names = names + ("unexpected",)
        symbols = tuple(
            JitResolvedSymbol(name, "symbol:%s" % name) for name in names
        )
        receipt = JitModuleLoadReceipt(
            spec_name=record.spec_name,
            spec_fingerprint=record.spec_fingerprint,
            cache_record_fingerprint=record.fingerprint,
            artifact_verification_fingerprint=verification.fingerprint,
            artifact_fingerprint=record.artifact.fingerprint,
            loader_id=self.loader_id,
            loader_version=self.loader_version,
            module_token="module:%s" % record.spec_name,
            load_generation=1,
            symbols=symbols,
        )
        return JitLoadedModule(receipt, object())


def loaded_module_binding(
    binding,
    artifact_binding,
    cache,
    events=None,
    symbol_mode="exact",
):
    loader = SyntheticModuleLoader(events, symbol_mode)
    resolver = ConfiguredAttentionJitModuleResolver(cache, loader)
    return resolver.resolve(binding, artifact_binding), loader


def callable_authority(plan, receipt, provider_id, operation_id):
    operation = load_packaged_attention_operator_catalog().get(operation_id)
    probe = AttentionOperatorProviderProbe(
        provider_id=provider_id,
        adapter_version="checkpoint-031-test",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    selection = AttentionOperatorProviderSelection(
        provider_id=provider_id,
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=digest("checkpoint-031-provider-record"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    observed = AttentionObservedOperatorCallable(
        provider_id=provider_id,
        package_name=operation.package_name,
        package_version="test-package",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=AttentionObservedCallableSignature(
            operation.positional_arguments,
            operation.keyword_arguments,
            observation_kind="checkpoint-031-test",
        ),
    )
    return selection, bind_attention_operator_callable(probe, operation, observed)


class FakeJitAutoResolver:
    """Synthetic resolver that never compiles, loads, imports, or calls an op."""

    def __init__(self):
        self.environment = runtime_environment()
        self.registry = JitSpecRegistry()
        self.cache_mode = "hit"
        self.artifact_mode = "valid"
        self.resolve_calls = []
        self.executors = []
        self.bindings = []
        self.artifact_bindings = []
        self.artifact_events = []
        self.module_bindings = []
        self.module_events = []
        self.module_symbol_mode = "exact"

    def resolve(self, plan, device):
        self.resolve_calls.append((plan, device))
        receipt = jit_receipt(plan, self.environment)
        module = gen_attention_jit_module_spec(
            plan, receipt, self.environment, registry=None
        )
        cache = JitCacheIndex()
        record = None
        if self.cache_mode == "hit":
            record = cache.publish(cache_record(module.jit_spec))
            policy = JitCompilationPolicy.CACHE_ONLY
        elif self.cache_mode == "build_required":
            policy = JitCompilationPolicy.ENABLED
        else:
            policy = JitCompilationPolicy.CACHE_ONLY
        resolver = ConfiguredAttentionJitPlanResolver(
            self.environment,
            cache,
            policy,
            registry=self.registry,
        )
        binding = resolver.resolve(plan, receipt)
        artifact_binding = None
        module_binding = None
        if binding.ready:
            payload = cache_payload(binding.module_spec.jit_spec)
            if self.artifact_mode == "corrupt":
                payload = payload[:-1] + bytes([payload[-1] ^ 1])
            artifact_binding = verified_artifact_binding(
                binding, cache, record, self.artifact_events, payload
            )
            module_binding, _ = loaded_module_binding(
                binding,
                artifact_binding,
                cache,
                self.module_events,
                self.module_symbol_mode,
            )
        selection, callable_binding = callable_authority(
            plan,
            receipt,
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
        )
        executor = FakeExecutor(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            plan.generation,
        )
        resolved = AttentionResolvedOperatorRuntime(
            framework_plan_fingerprint=plan.fingerprint,
            factory=FlashAttentionNpuV3PagedPlanFactory(),
            run_adapter=FlashAttentionNpuV3PagedRunAdapter(),
            executor=executor,
            receipt=receipt,
            selection=selection,
            callable_binding=callable_binding,
            jit_plan_binding=binding,
            jit_artifact_binding=artifact_binding,
            jit_module_binding=module_binding,
        )
        self.executors.append(executor)
        self.bindings.append(binding)
        self.artifact_bindings.append(artifact_binding)
        self.module_bindings.append(module_binding)
        return resolved


class FakeJitAuthorityResolver:
    provider_id = "cann"
    operation_id = fake_operation().operation_id

    def __init__(self, events, environment):
        self.events = events
        self.environment = environment

    def authorize(self, plan, device, operation, provider_probe):
        self.events.append("authorize")
        receipt = jit_receipt(plan, self.environment, "cann")
        selection = AttentionOperatorProviderSelection(
            provider_id="cann",
            provider_probe_fingerprint=provider_probe.fingerprint,
            provider_record_fingerprint=digest("checkpoint-031-package-record"),
            dispatch_receipt_fingerprint=receipt.fingerprint,
            profile_id=receipt.profile_id,
            profile_fingerprint=receipt.profile_fingerprint,
            backend=receipt.backend,
        )
        return AttentionOperatorRuntimeAuthority(
            framework_plan_fingerprint=plan.fingerprint,
            device=device,
            provider_probe_fingerprint=provider_probe.fingerprint,
            operation_id=operation.operation_id,
            operation_fingerprint=operation.fingerprint,
            receipt=receipt,
            selection=selection,
        )


class RecordingJitPlanResolver:
    def __init__(self, events, environment, cache_mode="hit"):
        self.events = events
        self.environment = environment
        self.cache_mode = cache_mode
        self.calls = 0
        self.cache = None
        self.record = None

    def resolve(self, plan, receipt):
        self.calls += 1
        self.events.append("jit_resolve")
        module = gen_attention_jit_module_spec(
            plan, receipt, self.environment, registry=None
        )
        cache = JitCacheIndex()
        if self.cache_mode == "hit":
            self.record = cache.publish(cache_record(module.jit_spec))
        self.cache = cache
        return resolve_attention_jit_plan(
            plan,
            receipt,
            self.environment,
            cache,
            JitCompilationPolicy.CACHE_ONLY,
        )


class RecordingJitArtifactResolver:
    def __init__(self, events, jit_resolver, artifact_mode="valid"):
        self.events = events
        self.jit_resolver = jit_resolver
        self.calls = 0
        self.artifact_mode = artifact_mode

    def resolve(self, binding):
        self.calls += 1
        payload = cache_payload(binding.module_spec.jit_spec)
        if self.artifact_mode == "corrupt":
            payload = payload[:-1] + bytes([payload[-1] ^ 1])
        return verified_artifact_binding(
            binding,
            self.jit_resolver.cache,
            self.jit_resolver.record,
            self.events,
            payload,
        )


class RecordingJitModuleResolver:
    def __init__(self, events, jit_resolver, symbol_mode="exact"):
        self.events = events
        self.jit_resolver = jit_resolver
        self.symbol_mode = symbol_mode
        self.calls = 0
        self.loader = None

    def resolve(self, plan_binding, artifact_binding):
        self.calls += 1
        result, self.loader = loaded_module_binding(
            plan_binding,
            artifact_binding,
            self.jit_resolver.cache,
            self.events,
            self.symbol_mode,
        )
        return result


def package_jit_implementation(
    cache_mode="hit", artifact_mode="valid", symbol_mode="exact"
):
    components = build_components()
    environment = runtime_environment()
    authority = FakeJitAuthorityResolver(components["events"], environment)
    jit_resolver = RecordingJitPlanResolver(
        components["events"], environment, cache_mode
    )
    artifact_resolver = RecordingJitArtifactResolver(
        components["events"], jit_resolver, artifact_mode
    )
    module_resolver = RecordingJitModuleResolver(
        components["events"], jit_resolver, symbol_mode
    )
    base = components["implementation"]
    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=base.priority,
        package_resolver=base._package_resolver,
        plan_gate=components["gate"],
        authority_resolver=authority,
        logical_factory=components["factory"],
        logical_run_adapter=FakeLogicalRunAdapter(),
        tensor_materializer=components["materializer"],
        jit_plan_resolver=jit_resolver,
        jit_artifact_resolver=artifact_resolver,
        jit_module_resolver=module_resolver,
    )
    return (
        components,
        implementation,
        jit_resolver,
        artifact_resolver,
        module_resolver,
    )


class AttentionJitActivePlanCheckpointTests(unittest.TestCase):
    """Checkpoint 031: wrapper owns and revalidates exact JIT readiness."""

    def wrapper(self, resolver):
        registry = AttentionOperatorRuntimeResolverRegistry((('npu', resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            wrapper = BatchAttention(kv_layout="NHD", device="npu:0")
        return wrapper

    def test_public_plan_run_signatures_stay_free_of_jit_controls(self):
        self.assertNotIn("jit", inspect.signature(BatchAttention).parameters)
        self.assertNotIn("jit", inspect.signature(BatchAttention.plan).parameters)
        self.assertNotIn("jit", inspect.signature(BatchAttention.run).parameters)
        self.assertNotIn("cache", inspect.signature(BatchAttention.plan).parameters)

    def test_public_plan_publishes_ready_jit_binding_and_run_reuses_it(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))

        runtime = wrapper._operator_runtime
        binding = runtime.jit_plan_binding
        artifact_binding = runtime.jit_artifact_binding
        self.assertTrue(binding.ready)
        self.assertEqual(
            runtime.operator_session.active_plan.jit_plan_binding_fingerprint,
            binding.fingerprint,
        )
        self.assertEqual(
            runtime.operator_session.active_plan.jit_artifact_binding_fingerprint,
            artifact_binding.fingerprint,
        )
        self.assertEqual(
            binding.module_spec.framework_plan_fingerprint,
            wrapper.plan_state.fingerprint,
        )
        first = wrapper.run("q1", ("k", "v"))
        second = wrapper.run("q2", ("k", "v"))
        self.assertEqual(first, second)
        self.assertIs(runtime.jit_plan_binding, binding)
        self.assertEqual(len(resolver.resolve_calls), 1)
        self.assertEqual(len(resolver.executors[0].calls), 2)

    def test_cache_only_miss_rejects_plan_without_partial_publication(self):
        resolver = FakeJitAutoResolver()
        resolver.cache_mode = "miss"
        wrapper = self.wrapper(resolver)
        with self.assertRaises(MissingJITCacheError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        self.assertFalse(wrapper._operator_runtime.is_planned)
        self.assertEqual(resolver.executors, [])

    def test_build_required_is_not_misreported_as_runnable(self):
        resolver = FakeJitAutoResolver()
        resolver.cache_mode = "build_required"
        wrapper = self.wrapper(resolver)
        with self.assertRaisesRegex(MissingJITCacheError, "authorized build"):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        self.assertFalse(wrapper._operator_runtime.is_planned)
        self.assertFalse(hasattr(__import__("flashinfer_npu").jit, "build_jit_specs"))

    def test_failed_jit_replan_preserves_old_plan_binding_and_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_binding = runtime.jit_plan_binding
        old_executor = resolver.executors[0]

        resolver.cache_mode = "miss"
        with self.assertRaises(MissingJITCacheError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))

        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_plan_binding, old_binding)
        self.assertEqual(
            wrapper.run("old", ("k", "v"))[0],
            "public-output-generation-1",
        )
        self.assertEqual(len(old_executor.calls), 1)

    def test_successful_replan_reuses_recipe_but_rebinds_exact_plan(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        first = wrapper._operator_runtime.jit_plan_binding
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        second = wrapper._operator_runtime.jit_plan_binding

        self.assertEqual(first.module_spec.jit_spec, second.module_spec.jit_spec)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            resolver.registry.get_stats(),
            {"total": 1, "source_materialized": 0, "recipe_only": 1},
        )

    def test_run_revalidates_internal_binding_before_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        first_executor = resolver.executors[0]

        other_wrapper = self.wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other_wrapper, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_plan_binding = (
            other_wrapper._operator_runtime.jit_plan_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "active-plan identity"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(first_executor.calls, [])

    def test_jit_runtime_requires_binding_and_aot_forbids_it(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        # The exact constructor gate is exercised through a real successful
        # resolve, then by rebuilding with only the binding field changed.
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        resolved_plan = resolver.resolve_calls[0][0]
        receipt = jit_receipt(resolved_plan, resolver.environment)
        selection, callable_binding = callable_authority(
            resolved_plan,
            receipt,
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
        )
        executor = FakeExecutor(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            1,
        )
        with self.assertRaisesRegex(SchemaError, "requires.*JIT plan binding"):
            AttentionResolvedOperatorRuntime(
                framework_plan_fingerprint=resolved_plan.fingerprint,
                factory=FlashAttentionNpuV3PagedPlanFactory(),
                run_adapter=FlashAttentionNpuV3PagedRunAdapter(),
                executor=executor,
                receipt=receipt,
                selection=selection,
                callable_binding=callable_binding,
            )
        binding = wrapper._operator_runtime.jit_plan_binding
        aot_receipt = replace(receipt, backend=Backend.ASCENDC_AOT)
        aot_selection = replace(
            selection,
            dispatch_receipt_fingerprint=aot_receipt.fingerprint,
            backend=Backend.ASCENDC_AOT,
        )
        with self.assertRaisesRegex(SchemaError, "non-JIT.*JIT bindings"):
            AttentionResolvedOperatorRuntime(
                framework_plan_fingerprint=resolved_plan.fingerprint,
                factory=FlashAttentionNpuV3PagedPlanFactory(),
                run_adapter=FlashAttentionNpuV3PagedRunAdapter(),
                executor=executor,
                receipt=aot_receipt,
                selection=aot_selection,
                callable_binding=callable_binding,
                jit_plan_binding=binding,
            )

    def test_package_jit_runtime_requires_configured_resolver_before_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
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
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "configured JIT plan resolver"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertNotIn("resolve_callable", components["events"])
        self.assertEqual(components["loader"].resolve_calls, 0)

    def test_package_cache_miss_stops_before_callable_import(self):
        components, implementation, jit_resolver, artifact_resolver, module_resolver = (
            package_jit_implementation("miss")
        )
        with self.assertRaises(MissingJITCacheError):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 1)
        self.assertEqual(artifact_resolver.calls, 0)
        self.assertEqual(module_resolver.calls, 0)
        self.assertIn("authorize", components["events"])
        self.assertIn("jit_resolve", components["events"])
        self.assertNotIn("resolve_callable", components["events"])
        self.assertEqual(components["loader"].resolve_calls, 0)

    def test_package_cache_hit_precedes_callable_import(self):
        components, implementation, jit_resolver, artifact_resolver, module_resolver = (
            package_jit_implementation()
        )
        resolved = implementation.resolve(framework_plan(), "npu:0")
        events = components["events"]
        self.assertTrue(resolved.jit_plan_binding.ready)
        self.assertLess(events.index("authorize"), events.index("jit_resolve"))
        self.assertLess(events.index("jit_resolve"), events.index("artifact_verify"))
        self.assertLess(events.index("artifact_verify"), events.index("resolve_callable"))
        self.assertLess(events.index("artifact_verify"), events.index("module_load"))
        self.assertLess(events.index("module_load"), events.index("resolve_callable"))
        self.assertEqual(jit_resolver.calls, 1)
        self.assertEqual(artifact_resolver.calls, 1)
        self.assertEqual(module_resolver.calls, 1)

    def test_registry_and_cache_publication_are_thread_safe_and_idempotent(self):
        environment = runtime_environment()
        plan = framework_plan()
        receipt = jit_receipt(plan, environment, "cann")
        module = gen_attention_jit_module_spec(plan, receipt, environment)
        registry = JitSpecRegistry()
        cache = JitCacheIndex()
        record = cache_record(module.jit_spec)

        def publish(_):
            return (
                registry.register(module.jit_spec).fingerprint,
                cache.publish(record).fingerprint,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = tuple(pool.map(publish, range(32)))
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(registry.get_stats()["total"], 1)
        self.assertEqual(cache.records, (record,))


if __name__ == "__main__":
    unittest.main()
