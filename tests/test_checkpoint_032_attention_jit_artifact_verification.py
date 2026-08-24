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
from flashinfer_npu.jit import (
    ConfiguredJitArtifactVerifier,
    JitCacheIndex,
    JitCompilationPolicy,
    verify_jit_cache_record_payload,
)
from flashinfer_npu.jit.attention import (
    ConfiguredAttentionJitArtifactResolver,
    resolve_attention_jit_plan,
)
from flashinfer_npu.runtime import ArtifactVerificationError, SchemaError
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
    RecordingJitPlanResolver,
    SyntheticArtifactReader,
    cache_payload,
    cache_record,
    jit_receipt,
    package_jit_implementation,
)


def direct_artifact_case():
    plan = framework_plan()
    environment = runtime_environment()
    receipt = jit_receipt(plan, environment, "cann")
    # The plan resolver creates the exact module spec before cache lookup.  A
    # first recipe-only pass lets the synthetic fixture create matching bytes.
    empty = resolve_attention_jit_plan(
        plan,
        receipt,
        environment,
        JitCacheIndex(),
        JitCompilationPolicy.CACHE_ONLY,
    )
    record = cache_record(empty.module_spec.jit_spec)
    cache = JitCacheIndex((record,))
    binding = resolve_attention_jit_plan(
        plan,
        receipt,
        environment,
        cache,
        JitCompilationPolicy.CACHE_ONLY,
    )
    payload = cache_payload(binding.module_spec.jit_spec)
    reader = SyntheticArtifactReader(
        {record.artifact.fingerprint: payload}
    )
    resolver = ConfiguredAttentionJitArtifactResolver(
        cache,
        ConfiguredJitArtifactVerifier("checkpoint032.synthetic", reader),
    )
    return plan, binding, record, cache, payload, reader, resolver


class NonBytesReader:
    def read(self, artifact):
        return "not-bytes"


class AttentionJitArtifactVerificationCheckpointTests(unittest.TestCase):
    """Checkpoint 032: byte identity is required before runtime publication."""

    def wrapper(self, resolver):
        registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            return BatchAttention(kv_layout="NHD", device="npu:0")

    def test_public_api_has_no_artifact_or_loader_controls(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("artifact", parameters)
            self.assertNotIn("loader", parameters)
        self.assertFalse(hasattr(flashinfer_npu.jit, "load_jit_module"))
        self.assertFalse(hasattr(flashinfer_npu.jit, "compile_jit_spec"))

    def test_exact_payload_produces_record_bound_receipt(self):
        _, binding, record, _, _, reader, resolver = direct_artifact_case()
        artifact_binding = resolver.resolve(binding)
        artifact_binding.validate_plan_binding(binding)
        artifact_binding.verification.validate_record(record)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(
            artifact_binding.verification.payload_digest,
            record.artifact.digest,
        )

    def test_wrong_size_and_digest_fail_closed(self):
        _, _, record, _, payload, _, _ = direct_artifact_case()
        with self.assertRaisesRegex(ArtifactVerificationError, "size"):
            verify_jit_cache_record_payload(
                record,
                payload[:-1],
                verifier_id="checkpoint032.synthetic",
            )
        corrupted = payload[:-1] + bytes([payload[-1] ^ 1])
        with self.assertRaisesRegex(ArtifactVerificationError, "digest"):
            verify_jit_cache_record_payload(
                record,
                corrupted,
                verifier_id="checkpoint032.synthetic",
            )

    def test_reader_must_return_bytes(self):
        _, binding, _, cache, _, _, _ = direct_artifact_case()
        resolver = ConfiguredAttentionJitArtifactResolver(
            cache,
            ConfiguredJitArtifactVerifier(
                "checkpoint032.nonbytes", NonBytesReader()
            ),
        )
        with self.assertRaisesRegex(TypeError, "must return bytes"):
            resolver.resolve(binding)

    def test_cache_record_must_still_match_resolution(self):
        _, binding, _, _, _, _, _ = direct_artifact_case()
        reader = SyntheticArtifactReader({})
        resolver = ConfiguredAttentionJitArtifactResolver(
            JitCacheIndex(),
            ConfiguredJitArtifactVerifier("checkpoint032.empty", reader),
        )
        with self.assertRaisesRegex(SchemaError, "record disappeared"):
            resolver.resolve(binding)
        self.assertEqual(reader.calls, [])

    def test_missing_artifact_resolver_stops_before_plan_or_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(
            components["events"], environment
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
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT artifact resolver"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_corrupt_artifact_stops_before_package_import(self):
        components, implementation, _, artifact_resolver, module_resolver = (
            package_jit_implementation(artifact_mode="corrupt")
        )
        with self.assertRaises(ArtifactVerificationError):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(artifact_resolver.calls, 1)
        self.assertEqual(module_resolver.calls, 0)
        self.assertIn("artifact_verify", components["events"])
        self.assertNotIn("resolve_callable", components["events"])
        self.assertEqual(components["loader"].resolve_calls, 0)

    def test_public_plan_owns_artifact_binding_and_run_does_not_reread(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        artifact_binding = runtime.jit_artifact_binding
        self.assertEqual(
            runtime.operator_session.active_plan.jit_artifact_binding_fingerprint,
            artifact_binding.fingerprint,
        )
        self.assertEqual(resolver.artifact_events, ["artifact_verify"])
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertEqual(resolver.artifact_events, ["artifact_verify"])

    def test_corrupt_replan_preserves_previous_plan_artifact_and_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_artifact = runtime.jit_artifact_binding
        old_executor = resolver.executors[0]

        resolver.artifact_mode = "corrupt"
        with self.assertRaises(ArtifactVerificationError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_artifact_binding, old_artifact)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_artifact_binding_drift_before_executor(self):
        first_resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(first_resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = first_resolver.executors[0]

        other = self.wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_artifact_binding = (
            other._operator_runtime.jit_artifact_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "artifact active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_artifact_identity_changes_active_plan_fingerprint(self):
        resolver = FakeJitAutoResolver()
        wrapper = self.wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        active = wrapper._operator_runtime.operator_session.active_plan
        changed = replace(
            active,
            jit_artifact_binding_fingerprint="0" * 64,
        )
        self.assertNotEqual(active.fingerprint, changed.fingerprint)


if __name__ == "__main__":
    unittest.main()
