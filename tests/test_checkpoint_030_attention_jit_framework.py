import hashlib
import json
import subprocess
import sys
import unittest

import flashinfer_npu
from flashinfer_npu.attention.capability import AttentionRuntimeEnvironment
from flashinfer_npu.attention.dispatch import AttentionDispatchReceipt
from flashinfer_npu.attention.planner import AttentionFrameworkSession
from flashinfer_npu.attention.schema import (
    AttentionMode,
    AttentionPlanSpec,
    SingleAttentionMetadata,
)
from flashinfer_npu.jit import (
    JitCacheIndex,
    JitCacheRecord,
    JitCompilationPolicy,
    JitEnvironment,
    JitResolutionState,
    JitSpec,
    JitSpecRegistry,
    MissingJITCacheError,
    require_jit_cache_hit,
    resolve_jit_spec,
)
from flashinfer_npu.jit.attention import (
    AttentionJitVariant,
    gen_attention_jit_module_spec,
    jit_environment_from_attention,
)
from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    Backend,
    QuantSpec,
    SchemaError,
)


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def runtime_environment(compiler_version="8.0.0"):
    return AttentionRuntimeEnvironment(
        soc_version="Ascend910B",
        soc_revision="2",
        driver_version="25.0.rc1",
        firmware_version="7.7.0",
        cann_version="8.0.0",
        torch_version="2.6.0",
        torch_npu_version="2.6.0",
        compiler_version=compiler_version,
        python_abi="cp310",
        ai_core_count=20,
        features=("bf16", "int8"),
    )


def framework_plan(kv_len=16, quant_spec=None):
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        causal=True,
        q_dtype="float16",
        kv_dtype="int8" if quant_spec is not None else "float16",
        kv_quant_spec=quant_spec,
    )
    session = AttentionFrameworkSession(AttentionMode.SINGLE_PREFILL)
    return session.plan(spec, SingleAttentionMetadata(qo_len=4, kv_len=kv_len))


def dispatch_receipt(plan, environment, backend=Backend.ASCENDC_JIT):
    return AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=digest("numerics"),
        profile_id="checkpoint.030.ascendc_jit",
        profile_fingerprint=digest("profile"),
        rule_id="checkpoint_030_rule",
        environment_fingerprint=environment.fingerprint,
        evidence_id="checkpoint-030-evidence",
        evidence_result_digest=digest("evidence"),
        kernel_id="checkpoint-030-attention-jit",
        kernel_fingerprint=digest("kernel"),
        artifact_fingerprint=digest("source-recipe"),
        launch_abi_fingerprint=digest("launch-abi"),
        binary_abi_fingerprint=digest("binary-abi"),
        backend=backend,
        float_workspace_bytes=4096,
        int_workspace_bytes=1024,
        float_workspace_alignment=64,
        int_workspace_alignment=64,
        selection_source="priority",
        requested_backend=backend.value,
    )


def generic_spec(name="attention.test"):
    return JitSpec(
        name=name,
        domain="attention",
        generator_id="attention.ascendc",
        generator_version="1",
        target_soc="Ascend910B",
        environment_fingerprint=digest("environment"),
        specialization_fingerprint=digest("variant"),
        input_fingerprints=(("kernel", digest("kernel")),),
    )


def compiled_artifact(target_soc="Ascend910B"):
    return ArtifactRef(
        kind=ArtifactKind.FILE,
        format=ArtifactFormat.ASCENDC_OBJECT,
        locator="jit-cache/attention.test.o",
        digest=digest("object-bytes"),
        target_soc=target_soc,
        build_id="checkpoint-030-build",
        size_bytes=123,
    )


def cache_record(spec, **changes):
    values = {
        "spec_name": spec.name,
        "spec_fingerprint": spec.fingerprint,
        "environment_fingerprint": spec.environment_fingerprint,
        "artifact": compiled_artifact(spec.target_soc),
        "producer_id": "checkpoint-030-fake-builder",
        "build_metadata_fingerprint": digest("build-metadata"),
    }
    values.update(changes)
    return JitCacheRecord(**values)


class AttentionJitFrameworkCheckpointTests(unittest.TestCase):
    """Checkpoint 030: JIT structure and decisions, never compilation."""

    def test_public_package_matches_flashinfer_jit_layering(self):
        self.assertIs(flashinfer_npu.jit.JitSpec, JitSpec)
        self.assertTrue(hasattr(flashinfer_npu.jit, "attention"))
        self.assertTrue(
            hasattr(flashinfer_npu.jit.attention, "gen_attention_jit_module_spec")
        )

    def test_environment_is_canonical_and_version_sensitive(self):
        first = JitEnvironment(
            "Ascend910B",
            "2",
            "8.0.0",
            "ascendc",
            "8.0.0",
            "2.6.0",
            "2.6.0",
            "cp310",
            features=("int8", "bf16"),
        )
        restored = JitEnvironment.from_dict(first.to_dict())
        changed = JitEnvironment.from_dict(
            dict(first.to_dict(), compiler_version="8.0.1")
        )
        self.assertEqual(first.features, ("bf16", "int8"))
        self.assertEqual(restored, first)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)

    def test_spec_round_trip_and_canonical_binding_order(self):
        spec = JitSpec(
            name="attention.roundtrip",
            domain="attention",
            generator_id="attention.ascendc",
            generator_version="1",
            target_soc="Ascend910B",
            environment_fingerprint=digest("environment"),
            specialization_fingerprint=digest("variant"),
            input_fingerprints=(
                ("launch_abi", digest("launch")),
                ("kernel", digest("kernel")),
            ),
            compile_options=("-O3", "--soc=Ascend910B"),
        )
        restored = JitSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(restored, spec)
        self.assertEqual(spec.input_fingerprints[0][0], "kernel")
        self.assertEqual(restored.fingerprint, spec.fingerprint)

    def test_spec_source_identity_never_reads_source_bytes(self):
        source = ArtifactRef(
            ArtifactKind.JIT_SOURCE,
            ArtifactFormat.ASCENDC_SOURCE,
            "kernels/attention/generated.asc",
            digest("source"),
            "Ascend910B",
            "generator-v1",
            42,
        )
        spec = JitSpec(
            name="attention.materialized",
            domain="attention",
            generator_id="attention.ascendc",
            generator_version="1",
            target_soc="Ascend910B",
            environment_fingerprint=digest("environment"),
            specialization_fingerprint=digest("variant"),
            source_artifacts=(source,),
            entry_points=("attention_run",),
        )
        self.assertTrue(spec.source_materialized)
        self.assertEqual(spec.source_artifacts, (source,))

    def test_spec_rejects_compiled_artifact_as_source(self):
        with self.assertRaisesRegex(SchemaError, "source artifacts"):
            JitSpec(
                name="attention.invalid",
                domain="attention",
                generator_id="attention.ascendc",
                generator_version="1",
                target_soc="Ascend910B",
                environment_fingerprint=digest("environment"),
                specialization_fingerprint=digest("variant"),
                source_artifacts=(compiled_artifact(),),
            )

    def test_registry_is_idempotent_but_rejects_name_drift(self):
        registry = JitSpecRegistry()
        spec = generic_spec()
        self.assertIs(registry.register(spec), spec)
        self.assertIs(registry.register(spec), spec)
        status = registry.get_spec_status(spec.name)
        self.assertEqual(status.registration_generation, 1)
        self.assertFalse(status.source_materialized)
        with self.assertRaisesRegex(SchemaError, "conflicting JIT spec"):
            registry.register(
                JitSpec.from_dict(
                    dict(spec.to_dict(), target_soc="Ascend310P")
                )
            )

    def test_registry_snapshots_do_not_expose_mutable_state(self):
        registry = JitSpecRegistry((generic_spec(),))
        snapshot = registry.get_all_specs()
        snapshot.clear()
        self.assertEqual(len(registry.specs("attention")), 1)
        self.assertEqual(
            registry.get_stats(),
            {"total": 1, "source_materialized": 0, "recipe_only": 1},
        )

    def test_cache_record_requires_exact_spec_environment_and_soc(self):
        spec = generic_spec()
        record = cache_record(spec)
        self.assertEqual(
            JitCacheRecord.from_dict(json.loads(json.dumps(record.to_dict()))),
            record,
        )
        self.assertIs(JitCacheIndex((record,)).lookup(spec), record)
        changed = JitSpec.from_dict(
            dict(spec.to_dict(), specialization_fingerprint=digest("changed"))
        )
        with self.assertRaisesRegex(SchemaError, "does not bind"):
            JitCacheIndex((record,)).lookup(changed)

    def test_cache_rejects_builtin_and_conflicting_publication(self):
        spec = generic_spec()
        builtin = ArtifactRef(
            ArtifactKind.BUILTIN,
            ArtifactFormat.ACLNN_BUILTIN,
            "builtin:aclnnAttention",
            digest("builtin"),
            "Ascend910B",
            "cann-8",
        )
        with self.assertRaisesRegex(SchemaError, "compiled file"):
            cache_record(spec, artifact=builtin)
        cache = JitCacheIndex((cache_record(spec),))
        with self.assertRaisesRegex(SchemaError, "conflicting JIT cache"):
            cache.publish(
                cache_record(
                    spec, build_metadata_fingerprint=digest("other")
                )
            )

    def test_resolution_is_pure_policy_decision(self):
        spec = generic_spec()
        hit = resolve_jit_spec(
            spec,
            JitCacheIndex((cache_record(spec),)),
            JitCompilationPolicy.DISABLED,
        )
        build = resolve_jit_spec(
            spec, JitCacheIndex(), JitCompilationPolicy.ENABLED
        )
        unavailable = resolve_jit_spec(
            spec, JitCacheIndex(), JitCompilationPolicy.CACHE_ONLY
        )
        self.assertEqual(hit.state, JitResolutionState.CACHE_HIT)
        self.assertEqual(build.state, JitResolutionState.BUILD_REQUIRED)
        self.assertEqual(unavailable.state, JitResolutionState.UNAVAILABLE)
        require_jit_cache_hit(hit)
        with self.assertRaises(MissingJITCacheError):
            require_jit_cache_hit(build)

    def test_attention_environment_projection_is_explicit(self):
        environment = runtime_environment()
        projected = jit_environment_from_attention(environment)
        self.assertEqual(projected.target_soc, environment.soc_version)
        self.assertEqual(projected.compiler_version, environment.compiler_version)
        self.assertEqual(projected.features, environment.features)
        self.assertNotEqual(projected.fingerprint, environment.fingerprint)

    def test_attention_jit_recipe_binds_selected_plan(self):
        environment = runtime_environment()
        plan = framework_plan()
        receipt = dispatch_receipt(plan, environment)
        registry = JitSpecRegistry()
        module = gen_attention_jit_module_spec(
            plan, receipt, environment, registry=registry
        )
        self.assertEqual(module.jit_spec.domain, "attention")
        self.assertEqual(module.jit_spec.target_soc, "Ascend910B")
        self.assertEqual(module.framework_plan_fingerprint, plan.fingerprint)
        self.assertEqual(module.dispatch_receipt_fingerprint, receipt.fingerprint)
        self.assertIs(registry.get(module.jit_spec.name), module.jit_spec)
        self.assertFalse(module.jit_spec.source_materialized)

    def test_non_jit_dispatch_cannot_enter_jit_generator(self):
        environment = runtime_environment()
        plan = framework_plan()
        receipt = dispatch_receipt(plan, environment, Backend.ASCENDC_AOT)
        with self.assertRaisesRegex(SchemaError, "only an ascendc_jit"):
            gen_attention_jit_module_spec(plan, receipt, environment)

    def test_stale_plan_and_environment_are_rejected(self):
        environment = runtime_environment()
        plan = framework_plan(16)
        receipt = dispatch_receipt(plan, environment)
        with self.assertRaisesRegex(SchemaError, "does not bind the plan"):
            gen_attention_jit_module_spec(
                framework_plan(32), receipt, environment
            )
        with self.assertRaisesRegex(SchemaError, "environment"):
            gen_attention_jit_module_spec(
                plan, receipt, runtime_environment("8.0.1")
            )

    def test_static_variant_reuses_recipe_across_dynamic_lengths(self):
        environment = runtime_environment()
        short = framework_plan(16)
        long = framework_plan(64)
        short_module = gen_attention_jit_module_spec(
            short, dispatch_receipt(short, environment), environment
        )
        long_module = gen_attention_jit_module_spec(
            long, dispatch_receipt(long, environment), environment
        )
        self.assertEqual(short_module.variant, long_module.variant)
        self.assertEqual(short_module.jit_spec, long_module.jit_spec)
        self.assertNotEqual(short_module.fingerprint, long_module.fingerprint)

    def test_quantization_participates_in_variant_and_cache_identity(self):
        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="float16",
            accumulator_dtype="float32",
            granularity="channel",
            axis=(2,),
        )
        plain_variant = AttentionJitVariant.from_plan_spec(framework_plan().spec)
        quant_variant = AttentionJitVariant.from_plan_spec(
            framework_plan(quant_spec=quant).spec
        )
        self.assertIsNone(plain_variant.kv_quant_spec_fingerprint)
        self.assertEqual(quant_variant.kv_quant_spec_fingerprint, quant.fingerprint)
        self.assertNotEqual(plain_variant.fingerprint, quant_variant.fingerprint)

    def test_import_has_no_torch_npu_cann_or_operator_package_side_effect(self):
        code = (
            "import sys; import flashinfer_npu.jit; "
            "forbidden=('torch','torch_npu','cann','flash_attn','flash_attention_npu'); "
            "print(','.join(name for name in forbidden if name in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "")

    def test_checkpoint_exposes_no_compiler_or_loader(self):
        self.assertFalse(hasattr(flashinfer_npu.jit, "build_jit_specs"))
        self.assertFalse(hasattr(flashinfer_npu.jit, "load_jit_module"))


if __name__ == "__main__":
    unittest.main()
