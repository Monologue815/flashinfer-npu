import hashlib
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionBackendCapabilityProfile,
    AttentionCapabilityEvidence,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionMetadataLimits,
    AttentionMode,
    AttentionNumericsPolicy,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizedKVInput,
    AttentionStateError,
    AttentionTrace,
    AttentionTraceCase,
    AttentionTraceCorpus,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    attention_operator_runtime_registry_snapshot,
    build_attention_operator_runtime_resolvers,
    build_framework_attention_corpus,
    framework_attention_coverage_policy,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.frontend import fp8_per_tensor_quant_spec
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import (
    attention_artifact,
    attention_launch_abi,
    bound_kernel,
    pinned_environment,
)
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_public_quantized_paged_prefill import (
    FakeNpuWorkspace,
    metadata_tensor,
)


def paged_decode_fp8_case():
    base = next(
        item
        for item in build_framework_attention_corpus().cases
        if item.case_id == "paged_decode_int8_token_alibi_empty"
    )
    quant_spec = fp8_per_tensor_quant_spec("float8_e4m3fn")
    spec = replace(
        base.trace.spec,
        q_dtype=quant_spec.storage_dtype,
        kv_dtype=quant_spec.storage_dtype,
        o_dtype=quant_spec.storage_dtype,
        kv_quant_spec=quant_spec,
    )
    q = ReferenceTensor(
        base.trace.q.shape,
        base.trace.q.data,
        dtype=quant_spec.storage_dtype,
    )
    original = base.trace.kv_data

    def component(value):
        return ReferenceQuantizedTensor(
            value.logical_shape,
            ReferenceTensor(
                value.storage.shape,
                value.storage.data,
                dtype=quant_spec.storage_dtype,
            ),
            ReferenceTensor((), (1.0,), dtype=quant_spec.scale_dtype),
            quant_spec,
        )

    kv_data = ReferenceQuantizedKVData(
        replace(
            original.spec,
            dtype=quant_spec.storage_dtype,
            quant_spec=quant_spec,
        ),
        component(original.key_data),
        component(original.value_data),
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=base.trace.metadata,
        q=q,
        kv_data=kv_data,
    )
    return AttentionTraceCase(
        "public_paged_decode_fp8_tensor",
        trace,
        "Synthetic framework fixture for bare FP8 paged decode routing.",
    )


def public_paged_decode_runtime(*, authorize_implicit_units=True):
    values = bootstrap_components()
    case = paged_decode_fp8_case()
    corpus = AttentionTraceCorpus(
        "public-paged-decode-fp8-v1",
        (case,),
        "Synthetic framework corpus for bare FP8 paged decode.",
    )
    policy = framework_attention_coverage_policy()
    coverage = policy.evaluate(corpus)
    spec = case.trace.spec
    metadata = case.trace.metadata
    rule = AttentionCapabilityRule(
        rule_id="public_paged_decode_fp8_v1",
        modes=(AttentionMode.BATCH_DECODE_PAGED,),
        kv_layouts=(spec.kv_layout,),
        dtype_signatures=((spec.q_dtype, spec.kv_dtype, spec.o_dtype),),
        supports_dense_kv=False,
        quant_specs=(spec.kv_quant_spec,),
        pos_encoding_modes=(spec.pos_encoding_mode,),
        mask_kinds=("none",),
        causal_values=(spec.effective_causal,),
        max_head_dim_qk=spec.head_dim_qk,
        max_head_dim_vo=spec.head_dim_vo,
        max_gqa_group_size=1,
        metadata_limits=AttentionMetadataLimits(
            max_batch_size=metadata.batch_size,
            max_total_qo_tokens=metadata.batch_size * spec.q_len_per_req,
            max_total_kv_tokens=sum(metadata.sequence_lengths),
            max_total_pages=len(metadata.indices),
            max_page_size=metadata.page_size,
        ),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-paged-decode-fp8-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(
            b"synthetic-public-paged-decode-fp8-record"
        ).hexdigest(),
    )
    profile = AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_paged_decode_fp8.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )
    descriptor = bound_kernel(
        profile,
        kernel_id="public_paged_decode_fp8_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_paged_decode_fp8_test.o"
        ),
        launch_abi=attention_launch_abi(
            "attention_paged_decode_fp8_test_entry"
        ),
    )
    original_binding = values["spec"].quantization_bindings[0]
    binding = replace(
        original_binding,
        quant_spec=spec.kv_quant_spec,
        argument_bindings=original_binding.argument_bindings
        + (
            AttentionOperatorQuantArgumentBinding(
                "run.q_scale", "runtime_query_scale"
            ),
            AttentionOperatorQuantArgumentBinding(
                "run.k_scale", "runtime_key_scale"
            ),
            AttentionOperatorQuantArgumentBinding(
                "run.v_scale", "runtime_value_scale"
            ),
        ),
        runtime_q_scale_policy="argument",
        runtime_k_scale_policy="argument",
        runtime_v_scale_policy="argument",
        implicit_unit_scale_sources=(
            ("kv.key.scale", "kv.value.scale")
            if authorize_implicit_units
            else ()
        ),
    )
    runtime_spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(descriptor,),
        observed_environment=profile.environment,
        quantization_bindings=(binding,),
        corpus=corpus,
        coverage_policy=policy,
    )
    registry = build_attention_operator_runtime_resolvers(
        (runtime_spec,),
        operation_catalog=values["catalog"],
        package_loader=values["loader"],
    )
    return values, case, registry


def plan_public_wrapper(wrapper, case):
    spec = case.trace.spec
    metadata = case.trace.metadata
    return wrapper.plan(
        metadata.indptr,
        metadata.indices,
        metadata.last_page_len,
        spec.num_qo_heads,
        spec.num_kv_heads,
        spec.head_dim_qk,
        metadata.page_size,
        pos_encoding_mode=spec.pos_encoding_mode.value,
        q_data_type=spec.q_dtype,
        kv_data_type=spec.kv_dtype,
        o_data_type=spec.o_dtype,
    )


class PublicQuantizedPagedDecodeTests(unittest.TestCase):
    """Bare paged FP8 inputs stay behind the FlashInfer-compatible run slot."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_paged_decode_runtime()
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=self.values["catalog"]
        )
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=self.case.trace.spec.kv_layout.value,
            backend="auto",
        )

    def raw_cache(self, *, dtype="float8_e4m3fn"):
        shape = self.case.trace.kv_data.key_data.logical_shape
        return (
            metadata_tensor("paged-decode-key", shape, dtype),
            metadata_tensor("paged-decode-value", shape, dtype),
        )

    def test_bare_fp8_plan_and_run_use_implicit_unit_kv_scales(self):
        self.assertEqual(
            fp8_per_tensor_quant_spec("torch.float8_e5m2").storage_dtype,
            "float8_e5m2",
        )
        wrapper = self.wrapper()
        self.assertIsNone(plan_public_wrapper(wrapper, self.case))
        self.assertEqual(
            wrapper.plan_state.spec.kv_quant_spec.fingerprint,
            fp8_per_tensor_quant_spec("float8_e4m3fn").fingerprint,
        )
        key, value = self.raw_cache()

        implicit_output = wrapper.run("q-implicit", (key, value))
        calibrated_output = wrapper.run(
            "q-calibrated",
            (key, value),
            q_scale=2.0,
            k_scale=1.5,
            v_scale=0.5,
        )

        self.assertEqual(implicit_output, "package-output:q-implicit")
        self.assertEqual(calibrated_output, "package-output:q-calibrated")
        implicit_call, calibrated_call = package_attention.calls
        self.assertIs(implicit_call[1], key)
        self.assertIs(implicit_call[2], value)
        self.assertEqual(implicit_call[6:11], (None, None, None, None, None))
        self.assertIs(calibrated_call[1], key)
        self.assertIs(calibrated_call[2], value)
        self.assertEqual(
            calibrated_call[6:11], (None, None, 1.5, 0.5, 2.0)
        )
        self.assertEqual(wrapper.plan_selection.route, "provider")

    def test_fp8_workspace_query_uses_the_same_non_publishing_plan(self):
        wrapper = self.wrapper()
        spec = self.case.trace.spec
        metadata = self.case.trace.metadata

        self.assertEqual(
            wrapper.workspace_size(
                metadata.indptr,
                metadata.indices,
                metadata.last_page_len,
                spec.num_qo_heads,
                spec.num_kv_heads,
                spec.head_dim_qk,
                metadata.page_size,
                pos_encoding_mode=spec.pos_encoding_mode.value,
                q_data_type=spec.q_dtype,
                kv_data_type=spec.kv_dtype,
                o_data_type=spec.o_dtype,
            ),
            (0, 0),
        )
        with self.assertRaisesRegex(AttentionStateError, "plan"):
            _ = wrapper.plan_state
        self.assertEqual(package_attention.calls, [])

    def test_host_oracle_keeps_logical_fp8_cache_without_provider_wrapper(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            ReferenceTensor.zeros((0,), dtype="uint8"),
            kv_layout="HND",
            backend="reference",
        )
        index = lambda values: ReferenceTensor(
            (len(values),), tuple(values), dtype="int32"
        )
        wrapper.plan(
            index((0, 1)),
            index((0,)),
            index((1,)),
            1,
            1,
            1,
            1,
            q_data_type="float8_e4m3fn",
            kv_data_type="float8_e4m3fn",
            o_data_type="float8_e4m3fn",
        )
        self.assertIsNone(wrapper.plan_state.spec.kv_quant_spec)
        q = ReferenceTensor.from_nested(
            [[[1.0]]], dtype="float8_e4m3fn"
        )
        key = ReferenceTensor.from_nested(
            [[[[1.0]]]], dtype="float8_e4m3fn"
        )
        value = ReferenceTensor.from_nested(
            [[[[4.0]]]], dtype="float8_e4m3fn"
        )

        output, lse = wrapper.run(
            q,
            (key, value),
            q_scale=2.0,
            k_scale=1.5,
            v_scale=0.5,
            return_lse=True,
        )

        self.assertEqual(output.data, (2.0,))
        self.assertEqual(lse.data, (3.0,))

    def test_binding_and_cache_structure_fail_closed_before_package_call(self):
        with self.assertRaisesRegex(
            SchemaError, "must authorize implicit unit scales"
        ):
            public_paged_decode_runtime(authorize_implicit_units=False)

        wrapper = self.wrapper()
        plan_public_wrapper(wrapper, self.case)
        combined = metadata_tensor(
            "combined-paged-decode",
            (1, 2, 1, 1, 1),
            "float8_e4m3fn",
        )
        with self.assertRaisesRegex(SchemaError, "separate.*key, value"):
            wrapper.run("q-combined", combined)
        with self.assertRaisesRegex(SchemaError, "storage.*dtype"):
            wrapper.run("q-wrong-dtype", self.raw_cache(dtype="float16"))
        self.assertEqual(package_attention.calls, [])

    def test_explicit_quantized_input_remains_supported(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper, self.case)
        key, value = self.raw_cache()
        key_scale = metadata_tensor("paged-decode-key-scale", (), "float32")
        value_scale = metadata_tensor("paged-decode-value-scale", (), "float32")
        explicit = AttentionOperatorQuantizedKVInput(
            quant_spec=wrapper.plan_state.spec.kv_quant_spec,
            key_storage=key,
            value_storage=value,
            key_scale=key_scale,
            value_scale=value_scale,
        )

        self.assertEqual(
            wrapper.run("q-explicit", explicit),
            "package-output:q-explicit",
        )
        call = package_attention.calls[0]
        self.assertIs(call[6], key_scale)
        self.assertIs(call[7], value_scale)


if __name__ == "__main__":
    unittest.main()
