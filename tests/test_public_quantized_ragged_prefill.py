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
    AttentionOperatorQuantizedTensorInput,
    AttentionPlanSpec,
    AttentionTrace,
    AttentionTraceCase,
    AttentionTraceCorpus,
    KVLayout,
    RaggedKVCacheSpec,
    RaggedKVMetadata,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    attention_operator_runtime_registry_snapshot,
    build_attention_operator_runtime_resolvers,
    framework_attention_coverage_policy,
    infer_quant_scale_shape,
    infer_quant_storage_shape,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.prefill import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer_npu.runtime import QuantSpec, SchemaError
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


def ragged_quantized_case():
    quant_spec = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        scale_dtype="float32",
        granularity="group",
        group_size=(1, 1, 1),
        axis=(0, 1, 2),
    )
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_RAGGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        kv_layout=KVLayout.HND,
        causal=False,
        q_dtype="float32",
        kv_dtype="int8",
        kv_quant_spec=quant_spec,
        o_dtype="int8",
    )
    metadata = RaggedKVMetadata((0, 1), (0, 2))
    cache = RaggedKVCacheSpec(
        total_kv_tokens=2,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        dtype="int8",
        layout=KVLayout.HND,
        device="cpu",
        quant_spec=quant_spec,
    )
    logical_shape = (1, 2, 1)
    key = ReferenceQuantizedTensor(
        logical_shape,
        ReferenceTensor.from_nested([[[1], [0]]], dtype="int8"),
        ReferenceTensor.from_nested([[[1.0], [1.0]]], dtype="float32"),
        quant_spec,
    )
    value = ReferenceQuantizedTensor(
        logical_shape,
        ReferenceTensor.from_nested([[[2], [4]]], dtype="int8"),
        ReferenceTensor.from_nested([[[1.0], [1.0]]], dtype="float32"),
        quant_spec,
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=metadata,
        q=ReferenceTensor.from_nested(
            [[[1.0], [0.5]]], dtype="float32"
        ),
        kv_data=ReferenceQuantizedKVData(cache, key, value),
    )
    return AttentionTraceCase(
        "public_ragged_prefill_int8_group",
        trace,
        "Exact no-mask ragged quantization fixture for public provider routing.",
    )


def public_ragged_runtime():
    values = bootstrap_components()
    case = ragged_quantized_case()
    corpus = AttentionTraceCorpus(
        "public-ragged-prefill-provider-v1",
        (case,),
        "Synthetic framework corpus for one exact ragged provider route.",
    )
    policy = framework_attention_coverage_policy()
    coverage = policy.evaluate(corpus)
    spec = case.trace.spec
    rule = AttentionCapabilityRule(
        rule_id="public_ragged_prefill_int8_group_v1",
        modes=(spec.mode,),
        kv_layouts=(spec.kv_layout,),
        dtype_signatures=((spec.q_dtype, spec.kv_dtype, spec.o_dtype),),
        supports_dense_kv=False,
        quant_specs=(spec.kv_quant_spec,),
        pos_encoding_modes=(spec.pos_encoding_mode,),
        mask_kinds=("none",),
        causal_values=(spec.effective_causal,),
        max_head_dim_qk=spec.head_dim_qk,
        max_head_dim_vo=spec.head_dim_vo,
        max_gqa_group_size=2,
        metadata_limits=AttentionMetadataLimits(
            max_batch_size=case.trace.metadata.batch_size,
            max_total_qo_tokens=case.trace.metadata.total_qo_tokens,
            max_total_kv_tokens=case.trace.metadata.total_kv_tokens,
            max_total_pages=0,
            max_page_size=0,
        ),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-ragged-framework-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(
            b"synthetic-public-ragged-framework-record"
        ).hexdigest(),
    )
    profile = AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_ragged.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )
    descriptor = bound_kernel(
        profile,
        kernel_id="public_ragged_prefill_int8_group_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_ragged_prefill_int8_test.o"
        ),
        launch_abi=attention_launch_abi("attention_ragged_prefill_test_entry"),
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
                "run.o_scale", "runtime_output_scale"
            ),
        ),
        runtime_q_scale_policy="argument",
        runtime_o_scale_policy="argument",
        runtime_o_scale_output_dtypes=("int8",),
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


def quantized_tensor_pair(
    quant_spec, *, value_quant_spec=None, logical_shape=(1, 2, 1)
):
    value_quant_spec = quant_spec if value_quant_spec is None else value_quant_spec

    def component(name, component_spec):
        storage_dtype = (
            "uint8"
            if component_spec.storage_dtype in {"int4_packed", "uint4_packed"}
            else component_spec.storage_dtype
        )
        return AttentionOperatorQuantizedTensorInput(
            quant_spec=component_spec,
            logical_shape=logical_shape,
            storage=metadata_tensor(
                name + "-storage",
                infer_quant_storage_shape(logical_shape, component_spec),
                storage_dtype,
            ),
            scale=metadata_tensor(
                name + "-scale",
                infer_quant_scale_shape(logical_shape, component_spec),
                component_spec.scale_dtype,
            ),
        )

    return component("key", quant_spec), component("value", value_quant_spec)


def plan_public_wrapper(wrapper, case):
    spec = case.trace.spec
    metadata = case.trace.metadata
    return wrapper.plan(
        metadata.qo_indptr,
        metadata.kv_indptr,
        spec.num_qo_heads,
        spec.num_kv_heads,
        spec.head_dim_qk,
        head_dim_vo=spec.head_dim_vo,
        causal=spec.causal,
        q_data_type=spec.q_dtype,
        kv_data_type=spec.kv_quant_spec,
        o_data_type=spec.o_dtype,
    )


class PublicQuantizedRaggedPrefillTests(unittest.TestCase):
    """Separate public K/V slots compose into one private quantized KV input."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_ragged_runtime()
        install_attention_operator_runtime_resolvers(
            registry,
            operation_catalog=self.values["catalog"],
        )
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def wrapper(self):
        return BatchPrefillWithRaggedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=self.case.trace.spec.kv_layout.value,
            backend="auto",
        )

    def test_public_separate_kv_inputs_inject_independent_storage_and_scale(self):
        wrapper = self.wrapper()
        self.assertIsNone(plan_public_wrapper(wrapper, self.case))
        key, value = quantized_tensor_pair(self.case.trace.spec.kv_quant_spec)
        query_scale = object()
        output_scale = object()

        output = wrapper.run(
            "q",
            key,
            value,
            q_scale=query_scale,
            o_scale=output_scale,
        )
        output_lse = wrapper.run("q-lse", key, value, return_lse=True)

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        first_call = package_attention.calls[0]
        self.assertIs(first_call[1], key.storage)
        self.assertIs(first_call[2], value.storage)
        self.assertIs(first_call[6], key.scale)
        self.assertIs(first_call[7], value.scale)
        self.assertIs(first_call[10], query_scale)
        self.assertIs(first_call[11], output_scale)
        self.assertEqual(wrapper.plan_selection.route, "provider")
        self.assertEqual(wrapper.plan_selection.provider_id, "cann")

    def test_bad_separate_inputs_fail_before_external_callable(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper, self.case)
        quant_spec = self.case.trace.spec.kv_quant_spec

        with self.assertRaisesRegex(SchemaError, "for key"):
            wrapper.run("plain", "key", "value")

        different = replace(quant_spec, group_size=(1, 2, 1))
        key, value = quantized_tensor_pair(
            quant_spec, value_quant_spec=different
        )
        with self.assertRaisesRegex(SchemaError, "different QuantSpec"):
            wrapper.run("mismatched", key, value)

        self.assertEqual(package_attention.calls, [])
        wrong_shape_key, wrong_shape_value = quantized_tensor_pair(
            quant_spec, logical_shape=(1, 2, 2)
        )
        with self.assertRaisesRegex(SchemaError, "key logical_shape"):
            wrapper.run("wrong-shape", wrong_shape_key, wrong_shape_value)
        self.assertEqual(package_attention.calls, [])
        valid_key, valid_value = quantized_tensor_pair(quant_spec)
        self.assertEqual(
            wrapper.run("valid", valid_key, valid_value),
            "package-output:valid",
        )

    def test_tensor_input_enforces_zero_point_presence_from_quantspec(self):
        quant_spec = self.case.trace.spec.kv_quant_spec
        asymmetric = replace(
            quant_spec,
            scheme="asymmetric",
            has_zero_point=True,
        )
        with self.assertRaisesRegex(SchemaError, "requires zero_point"):
            AttentionOperatorQuantizedTensorInput(
                asymmetric,
                logical_shape=(1, 2, 1),
                storage=object(),
                scale=object(),
            )


if __name__ == "__main__":
    unittest.main()
