import hashlib
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionBackendCapabilityProfile,
    AttentionCapabilityEvidence,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionMetadataLimits,
    AttentionNumericsPolicy,
    AttentionTraceCorpus,
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    build_attention_operator_runtime_resolvers,
    build_framework_attention_corpus,
    framework_attention_coverage_policy,
    install_attention_operator_runtime_resolvers,
)
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
    metadata_tensor,
    quantized_kv_input,
)


def mixed_quantized_case():
    corpus = build_framework_attention_corpus()
    case = next(
        item
        for item in corpus.cases
        if item.case_id == "mixed_int4_window_softcap_per_head_scale"
    )
    return corpus, case


def public_mixed_runtime():
    values = bootstrap_components()
    corpus, case = mixed_quantized_case()
    policy = framework_attention_coverage_policy()
    coverage = policy.evaluate(
        AttentionTraceCorpus(
            "public-mixed-evidence-subset",
            (case,),
            "Synthetic framework fixture for one exact mixed route.",
        )
    )
    spec = case.trace.spec
    rule = AttentionCapabilityRule(
        rule_id="public_mixed_paged_int4_v1",
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
        supports_sliding_window=True,
        supports_logits_soft_cap=True,
        metadata_limits=AttentionMetadataLimits(
            max_batch_size=case.trace.metadata.batch_size,
            max_total_qo_tokens=case.trace.metadata.total_qo_tokens,
            max_total_kv_tokens=sum(case.trace.metadata.kv_len_arr),
            max_total_pages=len(case.trace.metadata.kv_indices),
            max_page_size=case.trace.metadata.page_size,
        ),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-mixed-framework-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(
            b"synthetic-public-mixed-framework-record"
        ).hexdigest(),
    )
    profile = AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_mixed.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )
    descriptor = bound_kernel(
        profile,
        kernel_id="public_mixed_paged_int4_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_mixed_paged_int4_test.o"
        ),
        launch_abi=attention_launch_abi("attention_mixed_paged_test_entry"),
    )
    binding = replace(
        values["spec"].quantization_bindings[0],
        quant_spec=spec.kv_quant_spec,
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
        metadata.qo_indptr,
        metadata.kv_indptr,
        metadata.kv_indices,
        metadata.kv_len_arr,
        spec.num_qo_heads,
        spec.num_kv_heads,
        spec.head_dim_qk,
        spec.head_dim_vo,
        metadata.page_size,
        causal=spec.causal,
        sm_scale=spec.sm_scale,
        logits_soft_cap=spec.logits_soft_cap,
        q_data_type=spec.q_dtype,
        kv_data_type=spec.kv_quant_spec,
    )


def mixed_kv_input(quant_spec):
    logical_shape = (2, 2, 2, 3)
    return quantized_kv_input(
        quant_spec,
        key_shape=logical_shape,
        value_shape=logical_shape,
    )


def query_tensor(case, name="q"):
    spec = case.trace.spec
    return metadata_tensor(
        name,
        (
            case.trace.metadata.total_qo_tokens,
            spec.num_qo_heads,
            spec.head_dim_qk,
        ),
        spec.q_dtype,
    )


class PublicQuantizedBatchAttentionTests(unittest.TestCase):
    """Mixed prefill/decode keeps quantized provider routing wrapper-owned."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_mixed_runtime()
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
        return BatchAttention(
            kv_layout=self.case.trace.spec.kv_layout.value,
            device="npu:0",
        )

    def test_public_plan_run_routes_quantized_mixed_batch_and_returns_lse(self):
        wrapper = self.wrapper()
        self.assertIsNone(plan_public_wrapper(wrapper, self.case))
        kv_input = mixed_kv_input(self.case.trace.spec.kv_quant_spec)

        result = wrapper.run(
            query_tensor(self.case, "q"),
            kv_input,
            logits_soft_cap=self.case.trace.spec.logits_soft_cap,
        )

        self.assertEqual(result, ("package-output:q", "package-lse:0.25"))
        call = package_attention.calls[0]
        self.assertIs(call[1], kv_input.key_storage)
        self.assertIs(call[2], kv_input.value_storage)
        self.assertIs(call[6], kv_input.key_scale)
        self.assertIs(call[7], kv_input.value_scale)
        self.assertEqual(wrapper.plan_selection.route, "provider")
        self.assertEqual(wrapper.plan_selection.provider_id, "cann")

    def test_mixed_bad_quantized_input_fails_before_external_callable(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper, self.case)
        quant_spec = self.case.trace.spec.kv_quant_spec

        with self.assertRaisesRegex(SchemaError, "QuantizedKVInput"):
            wrapper.run(
                query_tensor(self.case, "plain"),
                ("key", "value"),
                logits_soft_cap=self.case.trace.spec.logits_soft_cap,
            )

        different = replace(quant_spec, packing_order="high_nibble_first")
        with self.assertRaisesRegex(SchemaError, "does not match"):
            wrapper.run(
                query_tensor(self.case, "mismatched"),
                mixed_kv_input(different),
                logits_soft_cap=self.case.trace.spec.logits_soft_cap,
            )

        self.assertEqual(package_attention.calls, [])
        self.assertEqual(
            wrapper.run(
                query_tensor(self.case, "valid"),
                mixed_kv_input(quant_spec),
                logits_soft_cap=self.case.trace.spec.logits_soft_cap,
            ),
            ("package-output:valid", "package-lse:0.25"),
        )


if __name__ == "__main__":
    unittest.main()
