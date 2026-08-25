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
    AttentionStateError,
    AttentionTraceCorpus,
    attention_operator_runtime_registry_snapshot,
    build_attention_operator_runtime_resolvers,
    build_framework_attention_corpus,
    framework_attention_coverage_policy,
    install_attention_operator_runtime_resolvers,
)
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
    quantized_kv_input,
)


def paged_decode_case():
    corpus = build_framework_attention_corpus()
    case = next(
        item
        for item in corpus.cases
        if item.case_id == "paged_decode_int8_token_alibi_empty"
    )
    return corpus, case


def public_decode_profile():
    corpus, case = paged_decode_case()
    policy = framework_attention_coverage_policy()
    subset = AttentionTraceCorpus(
        "public-quantized-decode-evidence-subset",
        (case,),
        "Synthetic framework fixture for one exact public decode route.",
    )
    coverage = policy.evaluate(subset)
    spec = case.trace.spec
    rule = AttentionCapabilityRule(
        rule_id="public_paged_decode_int8_token_v1",
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
        max_gqa_group_size=1,
        metadata_limits=AttentionMetadataLimits(
            max_batch_size=case.trace.metadata.batch_size,
            max_total_qo_tokens=case.trace.metadata.batch_size,
            max_total_kv_tokens=sum(case.trace.metadata.sequence_lengths),
            max_total_pages=len(case.trace.metadata.indices),
            max_page_size=case.trace.metadata.page_size,
        ),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-decode-framework-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(
            b"synthetic-public-decode-framework-record"
        ).hexdigest(),
    )
    return AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_decode.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )


def public_decode_runtime():
    values = bootstrap_components()
    corpus, case = paged_decode_case()
    profile = public_decode_profile()
    descriptor = bound_kernel(
        profile,
        kernel_id="public_paged_decode_int8_token_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_paged_decode_int8_test.o"
        ),
        launch_abi=attention_launch_abi("attention_paged_decode_test_entry"),
    )
    binding = replace(
        values["spec"].quantization_bindings[0],
        quant_spec=case.trace.spec.kv_quant_spec,
    )
    runtime_spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(descriptor,),
        observed_environment=profile.environment,
        quantization_bindings=(binding,),
        corpus=corpus,
        coverage_policy=framework_attention_coverage_policy(),
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
        kv_data_type=spec.kv_quant_spec,
        o_data_type=spec.o_dtype,
    )


def decode_kv_input(quant_spec, *, device="npu:0", scale_shape_override=None):
    logical_shape = (1, 1, 1, 1)
    return quantized_kv_input(
        quant_spec,
        key_shape=logical_shape,
        value_shape=logical_shape,
        scale_shape_override=scale_shape_override,
        device=device,
    )


class PublicQuantizedPagedDecodeTests(unittest.TestCase):
    """Paged decode uses the same wrapper-owned quantized provider contract."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_decode_runtime()
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
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=self.case.trace.spec.kv_layout.value,
            backend="auto",
        )

    def test_public_plan_and_run_inject_exact_int8_token_scales(self):
        wrapper = self.wrapper()
        self.assertIsNone(plan_public_wrapper(wrapper, self.case))
        kv_input = decode_kv_input(self.case.trace.spec.kv_quant_spec)

        output = wrapper.run("q", kv_input)
        output_lse = wrapper.run("q-lse", kv_input, return_lse=True)

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        first_call = package_attention.calls[0]
        self.assertIs(first_call[1], kv_input.key_storage)
        self.assertIs(first_call[2], kv_input.value_storage)
        self.assertIs(first_call[6], kv_input.key_scale)
        self.assertIs(first_call[7], kv_input.value_scale)
        self.assertEqual(wrapper.plan_selection.route, "provider")
        self.assertEqual(wrapper.plan_selection.provider_id, "cann")

    def test_decode_rejects_bad_quantized_input_before_external_callable(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper, self.case)
        quant_spec = self.case.trace.spec.kv_quant_spec

        with self.assertRaisesRegex(SchemaError, "QuantizedKVInput"):
            wrapper.run("plain", ("key", "value"))

        mismatched = replace(quant_spec, axis=(1,))
        with self.assertRaisesRegex(SchemaError, "does not match"):
            wrapper.run("mismatched", decode_kv_input(mismatched))

        with self.assertRaisesRegex(SchemaError, "scale.*shape"):
            wrapper.run(
                "bad-scale",
                decode_kv_input(quant_spec, scale_shape_override=()),
            )

        with self.assertRaisesRegex(SchemaError, "device"):
            wrapper.run(
                "bad-device",
                decode_kv_input(quant_spec, device="npu:1"),
            )

        self.assertEqual(package_attention.calls, [])
        self.assertEqual(
            wrapper.run("valid", decode_kv_input(quant_spec)),
            "package-output:valid",
        )

    def test_quantized_workspace_query_does_not_publish_a_plan_or_execute(self):
        wrapper = self.wrapper()
        spec = self.case.trace.spec
        metadata = self.case.trace.metadata

        size = wrapper.workspace_size(
            metadata.indptr,
            metadata.indices,
            metadata.last_page_len,
            spec.num_qo_heads,
            spec.num_kv_heads,
            spec.head_dim_qk,
            metadata.page_size,
            pos_encoding_mode=spec.pos_encoding_mode.value,
            q_data_type=spec.q_dtype,
            kv_data_type=spec.kv_quant_spec,
            o_data_type=spec.o_dtype,
        )

        self.assertEqual(size, (0, 0))
        self.assertEqual(package_attention.calls, [])
        with self.assertRaisesRegex(AttentionStateError, "plan"):
            _ = wrapper.plan_state


if __name__ == "__main__":
    unittest.main()
