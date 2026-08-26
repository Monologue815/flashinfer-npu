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
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizedKVInput,
    AttentionStateError,
    AttentionTraceCorpus,
    TensorView,
    attention_operator_runtime_registry_snapshot,
    build_attention_operator_runtime_resolvers,
    build_framework_attention_corpus,
    contiguous_strides,
    dtype_itemsize,
    framework_attention_coverage_policy,
    infer_quant_scale_shape,
    infer_quant_storage_shape,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import (
    attention_artifact,
    attention_launch_abi,
    bound_kernel,
    pinned_environment,
)
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components


class FakeNpuWorkspace:
    shape = (4096,)
    dtype = "uint8"
    device = "npu:0"


class MetadataTensor:
    def __init__(self, tensor_view):
        self.tensor_view = tensor_view
        self.shape = tensor_view.shape
        self.dtype = tensor_view.dtype
        self.device = tensor_view.device


def metadata_tensor(name, shape, dtype, *, device="npu:0"):
    shape = tuple(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    return MetadataTensor(
        TensorView(
            shape=shape,
            strides=contiguous_strides(shape),
            dtype=dtype,
            device=device,
            storage_id="public-quantized-prefill:" + name,
            storage_nbytes=numel * dtype_itemsize(dtype),
        )
    )


def paged_prefill_case():
    corpus = build_framework_attention_corpus()
    case = next(
        item
        for item in corpus.cases
        if item.case_id == "paged_prefill_int4_multi_request_shared_page"
    )
    return corpus, case


def public_prefill_profile():
    corpus, case = paged_prefill_case()
    policy = framework_attention_coverage_policy()
    subset = AttentionTraceCorpus(
        "public-quantized-prefill-evidence-subset",
        (case,),
        "Synthetic framework fixture for one exact public prefill route.",
    )
    coverage = policy.evaluate(subset)
    spec = case.trace.spec
    rule = AttentionCapabilityRule(
        rule_id="public_paged_prefill_int4_v1",
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
            max_total_qo_tokens=case.trace.metadata.total_qo_tokens,
            max_total_kv_tokens=sum(
                case.trace.metadata.paged_kv.sequence_lengths
            ),
            max_total_pages=len(case.trace.metadata.paged_kv.indices),
            max_page_size=case.trace.metadata.paged_kv.page_size,
        ),
        required_features=("int8-group-dequant",),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-prefill-framework-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(
            b"synthetic-public-prefill-framework-record"
        ).hexdigest(),
    )
    return AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_prefill.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )


def public_prefill_runtime():
    values = bootstrap_components()
    corpus, case = paged_prefill_case()
    profile = public_prefill_profile()
    descriptor = bound_kernel(
        profile,
        kernel_id="public_paged_prefill_int4_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_paged_prefill_int4_test.o"
        ),
        launch_abi=attention_launch_abi("attention_paged_prefill_test_entry"),
    )
    original_binding = values["spec"].quantization_bindings[0]
    binding = replace(
        original_binding,
        quant_spec=case.trace.spec.kv_quant_spec,
        argument_bindings=original_binding.argument_bindings
        + (
            AttentionOperatorQuantArgumentBinding(
                "run.q_scale", "runtime_query_scale"
            ),
        ),
        runtime_q_scale_policy="argument",
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


def quantized_kv_input(
    quant_spec,
    *,
    key_shape=(2, 2, 1, 3),
    value_shape=(2, 2, 1, 3),
    scale_shape_override=None,
    device="npu:0",
):
    storage_dtype = (
        "uint8"
        if quant_spec.storage_dtype in {"int4_packed", "uint4_packed"}
        else quant_spec.storage_dtype
    )
    key_scale_shape = (
        infer_quant_scale_shape(key_shape, quant_spec)
        if scale_shape_override is None
        else tuple(scale_shape_override)
    )
    return AttentionOperatorQuantizedKVInput(
        quant_spec=quant_spec,
        key_storage=metadata_tensor(
            "key-storage",
            infer_quant_storage_shape(key_shape, quant_spec),
            storage_dtype,
            device=device,
        ),
        value_storage=metadata_tensor(
            "value-storage",
            infer_quant_storage_shape(value_shape, quant_spec),
            storage_dtype,
            device=device,
        ),
        key_scale=metadata_tensor(
            "key-scale",
            key_scale_shape,
            quant_spec.scale_dtype,
            device=device,
        ),
        value_scale=metadata_tensor(
            "value-scale",
            infer_quant_scale_shape(value_shape, quant_spec),
            quant_spec.scale_dtype,
            device=device,
        ),
    )


def plan_public_wrapper(wrapper, case):
    spec = case.trace.spec
    metadata = case.trace.metadata
    return wrapper.plan(
        metadata.qo_indptr,
        metadata.paged_kv.indptr,
        metadata.paged_kv.indices,
        metadata.paged_kv.last_page_len,
        spec.num_qo_heads,
        spec.num_kv_heads,
        spec.head_dim_qk,
        metadata.paged_kv.page_size,
        head_dim_vo=spec.head_dim_vo,
        causal=spec.causal,
        q_data_type=spec.q_dtype,
        kv_data_type=spec.kv_quant_spec,
        o_data_type=spec.o_dtype,
    )


class PublicQuantizedPagedPrefillTests(unittest.TestCase):
    """Public wrapper owns exact quantized provider planning and lowering."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_prefill_runtime()
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
        return BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=self.case.trace.spec.kv_layout.value,
            backend="auto",
        )

    def test_public_plan_and_run_inject_exact_storage_and_independent_scales(self):
        wrapper = self.wrapper()
        self.assertIsNone(plan_public_wrapper(wrapper, self.case))
        kv_input = quantized_kv_input(self.case.trace.spec.kv_quant_spec)
        query_scale = object()

        output = wrapper.run("q", kv_input, q_scale=query_scale)
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
        self.assertIs(first_call[10], query_scale)
        self.assertIsNot(first_call[6], first_call[7])
        self.assertEqual(wrapper.plan_selection.route, "provider")
        self.assertEqual(wrapper.plan_selection.provider_id, "cann")

    def test_invalid_quantized_inputs_fail_before_external_callable(self):
        wrapper = self.wrapper()
        plan_public_wrapper(wrapper, self.case)
        quant_spec = self.case.trace.spec.kv_quant_spec

        with self.assertRaisesRegex(SchemaError, "QuantizedKVInput"):
            wrapper.run("plain", ("key", "value"))

        mismatched = replace(quant_spec, packing_order="high_nibble_first")
        with self.assertRaisesRegex(SchemaError, "does not match"):
            wrapper.run("mismatched", quantized_kv_input(mismatched))

        with self.assertRaisesRegex(SchemaError, "scale.*shape"):
            wrapper.run(
                "bad-scale",
                quantized_kv_input(quant_spec, scale_shape_override=(1,)),
            )

        with self.assertRaisesRegex(SchemaError, "device"):
            wrapper.run(
                "bad-device",
                quantized_kv_input(quant_spec, device="npu:1"),
            )

        self.assertEqual(package_attention.calls, [])
        self.assertEqual(
            wrapper.run("valid", quantized_kv_input(quant_spec)),
            "package-output:valid",
        )

    def test_quantized_workspace_query_is_non_mutating_and_non_executing(self):
        wrapper = self.wrapper()
        spec = self.case.trace.spec
        metadata = self.case.trace.metadata

        size = wrapper.workspace_size(
            metadata.qo_indptr,
            metadata.paged_kv.indptr,
            metadata.paged_kv.indices,
            metadata.paged_kv.last_page_len,
            spec.num_qo_heads,
            spec.num_kv_heads,
            spec.head_dim_qk,
            metadata.paged_kv.page_size,
            head_dim_vo=spec.head_dim_vo,
            causal=spec.causal,
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
