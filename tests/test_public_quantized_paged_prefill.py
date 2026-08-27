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
from flashinfer_npu.attention.frontend import fp8_per_tensor_quant_spec
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


def paged_prefill_fp8_case():
    _, base = paged_prefill_case()
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
        logical_numel = 1
        for dimension in value.logical_shape:
            logical_numel *= dimension
        return ReferenceQuantizedTensor(
            value.logical_shape,
            ReferenceTensor(
                value.logical_shape,
                (0.0,) * logical_numel,
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
        "public_paged_prefill_fp8_tensor",
        trace,
        "Synthetic framework fixture for bare FP8 paged prefill routing.",
    )


def public_paged_prefill_fp8_runtime(*, authorize_implicit_units=True):
    values = bootstrap_components()
    case = paged_prefill_fp8_case()
    corpus = AttentionTraceCorpus(
        "public-paged-prefill-fp8-v1",
        (case,),
        "Synthetic framework corpus for bare FP8 paged prefill.",
    )
    policy = framework_attention_coverage_policy()
    coverage = policy.evaluate(corpus)
    spec = case.trace.spec
    metadata = case.trace.metadata
    rule = AttentionCapabilityRule(
        rule_id="public_paged_prefill_fp8_v1",
        modes=(AttentionMode.BATCH_PREFILL_PAGED,),
        kv_layouts=(spec.kv_layout,),
        dtype_signatures=((spec.q_dtype, spec.kv_dtype, spec.o_dtype),),
        supports_dense_kv=False,
        quant_specs=(spec.kv_quant_spec,),
        pos_encoding_modes=(spec.pos_encoding_mode,),
        mask_kinds=("none",),
        causal_values=(spec.effective_causal,),
        max_head_dim_qk=spec.head_dim_qk,
        max_head_dim_vo=spec.head_dim_vo,
        max_gqa_group_size=spec.num_qo_heads // spec.num_kv_heads,
        metadata_limits=AttentionMetadataLimits(
            max_batch_size=metadata.batch_size,
            max_total_qo_tokens=metadata.total_qo_tokens,
            max_total_kv_tokens=sum(metadata.paged_kv.sequence_lengths),
            max_total_pages=len(metadata.paged_kv.indices),
            max_page_size=metadata.paged_kv.page_size,
        ),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-paged-prefill-fp8-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(
            b"synthetic-public-paged-prefill-fp8-record"
        ).hexdigest(),
    )
    profile = AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_paged_prefill_fp8.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )
    descriptor = bound_kernel(
        profile,
        kernel_id="public_paged_prefill_fp8_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_paged_prefill_fp8_test.o"
        ),
        launch_abi=attention_launch_abi(
            "attention_paged_prefill_fp8_test_entry"
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


class PublicFP8PagedPrefillTests(unittest.TestCase):
    """Bare FP8 paged prefill uses a wrapper-owned per-tensor provider plan."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_paged_prefill_fp8_runtime()
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
        return BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout=self.case.trace.spec.kv_layout.value,
            backend="auto",
        )

    def plan_wrapper(self, wrapper):
        spec = self.case.trace.spec
        metadata = self.case.trace.metadata
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
            kv_data_type=spec.kv_dtype,
            o_data_type=spec.o_dtype,
        )

    def raw_cache(self, *, dtype="float8_e4m3fn"):
        return (
            metadata_tensor(
                "paged-prefill-fp8-key",
                self.case.trace.kv_data.key_data.logical_shape,
                dtype,
            ),
            metadata_tensor(
                "paged-prefill-fp8-value",
                self.case.trace.kv_data.value_data.logical_shape,
                dtype,
            ),
        )

    def test_bare_fp8_plan_and_dynamic_calibration_reach_exact_binding(self):
        wrapper = self.wrapper()
        self.assertIsNone(self.plan_wrapper(wrapper))
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

    def test_implicit_binding_and_combined_storage_fail_closed(self):
        with self.assertRaisesRegex(
            SchemaError, "must authorize implicit unit scales"
        ):
            public_paged_prefill_fp8_runtime(authorize_implicit_units=False)

        wrapper = self.wrapper()
        self.plan_wrapper(wrapper)
        combined = metadata_tensor(
            "combined-paged-prefill-fp8",
            (2, 2, 2, 1, 3),
            "float8_e4m3fn",
        )
        with self.assertRaisesRegex(SchemaError, "separate.*key, value"):
            wrapper.run("q-combined", combined)
        with self.assertRaisesRegex(SchemaError, "storage.*dtype"):
            wrapper.run("q-wrong-dtype", self.raw_cache(dtype="float16"))
        self.assertEqual(package_attention.calls, [])

    def test_fp8_workspace_query_is_non_publishing(self):
        wrapper = self.wrapper()
        spec = self.case.trace.spec
        metadata = self.case.trace.metadata
        self.assertEqual(
            wrapper.workspace_size(
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
                kv_data_type=spec.kv_dtype,
                o_data_type=spec.o_dtype,
            ),
            (0, 0),
        )
        with self.assertRaisesRegex(AttentionStateError, "plan"):
            _ = wrapper.plan_state
        self.assertEqual(package_attention.calls, [])

    def test_host_oracle_keeps_logical_fp8_cache(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            ReferenceTensor.zeros((0,), dtype="uint8"),
            kv_layout="NHD",
            backend="reference",
        )
        index = lambda values: ReferenceTensor(
            (len(values),), tuple(values), dtype="int32"
        )
        wrapper.plan(
            index((0, 1)),
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


if __name__ == "__main__":
    unittest.main()
