import hashlib
import math
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
    AttentionOperatorQuantizedTensorInput,
    AttentionOperatorQuantArgumentBinding,
    AttentionPlanSpec,
    AttentionTrace,
    AttentionTraceCase,
    AttentionTraceCorpus,
    KVLayout,
    RaggedKVCacheSpec,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    SingleAttentionMetadata,
    attention_operator_runtime_registry_snapshot,
    build_attention_operator_runtime_resolvers,
    framework_attention_coverage_policy,
    infer_quant_scale_shape,
    infer_quant_storage_shape,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import single_decode_with_kv_cache
from flashinfer_npu.prefill import single_prefill_with_kv_cache
from flashinfer_npu.attention.frontend import single_fp8_per_head_quant_spec
from flashinfer_npu.runtime import KernelCapabilityBinding, QuantSpec, SchemaError
from tests.test_attention_capability import (
    attention_artifact,
    attention_launch_abi,
    bound_kernel,
    pinned_environment,
)
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_public_quantized_paged_prefill import metadata_tensor


class FakeNpuTensor:
    def __init__(self, name, shape, *, dtype="float32", device="npu:0"):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def __str__(self):
        return self.name


def single_quant_spec():
    return QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        scale_dtype="float32",
        granularity="group",
        group_size=(1, 1, 1),
        axis=(0, 1, 2),
    )


def single_quantized_case(
    mode, quant_spec, *, q_dtype="float32", o_dtype="float32"
):
    qo_len = 2 if mode == AttentionMode.SINGLE_PREFILL else 1
    q = (
        ReferenceTensor.from_nested(
            [[[1.0], [0.5]], [[0.2], [0.4]]], dtype=q_dtype
        )
        if mode == AttentionMode.SINGLE_PREFILL
        else ReferenceTensor.from_nested([[1.0], [0.5]], dtype=q_dtype)
    )
    logical_shape = (3, 1, 1)
    key = ReferenceQuantizedTensor(
        logical_shape,
        ReferenceTensor(
            infer_quant_storage_shape(logical_shape, quant_spec),
            (1.0, 0.0, 2.0),
            dtype=quant_spec.storage_dtype,
        ),
        ReferenceTensor(
            infer_quant_scale_shape(logical_shape, quant_spec),
            (1.0,) * math.prod(
                infer_quant_scale_shape(logical_shape, quant_spec)
            ),
            dtype=quant_spec.scale_dtype,
        ),
        quant_spec,
    )
    value = ReferenceQuantizedTensor(
        logical_shape,
        ReferenceTensor(
            infer_quant_storage_shape(logical_shape, quant_spec),
            (2.0, 4.0, 1.0),
            dtype=quant_spec.storage_dtype,
        ),
        ReferenceTensor(
            infer_quant_scale_shape(logical_shape, quant_spec),
            (1.0,) * math.prod(
                infer_quant_scale_shape(logical_shape, quant_spec)
            ),
            dtype=quant_spec.scale_dtype,
        ),
        quant_spec,
    )
    cache = RaggedKVCacheSpec(
        total_kv_tokens=3,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        dtype=quant_spec.storage_dtype,
        layout=KVLayout.NHD,
        device="cpu",
        quant_spec=quant_spec,
    )
    spec = AttentionPlanSpec(
        mode=mode,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        kv_layout=KVLayout.NHD,
        q_dtype=q_dtype,
        kv_dtype=quant_spec.storage_dtype,
        kv_quant_spec=quant_spec,
        o_dtype=o_dtype,
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(qo_len=qo_len, kv_len=3),
        q=q,
        kv_data=ReferenceQuantizedKVData(cache, key, value),
    )
    return AttentionTraceCase(
        "public_%s_int8_group" % mode.value,
        trace,
        "Synthetic exact single-request quantized provider fixture.",
    )


def public_single_quantized_runtime(
    *,
    runtime_scales=True,
    quant_spec=None,
    q_dtype="float32",
    o_dtype="float32",
    modes=(AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE),
):
    values = bootstrap_components()
    quant_spec = single_quant_spec() if quant_spec is None else quant_spec
    cases = tuple(
        single_quantized_case(
            mode, quant_spec, q_dtype=q_dtype, o_dtype=o_dtype
        )
        for mode in modes
    )
    corpus = AttentionTraceCorpus(
        "public-single-quantized-provider-v1",
        cases,
        "Synthetic framework corpus for exact single-request provider routes.",
    )
    policy = framework_attention_coverage_policy()
    coverage = policy.evaluate(corpus)
    rules = []
    for case in cases:
        spec = case.trace.spec
        metadata = case.trace.metadata
        rules.append(
            AttentionCapabilityRule(
                rule_id="public_%s_int8_group_v1" % spec.mode.value,
                modes=(spec.mode,),
                kv_layouts=(spec.kv_layout,),
                dtype_signatures=((spec.q_dtype, spec.kv_dtype, spec.o_dtype),),
                supports_dense_kv=False,
                quant_specs=(quant_spec,),
                pos_encoding_modes=(spec.pos_encoding_mode,),
                mask_kinds=("none",),
                causal_values=(spec.effective_causal,),
                max_head_dim_qk=spec.head_dim_qk,
                max_head_dim_vo=spec.head_dim_vo,
                max_gqa_group_size=2,
                metadata_limits=AttentionMetadataLimits(
                    max_batch_size=1,
                    max_total_qo_tokens=metadata.qo_len,
                    max_total_kv_tokens=metadata.kv_len,
                    max_total_pages=0,
                    max_page_size=0,
                ),
            )
        )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-public-single-quantized-framework-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-framework-fixture",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=tuple(case.case_id for case in cases),
        result_digest=hashlib.sha256(
            b"synthetic-public-single-quantized-framework-record"
        ).hexdigest(),
    )
    profile = AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.public_single_quantized.v1",
        backend="ascendc_aot",
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=tuple(rules),
        evidence=(evidence,),
    )
    prefill_descriptor = bound_kernel(
        profile,
        kernel_id="public_single_prefill_quantized_910b_test_v1",
        artifact=attention_artifact(
            "artifacts/ascend910b/public_single_prefill_quantized_test.o"
        ),
        launch_abi=attention_launch_abi(
            "attention_single_prefill_quantized_test_entry"
        ),
    )
    descriptors = [prefill_descriptor]
    for rule in rules[1:]:
        descriptors.append(
            replace(
                prefill_descriptor,
                kernel_id="public_%s_quantized_910b_test_v1" % rule.modes[0].value,
                op="attention.%s" % rule.modes[0].value,
                artifact=attention_artifact(
                    "artifacts/ascend910b/public_%s_quantized_test.o"
                    % rule.modes[0].value
                ),
                launch_abi=attention_launch_abi(
                    "attention_%s_quantized_test_entry" % rule.modes[0].value
                ),
                capability_binding=KernelCapabilityBinding(
                    domain="attention",
                    profile_id=profile.profile_id,
                    rule_id=rule.rule_id,
                    profile_fingerprint=profile.fingerprint,
                ),
            )
        )
    original_binding = values["spec"].quantization_bindings[0]
    if runtime_scales:
        binding = replace(
            original_binding,
            quant_spec=quant_spec,
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
                AttentionOperatorQuantArgumentBinding(
                    "run.q_head_scale", "runtime_query_head_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.k_head_scale", "runtime_key_head_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.v_head_scale", "runtime_value_head_scale"
                ),
            ),
            runtime_q_scale_policy="argument",
            runtime_k_scale_policy="argument",
            runtime_v_scale_policy="argument",
            runtime_q_head_scale_policy="argument",
            runtime_k_head_scale_policy="argument",
            runtime_v_head_scale_policy="argument",
            implicit_unit_scale_sources=(
                (
                    "kv.key.scale",
                    "kv.value.scale",
                    "run.q_head_scale",
                )
                if quant_spec.storage_dtype
                in {"float8_e4m3fn", "float8_e5m2"}
                else ()
            ),
        )
    else:
        binding = replace(original_binding, quant_spec=quant_spec)
    runtime_spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=tuple(descriptors),
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
    return values, quant_spec, registry


def quantized_tensor_pair(quant_spec, *, logical_shape=(3, 1, 1)):
    def component(name):
        return AttentionOperatorQuantizedTensorInput(
            quant_spec=quant_spec,
            logical_shape=logical_shape,
            storage=metadata_tensor(
                name + "-storage",
                infer_quant_storage_shape(logical_shape, quant_spec),
                quant_spec.storage_dtype,
            ),
            scale=metadata_tensor(
                name + "-scale",
                infer_quant_scale_shape(logical_shape, quant_spec),
                quant_spec.scale_dtype,
            ),
        )

    return component("key"), component("value")


class PublicQuantizedSingleAttentionTests(unittest.TestCase):
    """Single public APIs derive plans from explicit quantized logical shapes."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.quant_spec, registry = public_single_quantized_runtime()
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=self.values["catalog"]
        )
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def test_single_prefill_plans_and_injects_quantized_components(self):
        q = FakeNpuTensor("q-prefill", (2, 2, 1))
        key, value = quantized_tensor_pair(self.quant_spec)

        output = single_prefill_with_kv_cache(q, key, value)
        output_lse = single_prefill_with_kv_cache(
            q, key, value, return_lse=True
        )

        self.assertEqual(output, "package-output:q-prefill")
        self.assertEqual(
            output_lse, ("package-output:q-prefill", "package-lse:0.25")
        )
        first_call = package_attention.calls[0]
        self.assertIs(first_call[1], key.storage)
        self.assertIs(first_call[2], value.storage)
        self.assertIs(first_call[6], key.scale)
        self.assertIs(first_call[7], value.scale)

    def test_single_decode_plans_and_injects_quantized_components(self):
        q = FakeNpuTensor("q-decode", (2, 1))
        key, value = quantized_tensor_pair(self.quant_spec)

        output = single_decode_with_kv_cache(q, key, value)
        output_lse = single_decode_with_kv_cache(q, key, value, return_lse=True)

        self.assertEqual(output, "package-output:q-decode")
        self.assertEqual(
            output_lse, ("package-output:q-decode", "package-lse:0.25")
        )
        first_call = package_attention.calls[0]
        self.assertIs(first_call[1], key.storage)
        self.assertIs(first_call[2], value.storage)
        self.assertIs(first_call[6], key.scale)
        self.assertIs(first_call[7], value.scale)

    def test_single_quantized_contract_fails_before_external_callable(self):
        q = FakeNpuTensor("q-prefill", (2, 2, 1))
        key, value = quantized_tensor_pair(self.quant_spec)

        with self.assertRaisesRegex(TypeError, "provided together"):
            single_prefill_with_kv_cache(q, key, FakeNpuTensor("plain", (3, 1, 1)))
        with self.assertRaisesRegex(SchemaError, "head dimensions"):
            bad_key, bad_value = quantized_tensor_pair(
                self.quant_spec, logical_shape=(3, 1, 2)
            )
            single_prefill_with_kv_cache(q, bad_key, bad_value)
        self.assertEqual(package_attention.calls, [])

        bad_storage = AttentionOperatorQuantizedTensorInput(
            quant_spec=self.quant_spec,
            logical_shape=key.logical_shape,
            storage=metadata_tensor("bad-storage", (2, 1, 1), "int8"),
            scale=key.scale,
        )
        with self.assertRaisesRegex(SchemaError, "storage view shape"):
            single_prefill_with_kv_cache(q, bad_storage, value)
        self.assertEqual(package_attention.calls, [])

    def test_single_runtime_scales_require_and_follow_exact_binding(self):
        prefill_q = FakeNpuTensor("q-prefill-scale", (2, 2, 1))
        decode_q = FakeNpuTensor("q-decode-scale", (2, 1))
        key, value = quantized_tensor_pair(self.quant_spec)
        query_scale = metadata_tensor("query-head-scale", (2,), "float32")
        key_head_scale = metadata_tensor("key-head-scale", (1,), "float32")
        value_head_scale = metadata_tensor("value-head-scale", (1,), "float32")

        single_prefill_with_kv_cache(
            prefill_q,
            key,
            value,
            scale_q=query_scale,
            scale_k=key_head_scale,
            scale_v=value_head_scale,
            k_scale=1.25,
            v_scale=0.75,
        )
        single_decode_with_kv_cache(
            decode_q,
            key,
            value,
            q_scale=2.0,
            k_scale=1.5,
            v_scale=0.5,
        )

        prefill_call, decode_call = package_attention.calls
        self.assertEqual(prefill_call[8:10], (1.25, 0.75))
        self.assertIsNone(prefill_call[10])
        self.assertEqual(prefill_call[12:15], (
            query_scale,
            key_head_scale,
            value_head_scale,
        ))
        self.assertEqual(decode_call[8:10], (1.5, 0.5))
        self.assertIsNone(decode_call[10])
        self.assertEqual(decode_call[12:15], (None, None, None))

        package_attention.calls[:] = []
        with self.assertRaisesRegex(SchemaError, "q_head_scale shape"):
            single_prefill_with_kv_cache(
                prefill_q,
                key,
                value,
                scale_q=metadata_tensor("bad-query-head-scale", (1,), "float32"),
            )
        with self.assertRaisesRegex(SchemaError, "k_head_scale dtype"):
            single_prefill_with_kv_cache(
                prefill_q,
                key,
                value,
                scale_k=metadata_tensor("bad-key-head-scale", (1,), "float16"),
            )
        with self.assertRaisesRegex(SchemaError, "v_head_scale device"):
            single_prefill_with_kv_cache(
                prefill_q,
                key,
                value,
                scale_v=metadata_tensor(
                    "bad-value-head-scale", (1,), "float32", device="npu:1"
                ),
            )
        self.assertEqual(package_attention.calls, [])

        with self.assertRaisesRegex(SchemaError, "q_scale must be finite"):
            single_decode_with_kv_cache(
                decode_q, key, value, q_scale=float("nan")
            )
        self.assertEqual(package_attention.calls, [])

        values, _, registry = public_single_quantized_runtime(
            runtime_scales=False
        )
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=values["catalog"]
        )
        with self.assertRaisesRegex(
            SchemaError, "rejects run-time q_head_scale"
        ):
            single_prefill_with_kv_cache(prefill_q, key, value, scale_q=query_scale)
        with self.assertRaisesRegex(SchemaError, "rejects run-time k_scale"):
            single_decode_with_kv_cache(
                decode_q, key, value, k_scale=1.5
            )
        self.assertEqual(package_attention.calls, [])

    def test_bare_fp8_single_prefill_canonicalizes_public_head_scales(self):
        quant_spec = single_fp8_per_head_quant_spec("float8_e4m3fn", "NHD")
        self.assertEqual(quant_spec.axis, (1,))
        self.assertEqual(
            single_fp8_per_head_quant_spec("float8_e5m2", "HND").axis,
            (0,),
        )
        with self.assertRaisesRegex(
            SchemaError, "must authorize implicit unit scales"
        ):
            public_single_quantized_runtime(
                runtime_scales=False,
                quant_spec=quant_spec,
                q_dtype="float8_e4m3fn",
                o_dtype="float16",
                modes=(AttentionMode.SINGLE_PREFILL,),
            )
        values, _, registry = public_single_quantized_runtime(
            quant_spec=quant_spec,
            q_dtype="float8_e4m3fn",
            o_dtype="float16",
            modes=(AttentionMode.SINGLE_PREFILL,),
        )
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=values["catalog"]
        )
        package_attention.calls[:] = []
        q = FakeNpuTensor(
            "q-fp8", (2, 2, 1), dtype="float8_e4m3fn"
        )
        k = metadata_tensor("k-fp8", (3, 1, 1), "float8_e4m3fn")
        v = metadata_tensor("v-fp8", (3, 1, 1), "float8_e4m3fn")
        q_scale = metadata_tensor("q-fp8-scale", (2,), "float32")
        k_scale = metadata_tensor("k-fp8-scale", (1,), "float32")
        v_scale = metadata_tensor("v-fp8-scale", (1,), "float32")

        output = single_prefill_with_kv_cache(
            q,
            k,
            v,
            scale_q=q_scale,
            scale_k=k_scale,
            scale_v=v_scale,
            o_dtype="float16",
        )

        self.assertEqual(output, "package-output:q-fp8")
        call = package_attention.calls[0]
        self.assertIs(call[1], k)
        self.assertIs(call[2], v)
        self.assertIs(call[6], k_scale)
        self.assertIs(call[7], v_scale)
        self.assertIs(call[12], q_scale)
        self.assertEqual(call[13:15], (None, None))

        package_attention.calls[:] = []
        partial_output = single_prefill_with_kv_cache(
            q,
            k,
            v,
            scale_q=q_scale,
            scale_k=k_scale,
            o_dtype="float16",
        )
        implicit_output = single_prefill_with_kv_cache(
            q, k, v, o_dtype="float16"
        )

        self.assertEqual(partial_output, "package-output:q-fp8")
        self.assertEqual(implicit_output, "package-output:q-fp8")
        partial_call, implicit_call = package_attention.calls
        self.assertIs(partial_call[6], k_scale)
        self.assertIsNone(partial_call[7])
        self.assertIs(partial_call[12], q_scale)
        self.assertEqual(implicit_call[6:8], (None, None))
        self.assertIsNone(implicit_call[12])

if __name__ == "__main__":
    unittest.main()
