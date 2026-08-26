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
    AttentionOperatorQuantizedTensorInput,
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


def single_quantized_case(mode, quant_spec):
    qo_len = 2 if mode == AttentionMode.SINGLE_PREFILL else 1
    q = (
        ReferenceTensor.from_nested(
            [[[1.0], [0.5]], [[0.2], [0.4]]], dtype="float32"
        )
        if mode == AttentionMode.SINGLE_PREFILL
        else ReferenceTensor.from_nested([[1.0], [0.5]], dtype="float32")
    )
    logical_shape = (3, 1, 1)
    key = ReferenceQuantizedTensor(
        logical_shape,
        ReferenceTensor.from_nested([[[1]], [[0]], [[2]]], dtype="int8"),
        ReferenceTensor.from_nested(
            [[[1.0]], [[1.0]], [[1.0]]], dtype="float32"
        ),
        quant_spec,
    )
    value = ReferenceQuantizedTensor(
        logical_shape,
        ReferenceTensor.from_nested([[[2]], [[4]], [[1]]], dtype="int8"),
        ReferenceTensor.from_nested(
            [[[1.0]], [[1.0]], [[1.0]]], dtype="float32"
        ),
        quant_spec,
    )
    cache = RaggedKVCacheSpec(
        total_kv_tokens=3,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        dtype="int8",
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
        q_dtype="float32",
        kv_dtype="int8",
        kv_quant_spec=quant_spec,
        o_dtype="float32",
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


def public_single_quantized_runtime():
    values = bootstrap_components()
    quant_spec = single_quant_spec()
    cases = tuple(
        single_quantized_case(mode, quant_spec)
        for mode in (AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE)
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
    decode_rule = rules[1]
    decode_descriptor = replace(
        prefill_descriptor,
        kernel_id="public_single_decode_quantized_910b_test_v1",
        op="attention.%s" % AttentionMode.SINGLE_DECODE.value,
        artifact=attention_artifact(
            "artifacts/ascend910b/public_single_decode_quantized_test.o"
        ),
        launch_abi=attention_launch_abi(
            "attention_single_decode_quantized_test_entry"
        ),
        capability_binding=KernelCapabilityBinding(
            domain="attention",
            profile_id=profile.profile_id,
            rule_id=decode_rule.rule_id,
            profile_fingerprint=profile.fingerprint,
        ),
    )
    binding = replace(
        values["spec"].quantization_bindings[0], quant_spec=quant_spec
    )
    runtime_spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(prefill_descriptor, decode_descriptor),
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


if __name__ == "__main__":
    unittest.main()
