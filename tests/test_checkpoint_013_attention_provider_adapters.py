import sys
import unittest

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorTensorPlan,
    AttentionOperatorWrapperSession,
    AttentionPlanSpec,
    CannV2PagedPlanFactory,
    CannV2PagedPlanState,
    CannV2PagedRunAdapter,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedPlanState,
    FlashAttentionNpuV3PagedRunAdapter,
    KVLayout,
    PagedKVMetadata,
    PagedPrefillMetadata,
    AttentionMode,
    bind_attention_operator_callable,
    build_attention_dense_page_table,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, QuantSpec, SchemaError


def hash_value(character):
    return character * 64


def framework_plan(*, provider, page_size=None, layout=None, quant=None):
    if provider == "cann":
        page_size = 128 if page_size is None else page_size
        layout = KVLayout.HND if layout is None else layout
        last_page_len = (min(64, page_size), page_size)
    else:
        page_size = 16 if page_size is None else page_size
        layout = KVLayout.NHD if layout is None else layout
        last_page_len = (8, 16)
    kv_dtype = "bfloat16" if quant is None else quant.storage_dtype
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=layout,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype=kv_dtype,
        o_dtype="bfloat16",
        kv_quant_spec=quant,
    )
    metadata = PagedPrefillMetadata(
        qo_indptr=(0, 2, 3),
        paged_kv=PagedKVMetadata(
            indptr=(0, 2, 3),
            indices=(7, 3, 9),
            last_page_len=last_page_len,
            page_size=page_size,
        ),
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def runtime_authority(plan, provider, operation_id, backend):
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.provider.adapter.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_provider_adapter_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-provider-adapter-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-provider-adapter-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=backend,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )
    operation = load_packaged_attention_operator_catalog().get(operation_id)
    probe = AttentionOperatorProviderProbe(
        provider_id=provider,
        adapter_version="checkpoint-013-test",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    selection = AttentionOperatorProviderSelection(
        provider_id=provider,
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=hash_value("9"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    observed = AttentionObservedOperatorCallable(
        provider_id=provider,
        package_name=operation.package_name,
        package_version="test-package",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=AttentionObservedCallableSignature(
            operation.positional_arguments,
            operation.keyword_arguments,
            observation_kind="checkpoint-013-test",
        ),
    )
    binding = bind_attention_operator_callable(probe, operation, observed)
    return receipt, selection, binding


def planned_wrapper(provider):
    plan = framework_plan(provider=provider)
    if provider == "cann":
        factory = CannV2PagedPlanFactory()
        adapter = CannV2PagedRunAdapter()
        operation_id = CANN_V2_OPERATION_ID
        backend = Backend.ACLNN
    else:
        factory = FlashAttentionNpuV3PagedPlanFactory()
        adapter = FlashAttentionNpuV3PagedRunAdapter()
        operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID
        backend = Backend.ASCENDC_AOT
    receipt, selection, binding = runtime_authority(
        plan, provider, operation_id, backend
    )
    wrapper = AttentionOperatorWrapperSession(
        load_packaged_attention_operator_catalog()
    )
    wrapper.plan(factory, adapter, plan, receipt, selection, binding)
    return wrapper


class AttentionProviderAdaptersCheckpoint(unittest.TestCase):
    """Checkpoint 013: exact, non-executing provider-specific lowering."""

    def test_csr_page_table_is_densified_once_with_deterministic_padding(self):
        table = build_attention_dense_page_table(
            (0, 2, 3), (7, 3, 9), role="page_table"
        )

        self.assertEqual(table.tensor.shape, (2, 2))
        self.assertEqual(table.row_page_counts, (2, 1))
        self.assertEqual(table.rows, ((7, 3), (9, 0)))
        self.assertEqual(table.tensor.dtype, "int32")
        self.assertEqual(table.tensor.device_policy, "same_as_query")
        self.assertEqual(len(table.fingerprint), 64)

    def test_cann_v2_plan_owns_tnd_lengths_mask_and_exact_call(self):
        imported_before = set(sys.modules)
        wrapper = planned_wrapper("cann")
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertIsInstance(state, CannV2PagedPlanState)
        self.assertEqual(state.query_cumulative_lengths, (2, 3))
        self.assertEqual(state.kv_sequence_lengths, (192, 128))
        self.assertEqual(state.block_table.rows, ((7, 3), (9, 0)))
        self.assertEqual(state.minimum_kv_block_pool_size, 3)
        self.assertEqual(state.sparse_mode, 3)
        self.assertIsInstance(state.causal_mask, AttentionOperatorTensorPlan)
        self.assertEqual(state.causal_mask.materialization, "right_down_causal_2048")

        query, key, value = object(), object(), object()
        lowered = wrapper.run(query, (key, value))
        self.assertEqual(
            lowered.positional_arguments,
            (("query", query), ("key", key), ("value", value)),
        )
        keywords = dict(lowered.keyword_arguments)
        self.assertEqual(keywords["actual_seq_qlen"], [2, 3])
        self.assertEqual(keywords["actual_seq_kvlen"], [192, 128])
        self.assertIs(keywords["block_table"], state.block_table.tensor)
        self.assertIs(keywords["atten_mask"], state.causal_mask)
        self.assertEqual(keywords["input_layout"], "TND")
        self.assertEqual(keywords["block_size"], 128)
        self.assertTrue(keywords["return_softmax_lse"])
        self.assertEqual(lowered.return_names, ("output", "softmax_lse"))
        self.assertEqual(
            lowered.consumed_request_fields,
            ("query", "kv_cache", "return_lse", "logits_soft_cap"),
        )
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

        output_only = wrapper.run(query, (key, value), return_lse=False)
        output_only_keywords = dict(output_only.keyword_arguments)
        self.assertFalse(output_only_keywords["return_softmax_lse"])
        self.assertEqual(output_only.return_names, ("output",))

    def test_flash_attention_npu_v3_plan_owns_ragged_and_page_metadata(self):
        wrapper = planned_wrapper("flash_attention_npu")
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertIsInstance(state, FlashAttentionNpuV3PagedPlanState)
        self.assertEqual(state.cu_seqlens_q.values, (0, 2, 3))
        self.assertEqual(state.cache_seqlens.values, (24, 16))
        self.assertEqual(state.page_table.rows, ((7, 3), (9, 0)))
        self.assertEqual(state.max_seqlen_q, 2)
        self.assertEqual(state.window_size, (-1, -1))

        query, key, value = object(), object(), object()
        lowered = wrapper.run(query, (key, value))
        self.assertEqual(
            lowered.positional_arguments,
            (("q", query), ("k_cache", key), ("v_cache", value)),
        )
        keywords = dict(lowered.keyword_arguments)
        self.assertIs(keywords["cu_seqlens_q"], state.cu_seqlens_q)
        self.assertIs(keywords["cache_seqlens"], state.cache_seqlens)
        self.assertIs(keywords["page_table"], state.page_table.tensor)
        self.assertEqual(keywords["max_seqlen_q"], 2)
        self.assertTrue(keywords["causal"])
        self.assertEqual(keywords["window_size"], (-1, -1))
        self.assertTrue(keywords["return_softmax_lse"])
        self.assertEqual(lowered.mutable_argument_names, ("k_cache", "v_cache"))

    def test_unverified_layout_quantization_and_run_options_are_rejected(self):
        wrong_layout_plan = framework_plan(provider="cann", layout=KVLayout.NHD)
        receipt, selection, _ = runtime_authority(
            wrong_layout_plan, "cann", CANN_V2_OPERATION_ID, Backend.ACLNN
        )
        with self.assertRaisesRegex(SchemaError, "requires HND"):
            CannV2PagedPlanFactory().prepare(
                wrong_layout_plan, receipt, selection
            )

        wrong_page_plan = framework_plan(provider="cann", page_size=16)
        receipt, selection, _ = runtime_authority(
            wrong_page_plan, "cann", CANN_V2_OPERATION_ID, Backend.ACLNN
        )
        with self.assertRaisesRegex(SchemaError, "block size"):
            CannV2PagedPlanFactory().prepare(wrong_page_plan, receipt, selection)

        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="bfloat16",
            accumulator_dtype="float32",
        )
        quant_plan = framework_plan(provider="flash_attention_npu", quant=quant)
        receipt, selection, _ = runtime_authority(
            quant_plan,
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            Backend.ASCENDC_AOT,
        )
        with self.assertRaisesRegex(SchemaError, "no verified paged KV quantization"):
            FlashAttentionNpuV3PagedPlanFactory().prepare(
                quant_plan, receipt, selection
            )

        wrapper = planned_wrapper("flash_attention_npu")
        with self.assertRaisesRegex(SchemaError, "KV scale binding"):
            wrapper.run("q", ("k", "v"), k_scale="unverified-scale")
        with self.assertRaisesRegex(SchemaError, "separate"):
            wrapper.run("q", "packed-kv-cache")


if __name__ == "__main__":
    unittest.main()
