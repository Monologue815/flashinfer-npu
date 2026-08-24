import re
import sys
import unittest

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionMode,
    AttentionOperatorPlanGate,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionPlanSpec,
    CannV2PagedPlanFactory,
    CannV2PagedPlanGate,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedPlanGate,
    KVLayout,
    MixedPagedKVMetadata,
    PosEncodingMode,
    explain_cann_v2_paged_plan,
    explain_flash_attention_npu_v3_paged_plan,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, QuantSpec, SchemaError


def hash_value(character):
    return character * 64


def framework_plan(
    *,
    layout,
    page_size,
    q_dtype="bfloat16",
    kv_dtype=None,
    o_dtype=None,
    head_dim_qk=128,
    head_dim_vo=128,
    quant=None,
    pos_encoding_mode=PosEncodingMode.NONE,
    window_left=-1,
    window_right=0,
    logits_soft_cap=None,
    use_profiler=False,
):
    kv_dtype = q_dtype if kv_dtype is None else kv_dtype
    o_dtype = q_dtype if o_dtype is None else o_dtype
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        kv_layout=layout,
        causal=True,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=o_dtype,
        kv_quant_spec=quant,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        window_right=window_right,
        logits_soft_cap=logits_soft_cap,
        use_profiler=use_profiler,
    )
    metadata = MixedPagedKVMetadata(
        qo_indptr=(0, 2, 3),
        kv_indptr=(0, 2, 3),
        kv_indices=(7, 3, 9),
        kv_len_arr=(page_size + min(page_size, 64), page_size),
        page_size=page_size,
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def runtime_authority(plan, provider_id, operation_id, backend):
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="checkpoint.020.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="checkpoint_020_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="checkpoint-020-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="checkpoint-020-kernel",
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
        provider_id=provider_id,
        adapter_version="checkpoint-020-test",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    selection = AttentionOperatorProviderSelection(
        provider_id=provider_id,
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=hash_value("9"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    return receipt, selection


def prepare(factory, plan):
    if factory.provider_id == "cann":
        operation_id = CANN_V2_OPERATION_ID
        backend = Backend.ACLNN
    else:
        operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID
        backend = Backend.ASCENDC_AOT
    receipt, selection = runtime_authority(
        plan, factory.provider_id, operation_id, backend
    )
    return factory.prepare(plan, receipt, selection)


class ProviderPlanGatesCheckpoint(unittest.TestCase):
    """Checkpoint 020: auto gates and factories consume one rule source."""

    def test_documented_cann_and_flash_plans_are_accepted_without_imports(self):
        imported_before = set(sys.modules)
        cann_plan = framework_plan(layout=KVLayout.HND, page_size=128)
        flash_plan = framework_plan(layout=KVLayout.NHD, page_size=16)

        self.assertEqual(explain_cann_v2_paged_plan(cann_plan), ())
        self.assertEqual(
            explain_flash_attention_npu_v3_paged_plan(flash_plan), ()
        )
        self.assertEqual(
            CannV2PagedPlanGate().rejection_reasons(cann_plan, "npu:0"), ()
        )
        self.assertEqual(
            FlashAttentionNpuV3PagedPlanGate().rejection_reasons(
                flash_plan, "npu:0"
            ),
            (),
        )
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_gates_implement_the_package_runtime_plan_gate_protocol(self):
        self.assertIsInstance(CannV2PagedPlanGate(), AttentionOperatorPlanGate)
        self.assertIsInstance(
            FlashAttentionNpuV3PagedPlanGate(), AttentionOperatorPlanGate
        )

    def test_cann_gate_reports_every_independent_rejection_deterministically(self):
        plan = framework_plan(
            layout=KVLayout.NHD,
            page_size=16,
            q_dtype="float32",
            kv_dtype="float16",
            o_dtype="float16",
            head_dim_qk=64,
            head_dim_vo=64,
            window_left=32,
            window_right=0,
            logits_soft_cap=1.0,
            use_profiler=True,
        )

        reasons = explain_cann_v2_paged_plan(plan)

        self.assertEqual(reasons, explain_cann_v2_paged_plan(plan))
        joined = " | ".join(reasons)
        for fragment in (
            "logits soft cap",
            "profiler buffers",
            "requires HND",
            "float16 or bfloat16",
            "matching dtypes",
            "documented head dimensions",
            "sliding-window",
            "block size",
        ):
            self.assertIn(fragment, joined)

    def test_quantized_plan_is_rejected_by_both_unverified_bindings(self):
        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="bfloat16",
            accumulator_dtype="float32",
        )
        cann = framework_plan(
            layout=KVLayout.HND,
            page_size=128,
            kv_dtype="int8",
            quant=quant,
        )
        flash = framework_plan(
            layout=KVLayout.NHD,
            page_size=16,
            kv_dtype="int8",
            quant=quant,
        )

        self.assertIn(
            "provider operation has no verified paged KV quantization binding",
            explain_cann_v2_paged_plan(cann),
        )
        self.assertIn(
            "provider operation has no verified paged KV quantization binding",
            explain_flash_attention_npu_v3_paged_plan(flash),
        )

    def test_gate_and_factory_acceptance_are_consistent_for_plan_matrix(self):
        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="bfloat16",
            accumulator_dtype="float32",
        )
        cases = (
            (
                CannV2PagedPlanGate(),
                CannV2PagedPlanFactory(),
                framework_plan(layout=KVLayout.HND, page_size=128),
            ),
            (
                CannV2PagedPlanGate(),
                CannV2PagedPlanFactory(),
                framework_plan(layout=KVLayout.NHD, page_size=128),
            ),
            (
                CannV2PagedPlanGate(),
                CannV2PagedPlanFactory(),
                framework_plan(layout=KVLayout.HND, page_size=16),
            ),
            (
                FlashAttentionNpuV3PagedPlanGate(),
                FlashAttentionNpuV3PagedPlanFactory(),
                framework_plan(layout=KVLayout.NHD, page_size=16),
            ),
            (
                FlashAttentionNpuV3PagedPlanGate(),
                FlashAttentionNpuV3PagedPlanFactory(),
                framework_plan(layout=KVLayout.HND, page_size=128),
            ),
            (
                FlashAttentionNpuV3PagedPlanGate(),
                FlashAttentionNpuV3PagedPlanFactory(),
                framework_plan(
                    layout=KVLayout.NHD,
                    page_size=16,
                    kv_dtype="int8",
                    quant=quant,
                ),
            ),
        )

        for gate, factory, plan in cases:
            with self.subTest(operation=gate.operation_id, plan=plan.fingerprint):
                reasons = gate.rejection_reasons(plan, "npu:0")
                if reasons:
                    with self.assertRaisesRegex(
                        SchemaError, re.escape(reasons[0])
                    ):
                        prepare(factory, plan)
                else:
                    prepared = prepare(factory, plan)
                    self.assertEqual(prepared.implementation_id, gate.operation_id)


if __name__ == "__main__":
    unittest.main()
