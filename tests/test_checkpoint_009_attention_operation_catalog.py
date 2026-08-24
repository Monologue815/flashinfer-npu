import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionOperatorActivePlan,
    AttentionOperatorOperationCatalog,
    AttentionOperatorProviderSelection,
    AttentionPreparedOperatorPlan,
    bind_attention_operator_operation,
    build_framework_attention_corpus,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


def active_plan(operation_id, *, provider_id="cann", corpus_index=0):
    case = build_framework_attention_corpus().cases[corpus_index]
    session = AttentionFrameworkSession(case.trace.spec.mode)
    plan = session.plan(case.trace.spec, case.trace.metadata)
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.operation.catalog.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_operation_catalog_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-operation-catalog-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-operation-catalog-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=Backend.ACLNN,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )
    selection = AttentionOperatorProviderSelection(
        provider_id=provider_id,
        provider_probe_fingerprint=hash_value("9"),
        provider_record_fingerprint=hash_value("a"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    prepared = AttentionPreparedOperatorPlan(
        provider_id=provider_id,
        provider_selection_fingerprint=selection.fingerprint,
        framework_plan_fingerprint=plan.fingerprint,
        framework_plan_generation=plan.generation,
        implementation_id=operation_id,
        opaque_plan_token="fake-operation-catalog-plan",
        opaque_state={"test_only": True},
    )
    return AttentionOperatorActivePlan(plan, receipt, selection, prepared)


class AttentionOperationCatalogCheckpoint(unittest.TestCase):
    """Checkpoint 009: versioned documented API signatures, no package import."""

    def test_packaged_catalog_freezes_api_variants_and_round_trips(self):
        catalog = load_packaged_attention_operator_catalog()

        self.assertEqual(len(catalog.operations), 9)
        self.assertEqual(
            catalog,
            AttentionOperatorOperationCatalog.from_dict(catalog.to_dict()),
        )
        self.assertEqual(len(catalog.fingerprint), 64)
        self.assertIn(
            "cann.torch_npu.npu_fused_infer_attention_score@6.0.0",
            catalog.operation_ids,
        )
        self.assertIn(
            "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0",
            catalog.operation_ids,
        )
        self.assertIn(
            "flash_attention_npu.flash_attn_varlen_func@v4",
            catalog.operation_ids,
        )

    def test_critical_signature_differences_are_explicit(self):
        catalog = load_packaged_attention_operator_catalog()
        cann_v2 = catalog.get(
            "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"
        )
        self.assertEqual(cann_v2.positional_arguments, ("query", "key", "value"))
        self.assertEqual(
            cann_v2.host_sequence_arguments,
            ("actual_seq_qlen", "actual_seq_kvlen"),
        )
        self.assertIn("dequant_scale_key", cann_v2.quant_arguments)
        self.assertEqual(cann_v2.paged_table_argument, "block_table")

        flash_v2 = catalog.get(
            "flash_attention_npu.flash_attn_with_kvcache@v2"
        )
        flash_v3 = catalog.get(
            "flash_attention_npu.flash_attn_with_kvcache@v3"
        )
        self.assertFalse(flash_v2.supports_lse)
        self.assertIsNone(flash_v2.lse_control_argument)
        self.assertEqual(flash_v2.paged_table_argument, "block_table")
        self.assertTrue(flash_v3.supports_lse)
        self.assertEqual(flash_v3.lse_control_argument, "return_softmax_lse")
        self.assertEqual(flash_v3.paged_table_argument, "page_table")
        self.assertEqual(
            flash_v2.callable_path,
            flash_v3.callable_path,
        )
        self.assertNotEqual(flash_v2.fingerprint, flash_v3.fingerprint)

    def test_binding_requires_exact_version_provider_and_candidate_mode(self):
        catalog = load_packaged_attention_operator_catalog()
        operation_id = (
            "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"
        )
        active = active_plan(operation_id)
        binding = bind_attention_operator_operation(catalog, active)
        self.assertEqual(binding.operation_id, operation_id)
        self.assertEqual(binding.api_version, "7.3.0")
        self.assertEqual(len(binding.fingerprint), 64)

        unversioned = active_plan(
            "torch_npu.npu_fused_infer_attention_score_v2"
        )
        with self.assertRaisesRegex(SchemaError, "unknown Attention operation_id"):
            bind_attention_operator_operation(catalog, unversioned)

        wrong_provider = active_plan(
            operation_id, provider_id="flash_attention_npu"
        )
        with self.assertRaisesRegex(SchemaError, "active provider"):
            bind_attention_operator_operation(catalog, wrong_provider)

        mixed = active_plan(
            "cann.torch_npu.npu_fused_infer_attention_score@6.0.0",
            corpus_index=7,
        )
        with self.assertRaisesRegex(SchemaError, "planned mode"):
            bind_attention_operator_operation(catalog, mixed)

        operation = catalog.get(operation_id)
        with self.assertRaisesRegex(SchemaError, "declared argument"):
            replace(operation, paged_table_argument="unknown_table")
        with self.assertRaisesRegex(SchemaError, "duplicate operation_id"):
            AttentionOperatorOperationCatalog(
                "duplicate-test", catalog.operations + (catalog.operations[0],)
            )


if __name__ == "__main__":
    unittest.main()
