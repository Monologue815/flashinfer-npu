import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    CANN_ATTENTION_PROVIDER_ID,
    FLASH_ATTENTION_NPU_PROVIDER_ID,
    AttentionDispatchReceipt,
    AttentionBackendCapabilityProfile,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionNumericsPolicy,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderRecord,
    AttentionOperatorProviderSelectionError,
    AttentionRuntimeEnvironment,
    AttentionMode,
    KVLayout,
    bind_attention_operator_provider,
    explain_attention_operator_provider_binding,
)
from flashinfer_npu.runtime import Backend


def hash_value(character):
    return character * 64


def draft_profile(profile_id, backend):
    return AttentionBackendCapabilityProfile(
        profile_id=profile_id,
        backend=backend,
        environment=AttentionRuntimeEnvironment(
            soc_version="unknown",
            soc_revision="unknown",
            driver_version="unknown",
            firmware_version="unknown",
            cann_version="unknown",
            torch_version="unknown",
            torch_npu_version="unknown",
            compiler_version="unknown",
            python_abi="unknown",
        ),
        status=AttentionCapabilityStatus.DRAFT,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(
            AttentionCapabilityRule(
                rule_id="paged_prefill_dense_v1",
                modes=(AttentionMode.BATCH_PREFILL_PAGED,),
                kv_layouts=(KVLayout.NHD,),
                dtype_signatures=(("float16", "float16", "float16"),),
            ),
        ),
    )


def receipt_for(profile):
    return AttentionDispatchReceipt(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        plan_fingerprint=hash_value("1"),
        admission_fingerprint=hash_value("2"),
        workload_fingerprint=hash_value("3"),
        numerics_policy_fingerprint=profile.numerics_policy.fingerprint,
        profile_id=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        rule_id=profile.rules[0].rule_id,
        environment_fingerprint=profile.environment.fingerprint,
        evidence_id="fake-provider-binding-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-provider-binding-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=profile.backend,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )


def records():
    cann_profile = draft_profile("cann.test.attention.v1", Backend.ACLNN)
    flash_profile = draft_profile(
        "flash_attention_npu.test.attention.v1", Backend.ASCENDC_AOT
    )
    return (
        AttentionOperatorProviderRecord(
            AttentionOperatorProviderProbe(
                CANN_ATTENTION_PROVIDER_ID,
                "0.1",
                True,
                package_versions=(("torch_npu", "test-only"),),
            ),
            (cann_profile,),
        ),
        AttentionOperatorProviderRecord(
            AttentionOperatorProviderProbe(
                FLASH_ATTENTION_NPU_PROVIDER_ID,
                "0.1",
                False,
                unavailable_reasons=("package is not installed",),
            ),
            (flash_profile,),
        ),
    )


class AttentionProviderBindingCheckpoint(unittest.TestCase):
    """Checkpoint 005: receipt-to-provider binding, not operator execution."""

    def test_auto_binds_the_available_owner_and_preserves_full_identity(self):
        provider_records = records()
        receipt = receipt_for(provider_records[0].profiles[0])

        report = explain_attention_operator_provider_binding(
            provider_records, receipt
        )
        self.assertEqual(
            tuple(item.provider_id for item in report.accepted),
            (CANN_ATTENTION_PROVIDER_ID,),
        )
        selection = bind_attention_operator_provider(provider_records, receipt)

        self.assertEqual(selection.provider_id, CANN_ATTENTION_PROVIDER_ID)
        self.assertEqual(selection.dispatch_receipt_fingerprint, receipt.fingerprint)
        self.assertEqual(
            selection.profile_fingerprint, receipt.profile_fingerprint
        )
        self.assertEqual(selection.backend, Backend.ACLNN)
        self.assertEqual(len(selection.fingerprint), 64)

    def test_explicit_provider_policy_is_fail_fast_and_explainable(self):
        provider_records = records()
        receipt = receipt_for(provider_records[0].profiles[0])
        report = explain_attention_operator_provider_binding(
            provider_records,
            receipt,
            provider=FLASH_ATTENTION_NPU_PROVIDER_ID,
        )

        flash = next(
            item
            for item in report.candidates
            if item.provider_id == FLASH_ATTENTION_NPU_PROVIDER_ID
        )
        self.assertIn(
            "provider unavailable: package is not installed", flash.reasons
        )
        self.assertIn(
            "does not own selected profile cann.test.attention.v1", flash.reasons
        )
        with self.assertRaisesRegex(
            AttentionOperatorProviderSelectionError,
            "package is not installed",
        ):
            bind_attention_operator_provider(
                provider_records,
                receipt,
                provider=FLASH_ATTENTION_NPU_PROVIDER_ID,
            )

    def test_stale_profile_and_empty_registry_never_fall_back(self):
        provider_records = records()
        receipt = receipt_for(provider_records[0].profiles[0])
        stale_profile = replace(
            provider_records[0].profiles[0],
            status=provider_records[0].profiles[0].status,
        )
        # Change the profile identity without changing the receipt.
        stale_profile = replace(
            stale_profile,
            rules=(
                replace(stale_profile.rules[0], max_head_dim_qk=128),
            ),
        )
        stale_record = replace(provider_records[0], profiles=(stale_profile,))

        with self.assertRaisesRegex(
            AttentionOperatorProviderSelectionError, "fingerprint is stale"
        ):
            bind_attention_operator_provider(
                (stale_record, provider_records[1]), receipt
            )
        with self.assertRaisesRegex(
            AttentionOperatorProviderSelectionError, "no providers discovered"
        ):
            bind_attention_operator_provider((), receipt)


if __name__ == "__main__":
    unittest.main()
