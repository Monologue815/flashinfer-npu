import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionLoweredOperatorCall,
    AttentionOperatorActivePlan,
    AttentionOperatorProviderSelection,
    AttentionOperatorRunRequest,
    AttentionPreparedOperatorPlan,
    build_framework_attention_corpus,
    lower_attention_operator_run,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


def active_plan(
    provider_id="cann",
    operation_id="torch_npu.npu_fused_infer_attention_score",
):
    case = build_framework_attention_corpus().cases[0]
    framework_session = AttentionFrameworkSession(case.trace.spec.mode)
    plan = framework_session.plan(case.trace.spec, case.trace.metadata)
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.run.lowering.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_run_lowering_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-run-lowering-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-run-lowering-kernel",
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
        opaque_plan_token="fake-run-lowering-plan-1",
        opaque_state={"block_table": "plan-owned-block-table"},
    )
    return AttentionOperatorActivePlan(plan, receipt, selection, prepared)


class FakeCannRunAdapter:
    provider_id = "cann"

    def __init__(self, *, drop_optional=False, change_operation=False):
        self.lower_calls = 0
        self.drop_optional = drop_optional
        self.change_operation = change_operation

    def lower(self, active, request):
        self.lower_calls += 1
        key_cache, value_cache = request.kv_cache
        consumed = request.consumed_fields
        if self.drop_optional:
            consumed = tuple(item for item in consumed if item != "v_scale")
        return AttentionLoweredOperatorCall(
            provider_id=self.provider_id,
            operation_id=(
                "torch_npu.some_other_attention"
                if self.change_operation
                else active.prepared_plan.implementation_id
            ),
            active_plan_fingerprint=active.fingerprint,
            positional_arguments=(
                ("query", request.query),
                ("key", key_cache),
                ("value", value_cache),
            ),
            keyword_arguments=(
                ("block_table", active.prepared_plan.opaque_state["block_table"]),
                ("softmax_lse_flag", request.lse is not None),
            ),
            return_names=("output", "softmax_lse"),
            consumed_request_fields=consumed,
        )


class AttentionOperatorRunLoweringCheckpoint(unittest.TestCase):
    """Checkpoint 007: active-plan run lowering, with no external op call."""

    def test_run_request_reuses_the_planned_operation_and_opaque_state(self):
        active = active_plan()
        request = AttentionOperatorRunRequest.from_active_plan(
            active,
            "query-tensor",
            ("key-cache", "value-cache"),
            lse="lse-buffer",
            v_scale="value-scale",
            q_head_scale="query-head-scale",
        )
        adapter = FakeCannRunAdapter()

        lowered = lower_attention_operator_run(adapter, active, request)

        self.assertEqual(adapter.lower_calls, 1)
        self.assertEqual(
            lowered.operation_id,
            "torch_npu.npu_fused_infer_attention_score",
        )
        self.assertEqual(
            lowered.positional_arguments,
            (
                ("query", "query-tensor"),
                ("key", "key-cache"),
                ("value", "value-cache"),
            ),
        )
        self.assertIn(
            ("block_table", "plan-owned-block-table"),
            lowered.keyword_arguments,
        )
        self.assertEqual(lowered.consumed_request_fields, request.consumed_fields)
        self.assertIn("q_head_scale", request.consumed_fields)

    def test_non_default_run_fields_cannot_be_silently_dropped(self):
        active = active_plan()
        request = AttentionOperatorRunRequest.from_active_plan(
            active,
            "query",
            ("key", "value"),
            v_scale="scale",
        )

        with self.assertRaisesRegex(SchemaError, "consume every run field"):
            lower_attention_operator_run(
                FakeCannRunAdapter(drop_optional=True), active, request
            )

    def test_run_cannot_reselect_provider_operation_or_use_a_stale_plan(self):
        active = active_plan()
        request = AttentionOperatorRunRequest.from_active_plan(
            active, "query", ("key", "value")
        )

        with self.assertRaisesRegex(SchemaError, "changed the planned implementation"):
            lower_attention_operator_run(
                FakeCannRunAdapter(change_operation=True), active, request
            )

        stale = replace(request, framework_plan_generation=2)
        with self.assertRaisesRegex(SchemaError, "does not bind the active plan"):
            lower_attention_operator_run(FakeCannRunAdapter(), active, stale)

        wrong_provider = FakeCannRunAdapter()
        wrong_provider.provider_id = "flash_attention_npu"
        with self.assertRaisesRegex(SchemaError, "does not match the active provider"):
            lower_attention_operator_run(wrong_provider, active, request)


if __name__ == "__main__":
    unittest.main()
