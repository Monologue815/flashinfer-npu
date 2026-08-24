import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionOperatorPlanSession,
    AttentionOperatorProviderSelection,
    AttentionPreparedOperatorPlan,
    AttentionStateError,
    build_framework_attention_corpus,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


def framework_plan(generation=1):
    case = build_framework_attention_corpus().cases[0]
    session = AttentionFrameworkSession(case.trace.spec.mode)
    plan = None
    for _ in range(generation):
        plan = session.plan(case.trace.spec, case.trace.metadata)
    return plan


def dispatch_receipt(plan):
    return AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.active.plan.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_active_plan_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-active-plan-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-active-plan-kernel",
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


def provider_selection(receipt):
    return AttentionOperatorProviderSelection(
        provider_id="cann",
        provider_probe_fingerprint=hash_value("9"),
        provider_record_fingerprint=hash_value("a"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )


class FakePlanFactory:
    provider_id = "cann"
    operation_id = "fake.cann.attention"

    def __init__(self):
        self.prepare_calls = 0
        self.fail = False
        self.stale_selection = False

    def prepare(self, plan, receipt, selection):
        self.prepare_calls += 1
        if self.fail:
            raise RuntimeError("fake prepare failure")
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=(
                hash_value("f")
                if self.stale_selection
                else selection.fingerprint
            ),
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="fake-plan-generation-%d" % plan.generation,
            opaque_state={"test_only": True, "generation": plan.generation},
        )


class AttentionOperatorActivePlanCheckpoint(unittest.TestCase):
    """Checkpoint 006: provider prepare and atomic active-plan publication."""

    def test_plan_returns_none_and_keeps_opaque_state_inside_the_session(self):
        plan = framework_plan()
        receipt = dispatch_receipt(plan)
        selection = provider_selection(receipt)
        factory = FakePlanFactory()
        session = AttentionOperatorPlanSession()

        with self.assertRaisesRegex(AttentionStateError, "has not been planned"):
            _ = session.active_plan
        self.assertIsNone(session.plan(factory, plan, receipt, selection))

        active = session.active_plan
        self.assertEqual(factory.prepare_calls, 1)
        self.assertEqual(active.framework_plan, plan)
        self.assertEqual(active.provider_selection, selection)
        self.assertEqual(
            active.prepared_plan.opaque_state,
            {"test_only": True, "generation": 1},
        )
        self.assertEqual(len(active.fingerprint), 64)

    def test_failed_replan_preserves_the_last_committed_active_plan(self):
        first_plan = framework_plan(1)
        first_receipt = dispatch_receipt(first_plan)
        first_selection = provider_selection(first_receipt)
        factory = FakePlanFactory()
        session = AttentionOperatorPlanSession()
        session.plan(factory, first_plan, first_receipt, first_selection)
        first_active = session.active_plan

        second_plan = framework_plan(2)
        second_receipt = dispatch_receipt(second_plan)
        second_selection = provider_selection(second_receipt)
        factory.fail = True
        with self.assertRaisesRegex(RuntimeError, "fake prepare failure"):
            session.plan(factory, second_plan, second_receipt, second_selection)

        self.assertIs(session.active_plan, first_active)
        self.assertEqual(session.active_plan.framework_plan.generation, 1)

    def test_stale_prepared_identity_is_rejected_before_publication(self):
        plan = framework_plan()
        receipt = dispatch_receipt(plan)
        selection = provider_selection(receipt)
        factory = FakePlanFactory()
        factory.stale_selection = True
        session = AttentionOperatorPlanSession()

        with self.assertRaisesRegex(SchemaError, "identity is stale"):
            session.plan(factory, plan, receipt, selection)
        self.assertFalse(session.is_planned)

        wrong_receipt = replace(receipt, profile_fingerprint=hash_value("b"))
        with self.assertRaisesRegex(SchemaError, "does not bind"):
            session.plan(factory, plan, wrong_receipt, selection)
        self.assertFalse(session.is_planned)


if __name__ == "__main__":
    unittest.main()
