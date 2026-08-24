import unittest

from flashinfer_npu.attention import (
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionLoweredOperatorCall,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorPlanSession,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorWrapperSession,
    AttentionPreparedOperatorPlan,
    bind_attention_operator_callable,
    build_framework_attention_corpus,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, SchemaError


OPERATION_ID = "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"


def hash_value(character):
    return character * 64


def callable_authority(adapter_version="test-adapter"):
    operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)
    probe = AttentionOperatorProviderProbe(
        provider_id="cann",
        adapter_version=adapter_version,
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    observed = AttentionObservedOperatorCallable(
        provider_id="cann",
        package_name=operation.package_name,
        package_version="test-package",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=AttentionObservedCallableSignature(
            operation.positional_arguments,
            operation.keyword_arguments,
            observation_kind="test-adapter",
        ),
    )
    return probe, bind_attention_operator_callable(probe, operation, observed)


def plan_authority(probe):
    case = build_framework_attention_corpus().cases[0]
    framework_session = AttentionFrameworkSession(case.trace.spec.mode)
    plan = framework_session.plan(case.trace.spec, case.trace.metadata)
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.atomic.runtime.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_atomic_runtime_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-atomic-runtime-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-atomic-runtime-kernel",
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
        provider_id="cann",
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=hash_value("a"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    return plan, receipt, selection


class FakeFactory:
    provider_id = "cann"

    def __init__(self, prepared_operation_id=OPERATION_ID):
        self.operation_id = OPERATION_ID
        self.prepared_operation_id = prepared_operation_id
        self.prepare_calls = 0

    def prepare(self, plan, receipt, selection):
        self.prepare_calls += 1
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.prepared_operation_id,
            opaque_plan_token="atomic-runtime-plan",
            opaque_state={"test_only": True},
        )


class FakeRunAdapter:
    provider_id = "cann"

    def lower(self, active, request):
        key, value = request.kv_cache
        return AttentionLoweredOperatorCall(
            provider_id=self.provider_id,
            operation_id=active.prepared_plan.implementation_id,
            active_plan_fingerprint=active.fingerprint,
            positional_arguments=(
                ("query", request.query),
                ("key", key),
                ("value", value),
            ),
            keyword_arguments=(("return_softmax_lse", False),),
            consumed_request_fields=request.consumed_fields,
        )


class AttentionAtomicRuntimeBindingCheckpoint(unittest.TestCase):
    """Checkpoint 012: publish plan, operation and callable as one runtime."""

    def test_wrapper_publishes_a_closed_runtime_authority_chain(self):
        probe, callable_binding = callable_authority()
        plan, receipt, selection = plan_authority(probe)
        wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        wrapper.plan(
            FakeFactory(),
            FakeRunAdapter(),
            plan,
            receipt,
            selection,
            callable_binding,
        )

        runtime = wrapper.runtime_binding
        self.assertEqual(runtime.active_plan_fingerprint, wrapper.active_plan.fingerprint)
        self.assertEqual(
            runtime.operation_binding_fingerprint,
            wrapper.operation_binding.fingerprint,
        )
        self.assertEqual(
            runtime.callable_binding_fingerprint,
            wrapper.callable_binding.fingerprint,
        )
        self.assertEqual(wrapper.run("query", ("key", "value")).operation_id, OPERATION_ID)

    def test_stale_callable_is_rejected_before_prepare_and_preserves_runtime(self):
        probe, callable_binding = callable_authority()
        plan, receipt, selection = plan_authority(probe)
        wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        first_factory = FakeFactory()
        wrapper.plan(
            first_factory,
            FakeRunAdapter(),
            plan,
            receipt,
            selection,
            callable_binding,
        )
        active = wrapper.active_plan
        runtime = wrapper.runtime_binding

        _, stale_binding = callable_authority("different-adapter-generation")
        candidate_factory = FakeFactory()
        with self.assertRaisesRegex(SchemaError, "does not authorize"):
            wrapper.plan(
                candidate_factory,
                FakeRunAdapter(),
                plan,
                receipt,
                selection,
                stale_binding,
            )
        self.assertEqual(candidate_factory.prepare_calls, 0)
        self.assertIs(wrapper.active_plan, active)
        self.assertIs(wrapper.runtime_binding, runtime)

    def test_factory_cannot_change_declared_operation_during_prepare(self):
        probe, _ = callable_authority()
        plan, receipt, selection = plan_authority(probe)
        session = AttentionOperatorPlanSession()
        factory = FakeFactory(
            "cann.torch_npu.npu_fused_infer_attention_score@6.0.0"
        )

        with self.assertRaisesRegex(SchemaError, "changed its declared operation"):
            session.plan(factory, plan, receipt, selection)
        self.assertFalse(session.is_planned)


if __name__ == "__main__":
    unittest.main()
