import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionLoweredOperatorCall,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorWrapperSession,
    AttentionPreparedOperatorPlan,
    AttentionStateError,
    BatchAttention,
    bind_attention_operator_callable,
    build_framework_attention_corpus,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


OPERATION_ID = "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"


def callable_authority(operation_id=OPERATION_ID):
    operation = load_packaged_attention_operator_catalog().get(operation_id)
    probe = AttentionOperatorProviderProbe(
        provider_id=operation.provider_id,
        adapter_version="test-adapter",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    observed = AttentionObservedOperatorCallable(
        provider_id=operation.provider_id,
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


def plan_authority(generation=1, provider_id="cann"):
    case = build_framework_attention_corpus().cases[0]
    session = AttentionFrameworkSession(case.trace.spec.mode)
    plan = None
    for _ in range(generation):
        plan = session.plan(case.trace.spec, case.trace.metadata)
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.wrapper.session.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_wrapper_session_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-wrapper-session-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-wrapper-session-kernel",
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
    probe, _ = callable_authority()
    selection = AttentionOperatorProviderSelection(
        provider_id=provider_id,
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

    def __init__(
        self,
        operation_id=(
            OPERATION_ID
        ),
    ):
        self.operation_id = operation_id
        self.fail = False

    def prepare(self, plan, receipt, selection):
        if self.fail:
            raise RuntimeError("fake prepare failure")
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="wrapper-session-%d" % plan.generation,
            opaque_state={"generation": plan.generation},
        )


class FakeRunAdapter:
    provider_id = "cann"

    def __init__(self):
        self.lowered_generations = []

    def lower(self, active, request):
        generation = active.prepared_plan.opaque_state["generation"]
        self.lowered_generations.append(generation)
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


class AttentionOperatorWrapperSessionCheckpoint(unittest.TestCase):
    """Checkpoint 008: users see run(q, kv_cache, ...), not provider plumbing."""

    def test_run_surface_hides_mode_specific_internal_fields_without_plan_handles(self):
        internal = tuple(
            inspect.signature(AttentionOperatorWrapperSession.run).parameters
        )
        public = tuple(inspect.signature(BatchAttention.run).parameters)
        self.assertEqual(
            tuple(
                name
                for name in internal
                if name not in ("return_lse", "q_scale", "o_scale")
            ),
            public,
        )
        self.assertNotIn(
            "adapter", inspect.signature(AttentionOperatorWrapperSession.run).parameters
        )
        self.assertNotIn(
            "plan", inspect.signature(AttentionOperatorWrapperSession.run).parameters
        )

    def test_plan_returns_none_and_run_uses_only_wrapper_owned_state(self):
        wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        with self.assertRaisesRegex(AttentionStateError, "has not been planned"):
            wrapper.run("query", ("key", "value"))

        plan, receipt, selection = plan_authority()
        adapter = FakeRunAdapter()
        _, callable_binding = callable_authority()
        self.assertIsNone(
            wrapper.plan(
                FakeFactory(),
                adapter,
                plan,
                receipt,
                selection,
                callable_binding,
            )
        )
        lowered = wrapper.run("query", ("key", "value"))

        self.assertEqual(
            lowered.operation_id,
            OPERATION_ID,
        )
        self.assertEqual(adapter.lowered_generations, [1])

    def test_failed_replan_preserves_previous_plan_and_adapter(self):
        wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        first_plan, first_receipt, first_selection = plan_authority(1)
        first_adapter = FakeRunAdapter()
        _, callable_binding = callable_authority()
        wrapper.plan(
            FakeFactory(),
            first_adapter,
            first_plan,
            first_receipt,
            first_selection,
            callable_binding,
        )
        first_active = wrapper.active_plan

        second_plan, second_receipt, second_selection = plan_authority(2)
        failing_factory = FakeFactory()
        failing_factory.fail = True
        with self.assertRaisesRegex(RuntimeError, "fake prepare failure"):
            wrapper.plan(
                failing_factory,
                FakeRunAdapter(),
                second_plan,
                second_receipt,
                second_selection,
                callable_binding,
            )

        self.assertIs(wrapper.active_plan, first_active)
        wrapper.run("query", ("key", "value"))
        self.assertEqual(first_adapter.lowered_generations, [1])

        wrong_adapter = FakeRunAdapter()
        wrong_adapter.provider_id = "flash_attention_npu"
        with self.assertRaisesRegex(SchemaError, "selected provider"):
            wrapper.plan(
                FakeFactory(),
                wrong_adapter,
                second_plan,
                second_receipt,
                second_selection,
                callable_binding,
            )
        self.assertIs(wrapper.active_plan, first_active)


if __name__ == "__main__":
    unittest.main()
