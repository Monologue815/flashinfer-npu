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
    bind_attention_operator_callable,
    build_framework_attention_corpus,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, SchemaError


CANN_V2 = "cann.torch_npu.npu_fused_infer_attention_score_v2@7.3.0"
FLASH_KV_V3 = "flash_attention_npu.flash_attn_with_kvcache@v3"


def hash_value(character):
    return character * 64


def callable_authority(operation_id):
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


def plan_authority(
    *, provider_id="cann", backend=Backend.ACLNN, operation_id=CANN_V2
):
    case = build_framework_attention_corpus().cases[0]
    session = AttentionFrameworkSession(case.trace.spec.mode)
    plan = session.plan(case.trace.spec, case.trace.metadata)
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.catalog.lowering.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_catalog_lowering_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-catalog-lowering-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-catalog-lowering-kernel",
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
    probe, callable_binding = callable_authority(operation_id)
    selection = AttentionOperatorProviderSelection(
        provider_id=provider_id,
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=hash_value("a"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    return plan, receipt, selection, callable_binding


class FakeFactory:
    def __init__(self, provider_id, operation_id):
        self.provider_id = provider_id
        self.operation_id = operation_id

    def prepare(self, plan, receipt, selection):
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="catalog-bound-lowering-plan",
            opaque_state={"test_only": True},
        )


class FakeRunAdapter:
    def __init__(
        self,
        provider_id="cann",
        *,
        positional_names=("query", "key", "value"),
        keyword_arguments=(("return_softmax_lse", False),),
        return_names=("output",),
        mutable_argument_names=(),
    ):
        self.provider_id = provider_id
        self.positional_names = positional_names
        self.keyword_arguments = keyword_arguments
        self.return_names = return_names
        self.mutable_argument_names = mutable_argument_names

    def lower(self, active, request):
        key, value = request.kv_cache
        values = {
            "query": request.query,
            "key": key,
            "value": value,
            "q": request.query,
            "k_cache": key,
            "v_cache": value,
        }
        return AttentionLoweredOperatorCall(
            provider_id=self.provider_id,
            operation_id=active.prepared_plan.implementation_id,
            active_plan_fingerprint=active.fingerprint,
            positional_arguments=tuple(
                (name, values[name]) for name in self.positional_names
            ),
            keyword_arguments=self.keyword_arguments,
            return_names=self.return_names,
            mutable_argument_names=self.mutable_argument_names,
            consumed_request_fields=request.consumed_fields,
        )


class AttentionCatalogBoundLoweringCheckpoint(unittest.TestCase):
    """Checkpoint 010: lowered calls must match the plan-bound API variant."""

    def test_wrapper_atomically_binds_catalog_operation_and_valid_call(self):
        wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        plan, receipt, selection, callable_binding = plan_authority()
        wrapper.plan(
            FakeFactory("cann", CANN_V2),
            FakeRunAdapter(),
            plan,
            receipt,
            selection,
            callable_binding,
        )

        lowered = wrapper.run("query", ("key", "value"))
        self.assertEqual(wrapper.operation_binding.operation_id, CANN_V2)
        self.assertEqual(wrapper.operation_binding.api_version, "7.3.0")
        self.assertEqual(lowered.operation_id, CANN_V2)

    def test_unknown_keyword_position_return_and_lse_drift_are_rejected(self):
        plan, receipt, selection, callable_binding = plan_authority()

        def planned(adapter):
            wrapper = AttentionOperatorWrapperSession(
                load_packaged_attention_operator_catalog()
            )
            wrapper.plan(
                FakeFactory("cann", CANN_V2),
                adapter,
                plan,
                receipt,
                selection,
                callable_binding,
            )
            return wrapper

        with self.assertRaisesRegex(SchemaError, "unknown keyword"):
            planned(
                FakeRunAdapter(keyword_arguments=(("unknown_option", 1),))
            ).run("query", ("key", "value"))
        with self.assertRaisesRegex(SchemaError, "positional arguments"):
            planned(
                FakeRunAdapter(positional_names=("query", "value", "key"))
            ).run("query", ("key", "value"))
        with self.assertRaisesRegex(SchemaError, "unknown return"):
            planned(FakeRunAdapter(return_names=("output", "debug_tensor"))).run(
                "query", ("key", "value")
            )
        with self.assertRaisesRegex(SchemaError, "enables LSE"):
            planned(
                FakeRunAdapter(
                    keyword_arguments=(("return_softmax_lse", True),),
                    return_names=("output",),
                )
            ).run("query", ("key", "value"))

    def test_mutability_and_failed_replan_preserve_catalog_bound_state(self):
        (
            flash_plan,
            flash_receipt,
            flash_selection,
            flash_callable_binding,
        ) = plan_authority(
            provider_id="flash_attention_npu",
            backend=Backend.ASCENDC_AOT,
            operation_id=FLASH_KV_V3,
        )
        flash_wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        flash_wrapper.plan(
            FakeFactory("flash_attention_npu", FLASH_KV_V3),
            FakeRunAdapter(
                "flash_attention_npu",
                positional_names=("q", "k_cache", "v_cache"),
                keyword_arguments=(("return_softmax_lse", False),),
            ),
            flash_plan,
            flash_receipt,
            flash_selection,
            flash_callable_binding,
        )
        with self.assertRaisesRegex(SchemaError, "mutable arguments"):
            flash_wrapper.run("query", ("key", "value"))

        wrapper = AttentionOperatorWrapperSession(
            load_packaged_attention_operator_catalog()
        )
        plan, receipt, selection, callable_binding = plan_authority()
        adapter = FakeRunAdapter()
        wrapper.plan(
            FakeFactory("cann", CANN_V2),
            adapter,
            plan,
            receipt,
            selection,
            callable_binding,
        )
        active = wrapper.active_plan
        binding = wrapper.operation_binding
        with self.assertRaisesRegex(SchemaError, "unknown Attention operation_id"):
            wrapper.plan(
                FakeFactory(
                    "cann", "torch_npu.npu_fused_infer_attention_score_v2"
                ),
                adapter,
                plan,
                receipt,
                selection,
                callable_binding,
            )
        self.assertIs(wrapper.active_plan, active)
        self.assertIs(wrapper.operation_binding, binding)


if __name__ == "__main__":
    unittest.main()
