import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionLoweredOperatorCall,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolutionError,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionPlanSpec,
    AttentionResolvedOperatorRuntime,
    AttentionMode,
    BatchAttention,
    CannV2PagedPlanFactory,
    CannV2PagedRunAdapter,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedRunAdapter,
    KVLayout,
    MixedPagedKVMetadata,
    bind_attention_operator_callable,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


def framework_plan(layout=KVLayout.HND, page_size=128):
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=layout,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        o_dtype="bfloat16",
    )
    metadata = MixedPagedKVMetadata(
        qo_indptr=(0, 2, 3),
        kv_indptr=(0, 2, 3),
        kv_indices=(7, 3, 9),
        kv_len_arr=(
            (page_size + min(page_size, page_size // 2)),
            page_size,
        ),
        page_size=page_size,
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


class FakeExecutor:
    def __init__(self, provider_id, operation_id):
        self.provider_id = provider_id
        self.operation_id = operation_id
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        return ("%s-output" % self.provider_id, "%s-lse" % self.provider_id)


def authorized_runtime(plan, provider):
    if provider == "cann":
        operation_id = CANN_V2_OPERATION_ID
        backend = Backend.ACLNN
        factory = CannV2PagedPlanFactory()
        adapter = CannV2PagedRunAdapter()
    else:
        operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID
        backend = Backend.ASCENDC_AOT
        factory = FlashAttentionNpuV3PagedPlanFactory()
        adapter = FlashAttentionNpuV3PagedRunAdapter()
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.auto.selection.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_auto_selection_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-auto-selection-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-auto-selection-kernel",
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
        adapter_version="checkpoint-015-test",
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
            observation_kind="checkpoint-015-test",
        ),
    )
    callable_binding = bind_attention_operator_callable(
        probe, operation, observed
    )
    executor = FakeExecutor(provider, operation_id)
    return AttentionResolvedOperatorRuntime(
        framework_plan_fingerprint=plan.fingerprint,
        factory=factory,
        run_adapter=adapter,
        executor=executor,
        receipt=receipt,
        selection=selection,
        callable_binding=callable_binding,
    )


class FakeImplementation:
    def __init__(
        self,
        provider_id,
        operation_id,
        priority,
        reasons=(),
        resolved_provider=None,
    ):
        self.provider_id = provider_id
        self.operation_id = operation_id
        self.priority = priority
        self.reasons = tuple(reasons)
        self.resolved_provider = resolved_provider or provider_id
        self.explain_calls = 0
        self.resolve_calls = 0

    def rejection_reasons(self, plan, device):
        self.explain_calls += 1
        return self.reasons

    def resolve(self, plan, device):
        self.resolve_calls += 1
        return authorized_runtime(plan, self.resolved_provider)


def cann(priority=10, reasons=(), resolved_provider=None):
    return FakeImplementation(
        "cann",
        CANN_V2_OPERATION_ID,
        priority,
        reasons,
        resolved_provider,
    )


def flash(priority=10, reasons=(), resolved_provider=None):
    return FakeImplementation(
        "flash_attention_npu",
        FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
        priority,
        reasons,
        resolved_provider,
    )


class RuntimeImplementationAutoSelectionCheckpoint(unittest.TestCase):
    """Checkpoint 015: deterministic, explainable automatic implementation choice."""

    def test_highest_priority_unique_candidate_wins_independent_of_registration(self):
        plan = framework_plan()
        low = flash(priority=10)
        high = cann(priority=20)
        resolver = AttentionOperatorRuntimeImplementationRegistry((low, high))

        report = resolver.explain(plan, "npu:0")
        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(report.top_priority, 20)
        self.assertEqual(len(report.fingerprint), 64)
        resolved = resolver.resolve(plan, "npu:0")
        self.assertEqual(resolved.selection.provider_id, "cann")
        self.assertEqual(high.resolve_calls, 1)
        self.assertEqual(low.resolve_calls, 0)

    def test_same_priority_matches_are_rejected_as_ambiguous(self):
        plan = framework_plan()
        first = cann(priority=20)
        second = flash(priority=20)
        resolver = AttentionOperatorRuntimeImplementationRegistry((second, first))

        with self.assertRaisesRegex(
            AttentionOperatorRuntimeResolutionError, "ambiguous"
        ) as captured:
            resolver.resolve(plan, "npu")
        self.assertEqual(
            tuple(item.provider_id for item in captured.exception.report.finalists),
            ("cann", "flash_attention_npu"),
        )
        self.assertEqual(first.resolve_calls, 0)
        self.assertEqual(second.resolve_calls, 0)

    def test_no_match_preserves_every_rejection_reason(self):
        plan = framework_plan()
        first = cann(reasons=("package unavailable",))
        second = flash(reasons=("page size unsupported", "dtype unsupported"))
        resolver = AttentionOperatorRuntimeImplementationRegistry((first, second))

        with self.assertRaisesRegex(
            AttentionOperatorRuntimeResolutionError, "no Attention runtime"
        ) as captured:
            resolver.resolve(plan, "npu")
        report = captured.exception.report
        self.assertEqual(report.accepted, ())
        reasons = {
            item.provider_id: item.reasons for item in report.candidates
        }
        self.assertEqual(reasons["cann"], ("package unavailable",))
        self.assertEqual(
            reasons["flash_attention_npu"],
            ("page size unsupported", "dtype unsupported"),
        )

    def test_selected_implementation_cannot_change_provider_or_operation(self):
        plan = framework_plan()
        drifting = cann(priority=20, resolved_provider="flash_attention_npu")
        resolver = AttentionOperatorRuntimeImplementationRegistry((drifting,))

        with self.assertRaisesRegex(SchemaError, "changed its identity"):
            resolver.resolve(plan, "npu")

    def test_public_batch_attention_uses_the_registry_as_auto_resolver(self):
        selected = cann(priority=20)
        rejected = flash(priority=30, reasons=("HND layout unsupported",))
        implementation_registry = AttentionOperatorRuntimeImplementationRegistry(
            (rejected, selected)
        )
        device_registry = AttentionOperatorRuntimeResolverRegistry(
            (("npu", implementation_registry),)
        )
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            device_registry,
        ):
            wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        self.assertIsNone(
            wrapper.plan(
                (0, 2, 3),
                (0, 2, 3),
                (7, 3, 9),
                (192, 128),
                8,
                2,
                128,
                128,
                128,
                causal=True,
                q_data_type="bfloat16",
                kv_data_type="bfloat16",
            )
        )
        result = wrapper.run("query", ("key", "value"))
        self.assertEqual(result, ("cann-output", "cann-lse"))
        self.assertEqual(selected.resolve_calls, 1)
        self.assertEqual(rejected.resolve_calls, 0)


if __name__ == "__main__":
    unittest.main()
