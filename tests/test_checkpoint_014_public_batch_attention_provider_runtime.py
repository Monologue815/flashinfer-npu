import inspect
import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionLoweredOperatorCall,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionResolvedOperatorRuntime,
    BatchAttention,
    CannV2PagedPlanFactory,
    CannV2PagedRunAdapter,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedRunAdapter,
    bind_attention_operator_callable,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, DispatchError


def hash_value(character):
    return character * 64


class FakeTorchDType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "torch.%s" % self.name


class FakePlanTensor:
    dtype = FakeTorchDType("int32")

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class FakeExecutor:
    def __init__(self, provider_id, operation_id, generation):
        self.provider_id = provider_id
        self.operation_id = operation_id
        self.generation = generation
        self.calls = []

    def execute(self, call):
        if not isinstance(call, AttentionLoweredOperatorCall):
            raise TypeError("fake executor requires a lowered call")
        self.calls.append(call)
        return (
            "public-output-generation-%d" % self.generation,
            "public-lse-generation-%d" % self.generation,
        )


class FakeAutoResolver:
    def __init__(self):
        self.resolve_calls = []
        self.executors = []
        self.fail = False

    def resolve(self, plan, device):
        self.resolve_calls.append((plan, device))
        if self.fail:
            raise RuntimeError("fake automatic resolution failure")
        if plan.spec.kv_layout.value == "HND":
            provider = "cann"
            operation_id = CANN_V2_OPERATION_ID
            backend = Backend.ACLNN
            factory = CannV2PagedPlanFactory()
            adapter = CannV2PagedRunAdapter()
        else:
            provider = "flash_attention_npu"
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
            profile_id="fake.public.provider.profile.v1",
            profile_fingerprint=hash_value("2"),
            rule_id="fake_public_provider_rule_v1",
            environment_fingerprint=hash_value("3"),
            evidence_id="fake-public-provider-evidence",
            evidence_result_digest=hash_value("4"),
            kernel_id="fake-public-provider-kernel",
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
            adapter_version="checkpoint-014-test",
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
                observation_kind="checkpoint-014-test",
            ),
        )
        callable_binding = bind_attention_operator_callable(
            probe, operation, observed
        )
        executor = FakeExecutor(provider, operation_id, plan.generation)
        self.executors.append(executor)
        return AttentionResolvedOperatorRuntime(
            framework_plan_fingerprint=plan.fingerprint,
            factory=factory,
            run_adapter=adapter,
            executor=executor,
            receipt=receipt,
            selection=selection,
            callable_binding=callable_binding,
        )


def plan_public_wrapper(wrapper, *, page_size, kv_lengths, indices=(7, 3, 9)):
    return wrapper.plan(
        FakePlanTensor((0, 2, 3)),
        FakePlanTensor((0, 2, 3)),
        FakePlanTensor(indices),
        FakePlanTensor(kv_lengths),
        8,
        2,
        128,
        128,
        page_size,
        causal=True,
        q_data_type=FakeTorchDType("bfloat16"),
        kv_data_type=FakeTorchDType("bfloat16"),
    )


class PublicBatchAttentionProviderRuntimeCheckpoint(unittest.TestCase):
    """Checkpoint 014: public wrapper owns automatic provider runtime state."""

    def test_public_signature_has_no_provider_plan_factory_or_adapter(self):
        self.assertEqual(
            list(inspect.signature(BatchAttention).parameters),
            ["kv_layout", "device"],
        )
        self.assertNotIn("provider", inspect.signature(BatchAttention.plan).parameters)
        self.assertNotIn("plan", inspect.signature(BatchAttention.run).parameters)
        self.assertNotIn("adapter", inspect.signature(BatchAttention.run).parameters)

    def test_hnd_public_plan_auto_resolves_once_and_run_returns_public_result(self):
        resolver = FakeAutoResolver()
        registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        self.assertIsNone(
            plan_public_wrapper(wrapper, page_size=128, kv_lengths=(192, 128))
        )
        self.assertEqual(wrapper.plan_state.generation, 1)
        self.assertEqual(len(resolver.resolve_calls), 1)
        self.assertEqual(resolver.resolve_calls[0][1], "npu:0")

        first = wrapper.run("query-1", ("key-1", "value-1"))
        second = wrapper.run("query-2", ("key-2", "value-2"))
        self.assertEqual(
            first,
            ("public-output-generation-1", "public-lse-generation-1"),
        )
        self.assertEqual(second, first)
        self.assertEqual(len(resolver.resolve_calls), 1)
        self.assertEqual(len(resolver.executors[0].calls), 2)
        self.assertFalse(hasattr(wrapper, "operation_binding"))

    def test_nhd_public_plan_selects_flash_attention_npu_v3(self):
        resolver = FakeAutoResolver()
        registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            wrapper = BatchAttention(kv_layout="NHD", device="npu")
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        result = wrapper.run("query", ("key", "value"))

        self.assertEqual(
            resolver.executors[0].operation_id,
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
        )
        self.assertEqual(
            result, ("public-output-generation-1", "public-lse-generation-1")
        )

    def test_failed_replan_preserves_previous_framework_and_runtime(self):
        resolver = FakeAutoResolver()
        registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            wrapper = BatchAttention(kv_layout="HND", device="npu")
        plan_public_wrapper(wrapper, page_size=128, kv_lengths=(192, 128))
        first_plan = wrapper.plan_state
        first_executor = resolver.executors[0]

        resolver.fail = True
        with self.assertRaisesRegex(RuntimeError, "automatic resolution failure"):
            plan_public_wrapper(
                wrapper,
                page_size=128,
                kv_lengths=(256, 128),
                indices=(7, 3, 9),
            )
        self.assertIs(wrapper.plan_state, first_plan)
        self.assertEqual(wrapper.plan_state.generation, 1)
        result = wrapper.run("query", ("key", "value"))
        self.assertEqual(
            result, ("public-output-generation-1", "public-lse-generation-1")
        )
        self.assertEqual(len(first_executor.calls), 1)

    def test_missing_device_resolver_fails_at_plan_without_partial_state(self):
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
        ):
            wrapper = BatchAttention(kv_layout="NHD", device="npu")
        with self.assertRaisesRegex(DispatchError, "no Attention operator runtime"):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        with self.assertRaisesRegex(Exception, "has not been planned"):
            _ = wrapper.plan_state


if __name__ == "__main__":
    unittest.main()
