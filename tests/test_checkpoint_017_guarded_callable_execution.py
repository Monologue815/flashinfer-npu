import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionInjectedCallableExecutor,
    AttentionLoweredOperatorCall,
    AttentionObservedOperatorCallable,
    AttentionOperatorOperationSpec,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorRuntimeBinding,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionResolvedOperatorRuntime,
    AttentionMode,
    BatchAttention,
    CannV2PagedPlanFactory,
    CannV2PagedRunAdapter,
    bind_attention_operator_callable,
    load_packaged_attention_operator_catalog,
    observe_python_callable_signature,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


def fake_attention(query, key, value, *, scale=1.0, return_softmax_lse=False):
    fake_attention.calls.append(
        (query, key, value, scale, return_softmax_lse)
    )
    return ["output:%s" % query, "lse:%s" % scale]


fake_attention.calls = []


def small_operation():
    return AttentionOperatorOperationSpec(
        operation_id="cann.fake_attention@v1",
        provider_id="cann",
        package_name="fake_attention_package",
        callable_path="fake_attention_package.fake_attention",
        api_version="v1",
        candidate_modes=(AttentionMode.BATCH_MIXED_PAGED,),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=("scale", "return_softmax_lse"),
        return_names=("output", "softmax_lse"),
        lse_control_argument="return_softmax_lse",
        source_url="https://example.com/fake-attention-v1",
    )


def callable_binding(operation, callable_object=fake_attention):
    probe = AttentionOperatorProviderProbe(
        provider_id="cann",
        adapter_version="checkpoint-017-test",
        available=True,
        package_versions=((operation.package_name, "1.0.0"),),
    )
    observation = AttentionObservedOperatorCallable(
        provider_id="cann",
        package_name=operation.package_name,
        package_version="1.0.0",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=observe_python_callable_signature(callable_object),
    )
    return probe, bind_attention_operator_callable(probe, operation, observation)


def runtime_binding(operation, probe, binding, active_plan_fingerprint=None):
    return AttentionOperatorRuntimeBinding(
        active_plan_fingerprint=active_plan_fingerprint or hash_value("a"),
        provider_id="cann",
        provider_probe_fingerprint=probe.fingerprint,
        operation_binding_fingerprint=hash_value("b"),
        callable_binding_fingerprint=binding.fingerprint,
        operation_id=operation.operation_id,
        operation_fingerprint=operation.fingerprint,
        observation_fingerprint=binding.observation_fingerprint,
    )


def lowered_call(operation, active_plan_fingerprint=None):
    return AttentionLoweredOperatorCall(
        provider_id="cann",
        operation_id=operation.operation_id,
        active_plan_fingerprint=active_plan_fingerprint or hash_value("a"),
        positional_arguments=(
            ("query", "q"),
            ("key", "k"),
            ("value", "v"),
        ),
        keyword_arguments=(("scale", 0.125), ("return_softmax_lse", True)),
        return_names=("output", "softmax_lse"),
    )


def fake_cann_callable(
    query,
    key,
    value,
    *,
    query_rope=None,
    key_rope=None,
    pse_shift=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    block_table=None,
    dequant_scale_query=None,
    dequant_scale_key=None,
    dequant_offset_key=None,
    dequant_scale_value=None,
    dequant_offset_value=None,
    dequant_scale_key_rope=None,
    quant_scale_out=None,
    quant_offset_out=None,
    learnable_sink=None,
    num_query_heads=1,
    num_key_value_heads=1,
    softmax_scale=1.0,
    pre_tokens=0,
    next_tokens=0,
    input_layout="TND",
    sparse_mode=0,
    block_size=128,
    query_quant_mode=0,
    key_quant_mode=0,
    value_quant_mode=0,
    inner_precise=0,
    return_softmax_lse=False,
    query_dtype=None,
    key_dtype=None,
    value_dtype=None,
    query_rope_dtype=None,
    key_rope_dtype=None,
    key_shared_prefix_dtype=None,
    value_shared_prefix_dtype=None,
    dequant_scale_query_dtype=None,
    dequant_scale_key_dtype=None,
    dequant_scale_value_dtype=None,
    dequant_scale_key_rope_dtype=None,
):
    fake_cann_callable.calls.append(
        {
            "positional": (query, key, value),
            "return_softmax_lse": return_softmax_lse,
            "block_table": block_table,
        }
    )
    return ("public-cann-output", "public-cann-lse")


fake_cann_callable.calls = []


class FakeGuardedCannResolver:
    def __init__(self):
        self.candidate_executors = []

    def resolve(self, plan, device):
        operation = load_packaged_attention_operator_catalog().get(
            CANN_V2_OPERATION_ID
        )
        receipt = AttentionDispatchReceipt(
            mode=plan.spec.mode,
            plan_fingerprint=plan.fingerprint,
            admission_fingerprint=plan.admission_fingerprint,
            workload_fingerprint=plan.workload.fingerprint,
            numerics_policy_fingerprint=hash_value("1"),
            profile_id="checkpoint.017.profile.v1",
            profile_fingerprint=hash_value("2"),
            rule_id="checkpoint_017_rule_v1",
            environment_fingerprint=hash_value("3"),
            evidence_id="checkpoint-017-evidence",
            evidence_result_digest=hash_value("4"),
            kernel_id="checkpoint-017-fake-kernel",
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
        probe = AttentionOperatorProviderProbe(
            provider_id="cann",
            adapter_version="checkpoint-017-test",
            available=True,
            package_versions=((operation.package_name, "test-package"),),
        )
        selection = AttentionOperatorProviderSelection(
            provider_id="cann",
            provider_probe_fingerprint=probe.fingerprint,
            provider_record_fingerprint=hash_value("9"),
            dispatch_receipt_fingerprint=receipt.fingerprint,
            profile_id=receipt.profile_id,
            profile_fingerprint=receipt.profile_fingerprint,
            backend=receipt.backend,
        )
        observation = AttentionObservedOperatorCallable(
            provider_id="cann",
            package_name=operation.package_name,
            package_version="test-package",
            callable_path=operation.callable_path,
            api_version=operation.api_version,
            available=True,
            signature=observe_python_callable_signature(fake_cann_callable),
        )
        binding = bind_attention_operator_callable(probe, operation, observation)
        executor = AttentionInjectedCallableExecutor(
            operation, binding, fake_cann_callable
        )
        self.candidate_executors.append(executor)
        return AttentionResolvedOperatorRuntime(
            framework_plan_fingerprint=plan.fingerprint,
            factory=CannV2PagedPlanFactory(),
            run_adapter=CannV2PagedRunAdapter(),
            executor=executor,
            receipt=receipt,
            selection=selection,
            callable_binding=binding,
        )


def plan_public_wrapper(wrapper):
    return wrapper.plan(
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


class GuardedCallableExecutionCheckpoint(unittest.TestCase):
    """Checkpoint 017: exact callable execution requires final plan authority."""

    def setUp(self):
        fake_attention.calls[:] = []
        fake_cann_callable.calls[:] = []

    def test_exact_invocation_normalizes_multiple_returns_and_records_receipt(self):
        operation = small_operation()
        probe, binding = callable_binding(operation)
        candidate = AttentionInjectedCallableExecutor(
            operation, binding, fake_attention
        )
        self.assertFalse(candidate.is_runtime_bound)
        with self.assertRaisesRegex(RuntimeError, "not runtime-bound"):
            candidate.execute(lowered_call(operation))

        executor = candidate.bind_runtime(
            runtime_binding(operation, probe, binding)
        )
        self.assertIsNot(executor, candidate)
        self.assertEqual(
            executor.execute(lowered_call(operation)),
            ("output:q", "lse:0.125"),
        )
        self.assertEqual(fake_attention.calls, [("q", "k", "v", 0.125, True)])
        receipt = executor.last_execution_receipt
        self.assertEqual(receipt.active_plan_fingerprint, hash_value("a"))
        self.assertEqual(receipt.return_names, ("output", "softmax_lse"))
        self.assertEqual(len(receipt.fingerprint), 64)

    def test_return_arity_drift_is_rejected_without_success_receipt(self):
        def wrong_return(
            query, key, value, *, scale=1.0, return_softmax_lse=False
        ):
            return "only-output"

        operation = small_operation()
        probe, binding = callable_binding(operation, wrong_return)
        executor = AttentionInjectedCallableExecutor(
            operation, binding, wrong_return
        ).bind_runtime(runtime_binding(operation, probe, binding))

        with self.assertRaisesRegex(SchemaError, "multiple values"):
            executor.execute(lowered_call(operation))
        with self.assertRaisesRegex(RuntimeError, "not executed successfully"):
            _ = executor.last_execution_receipt

    def test_callable_exception_propagates_and_does_not_issue_receipt(self):
        def failing(query, key, value, *, scale=1.0, return_softmax_lse=False):
            raise ValueError("injected callable failure")

        operation = small_operation()
        probe, binding = callable_binding(operation, failing)
        executor = AttentionInjectedCallableExecutor(
            operation, binding, failing
        ).bind_runtime(runtime_binding(operation, probe, binding))

        with self.assertRaisesRegex(ValueError, "injected callable failure"):
            executor.execute(lowered_call(operation))
        with self.assertRaisesRegex(RuntimeError, "not executed successfully"):
            _ = executor.last_execution_receipt

    def test_stale_plan_identity_is_rejected_before_callable_invocation(self):
        operation = small_operation()
        probe, binding = callable_binding(operation)
        executor = AttentionInjectedCallableExecutor(
            operation, binding, fake_attention
        ).bind_runtime(runtime_binding(operation, probe, binding))

        with self.assertRaisesRegex(SchemaError, "not authorized"):
            executor.execute(lowered_call(operation, hash_value("c")))
        self.assertEqual(fake_attention.calls, [])

    def test_public_plan_binds_candidate_and_run_executes_injected_callable(self):
        resolver = FakeGuardedCannResolver()
        registry = AttentionOperatorRuntimeResolverRegistry((('npu', resolver),))
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        self.assertIsNone(plan_public_wrapper(wrapper))
        self.assertFalse(resolver.candidate_executors[0].is_runtime_bound)

        result = wrapper.run("q", ("k-cache", "v-cache"))
        self.assertEqual(result, ("public-cann-output", "public-cann-lse"))
        self.assertEqual(
            fake_cann_callable.calls[0]["positional"],
            ("q", "k-cache", "v-cache"),
        )
        self.assertTrue(fake_cann_callable.calls[0]["return_softmax_lse"])


if __name__ == "__main__":
    unittest.main()
