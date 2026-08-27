import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorCompletionReceipt,
    AttentionOperatorExecutionReceipt,
    AttentionOperatorRunReceipt,
    AttentionOperatorRuntime,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import framework_plan
from tests.test_checkpoint_031_attention_jit_active_plan import (
    package_jit_implementation,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor
from tests.test_checkpoint_062_runtime_completion_publication import (
    run_runtime,
    strict_results,
    strict_runtime,
    valid_result,
)


def digest(character):
    return character * 64


def planned_package_jit_runtime():
    components, implementation, _, _, _ = package_jit_implementation()
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (implementation,)
    )
    registry = AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )
    plan = framework_plan()
    runtime = AttentionOperatorRuntime(
        "npu:0",
        registry,
        components["catalog"],
        mode=plan.spec.mode,
    )
    runtime.plan(plan.spec, plan.metadata)
    return components, runtime


class AtomicRunReceiptTests(unittest.TestCase):
    """One public run joins executor success and result completion evidence."""

    def setUp(self):
        strict_results[:] = []

    def test_success_publishes_one_atomic_execution_completion_receipt(self):
        _, runtime = strict_runtime()
        expected = valid_result(runtime)
        strict_results.append(expected)

        self.assertIs(run_runtime(runtime), expected)

        receipt = runtime.last_run_receipt
        self.assertIs(receipt.execution, runtime._executor.execution_receipt())
        self.assertIs(receipt.completion, runtime.last_completion_receipt)
        self.assertEqual(
            receipt.active_plan_fingerprint,
            runtime.operator_session.active_plan.fingerprint,
        )
        self.assertEqual(receipt.return_names, ("output", "softmax_lse"))
        self.assertIsNone(receipt.jit_runtime_executor)
        self.assertEqual(len(receipt.fingerprint), 64)

    def test_execution_or_completion_failure_publishes_no_atomic_receipt(self):
        _, runtime = strict_runtime()
        strict_results.append(None)
        with self.assertRaisesRegex(SchemaError, "multiple values"):
            run_runtime(runtime)
        with self.assertRaisesRegex(AttentionStateError, "no successful atomic"):
            _ = runtime.last_run_receipt

        plan = runtime.plan_state
        strict_results.append(
            (
                buffer_tensor(
                    "bad-output",
                    plan.expected_output_shape[:-1]
                    + (plan.expected_output_shape[-1] + 1,),
                    plan.spec.o_dtype,
                ),
                buffer_tensor("lse", plan.expected_lse_shape, "float32"),
            )
        )
        with self.assertRaisesRegex(SchemaError, "output result shape"):
            run_runtime(runtime)
        with self.assertRaisesRegex(AttentionStateError, "no successful atomic"):
            _ = runtime.last_run_receipt

    def test_replan_clears_atomic_receipt_and_rebinds_next_run(self):
        _, runtime = strict_runtime()
        strict_results.append(valid_result(runtime, "-first"))
        run_runtime(runtime)
        first = runtime.last_run_receipt
        plan = runtime.plan_state

        runtime.plan(plan.spec, plan.metadata)

        with self.assertRaisesRegex(AttentionStateError, "no successful atomic"):
            _ = runtime.last_run_receipt
        strict_results.append(valid_result(runtime, "-second"))
        run_runtime(runtime)
        self.assertNotEqual(
            runtime.last_run_receipt.active_plan_fingerprint,
            first.active_plan_fingerprint,
        )

    def test_receipt_constructor_rejects_execution_completion_drift(self):
        _, runtime = strict_runtime()
        strict_results.append(valid_result(runtime))
        run_runtime(runtime)
        execution = runtime._executor.execution_receipt()
        completion = runtime.last_completion_receipt

        with self.assertRaisesRegex(SchemaError, "receipts differ"):
            AttentionOperatorRunReceipt(
                execution=execution,
                completion=replace(completion, provider_id="other_provider"),
            )

    def test_jit_runtime_binding_closes_pre_and_post_runtime_executor_objects(self):
        _, runtime = planned_package_jit_runtime()
        jit = runtime.jit_runtime_executor_binding
        pre_runtime_executor = runtime.jit_executor_binding.executor

        self.assertIsNot(pre_runtime_executor, runtime._executor)
        self.assertIs(
            pre_runtime_executor._callable_object,
            runtime._executor._callable_object,
        )
        self.assertIs(jit.executor, runtime._executor)
        self.assertEqual(
            jit.active_plan_fingerprint,
            runtime.operator_session.active_plan.fingerprint,
        )
        jit.validate(
            runtime.jit_executor_binding,
            runtime.operator_session.runtime_binding,
            runtime._executor,
        )

    def test_atomic_receipt_can_include_exact_jit_runtime_binding(self):
        _, runtime = planned_package_jit_runtime()
        jit = runtime.jit_runtime_executor_binding
        active = runtime.operator_session.active_plan
        execution = AttentionOperatorExecutionReceipt(
            runtime_binding_fingerprint=jit.runtime_binding_fingerprint,
            active_plan_fingerprint=active.fingerprint,
            provider_id=jit.provider_id,
            operation_id=jit.operation_id,
            return_names=("output",),
        )
        completion = AttentionOperatorCompletionReceipt(
            active_plan_fingerprint=active.fingerprint,
            framework_plan_fingerprint=runtime.plan_state.fingerprint,
            operation_fingerprint=(
                runtime.operator_session.callable_binding.operation_fingerprint
            ),
            access_policy_fingerprint=digest("a"),
            provider_id=jit.provider_id,
            operation_id=jit.operation_id,
            expected_device="npu:0",
            return_names=("output",),
            input_view_fingerprints=(("query", digest("b")),),
            result_view_fingerprints=(("output", digest("c")),),
        )

        receipt = AttentionOperatorRunReceipt(
            execution=execution,
            completion=completion,
            jit_runtime_executor=jit,
        )

        self.assertIs(receipt.jit_runtime_executor, jit)
        self.assertEqual(
            receipt.to_dict()["jit_runtime_executor_binding_fingerprint"],
            jit.fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
