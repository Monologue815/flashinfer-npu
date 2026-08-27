import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorCompletionValidator,
    AttentionTensorAccessPolicy,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    quantized_input,
    query_input,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor


def completion_fixture(*, access_policy=None):
    values = bootstrap_components()
    plan, session = active_session(values)
    lowered = session.run(
        query_input(plan), quantized_input(plan.spec.kv_quant_spec)
    )
    validator = AttentionOperatorCompletionValidator(
        values["operation"],
        session.active_plan,
        values["tensor_metadata_inspector"],
        access_policy or AttentionTensorAccessPolicy(),
        "npu:0",
    )
    return values, plan, session, lowered, validator


class ProviderResultTensorContractTests(unittest.TestCase):
    """Provider-produced public tensors remain bound to the active plan."""

    def test_output_and_lse_metadata_produce_a_plan_bound_receipt(self):
        _, plan, session, lowered, validator = completion_fixture()
        output = buffer_tensor(
            "completion-output", plan.expected_output_shape, plan.spec.o_dtype
        )
        lse = buffer_tensor(
            "completion-lse", plan.expected_lse_shape, "float32"
        )

        receipt = validator.validate(lowered, (output, lse))

        self.assertEqual(receipt.return_names, ("output", "softmax_lse"))
        self.assertEqual(receipt.active_plan_fingerprint, session.active_plan.fingerprint)
        self.assertEqual(receipt.framework_plan_fingerprint, plan.fingerprint)
        self.assertEqual(receipt.expected_device, "npu:0")
        self.assertEqual(len(receipt.result_view_fingerprints), 2)
        self.assertEqual(len(receipt.fingerprint), 64)

    def test_result_shape_dtype_device_and_writable_fail_closed(self):
        _, plan, _, lowered, validator = completion_fixture()
        valid_lse = buffer_tensor(
            "valid-lse", plan.expected_lse_shape, "float32"
        )
        cases = (
            (
                buffer_tensor(
                    "bad-shape",
                    plan.expected_output_shape[:-1] + (plan.spec.head_dim_vo + 1,),
                    plan.spec.o_dtype,
                ),
                "output result shape",
            ),
            (
                buffer_tensor(
                    "bad-dtype", plan.expected_output_shape, "float16"
                ),
                "output result dtype",
            ),
            (
                buffer_tensor(
                    "bad-device",
                    plan.expected_output_shape,
                    plan.spec.o_dtype,
                    device="npu:1",
                ),
                "output result device",
            ),
            (
                buffer_tensor(
                    "read-only",
                    plan.expected_output_shape,
                    plan.spec.o_dtype,
                    writable=False,
                ),
                "output result must be writable",
            ),
        )
        for output, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    validator.validate(lowered, (output, valid_lse))

    def test_lse_is_exact_fp32_plan_shape_on_provider_device(self):
        _, plan, _, lowered, validator = completion_fixture()
        output = buffer_tensor(
            "valid-output", plan.expected_output_shape, plan.spec.o_dtype
        )
        cases = (
            (
                buffer_tensor(
                    "bad-lse-shape", plan.expected_lse_shape + (1,), "float32"
                ),
                "softmax_lse result shape",
            ),
            (
                buffer_tensor(
                    "bad-lse-dtype", plan.expected_lse_shape, "float16"
                ),
                "softmax_lse result dtype",
            ),
            (
                buffer_tensor(
                    "bad-lse-device",
                    plan.expected_lse_shape,
                    "float32",
                    device="npu:1",
                ),
                "softmax_lse result device",
            ),
        )
        for lse, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    validator.validate(lowered, (output, lse))

    def test_alignment_contiguity_and_output_lse_alias_are_rejected(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_output=True, required_alignment=16
        )
        _, plan, _, lowered, validator = completion_fixture(access_policy=policy)
        lse = buffer_tensor("lse", plan.expected_lse_shape, "float32")
        non_contiguous = buffer_tensor(
            "strided",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            strides=(
                plan.expected_output_shape[1] * plan.expected_output_shape[2] + 1,
                plan.expected_output_shape[2],
                1,
            ),
        )
        with self.assertRaisesRegex(SchemaError, "output result must be contiguous"):
            validator.validate(lowered, (non_contiguous, lse))

        under_aligned = buffer_tensor(
            "under-aligned",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            alignment=8,
        )
        with self.assertRaisesRegex(SchemaError, "output must be 16-byte aligned"):
            validator.validate(lowered, (under_aligned, lse))

        shared_storage = "checkpoint-061:shared-result-storage"
        output = buffer_tensor(
            "aliased-output",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=shared_storage,
        )
        aliased_lse = buffer_tensor(
            "aliased-lse",
            plan.expected_lse_shape,
            "float32",
            storage_id=shared_storage,
        )
        with self.assertRaisesRegex(SchemaError, "results cannot alias"):
            validator.validate(lowered, (output, aliased_lse))

    def test_return_arity_and_active_plan_identity_are_rechecked(self):
        _, plan, _, lowered, validator = completion_fixture()
        output = buffer_tensor(
            "output", plan.expected_output_shape, plan.spec.o_dtype
        )
        with self.assertRaisesRegex(SchemaError, "return arity"):
            validator.validate(lowered, output)
        with self.assertRaisesRegex(SchemaError, "return arity"):
            validator.validate(lowered, (output,))
        with self.assertRaisesRegex(SchemaError, "differs from active plan"):
            validator.validate(
                replace(lowered, active_plan_fingerprint="f" * 64),
                (output, buffer_tensor("lse", plan.expected_lse_shape, "float32")),
            )


if __name__ == "__main__":
    unittest.main()
