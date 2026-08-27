import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionInjectedCallableExecutor,
    AttentionLoweredOperatorCall,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_017_guarded_callable_execution import (
    callable_binding,
    hash_value,
    runtime_binding,
    small_operation,
)


def completion_operation():
    operation = small_operation()
    return replace(
        operation,
        callable_path="fake_attention_package.attention_with_buffers",
        keyword_arguments=operation.keyword_arguments + ("out", "lse"),
        mutable_arguments=("out", "lse"),
        output_buffer_argument="out",
        lse_buffer_argument="lse",
    )


def completion_call(operation, *, out=None, lse=None, return_lse=True, mutable=None):
    keywords = [
        ("scale", 0.125),
        ("return_softmax_lse", return_lse),
        ("out", out),
    ]
    if return_lse or lse is not None:
        keywords.append(("lse", lse))
    if mutable is None:
        mutable = tuple(
            name
            for name, value in (("out", out), ("lse", lse))
            if value is not None
        )
    return AttentionLoweredOperatorCall(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        active_plan_fingerprint=hash_value("a"),
        positional_arguments=(("query", "q"), ("key", "k"), ("value", "v")),
        keyword_arguments=tuple(keywords),
        return_names=(
            ("output", "softmax_lse") if return_lse else ("output",)
        ),
        mutable_argument_names=mutable,
    )


def bound_executor(callable_object):
    operation = completion_operation()
    probe, binding = callable_binding(operation, callable_object)
    executor = AttentionInjectedCallableExecutor(
        operation, binding, callable_object
    ).bind_runtime(runtime_binding(operation, probe, binding))
    return operation, executor


class ProviderCompletionContractTests(unittest.TestCase):
    """Provider completion preserves FlashInfer return arity and buffer identity."""

    def test_caller_owned_output_and_lse_are_returned_by_identity(self):
        output = object()
        lse_result = object()

        def attention(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            return (out, lse) if return_softmax_lse else out

        operation, executor = bound_executor(attention)
        result = executor.execute(
            completion_call(operation, out=output, lse=lse_result)
        )

        self.assertIs(result[0], output)
        self.assertIs(result[1], lse_result)
        self.assertEqual(
            executor.last_execution_receipt.caller_owned_return_names,
            ("output", "softmax_lse"),
        )

    def test_output_only_is_one_value_and_preserves_caller_buffer(self):
        output = object()

        def attention(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            return out

        operation, executor = bound_executor(attention)
        result = executor.execute(
            completion_call(operation, out=output, return_lse=False)
        )

        self.assertIs(result, output)
        self.assertEqual(
            executor.last_execution_receipt.caller_owned_return_names,
            ("output",),
        )

    def test_allocated_provider_results_are_allowed_without_caller_buffers(self):
        output = object()
        lse_result = object()

        def attention(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            return output, lse_result

        operation, executor = bound_executor(attention)
        result = executor.execute(completion_call(operation))

        self.assertEqual(result, (output, lse_result))
        self.assertEqual(
            executor.last_execution_receipt.caller_owned_return_names, ()
        )

    def test_different_return_object_cannot_replace_caller_output(self):
        def attention(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            return object(), lse

        operation, executor = bound_executor(attention)
        with self.assertRaisesRegex(SchemaError, "caller-owned output buffer"):
            executor.execute(
                completion_call(operation, out=object(), lse=object())
            )
        with self.assertRaisesRegex(RuntimeError, "not executed successfully"):
            _ = executor.last_execution_receipt

    def test_missing_or_ambiguous_public_results_are_rejected(self):
        def missing(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            return out, None

        operation, executor = bound_executor(missing)
        with self.assertRaisesRegex(SchemaError, "returned no value"):
            executor.execute(
                completion_call(operation, out=object(), lse=None)
            )

        def ambiguous(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            return (out,)

        operation, executor = bound_executor(ambiguous)
        with self.assertRaisesRegex(SchemaError, "return arity"):
            executor.execute(
                completion_call(operation, out=object(), return_lse=False)
            )

    def test_caller_buffer_must_be_lowered_as_mutable_before_invocation(self):
        calls = []

        def attention(
            query,
            key,
            value,
            *,
            scale=1.0,
            return_softmax_lse=False,
            out=None,
            lse=None,
        ):
            calls.append(True)
            return out

        operation, executor = bound_executor(attention)
        with self.assertRaisesRegex(SchemaError, "not lowered as mutable"):
            executor.execute(
                completion_call(
                    operation,
                    out=object(),
                    return_lse=False,
                    mutable=(),
                )
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
