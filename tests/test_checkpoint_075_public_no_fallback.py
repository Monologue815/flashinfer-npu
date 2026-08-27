from dataclasses import replace
import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
    BatchAttention,
)
from tests.test_checkpoint_015_runtime_implementation_auto_selection import (
    CANN_V2_OPERATION_ID,
    FakeImplementation,
    authorized_runtime,
    flash,
)


class FailingExecutor:
    def __init__(self):
        self.provider_id = "cann"
        self.operation_id = CANN_V2_OPERATION_ID
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        raise RuntimeError("selected CANN executor failed")


class FailingCannImplementation(FakeImplementation):
    def __init__(self, priority=20):
        super().__init__(
            "cann",
            CANN_V2_OPERATION_ID,
            priority,
        )
        self.executor = FailingExecutor()

    def resolve(self, plan, device):
        self.resolve_calls += 1
        resolved = authorized_runtime(plan, "cann")
        return replace(resolved, executor=self.executor)


def plan(wrapper):
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


class PublicAttentionNoFallbackTests(unittest.TestCase):
    def test_selected_provider_failure_never_reselects_or_retries_alternative(self):
        selected = FailingCannImplementation(priority=20)
        alternative = flash(priority=10)
        implementations = AttentionOperatorRuntimeImplementationRegistry(
            (alternative, selected)
        )
        registry = AttentionOperatorRuntimeResolverRegistry(
            (("npu", implementations),)
        )
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            wrapper = BatchAttention(kv_layout="HND", device="npu:0")

        self.assertIsNone(plan(wrapper))
        selection = wrapper.plan_selection
        self.assertEqual(selection.provider_id, "cann")
        self.assertEqual(selected.resolve_calls, 1)
        self.assertEqual(alternative.resolve_calls, 0)

        for expected_calls in (1, 2):
            with self.subTest(expected_calls=expected_calls):
                with self.assertRaisesRegex(
                    RuntimeError, "selected CANN executor failed"
                ):
                    wrapper.run("query", ("key", "value"))
                self.assertEqual(len(selected.executor.calls), expected_calls)
                self.assertEqual(selected.resolve_calls, 1)
                self.assertEqual(alternative.resolve_calls, 0)
                self.assertEqual(wrapper.plan_selection, selection)
                with self.assertRaises(AttentionStateError):
                    _ = wrapper.last_run_receipt


if __name__ == "__main__":
    unittest.main()
