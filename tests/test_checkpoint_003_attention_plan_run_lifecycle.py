import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionStateError,
    BatchAttention,
    ReferenceTensor,
)
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper


def tensor(value, dtype="float32"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device="cpu")


def index(values):
    return tensor(values, dtype="int32")


class AttentionPlanRunLifecycleCheckpoint(unittest.TestCase):
    """Checkpoint 003: freeze the FlashInfer-compatible public lifecycle only."""

    def test_public_surface_keeps_plan_internal_to_the_wrapper(self):
        self.assertNotIn("plan", inspect.signature(BatchAttention.run).parameters)
        self.assertNotIn("plan", inspect.signature(
            BatchPrefillWithPagedKVCacheWrapper.run
        ).parameters)
        self.assertEqual(
            inspect.signature(
                BatchPrefillWithPagedKVCacheWrapper
            ).parameters["backend"].default,
            "auto",
        )
        # Upstream holistic BatchAttention does not expose a backend argument.
        self.assertNotIn("backend", inspect.signature(BatchAttention).parameters)

    def test_plan_returns_none_and_run_consumes_wrapper_owned_state(self):
        wrapper = BatchAttention(device="cpu")
        with self.assertRaisesRegex(AttentionStateError, "has not been planned"):
            wrapper.run(
                tensor([[[0.0]]]),
                (tensor([[[[0.0]]]]), tensor([[[[1.0]]]])),
            )

        result = wrapper.plan(
            index([0, 1]),
            index([0, 1]),
            index([0]),
            index([1]),
            1,
            1,
            1,
            1,
            1,
            q_data_type="float32",
            kv_data_type="float32",
        )
        self.assertIsNone(result)
        self.assertEqual(wrapper.plan_state.generation, 1)

        output, lse = wrapper.run(
            tensor([[[0.0]]]),
            (tensor([[[[0.0]]]]), tensor([[[[7.0]]]])),
        )
        self.assertEqual(output.data, (7.0,))
        self.assertEqual(lse.data, (0.0,))

    def test_replan_atomically_replaces_the_internal_plan(self):
        wrapper = BatchAttention(device="cpu")
        args = (
            index([0, 1]),
            index([0, 1]),
            index([0]),
            index([1]),
            1,
            1,
            1,
            1,
            1,
        )
        self.assertIsNone(
            wrapper.plan(
                *args,
                causal=False,
                q_data_type="float32",
                kv_data_type="float32",
            )
        )
        first = wrapper.plan_state

        self.assertIsNone(
            wrapper.plan(
                *args,
                causal=True,
                q_data_type="float32",
                kv_data_type="float32",
            )
        )
        second = wrapper.plan_state

        self.assertEqual(first.generation, 1)
        self.assertEqual(second.generation, 2)
        self.assertFalse(first.spec.causal)
        self.assertTrue(second.spec.causal)


if __name__ == "__main__":
    unittest.main()
