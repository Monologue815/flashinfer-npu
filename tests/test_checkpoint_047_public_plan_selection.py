import unittest

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
    BatchAttention,
    ReferenceTensor,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from tests.test_checkpoint_019_package_runtime_integration import build_components


class FakeNpuWorkspace:
    def __init__(self, size=2048, *, device="npu:0"):
        self.shape = (size,)
        self.dtype = "uint8"
        self.device = device


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def provider_wrappers():
    return (
        (
            AttentionMode.BATCH_MIXED_PAGED,
            BatchAttention(kv_layout="HND", device="npu:0"),
        ),
        (
            AttentionMode.BATCH_PREFILL_PAGED,
            BatchPrefillWithPagedKVCacheWrapper(
                FakeNpuWorkspace(), kv_layout="HND", backend="auto"
            ),
        ),
        (
            AttentionMode.BATCH_PREFILL_RAGGED,
            BatchPrefillWithRaggedKVCacheWrapper(
                FakeNpuWorkspace(), kv_layout="HND", backend="auto"
            ),
        ),
        (
            AttentionMode.BATCH_DECODE_PAGED,
            BatchDecodeWithPagedKVCacheWrapper(
                FakeNpuWorkspace(), kv_layout="HND", backend="auto"
            ),
        ),
    )


def plan_wrapper(mode, wrapper, *, kv_length=64):
    if mode == AttentionMode.BATCH_MIXED_PAGED:
        return wrapper.plan(
            [0, 1],
            [0, 1],
            [7],
            [kv_length],
            8,
            2,
            128,
            128,
            128,
            causal=True,
        )
    if mode == AttentionMode.BATCH_PREFILL_PAGED:
        return wrapper.plan(
            [0, 1],
            [0, 1],
            [7],
            [kv_length],
            8,
            2,
            128,
            128,
            causal=True,
            q_data_type="bfloat16",
            kv_data_type="bfloat16",
            o_data_type="bfloat16",
        )
    if mode == AttentionMode.BATCH_PREFILL_RAGGED:
        return wrapper.plan(
            [0, 1],
            [0, kv_length],
            8,
            2,
            128,
            causal=True,
            q_data_type="bfloat16",
            kv_data_type="bfloat16",
            o_data_type="bfloat16",
        )
    return wrapper.plan(
        [0, 1],
        [7],
        [kv_length],
        8,
        2,
        128,
        128,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        o_data_type="bfloat16",
    )


class PublicPlanSelectionCheckpointTests(unittest.TestCase):
    """Plan diagnostics reveal identities but never executable provider state."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.components = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(self.components),
            operation_catalog=self.components["catalog"],
        )
        self.registry_generation = (
            attention_operator_runtime_registry_snapshot().generation
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def test_all_batch_modes_publish_provider_selection_after_plan(self):
        for mode, wrapper in provider_wrappers():
            with self.subTest(mode=mode.value):
                with self.assertRaises(AttentionStateError):
                    _ = wrapper.plan_selection
                self.assertIsNone(plan_wrapper(mode, wrapper))

                selection = wrapper.plan_selection
                self.assertEqual(selection.mode, mode)
                self.assertEqual(selection.route, "provider")
                self.assertEqual(selection.backend, "aclnn")
                self.assertEqual(selection.provider_id, "cann")
                self.assertEqual(
                    selection.operation_id,
                    self.components["operation"].operation_id,
                )
                self.assertEqual(
                    selection.registry_generation, self.registry_generation
                )
                self.assertEqual(
                    selection.framework_plan_fingerprint,
                    wrapper.plan_state.fingerprint,
                )
                self.assertEqual(len(selection.fingerprint), 64)
                self.assertFalse(hasattr(selection, "executor"))
                self.assertFalse(hasattr(selection, "callable"))
                self.assertFalse(hasattr(selection, "opaque_state"))

    def test_reference_selection_contains_no_provider_identity(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            ReferenceTensor.zeros((16,), dtype="uint8", device="cpu"),
            kv_layout="HND",
            backend="reference",
        )
        index = lambda values: ReferenceTensor.from_nested(
            values, dtype="int32", device="cpu"
        )
        wrapper.plan(
            index([0, 1]),
            index([7]),
            index([64]),
            8,
            2,
            128,
            128,
            q_data_type="bfloat16",
            kv_data_type="bfloat16",
            o_data_type="bfloat16",
        )

        selection = wrapper.plan_selection
        self.assertEqual(selection.route, "reference")
        self.assertEqual(selection.backend, "reference")
        self.assertIsNone(selection.provider_id)
        self.assertIsNone(selection.operation_id)
        self.assertIsNone(selection.active_plan_fingerprint)
        self.assertIsNone(selection.registry_generation)

    def test_failed_replan_preserves_the_previous_selection(self):
        wrapper = BatchAttention(kv_layout="HND", device="npu:0")
        plan_wrapper(AttentionMode.BATCH_MIXED_PAGED, wrapper)
        original = wrapper.plan_selection
        self.components["authority"].fail = True

        with self.assertRaisesRegex(RuntimeError, "authority failure"):
            plan_wrapper(
                AttentionMode.BATCH_MIXED_PAGED,
                wrapper,
                kv_length=32,
            )

        self.assertEqual(wrapper.plan_selection, original)
        self.assertEqual(wrapper.plan_state.generation, original.plan_generation)


if __name__ == "__main__":
    unittest.main()
