import unittest

from flashinfer_npu.attention import (
    AttentionStateError,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)
from tests.test_checkpoint_039_public_paged_decode_provider_runtime import (
    FakeNpuWorkspace,
    plan_public_wrapper,
    runtime_registry,
)


def query_workspace(value, **kwargs):
    return value.workspace_size(
        [0, 1],
        [7],
        [64],
        8,
        2,
        128,
        128,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        o_data_type="bfloat16",
        **kwargs,
    )


class PagedDecodeWorkspaceQueryCheckpointTests(unittest.TestCase):
    """Paged decode shares the provider query fork without changing plan state."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.components = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(self.components),
            operation_catalog=self.components["catalog"],
        )
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

    def test_query_before_plan_does_not_publish_runtime_state(self):
        value = self.wrapper()
        initial_workspace = value.workspace_contract

        self.assertEqual(query_workspace(value), (0, 0))

        self.assertFalse(value._operator_runtime.is_planned)
        self.assertIs(value.workspace_contract, initial_workspace)
        self.assertFalse(value.workspace_contract.requirements_known)
        with self.assertRaises(AttentionStateError):
            _ = value.plan_selection
        self.assertEqual(package_attention.calls, [])

    def test_query_after_plan_preserves_selection_workspace_and_executor(self):
        value = self.wrapper()
        plan_public_wrapper(value)
        original_plan = value.plan_state
        original_selection = value.plan_selection
        original_workspace = value.workspace_contract
        original_executor = value._operator_runtime._executor

        self.assertEqual(query_workspace(value, window_left=8), (0, 0))

        self.assertIs(value.plan_state, original_plan)
        self.assertEqual(value.plan_selection, original_selection)
        self.assertIs(value.workspace_contract, original_workspace)
        self.assertIs(value._operator_runtime._executor, original_executor)
        self.assertEqual(
            value.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_query_rejects_unbound_controls_before_provider_resolution(self):
        value = self.wrapper()

        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            query_workspace(value, fixed_split_size=1)

        self.assertEqual(self.components["loader"].resolve_calls, 0)
        self.assertEqual(self.components["authority"].calls, 0)

    def test_failed_query_preserves_existing_plan(self):
        value = self.wrapper()
        plan_public_wrapper(value)
        original_plan = value.plan_state
        original_selection = value.plan_selection
        self.components["authority"].fail = True

        with self.assertRaisesRegex(RuntimeError, "authority failure"):
            query_workspace(value, window_left=8)

        self.assertIs(value.plan_state, original_plan)
        self.assertEqual(value.plan_selection, original_selection)


if __name__ == "__main__":
    unittest.main()
