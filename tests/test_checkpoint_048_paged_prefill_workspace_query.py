import unittest

from flashinfer_npu.attention import (
    EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    package_attention,
)


class FakeNpuWorkspace:
    def __init__(self, size=2048):
        self.shape = (size,)
        self.dtype = "uint8"
        self.device = "npu:0"


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    return AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )


def wrapper():
    return BatchPrefillWithPagedKVCacheWrapper(
        FakeNpuWorkspace(), kv_layout="HND", backend="auto"
    )


def query_workspace(value, *, kv_length=64):
    return value.workspace_size(
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


def plan(value, *, kv_length=64):
    return value.plan(
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


class PagedPrefillWorkspaceQueryCheckpointTests(unittest.TestCase):
    """Provider size queries plan on a fork and never publish wrapper state."""

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

    def test_query_before_plan_returns_exact_caller_workspace_requirement(self):
        value = wrapper()
        initial_workspace = value.workspace_contract

        self.assertEqual(query_workspace(value), (0, 0))

        self.assertIs(value.workspace_contract, initial_workspace)
        self.assertFalse(value.workspace_contract.requirements_known)
        self.assertFalse(value._operator_runtime.is_planned)
        with self.assertRaises(AttentionStateError):
            _ = value.plan_selection
        self.assertEqual(self.components["loader"].resolve_calls, 1)
        self.assertEqual(self.components["authority"].calls, 1)
        self.assertEqual(package_attention.calls, [])

    def test_query_after_plan_preserves_every_published_runtime_object(self):
        value = wrapper()
        self.assertIsNone(plan(value))
        original_plan = value.plan_state
        original_selection = value.plan_selection
        original_workspace = value.workspace_contract
        original_session = value._operator_runtime.operator_session
        original_executor = value._operator_runtime._executor

        self.assertEqual(query_workspace(value, kv_length=32), (0, 0))

        self.assertIs(value.plan_state, original_plan)
        self.assertEqual(value.plan_selection, original_selection)
        self.assertIs(value.workspace_contract, original_workspace)
        self.assertIs(value._operator_runtime.operator_session, original_session)
        self.assertIs(value._operator_runtime._executor, original_executor)
        self.assertEqual(
            value.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_query_uses_the_wrapper_frozen_registry_snapshot(self):
        value = wrapper()
        frozen_generation = value._operator_runtime_registry_snapshot.generation
        install_attention_operator_runtime_resolvers(
            EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
            operation_catalog=self.components["catalog"],
        )
        self.assertNotEqual(
            attention_operator_runtime_registry_snapshot().generation,
            frozen_generation,
        )

        self.assertEqual(query_workspace(value), (0, 0))
        self.assertEqual(self.components["loader"].resolve_calls, 1)

    def test_failed_query_preserves_an_existing_plan(self):
        value = wrapper()
        plan(value)
        original_plan = value.plan_state
        original_selection = value.plan_selection
        original_workspace = value.workspace_contract
        self.components["authority"].fail = True

        with self.assertRaisesRegex(RuntimeError, "authority failure"):
            query_workspace(value, kv_length=32)

        self.assertIs(value.plan_state, original_plan)
        self.assertEqual(value.plan_selection, original_selection)
        self.assertIs(value.workspace_contract, original_workspace)


if __name__ == "__main__":
    unittest.main()
