import unittest

from flashinfer_npu.attention import (
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.frontend import framework_index_values
from flashinfer_npu.attention.reference import ReferenceTensor
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_043_provider_workspace_reset import (
    FakeNpuWorkspace, plan_wrapper, runtime_registry,
)
from tests.test_checkpoint_116_metadata_integer_contract import IndexValue, IntOnly


class IndexTensor:
    dtype = "int32"

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


class PlanIntegerAdmissionCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.components = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(self.components),
            operation_catalog=self.components["catalog"],
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def test_reader_accepts_integer_protocol_and_reference_storage(self):
        for value in ([IndexValue(0), IndexValue(7)],
                      IndexTensor([IndexValue(0), IndexValue(7)]),
                      ReferenceTensor((2,), (0, 7), "int32")):
            self.assertEqual(framework_index_values(value, "indices"), (0, 7))

    def test_reader_rejects_non_integer_sequences_and_tensor_payloads(self):
        for value in (True, 1.0, 1.5, "1", float("inf"), float("nan"), IntOnly()):
            for wrap in (list, IndexTensor):
                with self.subTest(value=value, wrapper=wrap):
                    with self.assertRaisesRegex(SchemaError, "indices values must be integers"):
                        framework_index_values(wrap([value]), "indices")

    def test_public_wrappers_reject_before_provider_resolution(self):
        fixtures = (
            (BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace()),
             [[0, 1], [7], [64], 8, 2, 128, 128], 1),
            (BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace()),
             [[0, 1], [0, 1], [7], [64], 8, 2, 128, 128], 2),
            (BatchPrefillWithRaggedKVCacheWrapper(FakeNpuWorkspace()),
             [[0, 1], [0, 64], 8, 2, 128], 1),
            (BatchAttention(device="npu:0"),
             [[0, 1], [0, 1], [7], [64], 8, 2, 128, 128, 128], 2),
        )
        for wrapper, args, slot in fixtures:
            for value in (float(args[slot][-1]), float("inf")):
                with self.subTest(wrapper=type(wrapper).__name__, value=value):
                    invalid = list(args)
                    invalid[slot] = args[slot][:-1] + [value]
                    with self.assertRaises(SchemaError):
                        wrapper.plan(*invalid)
        self.assertEqual(self.components["events"], [])

    def test_failed_replan_preserves_plan_workspace_and_execution(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="HND")
        plan_wrapper(wrapper)
        plan = wrapper.plan_state
        workspace = wrapper.workspace_contract
        events = list(self.components["events"])
        for slot in range(3):
            valid_args = [[0, 1], [7], [64]]
            for value in (float(valid_args[slot][-1]), float("inf")):
                with self.subTest(slot=slot, value=value):
                    args = list(valid_args)
                    args[slot] = [0, value] if slot == 0 else [value]
                    with self.assertRaises(SchemaError):
                        wrapper.plan(*args, 8, 2, 128, 128,
                                     q_data_type="bfloat16", kv_data_type="bfloat16")
                    self.assertIs(wrapper.plan_state, plan)
                    self.assertIs(wrapper.workspace_contract, workspace)
                    self.assertEqual(self.components["events"], events)
        self.assertEqual(wrapper.run("q", ("k", "v")), "package-output:q")
