import inspect
import unittest

from flashinfer_npu.attention import (
    AttentionStateError,
    AttentionWorkspaceContract,
    ReferenceTensor,
    WorkspaceRequirementUnknownError,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def index(values, device="cpu"):
    return tensor(values, dtype="int32", device=device)


def workspace(size=256, device="cpu"):
    return ReferenceTensor.zeros((size,), dtype="uint8", device=device)


class WorkspaceContractTests(unittest.TestCase):
    def test_unknown_backend_requirement_is_not_interpreted_as_zero(self):
        contract = AttentionWorkspaceContract(
            backend="ascendc_aot",
            device="npu:0",
            float_capacity_bytes=1024,
            int_capacity_bytes=512,
        )
        self.assertFalse(contract.requirements_known)
        with self.assertRaisesRegex(WorkspaceRequirementUnknownError, "unknown"):
            _ = contract.required_sizes

    def test_known_requirements_validate_each_caller_owned_buffer(self):
        with self.assertRaisesRegex(SchemaError, "float workspace capacity"):
            AttentionWorkspaceContract(
                backend="test",
                device="cpu",
                float_capacity_bytes=63,
                int_capacity_bytes=32,
                required_float_bytes=64,
                required_int_bytes=32,
            )
        contract = AttentionWorkspaceContract.for_host_reference(
            device="cpu",
            float_capacity_bytes=0,
            int_capacity_bytes=0,
        )
        self.assertEqual(contract.required_sizes, (0, 0))

    def test_reset_tracks_binding_generation_and_preserves_active_plan(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(index([0, 1]), index([0]), index([1]), 1, 1, 1, 1)
        initial = wrapper.workspace_contract
        self.assertEqual(initial.plan_generation, wrapper.plan_state.generation)
        wrapper.reset_workspace_buffer(workspace(64), workspace(32))
        rebound = wrapper.workspace_contract
        self.assertEqual(rebound.binding_generation, initial.binding_generation + 1)
        self.assertEqual(rebound.plan_generation, initial.plan_generation)
        self.assertEqual(rebound.float_capacity_bytes, 64)
        output = wrapper.run(
            tensor([[[0.0]]], dtype="float16"),
            (
                tensor([[[[0.0]]]], dtype="float16"),
                tensor([[[[2.0]]]], dtype="float16"),
            ),
        )
        self.assertEqual(output.data, (2.0,))

    def test_reset_rejects_alias_and_device_change_after_plan(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        shared = workspace(64)
        with self.assertRaisesRegex(SchemaError, "cannot alias"):
            wrapper.reset_workspace_buffer(shared, shared)
        wrapper.plan(index([0, 1]), index([0]), index([1]), 1, 1, 1, 1)
        with self.assertRaisesRegex(SchemaError, "device cannot change"):
            wrapper.reset_workspace_buffer(
                workspace(64, "cpu:1"), workspace(32, "cpu:1")
            )


class WorkspaceSizeFacadeTests(unittest.TestCase):
    def test_workspace_size_signatures_match_upstream_snapshot(self):
        self.assertEqual(
            list(
                inspect.signature(
                    BatchPrefillWithPagedKVCacheWrapper.workspace_size
                ).parameters
            ),
            [
                "self", "qo_indptr", "paged_kv_indptr", "paged_kv_indices",
                "paged_kv_last_page_len", "num_qo_heads", "num_kv_heads",
                "head_dim_qk", "page_size", "head_dim_vo", "custom_mask",
                "packed_custom_mask", "causal", "pos_encoding_mode",
                "use_fp16_qk_reduction", "sm_scale", "window_left",
                "logits_soft_cap", "rope_scale", "rope_theta", "q_data_type",
                "kv_data_type", "o_data_type", "prefix_len_ptr",
                "token_pos_in_items_ptr", "token_pos_in_items_len",
                "max_item_len_ptr", "seq_lens", "seq_lens_q", "block_tables",
                "max_token_per_sequence", "max_sequence_kv", "fixed_split_size",
                "disable_split_kv",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    BatchDecodeWithPagedKVCacheWrapper.workspace_size
                ).parameters
            ),
            [
                "self", "indptr", "indices", "last_page_len", "num_qo_heads",
                "num_kv_heads", "head_dim", "page_size", "pos_encoding_mode",
                "window_left", "logits_soft_cap", "q_data_type", "kv_data_type",
                "o_data_type", "data_type", "sm_scale", "rope_scale",
                "rope_theta", "block_tables", "seq_lens", "fixed_split_size",
                "disable_split_kv", "q_len_per_req",
            ],
        )

    def test_prefill_workspace_query_is_zero_for_host_and_does_not_plan(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        sizes = wrapper.workspace_size(
            index([0, 1]), index([0, 1]), index([0]), index([1]),
            1, 1, 1, 1,
        )
        self.assertEqual(sizes, (0, 0))
        with self.assertRaises(AttentionStateError):
            _ = wrapper.plan_state

    def test_decode_workspace_query_preserves_existing_plan(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(index([0, 1]), index([0]), index([1]), 1, 1, 1, 1)
        planned = wrapper.plan_state
        sizes = wrapper.workspace_size(
            index([0, 1]), index([0]), index([1]), 2, 1, 2, 1
        )
        self.assertEqual(sizes, (0, 0))
        self.assertIs(wrapper.plan_state, planned)

    def test_workspace_query_runs_plan_validation(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        with self.assertRaisesRegex(SchemaError, "last_page_len"):
            wrapper.workspace_size(
                index([0, 1]), index([0]), index([0]), 1, 1, 1, 1
            )


if __name__ == "__main__":
    unittest.main()
