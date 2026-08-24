from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionCaptureCompatibilityError,
    AttentionCapturedExecution,
    AttentionDispatchError,
    AttentionGraphResourceContract,
    AttentionWorkspaceContract,
    ReferenceBuffer,
    ReferenceTensor,
    build_kernel_execution_identity,
    select_attention_dispatch,
    validate_reference_attention_views,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import (
    bound_kernel,
    functional_profile,
    group_case,
    group_plan,
)


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def index(values, device="cpu"):
    return tensor(values, dtype="int32", device=device)


def workspace(size=256, device="cpu"):
    return ReferenceTensor.zeros((size,), dtype="uint8", device=device)


def graph_decode_wrapper():
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace(),
        use_cuda_graph=True,
        paged_kv_indptr_buffer=index([0, 0]),
        paged_kv_indices_buffer=index([0, 0]),
        paged_kv_last_page_len_buffer=index([0]),
        backend="reference",
    )
    wrapper.plan(
        index([0, 1]),
        index([0]),
        index([1]),
        1,
        1,
        1,
        1,
        q_data_type="float32",
    )
    return wrapper


def decode_inputs():
    return (
        tensor([[[0.0]]]),
        (tensor([[[[0.0]]]]), tensor([[[[7.0]]]])),
    )


class AttentionExecutionIdentitySchemaTests(unittest.TestCase):
    def test_host_capture_round_trip_preserves_identity(self):
        wrapper = graph_decode_wrapper()
        q, kv = decode_inputs()
        wrapper.run(q, kv)

        capture = wrapper.capture_record
        self.assertIsNotNone(capture)
        self.assertEqual(capture.capture_kind, "host_contract")
        self.assertEqual(capture.identity.binding_kind, "reference_contract")
        restored = AttentionCapturedExecution.from_dict(capture.to_dict())
        self.assertEqual(restored, capture)
        self.assertEqual(restored.fingerprint, capture.fingerprint)
        self.assertNotEqual(
            capture.identity.plan_fingerprint,
            capture.identity.workspace_fingerprint,
        )

    def test_kernel_binding_fields_are_atomic_and_forbidden_on_reference(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs())
        identity = wrapper.capture_record.identity
        with self.assertRaisesRegex(SchemaError, "present together"):
            replace(identity, kernel_id="kernel")
        with self.assertRaisesRegex(SchemaError, "cannot claim"):
            replace(
                identity,
                capability_profile_id="profile",
                capability_profile_fingerprint="1" * 64,
                capability_rule_id="rule",
                capability_evidence_id="evidence",
                kernel_id="kernel",
                kernel_fingerprint="2" * 64,
            )
        with self.assertRaisesRegex(SchemaError, "requires kernel binding"):
            replace(identity, backend="ascend")

    def test_capture_reports_each_mutated_identity_dimension(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs())
        capture = wrapper.capture_record
        mutations = {
            "plan_fingerprint": "1" * 64,
            "admission_fingerprint": "2" * 64,
            "numerics_policy_fingerprint": "3" * 64,
            "workspace_fingerprint": "4" * 64,
            "graph_resources_fingerprint": "5" * 64,
            "tensor_signature_fingerprint": "6" * 64,
            "access_policy_fingerprint": "7" * 64,
            "return_lse": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                candidate = replace(capture.identity, **{field: value})
                with self.assertRaisesRegex(
                    AttentionCaptureCompatibilityError, field
                ):
                    capture.validate_reuse(candidate)

    def test_graph_resources_are_versioned_and_capacity_sensitive(self):
        wrapper = graph_decode_wrapper()
        resources = wrapper.graph_resource_contract
        self.assertEqual(
            tuple(item.name for item in resources.persistent_buffers),
            (
                "paged_kv_indptr",
                "paged_kv_indices",
                "paged_kv_last_page_len",
            ),
        )
        self.assertEqual(
            AttentionGraphResourceContract.from_dict(resources.to_dict()),
            resources,
        )
        changed = replace(
            resources,
            persistent_buffers=(
                resources.persistent_buffers[0],
                replace(resources.persistent_buffers[1], capacity=3),
                resources.persistent_buffers[2],
            ),
        )
        self.assertNotEqual(changed.fingerprint, resources.fingerprint)
        with self.assertRaisesRegex(SchemaError, "unique"):
            replace(
                resources,
                persistent_buffers=(
                    resources.persistent_buffers[0],
                    replace(
                        resources.persistent_buffers[1],
                        name=resources.persistent_buffers[0].name,
                    ),
                    resources.persistent_buffers[2],
                ),
            )


class HostGraphContractLifecycleTests(unittest.TestCase):
    def test_equivalent_new_allocations_reuse_structural_contract(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs())
        first = wrapper.capture_record

        # Storage identity is deliberately excluded from the Host structural ABI.
        wrapper.run(*decode_inputs())
        self.assertIs(wrapper.capture_record, first)
        self.assertEqual(wrapper.capture_record.capture_generation, 1)

    def test_return_shape_change_rejects_stale_contract(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs())
        with self.assertRaisesRegex(
            AttentionCaptureCompatibilityError, "return_lse"
        ):
            wrapper.run(*decode_inputs(), return_lse=True)

    def test_run_options_change_rejects_stale_contract(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs(), q_scale=1.0)
        with self.assertRaisesRegex(
            AttentionCaptureCompatibilityError, "run_options_fingerprint"
        ):
            wrapper.run(*decode_inputs(), q_scale=2.0)

    def test_caller_lse_buffer_does_not_hide_return_shape_change(self):
        wrapper = graph_decode_wrapper()
        q, kv = decode_inputs()
        lse = ReferenceBuffer.zeros((1, 1), dtype="float32")
        wrapper.run(q, kv, lse=lse, return_lse=False)
        self.assertFalse(wrapper.capture_record.identity.return_lse)
        with self.assertRaisesRegex(
            AttentionCaptureCompatibilityError, "return_lse"
        ):
            wrapper.run(q, kv, lse=lse, return_lse=True)

    def test_workspace_rebind_rejects_stale_contract(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs())
        wrapper.reset_workspace_buffer(workspace(512), workspace(16))
        with self.assertRaisesRegex(
            AttentionCaptureCompatibilityError, "workspace_fingerprint"
        ):
            wrapper.run(*decode_inputs())

    def test_replan_invalidates_then_recaptures_contract(self):
        wrapper = graph_decode_wrapper()
        wrapper.run(*decode_inputs())
        first = wrapper.capture_record

        wrapper.plan(
            index([0, 1]),
            index([0]),
            index([1]),
            1,
            1,
            1,
            1,
            q_data_type="float32",
        )
        self.assertIsNone(wrapper.capture_record)
        wrapper.run(*decode_inputs())
        self.assertEqual(wrapper.capture_record.capture_generation, 2)
        self.assertNotEqual(
            wrapper.capture_record.identity.workspace_fingerprint,
            first.identity.workspace_fingerprint,
        )

    def test_non_graph_wrapper_never_publishes_capture(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace(), backend="reference"
        )
        wrapper.plan(
            index([0, 1]), index([0]), index([1]), 1, 1, 1, 1,
            q_data_type="float32",
        )
        wrapper.run(*decode_inputs())
        self.assertIsNone(wrapper.capture_record)
        self.assertFalse(wrapper.graph_resource_contract.graph_enabled)


class KernelExecutionIdentityTests(unittest.TestCase):
    def kernel_inputs(self):
        profile = functional_profile()
        kernel = bound_kernel(profile)
        plan = group_plan()
        receipt = select_attention_dispatch(
            plan, (profile,), (kernel,), profile.environment
        )
        _, case = group_case()
        float_workspace = ReferenceTensor.zeros(
            (receipt.float_workspace_bytes,), dtype="uint8"
        )
        int_workspace = ReferenceTensor.zeros(
            (receipt.int_workspace_bytes,), dtype="uint8"
        )
        workspace_contract = AttentionWorkspaceContract(
            backend=receipt.backend.value,
            device="cpu",
            float_capacity_bytes=receipt.float_workspace_bytes,
            int_capacity_bytes=receipt.int_workspace_bytes,
            required_float_bytes=receipt.float_workspace_bytes,
            required_int_bytes=receipt.int_workspace_bytes,
            plan_generation=plan.generation,
        )
        tensors = validate_reference_attention_views(
            case.trace.q,
            case.trace.kv_data,
            workspace_float=float_workspace,
            workspace_int=int_workspace,
            plan=plan,
        )
        return (
            plan,
            profile,
            kernel,
            receipt,
            workspace_contract,
            tensors,
        )

    def test_dispatch_receipt_builds_bound_non_reference_identity(self):
        plan, profile, kernel, receipt, workspace_contract, tensors = (
            self.kernel_inputs()
        )
        identity = build_kernel_execution_identity(
            plan,
            workspace_contract,
            AttentionGraphResourceContract.disabled(plan.spec.mode),
            tensors,
            receipt,
            profile,
            kernel,
            profile.environment,
            return_lse=True,
        )
        self.assertEqual(identity.binding_kind, "kernel")
        self.assertEqual(identity.backend, profile.backend.value)
        self.assertEqual(identity.capability_profile_id, profile.profile_id)
        self.assertEqual(identity.capability_evidence_id, receipt.evidence_id)
        self.assertEqual(identity.kernel_id, kernel.kernel_id)
        self.assertTrue(identity.return_lse)
        self.assertFalse(identity.graph_enabled)

    def test_unknown_or_mismatched_workspace_cannot_authorize_execution(self):
        plan, profile, kernel, receipt, workspace_contract, tensors = (
            self.kernel_inputs()
        )
        graph = AttentionGraphResourceContract.disabled(plan.spec.mode)
        unknown = replace(
            workspace_contract,
            required_float_bytes=None,
            required_int_bytes=None,
        )
        with self.assertRaisesRegex(SchemaError, "known workspace"):
            build_kernel_execution_identity(
                plan,
                unknown,
                graph,
                tensors,
                receipt,
                profile,
                kernel,
                profile.environment,
            )
        mismatched = replace(
            workspace_contract,
            required_float_bytes=receipt.float_workspace_bytes + 1,
            float_capacity_bytes=receipt.float_workspace_bytes + 1,
        )
        with self.assertRaisesRegex(SchemaError, "dispatch formula"):
            build_kernel_execution_identity(
                plan,
                mismatched,
                graph,
                tensors,
                receipt,
                profile,
                kernel,
                profile.environment,
            )

    def test_workspace_tensor_view_must_match_resource_contract(self):
        plan, profile, kernel, receipt, workspace_contract, tensors = (
            self.kernel_inputs()
        )
        changed_float = ReferenceTensor.zeros(
            (receipt.float_workspace_bytes + 1,), dtype="uint8"
        )
        changed_tensors = validate_reference_attention_views(
            group_case()[1].trace.q,
            group_case()[1].trace.kv_data,
            workspace_float=changed_float,
            workspace_int=ReferenceTensor.zeros(
                (receipt.int_workspace_bytes,), dtype="uint8"
            ),
            plan=plan,
        )
        with self.assertRaisesRegex(SchemaError, "capacity/device contract"):
            build_kernel_execution_identity(
                plan,
                workspace_contract,
                AttentionGraphResourceContract.disabled(plan.spec.mode),
                changed_tensors,
                receipt,
                profile,
                kernel,
                profile.environment,
            )

    def test_receipt_is_revalidated_before_identity_construction(self):
        plan, profile, kernel, receipt, workspace_contract, tensors = (
            self.kernel_inputs()
        )
        stale = replace(receipt, kernel_fingerprint="f" * 64)
        with self.assertRaisesRegex(AttentionDispatchError, "kernel"):
            build_kernel_execution_identity(
                plan,
                workspace_contract,
                AttentionGraphResourceContract.disabled(plan.spec.mode),
                tensors,
                stale,
                profile,
                kernel,
                profile.environment,
            )


if __name__ == "__main__":
    unittest.main()
