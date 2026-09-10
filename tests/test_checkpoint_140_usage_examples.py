"""Execute the caller-facing documentation against synthetic integrations."""

import re
import unittest
from pathlib import Path

from flashinfer_npu.attention import attention_operator_runtime_registry_snapshot, install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.holistic import _install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.operator_mask_binding import AttentionBatchMaskIntegration, AttentionOperatorMaskArgumentSpec
from flashinfer_npu.attention.operator_runtime_owner import AttentionBatchRuntimeOwner
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_025_quantized_provider_run_lowering import metadata_tensor, quantized_input, query_input
from tests.test_checkpoint_043_provider_workspace_reset import FakeNpuWorkspace
from tests.test_checkpoint_135_quantized_mask_composition import components, result_pair, results, calls
from tests.test_checkpoint_136_framework_mask_frontend import Unreadable
from tests.test_checkpoint_137_batch_recorder_bootstrap import Factory
from tests.test_checkpoint_139_public_batch_mask import mask


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "attention_usage.md"


def snippets():
    blocks = re.findall(r"<!-- example: ([a-z_]+) -->\n```python\n(.*?)\n```", GUIDE.read_text(), re.S)
    if [name for name, _ in blocks] != ["paged_quant_mask", "reuse_plan_buffers", "inspect_selection"]:
        raise AssertionError("usage guide examples must have unique expected markers")
    return {name: compile(source, str(GUIDE) + ":" + name, "exec") for name, source in blocks}


class UsageExamplesCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.runtime, _, _, self.inspector, _ = components()
        self.owner, self.factory = AttentionBatchRuntimeOwner(), Factory()
        self.blocks = snippets()
        calls.clear()
        results.clear()

    def tearDown(self):
        for recorder in self.factory.recorders:
            for event in recorder.events:
                event.ready = True
        self.owner.close()
        old = self.original
        _install_attention_operator_runtime_resolvers(
            old.registry, operation_catalog=old.operation_catalog,
            runtime_declarations=old.runtime_declarations,
            plan_scoring_manifest_binding=old.plan_scoring_manifest_binding,
            provider_integration_bundle_binding=old.provider_integration_bundle_binding,
            batch_completion_event_recorder_factory=old.batch_completion_event_recorder_factory,
            batch_runtime_owner=old.batch_runtime_owner,
            batch_mask_integration=old.batch_mask_integration)
        calls.clear()
        results.clear()

    def install(self):
        operation = self.runtime._operation_catalog.operations[0]
        mappings = (
            AttentionOperatorMaskArgumentSpec(operation.fingerprint, "bool_allow_flat", "mask", "logical_offsets"),
            AttentionOperatorMaskArgumentSpec(operation.fingerprint, "packed_allow_little_segments", "mask", "logical_offsets", "byte_offsets"),
        )
        install_attention_operator_runtime_resolvers(
            self.runtime._resolver_registry, operation_catalog=self.runtime._operation_catalog,
            batch_completion_event_recorder_factory=self.factory, batch_runtime_owner=self.owner,
            batch_mask_integration=AttentionBatchMaskIntegration(mappings, self.inspector))

    def inputs(self, packed=False):
        plan = self.runtime.plan_state
        kv = quantized_input(plan.spec.kv_quant_spec)
        return dict(
            workspace_buffer=FakeNpuWorkspace(), qo_indptr=[0, 1, 2],
            paged_kv_indptr=[0, 1, 2], paged_kv_indices=[1, 0], paged_kv_last_page_len=[2, 1],
            num_qo_heads=2, num_kv_heads=1, head_dim_qk=3, head_dim_vo=2, page_size=2,
            q_dtype="float32", o_dtype="float32", quant_spec=kv.quant_spec,
            k_storage=kv.key_storage, v_storage=kv.value_storage,
            k_scale=kv.key_scale, v_scale=kv.value_scale,
            k_logical_shape=(2, 2, 1, 3), v_logical_shape=(2, 2, 1, 2),
            q=query_input(plan), q_next=query_input(plan, "next-query"),
            custom_mask=Unreadable() if packed else mask(numel=3),
            packed_custom_mask=mask(True, numel=2) if packed else None,
        )

    def test_documented_plan_run_reuse_and_diagnostics_execute_for_both_masks(self):
        self.install()
        for packed in (False, True):
            with self.subTest(packed=packed):
                namespace = self.inputs(packed)
                expected = result_pair(self.runtime, str(packed))
                results.append(expected)
                exec(self.blocks["paged_quant_mask"], namespace)
                wrapper = namespace["attention"]
                planned = wrapper.plan_state
                self.assertIs(namespace["output"], expected[0])
                self.assertIs(namespace["lse"], expected[1])
                self.assertIs(calls[-1][2], namespace["k_scale"])
                self.assertIs(calls[-1][3], namespace["v_scale"])
                self.assertIs(calls[-1][4], namespace["packed_custom_mask" if packed else "custom_mask"])
                # The guide requires completion before reusing the output buffers.
                self.factory.recorders[-1].events[0].ready = True
                exec(self.blocks["reuse_plan_buffers"], namespace)
                self.assertIs(wrapper.plan_state, planned)
                self.assertIs(namespace["output_again"], expected[0])
                self.assertIs(namespace["lse_again"], expected[1])
                exec(self.blocks["inspect_selection"], namespace)
                self.assertEqual(namespace["selected_operation"], self.runtime._operation_catalog.operations[0].operation_id)
                self.assertEqual(len(wrapper._operator_runtime.call_retention.pending_tokens), 1)

    def test_documented_scale_constraints_fail_before_invocation(self):
        self.install()
        namespace = self.inputs()
        namespace["k_scale"] = metadata_tensor("invalid-scale", (1,), "float32")
        with self.assertRaisesRegex(SchemaError, "scale.*shape"):
            exec(self.blocks["paged_quant_mask"], namespace)
        self.assertEqual(calls, [])
        self.assertEqual(self.factory.recorders[0].events, [])

    def test_documented_bootstrap_requirement_is_not_silently_bypassed(self):
        install_attention_operator_runtime_resolvers(
            self.runtime._resolver_registry, operation_catalog=self.runtime._operation_catalog)
        namespace = self.inputs()
        namespace["custom_mask"] = Unreadable()
        with self.assertRaisesRegex(NotImplementedError, "custom-mask"):
            exec(self.blocks["paged_quant_mask"], namespace)
        self.assertEqual(calls, [])
