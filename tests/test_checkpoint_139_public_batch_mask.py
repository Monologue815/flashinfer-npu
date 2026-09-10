import gc
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog, AttentionOperatorRuntimeResolutionError,
    attention_operator_runtime_registry_snapshot, install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.holistic import _install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.operator_mask_binding import AttentionBatchMaskIntegration, AttentionOperatorMaskArgumentSpec
from flashinfer_npu.attention.operator_runtime_owner import AttentionBatchRuntimeOwner
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper, single_prefill_with_kv_cache
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import FakeTensorMetadataInspector
from tests.test_checkpoint_025_quantized_provider_run_lowering import metadata_tensor, quantized_input, query_input
from tests.test_checkpoint_043_provider_workspace_reset import FakeNpuWorkspace
from tests.test_checkpoint_102_public_nvfp4_canonicalization import FakeNpuTensor
from tests.test_checkpoint_131_transactional_mask_plan import mask_calls
from tests.test_checkpoint_132_mask_candidate_admission import selection_inputs
from tests.test_checkpoint_135_quantized_mask_composition import components, result_pair, results, calls
from tests.test_checkpoint_136_framework_mask_frontend import MaskTensor, Unreadable
from tests.test_checkpoint_137_batch_recorder_bootstrap import Factory


def mask(packed=False, numel=None):
    payload = MaskTensor((numel if numel is not None else (3 if packed else 12),), "uint8" if packed else "bool")
    payload.tensor_view = metadata_tensor("public-mask", payload.shape, payload.dtype).tensor_view
    return payload


def plan(wrapper, **kwargs):
    if isinstance(wrapper, BatchPrefillWithPagedKVCacheWrapper):
        return wrapper.plan([0, 1, 2, 2], [0, 1, 4, 4], [0, 1, 2, 3], [3, 3, 0], 1, 1, 1, 3, **kwargs)
    return wrapper.plan([0, 1, 2, 2], [0, 3, 12, 12], 1, 1, 1, **kwargs)


def run(wrapper):
    return wrapper.run("q", ("k", "v")) if isinstance(wrapper, BatchPrefillWithPagedKVCacheWrapper) else wrapper.run("q", "k", "v")


class PublicBatchMaskCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        runtime, _, self.operations, self.mappings, self.events = selection_inputs()
        self.registry, self.catalog = runtime._resolver_registry, runtime._operation_catalog
        self.inspector = FakeTensorMetadataInspector()
        self.integration = AttentionBatchMaskIntegration(self.mappings, self.inspector)
        self.factory, self.owner = Factory(), AttentionBatchRuntimeOwner()
        mask_calls.clear()
        calls.clear()
        results.clear()

    def install(self, integration=None):
        return install_attention_operator_runtime_resolvers(
            self.registry, operation_catalog=self.catalog,
            batch_completion_event_recorder_factory=self.factory,
            batch_runtime_owner=self.owner,
            batch_mask_integration=self.integration if integration is None else integration)

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
        mask_calls.clear()
        calls.clear()
        results.clear()

    def test_paged_and_ragged_plan_select_mask_provider_without_caller_controls(self):
        self.install()
        self.assertEqual(self.inspector.calls, [])
        for wrapper_type in (BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper):
            wrapper = wrapper_type(FakeNpuWorkspace())
            for packed in (False, True):
                with self.subTest(wrapper=wrapper_type.__name__, packed=packed):
                    for events in self.events:
                        events.clear()
                    payload = mask(packed)
                    self.assertIsNone(plan(wrapper, **{"packed_custom_mask" if packed else "custom_mask": payload}))
                    self.assertEqual(wrapper.plan_selection.operation_id, self.operations[int(packed)].operation_id)
                    self.assertEqual(self.events[1 - int(packed)], [])
                    self.assertEqual(run(wrapper), "output")
                    self.assertIs(mask_calls[-1][0], payload)
                    self.assertEqual(mask_calls[-1][1:], ((0, 3, 12, 12), (0, 1, 3, 3) if packed else None))

    def test_packed_precedence_ignores_unreadable_bool_source(self):
        self.install()
        wrapper = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        payload = mask(True)
        plan(wrapper, custom_mask=Unreadable(), packed_custom_mask=payload)
        self.assertTrue(wrapper.plan_state.spec.custom_mask.packed)
        run(wrapper)
        self.assertIs(mask_calls[-1][0], payload)

    def test_invalid_replan_preserves_previous_plan_workspace_and_mask(self):
        self.install()
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(FakeNpuWorkspace())
        payload = mask()
        plan(wrapper, custom_mask=payload)
        previous, workspace, selection = wrapper.plan_state, wrapper.workspace_contract, wrapper.plan_selection
        with self.assertRaisesRegex(SchemaError, "shape"):
            plan(wrapper, packed_custom_mask=mask(True, 2))
        bad = mask()
        bad.tensor_view = replace(bad.tensor_view, strides=(2,), storage_nbytes=64)
        with self.assertRaisesRegex(SchemaError, "contiguous"):
            plan(wrapper, custom_mask=bad)
        self.assertIs(wrapper.plan_state, previous)
        self.assertIs(wrapper.workspace_contract, workspace)
        self.assertEqual(wrapper.plan_selection, selection)
        self.assertEqual(mask_calls, [])
        run(wrapper)
        self.assertIs(mask_calls[-1][0], payload)

    def test_source_drift_is_rejected_before_invocation(self):
        self.install()
        wrapper = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        payload = mask()
        plan(wrapper, custom_mask=payload)
        payload.tensor_view = replace(payload.tensor_view, storage_id="changed-storage")
        with self.assertRaises(SchemaError):
            run(wrapper)
        self.assertEqual(mask_calls, [])
        self.assertEqual(self.factory.recorders[0].events, [])

    def test_unmasked_replan_removes_mask_but_pending_call_keeps_source_alive(self):
        self.install()
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(FakeNpuWorkspace())
        payload = mask()
        reference = weakref.ref(payload)
        plan(wrapper, custom_mask=payload)
        run(wrapper)
        plan(wrapper)
        run(wrapper)
        self.assertIsNone(wrapper.plan_state.spec.custom_mask)
        self.assertEqual(mask_calls[-1], (None, None, None))
        mask_calls.clear()
        self.inspector.calls.clear()
        del payload
        gc.collect()
        self.assertIsNotNone(reference())
        self.factory.recorders[0].events[0].ready = True
        run(wrapper)
        gc.collect()
        self.assertIsNone(reference())

    def test_missing_format_is_rejected_before_any_candidate_probe(self):
        self.install(AttentionBatchMaskIntegration((self.mappings[1],), self.inspector))
        wrapper = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        with self.assertRaises(AttentionOperatorRuntimeResolutionError):
            plan(wrapper, custom_mask=mask())
        self.assertEqual(self.events, ([], []))
        self.assertEqual(self.inspector.calls, [])
        self.assertEqual(mask_calls, [])

    def test_install_requires_complete_lifetime_configuration_and_exact_catalog(self):
        baseline = attention_operator_runtime_registry_snapshot().generation
        with self.assertRaisesRegex(SchemaError, "ownership and completion"):
            install_attention_operator_runtime_resolvers(self.registry, operation_catalog=self.catalog,
                                                        batch_mask_integration=self.integration)
        changed = AttentionOperatorOperationCatalog("changed", (replace(self.operations[0], api_version="v2"), self.operations[1]))
        with self.assertRaisesRegex(SchemaError, "absent.*catalog"):
            install_attention_operator_runtime_resolvers(
                self.registry, operation_catalog=changed,
                batch_completion_event_recorder_factory=self.factory, batch_runtime_owner=self.owner,
                batch_mask_integration=self.integration)
        self.assertEqual(attention_operator_runtime_registry_snapshot().generation, baseline)
        self.assertEqual(self.events, ([], []))
        self.assertEqual(self.factory.calls, [])

    def test_integration_freezes_mapping_sequence_and_validates_dependencies(self):
        values = list(self.mappings)
        integration = AttentionBatchMaskIntegration(values, self.inspector)
        values.clear()
        self.assertEqual(integration.mappings, self.mappings)
        with self.assertRaisesRegex(SchemaError, "unique"):
            AttentionBatchMaskIntegration((self.mappings[0], self.mappings[0]), self.inspector)
        with self.assertRaises(SchemaError):
            AttentionBatchMaskIntegration(self.mappings, self.inspector, required_alignment=True)
        with self.assertRaises(TypeError):
            AttentionBatchMaskIntegration(self.mappings, type("BadInspector", (), {"to_view": 0})())
        self.assertEqual(self.inspector.calls, [])

    def test_default_reinstall_changes_future_wrappers_only_and_single_graph_stay_closed(self):
        self.install()
        old = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        q = FakeNpuTensor("q", (1, 1, 1), dtype="bfloat16")
        kv = FakeNpuTensor("kv", (3, 1, 1), dtype="bfloat16")
        with self.assertRaisesRegex(NotImplementedError, "custom-mask"):
            single_prefill_with_kv_cache(q, kv, kv, custom_mask=Unreadable())
        with self.assertRaisesRegex(NotImplementedError, "graph resources"):
            BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace(), use_cuda_graph=True)
        install_attention_operator_runtime_resolvers(self.registry, operation_catalog=self.catalog)
        new = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        with self.assertRaisesRegex(NotImplementedError, "custom-mask"):
            plan(new, custom_mask=Unreadable())
        plan(old, custom_mask=mask())
        self.assertEqual(run(old), "output")

    def test_quantized_kv_and_mask_compose_through_public_paged_plan_run(self):
        runtime, _, _, inspector, _ = components()
        operation = runtime._operation_catalog.operations[0]
        mapping = AttentionOperatorMaskArgumentSpec(operation.fingerprint, "bool_allow_flat", "mask", "logical_offsets")
        install_attention_operator_runtime_resolvers(
            runtime._resolver_registry, operation_catalog=runtime._operation_catalog,
            batch_completion_event_recorder_factory=self.factory, batch_runtime_owner=self.owner,
            batch_mask_integration=AttentionBatchMaskIntegration((mapping,), inspector))
        wrapper = BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace())
        quant_spec = runtime.plan_state.spec.kv_quant_spec
        payload = mask(numel=3)
        wrapper.plan([0, 1, 2], [0, 1, 2], [1, 0], [2, 1], 2, 1, 3, 2,
                     q_data_type="float32", kv_data_type=quant_spec, o_data_type="float32",
                     head_dim_vo=2, custom_mask=payload)
        plan_state = wrapper.plan_state
        kv = quantized_input(quant_spec)
        expected = result_pair(wrapper._operator_runtime)
        results.append(expected)
        result = wrapper.run(query_input(plan_state), kv, return_lse=True)
        self.assertIs(result[0], expected[0])
        self.assertIs(result[1], expected[1])
        for actual, source in zip(calls[-1][:5], (kv.key_storage, kv.value_storage, kv.key_scale, kv.value_scale, payload)):
            self.assertIs(actual, source)
