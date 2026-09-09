import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionMode, AttentionOperatorCompletionValidatorFactory,
    AttentionOperatorOperationCatalog, AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageResolver, AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorQuantArgumentBinding, AttentionOperatorQuantizationBinding,
    AttentionOperatorRuntime, AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry, AttentionTensorAccessPolicy,
    AttentionStateError,
    CustomMaskSpec, EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG, PagedPrefillMetadata,
)
from flashinfer_npu.attention.operator_mask_binding import (
    AttentionMaskPlanRunAdapterBinder, AttentionOperatorMaskArgumentSpec,
)
from flashinfer_npu.attention.operator_quantization import AttentionOperatorQuantizationRunAdapterFactory
from flashinfer_npu.attention.operator_run import (
    AttentionOperatorCallerBufferRunAdapterFactory, AttentionOperatorRunAdapterFactoryChain,
    AttentionOperatorRunTensorValidationAdapterFactory,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan, pinned_environment
from tests.test_checkpoint_019_package_runtime_integration import FakeLogicalRunAdapter
from tests.test_checkpoint_022_operator_runtime_bootstrap import FakeTensorMetadataInspector
from tests.test_checkpoint_025_quantized_provider_run_lowering import metadata_tensor, quantized_input, query_input
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor
from tests.test_checkpoint_129_call_retention import Recorder
from tests.test_checkpoint_131_transactional_mask_plan import MaskLoader, mask_runtime


calls = []
results = []


def record_attention(
    query, key, value, *, table=None, scale=1.0, return_softmax_lse=False,
    key_scale=None, value_scale=None, runtime_key_scale=None, runtime_value_scale=None,
    runtime_query_scale=None, runtime_output_scale=None, runtime_query_head_scale=None,
    runtime_key_head_scale=None, runtime_value_head_scale=None,
    mask=None, logical_offsets=None, byte_offsets=None, out=None, lse=None,
):
    # No arithmetic: record exact arguments, then return prearranged metadata tensors.
    calls.append((key, value, key_scale, value_scale, mask, logical_offsets, byte_offsets))
    if out is None:
        return results.pop(0)
    return (out, lse) if return_softmax_lse else out


class Loader(MaskLoader):
    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append("resolve_callable")
        return record_attention


def components(*, packed=False, policy=None):
    values, original, _ = mask_runtime(Recorder())
    operation = replace(original, keyword_arguments=original.keyword_arguments + ("out", "lse"),
                        mutable_arguments=("out", "lse"), output_buffer_argument="out", lse_buffer_argument="lse")
    catalog = AttentionOperatorOperationCatalog("synthetic_quantized_mask", (operation,))
    inspector = FakeTensorMetadataInspector()
    policy = policy or AttentionTensorAccessPolicy()
    base = group_plan()
    spec = replace(base.spec, mode=AttentionMode.BATCH_PREFILL_PAGED,
                   custom_mask=CustomMaskSpec(2 if packed else 3, packed=packed))
    metadata = PagedPrefillMetadata((0, 1, 2), base.metadata)
    quant_binding = AttentionOperatorQuantizationBinding(
        operation.provider_id, operation.operation_id, spec.kv_quant_spec,
        (AttentionOperatorQuantArgumentBinding("kv.key.scale", "key_scale"),
         AttentionOperatorQuantArgumentBinding("kv.value.scale", "value_scale")))
    # Logical-layout metadata contracts use a synthetic authority. No physical
    # layout capability/evidence or real device correctness is asserted here.
    factories = AttentionOperatorRunAdapterFactoryChain(operation.provider_id, operation.operation_id, (
        AttentionOperatorQuantizationRunAdapterFactory(
            operation, (quant_binding,), inspector, policy, EMPTY_QUANT_PHYSICAL_LAYOUT_CATALOG,
            (), (), pinned_environment(), ()),
        AttentionOperatorCallerBufferRunAdapterFactory(operation),
        AttentionOperatorRunTensorValidationAdapterFactory(
            operation.provider_id, operation.operation_id, inspector, policy),
    ))
    resolver = AttentionOperatorPackageResolver(
        catalog, AttentionOperatorPackageCompatibility(
            operation.provider_id, operation.operation_id, "synthetic-quant-mask-v1", ("1.0.0",)),
        Loader(values["events"]))
    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=100, package_resolver=resolver, plan_gate=values["gate"],
        authority_resolver=values["authority"], logical_factory=values["factory"],
        logical_run_adapter=FakeLogicalRunAdapter(), tensor_materializer=values["materializer"],
        run_adapter_factory=factories,
        completion_validator_factory=AttentionOperatorCompletionValidatorFactory(operation, inspector, policy))
    registry = AttentionOperatorRuntimeResolverRegistry(
        (("npu", AttentionOperatorRuntimeImplementationRegistry((implementation,))),))
    recorder = Recorder()
    runtime = AttentionOperatorRuntime("npu:0", registry, catalog, mode=spec.mode,
                                       completion_event_recorder=recorder)
    mask = metadata_tensor("mask", (spec.custom_mask.numel,), "uint8" if packed else "bool")
    mask.tensor_view = replace(mask.tensor_view, storage_nbytes=64)
    mapping = AttentionOperatorMaskArgumentSpec(
        operation.fingerprint, "packed_allow_little_segments" if packed else "bool_allow_flat",
        "mask", "logical_offsets", "byte_offsets" if packed else None)
    binder = AttentionMaskPlanRunAdapterBinder(mask, mask, inspector, mapping, "npu:0")
    runtime.plan(spec, metadata, run_adapter_plan_binder=binder)
    return runtime, mask, recorder, inspector, binder


def result_pair(runtime, suffix=""):
    plan = runtime.plan_state
    return (buffer_tensor("output" + suffix, plan.expected_output_shape, plan.spec.o_dtype),
            buffer_tensor("lse" + suffix, plan.expected_lse_shape, "float32"))


class QuantizedMaskCompositionCheckpoint(unittest.TestCase):
    def setUp(self):
        calls[:] = []
        results[:] = []

    def test_output_aliasing_quant_scale_is_rejected_before_invocation(self):
        runtime, _, _, _, _ = components()
        plan = runtime.plan_state
        kv = quantized_input(plan.spec.kv_quant_spec)
        out = buffer_tensor("aliased-output", plan.expected_output_shape, plan.spec.o_dtype,
                            storage_id=kv.key_scale.tensor_view.storage_id)
        with self.assertRaisesRegex(SchemaError, "alias.*kv.key_scale"):
            runtime.run(query_input(plan), kv, return_lse=False, out=out)
        self.assertEqual(calls, [])

    def test_quantized_storage_scales_and_each_mask_encoding_reach_the_exact_arguments(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                runtime, mask, recorder, _, _ = components(packed=packed)
                plan = runtime.plan_state
                kv = quantized_input(plan.spec.kv_quant_spec)
                expected = result_pair(runtime)
                results.append(expected)
                actual = runtime.run(query_input(plan), kv, return_lse=True)
                self.assertIs(actual[0], expected[0])
                self.assertIs(actual[1], expected[1])
                for actual_arg, source in zip(calls[-1][:5], (
                    kv.key_storage, kv.value_storage, kv.key_scale, kv.value_scale, mask,
                )):
                    self.assertIs(actual_arg, source)
                self.assertEqual(calls[-1][5:], ((0, 2, 3), (0, 1, 2) if packed else None))
                names = tuple(name for name, _ in runtime.last_completion_receipt.input_view_fingerprints)
                self.assertEqual(names, ("query", "kv.key_storage", "kv.key_scale",
                                         "kv.value_storage", "kv.value_scale", "custom_mask"))
                self.assertEqual(len(runtime.last_lowered_call.retained_resources), 1)
                self.assertEqual(len(runtime.call_retention.pending_tokens), 1)
                recorder.events[0].ready = True
                runtime.close()

    def test_caller_buffers_work_with_packed_mask_and_keep_return_identity(self):
        runtime, mask, _, _, _ = components(packed=True)
        plan = runtime.plan_state
        kv = quantized_input(plan.spec.kv_quant_spec)
        out, lse = result_pair(runtime)
        actual = runtime.run(query_input(plan), kv, return_lse=True, out=out, lse=lse)
        self.assertIs(actual[0], out)
        self.assertIs(actual[1], lse)
        self.assertIs(calls[-1][4], mask)
        self.assertEqual(runtime.last_lowered_call.mutable_argument_names, ("out", "lse"))

    def test_mask_remains_protected_when_general_output_input_aliasing_is_allowed(self):
        runtime, mask, _, _, _ = components(policy=AttentionTensorAccessPolicy(permit_output_input_alias=True))
        plan = runtime.plan_state
        out = buffer_tensor("mask-alias", plan.expected_output_shape, plan.spec.o_dtype,
                            storage_id=mask.tensor_view.storage_id)
        out.tensor_view = replace(out.tensor_view, storage_nbytes=64)
        with self.assertRaisesRegex(SchemaError, "alias.*custom mask"):
            runtime.run(query_input(plan), quantized_input(plan.spec.kv_quant_spec), return_lse=False, out=out)
        self.assertEqual(calls, [])
        self.assertEqual(runtime.call_retention.pending_tokens, ())

    def test_wrong_quant_spec_fails_before_mask_reinspection_or_invocation(self):
        runtime, _, recorder, inspector, _ = components()
        plan = runtime.plan_state
        original = runtime.operator_session.active_plan
        different = replace(plan.spec.kv_quant_spec, group_size=(1, 1, 1, 1))
        inspector.calls.clear()
        with self.assertRaisesRegex(SchemaError, "active QuantSpec"):
            runtime.run(query_input(plan), quantized_input(different), return_lse=False)
        self.assertFalse(any(name == "custom_mask" for _, name, _ in inspector.calls))
        self.assertEqual(calls, [])
        self.assertEqual(recorder.events, [])
        self.assertIs(runtime.operator_session.active_plan, original)

    def test_bad_scale_shape_is_rejected_without_poisoning_the_plan(self):
        runtime, _, recorder, _, _ = components(packed=True)
        plan = runtime.plan_state
        bad = quantized_input(plan.spec.kv_quant_spec,
                              key_scale=metadata_tensor("bad-scale", (1,), "float32"))
        with self.assertRaisesRegex(SchemaError, "scale.*shape"):
            runtime.run(query_input(plan), bad, return_lse=False)
        self.assertEqual(calls, [])
        self.assertEqual(recorder.events, [])
        out, _ = result_pair(runtime)
        results.append(out)
        self.assertIs(runtime.run(query_input(plan), quantized_input(plan.spec.kv_quant_spec),
                                  return_lse=False), out)

    def test_invalid_returned_alias_preserves_submitted_call_but_no_success_receipt(self):
        runtime, _, recorder, _, _ = components()
        plan = runtime.plan_state
        kv = quantized_input(plan.spec.kv_quant_spec)
        returned = buffer_tensor("returned-scale-alias", plan.expected_output_shape, plan.spec.o_dtype,
                                 storage_id=kv.key_scale.tensor_view.storage_id)
        results.append(returned)
        with self.assertRaisesRegex(SchemaError, "output result cannot alias kv.key_scale"):
            runtime.run(query_input(plan), kv, return_lse=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(runtime.call_retention.pending_tokens), 1)
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_run_receipt
        recorder.events[0].ready = True
        runtime.close()

    def test_quantized_dense_quantized_replanning_does_not_leave_stale_scale_arguments(self):
        runtime, mask, recorder, _, binder = components()
        original = runtime.plan_state
        for quantized in (True, False, True):
            spec = original.spec if quantized else replace(original.spec, kv_dtype="float32", kv_quant_spec=None)
            runtime.plan(spec, original.metadata, run_adapter_plan_binder=binder)
            plan = runtime.plan_state
            if quantized:
                kv = quantized_input(spec.kv_quant_spec)
                expected_scales = (kv.key_scale, kv.value_scale)
            else:
                kv = (metadata_tensor("dense-key", (2, 2, 1, 3), "float32"),
                      metadata_tensor("dense-value", (2, 2, 1, 2), "float32"))
                expected_scales = (None, None)
            out, _ = result_pair(runtime, str(plan.generation))
            results.append(out)
            self.assertIs(runtime.run(query_input(plan), kv, return_lse=False), out)
            self.assertIs(calls[-1][2], expected_scales[0])
            self.assertIs(calls[-1][3], expected_scales[1])
            self.assertIs(calls[-1][4], mask)
        tokens = runtime.call_retention.pending_tokens
        self.assertEqual(len({token.active_plan_fingerprint for token in tokens}), 3)
        for event in recorder.events:
            event.ready = True
        runtime.close()
