import gc
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog, AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageResolver, AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorRuntime, AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry, AttentionStateError,
)
from flashinfer_npu.attention.operator_mask_binding import (
    AttentionMaskPlanRunAdapterBinder, AttentionOperatorMaskArgumentSpec,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    FakePackageLoader, FakeLogicalRunAdapter, build_components, fake_operation,
)
from tests.test_checkpoint_125_mask_tensor_inspection import inspection_inputs
from tests.test_checkpoint_129_call_retention import Recorder


mask_calls = []


def record_mask_attention(
    query, key, value, *, table=None, scale=1.0, return_softmax_lse=False,
    key_scale=None, value_scale=None, runtime_key_scale=None, runtime_value_scale=None,
    runtime_query_scale=None, runtime_output_scale=None, runtime_query_head_scale=None,
    runtime_key_head_scale=None, runtime_value_head_scale=None,
    mask=None, logical_offsets=None, byte_offsets=None,
):
    # Argument recorder only: no attention computation or tensor content access.
    mask_calls.append((mask, logical_offsets, byte_offsets))
    return ("output", "lse") if return_softmax_lse else "output"


class MaskLoader(FakePackageLoader):
    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append("resolve_callable")
        return record_mask_attention


def mask_runtime(recorder):
    values = build_components()
    original = fake_operation()
    operation = replace(original,
                        keyword_arguments=original.keyword_arguments + ("mask", "logical_offsets", "byte_offsets"),
                        host_sequence_arguments=("logical_offsets", "byte_offsets"))
    catalog = AttentionOperatorOperationCatalog("synthetic-mask-runtime", (operation,))
    resolver = AttentionOperatorPackageResolver(
        catalog, AttentionOperatorPackageCompatibility(
            "cann", operation.operation_id, "synthetic-mask-v1", ("1.0.0",)),
        MaskLoader(values["events"]))
    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=100, package_resolver=resolver, plan_gate=values["gate"],
        authority_resolver=values["authority"], logical_factory=values["factory"],
        logical_run_adapter=FakeLogicalRunAdapter(), tensor_materializer=values["materializer"])
    registry = AttentionOperatorRuntimeResolverRegistry(
        (("npu", AttentionOperatorRuntimeImplementationRegistry((implementation,))),))
    _, plan, _, _ = inspection_inputs()
    runtime = AttentionOperatorRuntime("npu:0", registry, catalog, mode=plan.spec.mode,
                                       completion_event_recorder=recorder)
    return values, operation, runtime


def mask_request(operation, packed=False):
    _, plan, resource, inspector = inspection_inputs(packed)
    mapping = AttentionOperatorMaskArgumentSpec(
        operation.fingerprint, "packed_allow_little_segments" if packed else "bool_allow_flat",
        "mask", "logical_offsets", "byte_offsets" if packed else None)
    binder = AttentionMaskPlanRunAdapterBinder(
        resource.payload, resource.owner, inspector, mapping, "npu:0", required_alignment=16)
    return plan, binder, inspector, resource.payload, weakref.ref(resource.owner)


class TransactionalMaskPlanCheckpoint(unittest.TestCase):
    def setUp(self):
        mask_calls[:] = []

    def test_masked_plan_run_and_unmasked_replan_use_the_existing_runtime(self):
        recorder = Recorder()
        _, operation, runtime = mask_runtime(recorder)
        plan, binder, inspector, payload, owner = mask_request(operation)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(len(inspector.calls), 1)
        self.assertEqual(mask_calls, [])
        self.assertIsNone(runtime.plan_state.spec.custom_mask.bit_order)
        self.assertEqual(runtime.run("q", ("k", "v"), return_lse=False), "output")
        self.assertIs(mask_calls[0][0], payload)
        self.assertEqual(mask_calls[0][1:], ((0, 3, 12, 12), None))
        self.assertEqual(len(inspector.calls), 2)
        first = runtime.call_retention.pending_tokens[0]
        del binder
        runtime.plan(replace(plan.spec, custom_mask=None), plan.metadata)
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertEqual(runtime.run("q", ("k", "v"), return_lse=True), ("output", "lse"))
        self.assertEqual(mask_calls[1], (None, None, None))
        self.assertEqual(runtime.last_lowered_call.retained_resources, ())
        recorder.events[0].ready = True
        runtime.call_retention.poll(first)
        gc.collect()
        self.assertIsNone(owner())

    def test_equivalent_replans_bind_new_payloads_to_new_generations(self):
        recorder = Recorder()
        _, operation, runtime = mask_runtime(recorder)
        first_plan, first_binder, _, first_payload, _ = mask_request(operation, packed=True)
        runtime.plan(first_plan.spec, first_plan.metadata, run_adapter_plan_binder=first_binder)
        first_generation = runtime.plan_state.generation
        runtime.run("q", ("k", "v"), return_lse=False)
        second_plan, second_binder, _, second_payload, _ = mask_request(operation, packed=True)
        runtime.plan(second_plan.spec, second_plan.metadata, run_adapter_plan_binder=second_binder)
        self.assertGreater(runtime.plan_state.generation, first_generation)
        runtime.run("q", ("k", "v"), return_lse=False)
        self.assertIs(mask_calls[0][0], first_payload)
        self.assertIs(mask_calls[1][0], second_payload)
        self.assertEqual(mask_calls[1][1:], ((0, 3, 12, 12), (0, 1, 3, 3)))
        tokens = runtime.call_retention.pending_tokens
        self.assertNotEqual(tokens[0].active_plan_fingerprint, tokens[1].active_plan_fingerprint)

    def test_bad_new_mask_preserves_plan_adapter_and_old_executable(self):
        _, operation, runtime = mask_runtime(Recorder())
        plan, binder, _, payload, _ = mask_request(operation)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        active = runtime.operator_session.active_plan
        generation = runtime.plan_state.generation
        session = runtime.operator_session
        _, bad_binder, bad_inspector, _, _ = mask_request(operation)
        bad_inspector.view = replace(bad_inspector.view, shape=(11,))
        with self.assertRaisesRegex(SchemaError, "element count"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=bad_binder)
        self.assertIs(runtime.operator_session, session)
        self.assertIs(runtime.operator_session.active_plan, active)
        self.assertEqual(runtime.plan_state.generation, generation)
        self.assertEqual(mask_calls, [])
        runtime.run("q", ("k", "v"), return_lse=False)
        self.assertIs(mask_calls[0][0], payload)

    def test_later_executor_binding_failure_rolls_back_prepared_mask(self):
        _, operation, runtime = mask_runtime(Recorder())
        plan, binder, _, payload, _ = mask_request(operation)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        old_session = runtime.operator_session
        _, next_binder, next_inspector, _, _ = mask_request(operation)
        with patch("flashinfer_npu.attention.operator_execution.AttentionInjectedCallableExecutor.bind_runtime",
                   side_effect=RuntimeError("executor binding failed")):
            with self.assertRaisesRegex(RuntimeError, "executor binding failed"):
                runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=next_binder)
        self.assertEqual(len(next_inspector.calls), 1)
        self.assertIs(runtime.operator_session, old_session)
        runtime.run("q", ("k", "v"), return_lse=False)
        self.assertIs(mask_calls[-1][0], payload)

    def test_missing_resource_binding_or_recorder_fails_before_package_resolution(self):
        values, operation, runtime = mask_runtime(None)
        plan, binder, inspector, _, _ = mask_request(operation)
        with self.assertRaisesRegex(AttentionStateError, "plan-bound resource adapter"):
            runtime.plan(plan.spec, plan.metadata)
        with self.assertRaisesRegex(AttentionStateError, "completion event recorder"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(values["events"], [])
        self.assertEqual(inspector.calls, [])
        self.assertFalse(runtime.is_planned)

    def test_mapping_and_encoding_drift_fail_before_tensor_inspection(self):
        _, operation, runtime = mask_runtime(Recorder())
        plan, binder, inspector, _, _ = mask_request(replace(operation, api_version="v2"))
        with self.assertRaisesRegex(SchemaError, "exact operation"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(inspector.calls, [])
        unpacked, _, _, _, _ = mask_request(operation)
        _, packed_binder, packed_inspector, _, _ = mask_request(operation, packed=True)
        with self.assertRaisesRegex(SchemaError, "encoding transformation"):
            runtime.plan(unpacked.spec, unpacked.metadata, run_adapter_plan_binder=packed_binder)
        self.assertEqual(packed_inspector.calls, [])

    def test_invalid_binder_result_never_publishes_a_candidate(self):
        _, operation, runtime = mask_runtime(Recorder())
        plan, _, _, _, _ = mask_request(operation)

        class WrongAdapter:
            provider_id = "other"
            operation_id = operation.operation_id

            def lower(self, active, request):
                raise AssertionError("wrong adapter must not be used")

        class BadBinder:
            requires_call_retention = True
            result = None

            def bind(self, base, active, selected):
                return self.result

        binder = BadBinder()
        with self.assertRaisesRegex(TypeError, "invalid adapter"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        binder.result = WrongAdapter()
        with self.assertRaisesRegex(SchemaError, "selected operation"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertFalse(runtime.is_planned)
        self.assertEqual(mask_calls, [])

    def test_unplanned_fork_preserves_recorder_with_separate_pending_registry(self):
        recorder = Recorder()
        _, operation, runtime = mask_runtime(recorder)
        plan, binder, _, _, _ = mask_request(operation)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        runtime.run("q", ("k", "v"), return_lse=False)
        tokens = runtime.call_retention.pending_tokens
        forked = runtime.fork_unplanned()
        self.assertFalse(forked.is_planned)
        self.assertIsNot(forked.call_retention, runtime.call_retention)
        self.assertEqual(forked.call_retention.pending_tokens, ())
        forked.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        forked.run("q", ("k", "v"), return_lse=False)
        self.assertEqual(len(recorder.events), 2)
        self.assertEqual(runtime.call_retention.pending_tokens, tokens)
        self.assertEqual(len(forked.call_retention.pending_tokens), 1)
