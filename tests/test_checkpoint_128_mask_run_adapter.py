import gc
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionLoweredOperatorCall, AttentionOperatorActivePlan,
    AttentionOperatorProviderSelection, AttentionOperatorRunRequest,
    AttentionPreparedOperatorPlan,
)
from flashinfer_npu.attention.operator_mask_binding import AttentionOperatorMaskRunAdapter
from flashinfer_npu.attention.operator_run import (
    AttentionOperatorRunTensorValidationAdapter, lower_attention_operator_run,
)
from flashinfer_npu.attention.operator_completion import AttentionOperatorCompletionValidator
from flashinfer_npu.attention.tensor_contract import (
    AttentionTensorAccessPolicy, TensorView, contiguous_strides,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import receipt_for
from tests.test_checkpoint_126_mask_resource_revalidation import prepared_source
from tests.test_checkpoint_127_mask_argument_binding import argument_spec, synthetic_operation


def active_for(plan, operation, token="synthetic-mask-plan"):
    receipt = receipt_for(plan)
    selection = AttentionOperatorProviderSelection(
        provider_id=operation.provider_id, provider_probe_fingerprint="a" * 64,
        provider_record_fingerprint="b" * 64, dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id, profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend)
    prepared = AttentionPreparedOperatorPlan(
        provider_id=operation.provider_id, provider_selection_fingerprint=selection.fingerprint,
        framework_plan_fingerprint=plan.fingerprint, framework_plan_generation=plan.generation,
        implementation_id=operation.operation_id, opaque_plan_token=token, opaque_state=None)
    return AttentionOperatorActivePlan(plan, receipt, selection, prepared)


class BaseAdapter:
    provider_id = "synthetic"

    def __init__(self):
        self.calls = 0
        self.extra = {}

    def lower(self, active, request):
        self.calls += 1
        call = AttentionLoweredOperatorCall(
            provider_id=self.provider_id, operation_id=active.prepared_plan.implementation_id,
            active_plan_fingerprint=active.fingerprint,
            positional_arguments=(("query", request.query),
                                  ("key", request.kv_cache[0]), ("value", request.kv_cache[1])),
            consumed_request_fields=request.consumed_fields)
        return replace(call, **self.extra)


def adapter_inputs(packed=False, operation=None):
    session, plan, inspected, inspector = prepared_source(packed)
    operation = operation or synthetic_operation()
    active = active_for(plan, operation)
    base = BaseAdapter()
    adapter = AttentionOperatorMaskRunAdapter(
        base, active, operation, inspected, argument_spec(operation, packed), inspector)
    request = AttentionOperatorRunRequest.from_active_plan(active, "q", ("k", "v"), return_lse=False)
    return session, active, inspected, inspector, operation, base, adapter, request


class MaskRunAdapterCheckpoint(unittest.TestCase):
    def test_tensor_and_completion_adapters_preserve_mask_protection(self):
        _, active, inspected, _, operation, base, _, request = adapter_inputs()
        plan = active.framework_plan

        def view(shape, dtype, storage, writable=False):
            return TensorView(shape, contiguous_strides(shape), dtype, "npu:0",
                              storage, 64, writable=writable)

        views = {
            "query": view(plan.expected_query_shape, plan.spec.q_dtype, "query"),
            "kv.key": view((12, 1, 1), plan.spec.kv_dtype, "key"),
            "kv.value": view((12, 1, 1), plan.spec.kv_dtype, "value"),
            "custom_mask": inspected.view,
            "output": view(plan.expected_output_shape, plan.spec.o_dtype, "output", True),
        }

        class Inspector:
            def to_view(self, tensor, *, name, writable=False):
                return views[name]

        inspector = Inspector()
        # Even a profile permitting general output/input aliasing must protect
        # the borrowed plan mask separately.
        policy = AttentionTensorAccessPolicy(permit_output_input_alias=True)
        tensors = AttentionOperatorRunTensorValidationAdapter(base, inspector, "npu:0", policy)
        adapter = AttentionOperatorMaskRunAdapter(
            tensors, active, operation, inspected, argument_spec(operation), inspector)
        call = lower_attention_operator_run(adapter, active, request)
        self.assertEqual(tuple(name for name, _ in call.validated_input_views),
                         ("query", "kv.key_storage", "kv.value_storage", "custom_mask"))
        validator = AttentionOperatorCompletionValidator(operation, active, inspector, policy, "npu:0")
        receipt = validator.validate(call, object())
        self.assertIn(("custom_mask", inspected.view.fingerprint), receipt.input_view_fingerprints)
        views["output"] = replace(views["output"], storage_id="query")
        validator.validate(call, object())
        views["output"] = replace(views["output"], storage_id=inspected.view.storage_id)
        with self.assertRaisesRegex(SchemaError, "output result cannot alias custom_mask"):
            validator.validate(call, object())

    def test_existing_lowering_pipeline_injects_mask_without_new_run_fields(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                _, active, inspected, inspector, _, base, adapter, request = adapter_inputs(packed)
                call = lower_attention_operator_run(adapter, active, request)
                self.assertIsInstance(call, AttentionLoweredOperatorCall)
                self.assertEqual(call.active_plan_fingerprint, active.fingerprint)
                self.assertEqual(call.positional_arguments, (("query", "q"), ("key", "k"), ("value", "v")))
                self.assertEqual(call.consumed_request_fields, request.consumed_fields)
                self.assertIs(dict(call.keyword_arguments)["mask"], inspected.resource.payload)
                self.assertEqual(dict(call.keyword_arguments)["logical_offsets"], (0, 3, 12, 12))
                self.assertEqual("byte_offsets" in dict(call.keyword_arguments), packed)
                self.assertEqual(call.validated_input_views, (("custom_mask", inspected.view),))
                self.assertIs(call.retained_resources[0].inspected, inspected)
                self.assertEqual((base.calls, len(inspector.calls)), (1, 1))

    def test_different_active_state_and_stale_request_fail_before_base_lowering(self):
        session, active, _, inspector, operation, base, adapter, request = adapter_inputs()
        framework = active.framework_plan
        for changed in (active_for(framework, operation, token="different-prepared-state"),
                        active_for(session.plan(framework.spec, framework.metadata), operation)):
            with self.assertRaisesRegex(SchemaError, "active plan"):
                adapter.lower(changed, AttentionOperatorRunRequest.from_active_plan(
                    changed, "q", ("k", "v"), return_lse=False))
        with self.assertRaisesRegex(SchemaError, "request.*active plan"):
            adapter.lower(active, replace(request, active_plan_fingerprint="c" * 64))
        self.assertEqual((base.calls, inspector.calls), (0, []))

    def test_argument_and_view_collisions_never_overwrite_even_none(self):
        _, active, inspected, inspector, _, base, adapter, request = adapter_inputs(packed=True)
        for changes in (
            {"keyword_arguments": (("mask", None),)},
            {"keyword_arguments": (("logical_offsets", (0,)),)},
            {"keyword_arguments": (("byte_offsets", (0,)),)},
            {"validated_input_views": (("custom_mask", inspected.view),)},
        ):
            with self.subTest(changes=tuple(changes)):
                base.extra = changes
                with self.assertRaisesRegex(SchemaError, "collides"):
                    adapter.lower(active, request)
        self.assertEqual(inspector.calls, [])

    def test_base_call_identity_and_signature_are_checked_before_mask_inspection(self):
        _, active, _, inspector, _, base, adapter, request = adapter_inputs()
        for changes, message in (
            ({"active_plan_fingerprint": "c" * 64}, "active plan"),
            ({"operation_id": "synthetic.other@v1"}, "implementation"),
            ({"provider_id": "other"}, "active plan"),
            ({"keyword_arguments": (("unknown", None),)}, "unknown keyword"),
            ({"consumed_request_fields": ()}, "consume"),
        ):
            with self.subTest(changes=changes):
                base.extra = changes
                with self.assertRaisesRegex(SchemaError, message):
                    adapter.lower(active, request)
        self.assertEqual(inspector.calls, [])

    def test_revalidation_failure_does_not_change_previously_lowered_calls(self):
        _, active, inspected, inspector, _, _, adapter, request = adapter_inputs()
        first = adapter.lower(active, request)
        inspector.view = replace(inspected.view, storage_id="different-storage")
        with self.assertRaisesRegex(SchemaError, "changed.*replan"):
            adapter.lower(active, request)
        self.assertIs(first.retained_resources[0].inspected, inspected)
        inspector.view = inspected.view
        self.assertIs(adapter.lower(active, request).retained_resources[0].inspected, inspected)

    def test_mutable_argument_must_not_alias_borrowed_mask(self):
        operation = replace(synthetic_operation(),
                            keyword_arguments=("mask", "logical_offsets", "byte_offsets", "out"),
                            mutable_arguments=("out",), output_buffer_argument="out")
        _, active, inspected, _, _, base, _, request = adapter_inputs(operation=operation)
        output = object()
        base.extra = {"keyword_arguments": (("out", output),), "mutable_argument_names": ("out",)}

        class Inspector:
            output_view = replace(inspected.view, writable=True)

            def to_view(self, tensor, *, name, writable=False):
                return self.output_view if tensor is output else inspected.view

        inspector = Inspector()
        adapter = AttentionOperatorMaskRunAdapter(
            base, active, operation, inspected, argument_spec(operation), inspector)
        request = replace(request, out=output)
        with self.assertRaisesRegex(SchemaError, "alias.*custom mask"):
            adapter.lower(active, request)
        inspector.output_view = replace(inspector.output_view, storage_id="separate-output")
        self.assertIs(dict(adapter.lower(active, request).keyword_arguments)["out"], output)
        inspector.output_view = replace(inspector.output_view, device="npu:1")
        with self.assertRaisesRegex(SchemaError, "planned device"):
            adapter.lower(active, request)

    def test_calls_keep_prior_owners_and_mask_owner_across_adapter_replacement(self):
        _, active, inspected, inspector, _, base, adapter, request = adapter_inputs()
        owner_ref = weakref.ref(inspected.resource.owner)
        prior_owner = object()
        base.extra = {"retained_resources": (prior_owner,)}
        call = adapter.lower(active, request)
        self.assertIs(call.retained_resources[0], prior_owner)
        copied_call = replace(call)
        del inspected, adapter, call
        gc.collect()
        self.assertIsNotNone(owner_ref())
        del copied_call
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertEqual(len(inspector.calls), 1)

    def test_retained_resources_are_normalized_and_excluded_from_repr(self):
        class OpaqueOwner:
            def __repr__(self):
                raise AssertionError("owner must not be represented")

        base = AttentionLoweredOperatorCall("synthetic", "op", "a" * 64, (("query", "q"),))
        owner = OpaqueOwner()
        retained = replace(base, retained_resources=[owner])
        self.assertIs(retained.retained_resources[0], owner)
        self.assertIsInstance(retained.retained_resources, tuple)
        self.assertEqual(repr(retained), repr(base))
        self.assertEqual(retained, base)
        for invalid in (None, (None,)):
            with self.assertRaisesRegex(SchemaError, "retained resources"):
                replace(base, retained_resources=invalid)
