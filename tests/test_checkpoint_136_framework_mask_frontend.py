import unittest
from dataclasses import replace

from flashinfer_npu.attention.frontend import adapt_framework_batch_custom_mask
from flashinfer_npu.attention.operator_mask_binding import (
    AttentionMaskPlanRunAdapterBinder, AttentionOperatorMaskArgumentSpec,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_022_operator_runtime_bootstrap import FakeTensorMetadataInspector
from tests.test_checkpoint_025_quantized_provider_run_lowering import metadata_tensor
from tests.test_checkpoint_124_mask_plan_resources import mask_plan
from tests.test_checkpoint_129_call_retention import Recorder
from tests.test_checkpoint_131_transactional_mask_plan import mask_calls, mask_runtime


class Unreadable:
    def __getattr__(self, name):
        raise AssertionError("unexpected tensor access: " + name)

    def __bool__(self):
        raise AssertionError("tensor truthiness must not be evaluated")

    def __iter__(self):
        raise AssertionError("tensor contents must not be iterated")


class MaskTensor(Unreadable):
    def __init__(self, shape, dtype="bool", device="npu:0"):
        self.shape, self.dtype, self.device = shape, dtype, device


class IntegerLike:
    def __index__(self):
        return 3


def adapt(custom=None, packed=None, sizes=(3, 9, 0), device="npu:0"):
    return adapt_framework_batch_custom_mask(
        custom, packed, segment_sizes=sizes, device=device)


class FrameworkMaskFrontendCheckpoint(unittest.TestCase):
    def setUp(self):
        mask_calls.clear()

    def test_absent_masks_do_not_inspect_unused_metadata(self):
        self.assertEqual(adapt(sizes=Unreadable(), device=Unreadable()), (None, None))

    def test_bool_and_packed_payloads_are_preserved_without_content_access(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                payload = MaskTensor((3 if packed else 12,), "torch.uint8" if packed else "torch.bool")
                spec, result = adapt(packed=payload) if packed else adapt(custom=payload)
                self.assertIs(result, payload)
                self.assertEqual(spec.numel, 3 if packed else 12)
                self.assertEqual(spec.packed, packed)

    def test_packed_precedence_never_inspects_the_ignored_mask(self):
        packed = MaskTensor((3,), "uint8")
        spec, result = adapt(Unreadable(), packed)
        self.assertTrue(spec.packed)
        self.assertIs(result, packed)
        # Invalid packed input is not silently replaced with the bool alternative.
        with self.assertRaisesRegex(SchemaError, "packed_custom_mask shape"):
            adapt(Unreadable(), MaskTensor((2,), "uint8"))

    def test_packed_size_rounds_each_segment_separately(self):
        with self.assertRaisesRegex(SchemaError, "shape must be \\(3,\\)"):
            adapt(packed=MaskTensor((2,), "uint8"))
        spec, _ = adapt(packed=MaskTensor((3,), "uint8"), sizes=(1, 0, 1, 1))
        self.assertEqual(spec.numel, 3)

    def test_segment_sizes_require_nonnegative_integers_before_tensor_access(self):
        for sizes in ((True,), (3.0,), ("3",), (-1,), None):
            with self.subTest(sizes=sizes), self.assertRaises(SchemaError):
                adapt(Unreadable(), sizes=sizes)
        spec, _ = adapt(MaskTensor((3,)), sizes=(IntegerLike(),))
        self.assertEqual(spec.numel, 3)
        for sizes in ((), (0, 0)):
            for packed in (False, True):
                payload = MaskTensor((0,), "uint8" if packed else "bool")
                spec, _ = adapt(packed=payload, sizes=sizes) if packed else adapt(payload, sizes=sizes)
                self.assertEqual(spec.numel, 0)

    def test_rank_shape_dtype_and_device_fail_without_conversion(self):
        for payload, message in (
            (MaskTensor((3, 4)), "shape"),
            (MaskTensor((12.0,)), "integer shape"),
            (MaskTensor((True,)), "integer shape"),
            (MaskTensor((-1,)), "negative"),
            (MaskTensor((12,), "uint8"), "dtype"),
            (MaskTensor((12,), device="npu:1"), "workspace device"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(SchemaError, message):
                adapt(payload)
        for device in (None, "", 0):
            with self.subTest(device=device), self.assertRaisesRegex(SchemaError, "workspace device"):
                adapt(Unreadable(), device=device)

    def test_frontend_result_reaches_existing_private_plan_and_run_chain(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                recorder = Recorder()
                _, operation, runtime = mask_runtime(recorder)
                _, original = mask_plan(packed=packed)
                payload = MaskTensor((3 if packed else 12,), "uint8" if packed else "bool")
                payload.tensor_view = metadata_tensor("frontend-mask", payload.shape, payload.dtype).tensor_view
                spec, selected = adapt(packed=payload) if packed else adapt(payload)
                mapping = AttentionOperatorMaskArgumentSpec(
                    operation.fingerprint,
                    "packed_allow_little_segments" if packed else "bool_allow_flat",
                    "mask", "logical_offsets", "byte_offsets" if packed else None)
                binder = AttentionMaskPlanRunAdapterBinder(
                    selected, selected, FakeTensorMetadataInspector(), mapping, "npu:0")
                runtime.plan(replace(original.spec, custom_mask=spec), original.metadata,
                             run_adapter_plan_binder=binder)
                self.assertEqual(runtime.run("q", ("k", "v"), return_lse=False), "output")
                self.assertIs(mask_calls[-1][0], payload)
                self.assertEqual(mask_calls[-1][1:], ((0, 3, 12, 12), (0, 1, 3, 3) if packed else None))
                self.assertEqual(len(runtime.call_retention.pending_tokens), 1)
                recorder.events[0].ready = True
                runtime.close()

    def test_frontend_shape_acceptance_does_not_bypass_storage_inspection(self):
        _, operation, runtime = mask_runtime(Recorder())
        _, original = mask_plan()
        payload = MaskTensor((12,))
        payload.tensor_view = metadata_tensor("noncontiguous-mask", (12,), "bool").tensor_view
        payload.tensor_view = replace(payload.tensor_view, strides=(2,), storage_nbytes=64)
        spec, selected = adapt(payload)
        mapping = AttentionOperatorMaskArgumentSpec(
            operation.fingerprint, "bool_allow_flat", "mask", "logical_offsets")
        binder = AttentionMaskPlanRunAdapterBinder(
            selected, selected, FakeTensorMetadataInspector(), mapping, "npu:0")
        with self.assertRaisesRegex(SchemaError, "contiguous"):
            runtime.plan(replace(original.spec, custom_mask=spec), original.metadata,
                         run_adapter_plan_binder=binder)
        self.assertFalse(runtime.is_planned)
        self.assertEqual(mask_calls, [])
        self.assertEqual(runtime.call_retention.pending_tokens, ())
