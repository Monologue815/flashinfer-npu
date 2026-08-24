import inspect
import unittest
from concurrent.futures import ThreadPoolExecutor

import flashinfer_npu
from flashinfer_npu.attention import (
    ATTENTION_SINGLE_JIT_TEMP_BYTES,
    AttentionJITTemporaryBuffer,
    AttentionProtocolRoute,
    AttentionProtocolState,
    ReferenceBuffer,
    ReferenceTensor,
    capture_attention_protocol,
)
from flashinfer_npu.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer_npu.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer_npu.runtime import SchemaError


def tensor(value, dtype="float32"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device="cpu")


class RecordingJITModule:
    def __init__(self):
        self.calls = []

    def run(self, *args):
        self.calls.append(args)
        output = args[4]
        lse = args[5]
        self._fill(output, 1.0)
        if lse is not None:
            self._fill(lse, 10.0)

    @staticmethod
    def _fill(buffer, start):
        count = 1
        for dim in buffer.shape:
            count *= dim
        buffer.data = tuple(start + index for index in range(count))


class AttentionSingleJITProtocolTests(unittest.TestCase):
    def test_public_signatures_and_package_exports_match_upstream_surface(self):
        self.assertEqual(
            list(
                inspect.signature(
                    single_prefill_with_kv_cache_with_jit_module
                ).parameters
            ),
            [
                "jit_module",
                "q",
                "k",
                "v",
                "args",
                "kv_layout",
                "mask_mode",
                "window_left",
                "return_lse",
            ],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    single_decode_with_kv_cache_with_jit_module
                ).parameters
            ),
            [
                "jit_module",
                "q",
                "k",
                "v",
                "args",
                "kv_layout",
                "window_left",
                "return_lse",
            ],
        )
        self.assertIs(
            flashinfer_npu.single_prefill_with_kv_cache_with_jit_module,
            single_prefill_with_kv_cache_with_jit_module,
        )
        self.assertIs(
            flashinfer_npu.single_decode_with_kv_cache_with_jit_module,
            single_decode_with_kv_cache_with_jit_module,
        )

    def test_prefill_allocates_exact_buffers_and_forwards_upstream_codes(self):
        module = RecordingJITModule()
        q = tensor([[[0.0], [0.0]], [[0.0], [0.0]]])
        k = tensor([[[0.0], [0.0], [0.0]]])
        v = tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
        output, lse = single_prefill_with_kv_cache_with_jit_module(
            module,
            q,
            k,
            v,
            "extra-buffer",
            7,
            kv_layout="HND",
            mask_mode=2,
            window_left=3,
            return_lse=True,
        )

        self.assertEqual(output.shape, (2, 2, 2))
        self.assertEqual(lse.shape, (2, 2))
        self.assertEqual(output.data, tuple(float(i) for i in range(1, 9)))
        call = module.calls[0]
        self.assertIs(call[0], q)
        self.assertIs(call[1], k)
        self.assertIs(call[2], v)
        self.assertIsInstance(call[3], AttentionJITTemporaryBuffer)
        self.assertEqual(call[3].capacity_bytes, ATTENTION_SINGLE_JIT_TEMP_BYTES)
        self.assertIsInstance(call[4], ReferenceBuffer)
        self.assertIsInstance(call[5], ReferenceBuffer)
        self.assertEqual(call[6:9], (2, 1, 3))
        self.assertEqual(call[9:], ("extra-buffer", 7))

    def test_decode_discards_requested_lse_but_forwards_its_buffer(self):
        module = RecordingJITModule()
        q = tensor([[0.0], [0.0]])
        k = tensor([[[0.0]], [[0.0]], [[0.0]]])
        v = tensor([[[0.0]], [[0.0]], [[0.0]]])
        output = single_decode_with_kv_cache_with_jit_module(
            module,
            q,
            k,
            v,
            "extra",
            kv_layout="NHD",
            window_left=1,
            return_lse=True,
        )

        self.assertIsInstance(output, ReferenceTensor)
        self.assertEqual(output.shape, (2, 1))
        call = module.calls[0]
        self.assertEqual(call[5].shape, (2,))
        self.assertEqual(call[6:8], (0, 1))
        self.assertEqual(call[8:], ("extra",))

    def test_invalid_module_mask_window_and_mutated_output_are_rejected(self):
        q = tensor([[[0.0]]])
        k = tensor([[[0.0]]])
        v = tensor([[[0.0]]])
        with self.assertRaisesRegex(TypeError, "callable run"):
            single_prefill_with_kv_cache_with_jit_module(object(), q, k, v)
        with self.assertRaisesRegex(SchemaError, "mask_mode"):
            single_prefill_with_kv_cache_with_jit_module(
                RecordingJITModule(), q, k, v, mask_mode=True
            )
        with self.assertRaisesRegex(SchemaError, "window_left"):
            single_prefill_with_kv_cache_with_jit_module(
                RecordingJITModule(), q, k, v, window_left=-2
            )

        class CorruptModule:
            def run(self, *args):
                args[4].data = (1.0, 2.0)

        with self.assertRaisesRegex(SchemaError, "invalid output buffer"):
            single_prefill_with_kv_cache_with_jit_module(
                CorruptModule(), q, k, v
            )

    def test_opt_in_capture_automatically_records_success_and_failure(self):
        q = tensor([[[0.0]]])
        k = tensor([[[0.0]]])
        v = tensor([[[0.0]]])
        with capture_attention_protocol("jit-auto") as capture:
            single_prefill_with_kv_cache_with_jit_module(
                RecordingJITModule(), q, k, v
            )

            class FailingModule:
                def run(self, *args):
                    raise RuntimeError("synthetic JIT failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic JIT"):
                single_prefill_with_kv_cache_with_jit_module(
                    FailingModule(), q, k, v
                )

        self.assertEqual(len(capture.traces), 2)
        self.assertEqual(capture.incomplete_count, 0)
        self.assertEqual(
            tuple(event.state for event in capture.traces[0].events),
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.INVOKED,
                AttentionProtocolState.COMPLETED,
                AttentionProtocolState.RELEASED,
            ),
        )
        self.assertEqual(
            tuple(event.state for event in capture.traces[1].events),
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.INVOKED,
                AttentionProtocolState.FAILED_SYNC,
                AttentionProtocolState.RELEASED,
            ),
        )
        self.assertTrue(
            all(trace.route == AttentionProtocolRoute.SINGLE_JIT for trace in capture.traces)
        )
        corpus = capture.to_corpus("jit-auto-corpus")
        self.assertEqual(corpus.route_counts, {"single_jit": 2, "provider": 0})

    def test_nested_capture_restores_outer_context(self):
        q = tensor([[[0.0]]])
        k = tensor([[[0.0]]])
        v = tensor([[[0.0]]])
        with capture_attention_protocol("outer") as outer:
            single_prefill_with_kv_cache_with_jit_module(
                RecordingJITModule(), q, k, v
            )
            with capture_attention_protocol("inner") as inner:
                single_prefill_with_kv_cache_with_jit_module(
                    RecordingJITModule(), q, k, v
                )
            single_prefill_with_kv_cache_with_jit_module(
                RecordingJITModule(), q, k, v
            )
        self.assertEqual(
            tuple(trace.trace_id for trace in outer.traces), ("outer:1", "outer:2")
        )
        self.assertEqual(tuple(trace.trace_id for trace in inner.traces), ("inner:1",))

    def test_thread_contexts_do_not_cross_publish_traces(self):
        def capture_one(index):
            q = tensor([[[0.0]]])
            k = tensor([[[0.0]]])
            v = tensor([[[0.0]]])
            with capture_attention_protocol("thread-%d" % index) as capture:
                single_prefill_with_kv_cache_with_jit_module(
                    RecordingJITModule(), q, k, v
                )
            return tuple(trace.trace_id for trace in capture.traces)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(capture_one, range(8)))
        self.assertEqual(
            results,
            tuple(("thread-%d:1" % index,) for index in range(8)),
        )


if __name__ == "__main__":
    unittest.main()
