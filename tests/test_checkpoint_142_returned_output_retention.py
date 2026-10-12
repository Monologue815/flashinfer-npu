import gc
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention.operator_retention import (
    AttentionCallRetentionError, AttentionOperatorCallRetention,
    AttentionRetainedInvocationError, execute_attention_retained_call,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_025_quantized_provider_run_lowering import quantized_input, query_input
from tests.test_checkpoint_129_call_retention import Event, Recorder, masked_call
from tests.test_checkpoint_135_quantized_mask_composition import components, result_pair, results, calls


class Output:
    def __iter__(self):
        raise AssertionError("tensor contents must not be read")

    def __bool__(self):
        raise AssertionError("tensor truthiness must not be evaluated")


class ReturnedOutputRetentionCheckpoint(unittest.TestCase):
    def setUp(self):
        calls.clear()
        results.clear()

    def tearDown(self):
        calls.clear()
        results.clear()

    def test_provider_allocated_outputs_survive_caller_drop_until_completion(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        references = []
        def invoke(_):
            output, lse = Output(), Output()
            references.extend((weakref.ref(output), weakref.ref(lse)))
            return output, lse
        result, token = execute_attention_retained_call(registry, call, invoke, recorder)
        del result
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        recorder.events[0].ready = True
        registry.poll(token)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_mutating_return_list_does_not_discard_its_original_output_members(self):
        class ReturnList(list):
            def __iter__(self):
                raise AssertionError("custom iteration must not run during retention")
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        output = Output()
        reference = weakref.ref(output)
        value = ReturnList((output,))
        actual, token = execute_attention_retained_call(registry, call, lambda _: value, recorder)
        self.assertIs(actual, value)
        actual.clear()
        del output
        gc.collect()
        self.assertIsNotNone(reference())
        recorder.events[0].ready = True
        registry.poll(token)
        gc.collect()
        self.assertIsNone(reference())

    def test_recording_failure_keeps_returned_allocations_without_exception_ownership(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        recorder.failure = RuntimeError("synthetic record failure")
        call, _ = masked_call()
        references = []
        def invoke(_):
            value = Output()
            references.append(weakref.ref(value))
            return value
        def attempt():
            try:
                execute_attention_retained_call(registry, call, invoke, recorder)
            except AttentionRetainedInvocationError as error:
                return error.token
            self.fail("recording must fail")
        token = attempt()
        recorder.failure = None  # Drop the synthetic exception's traceback owner.
        gc.collect()
        self.assertIsNotNone(references[0]())
        self.assertEqual(registry.collect_completed().unrecorded, (token,))
        registry.bind_event(token, Event(token, ready=True))
        registry.poll(token)
        gc.collect()
        self.assertIsNone(references[0]())

    def test_result_validation_failure_keeps_provider_output_until_runtime_close(self):
        runtime, _, recorder, inspector, _ = components()
        plan = runtime.plan_state
        output, _ = result_pair(runtime)
        output.tensor_view = replace(output.tensor_view, shape=(1,), strides=(1,))
        reference = weakref.ref(output)
        results.append(output)
        del output
        with self.assertRaisesRegex(SchemaError, "shape must be"):
            runtime.run(query_input(plan), quantized_input(plan.spec.kv_quant_spec), return_lse=False)
        inspector.calls.clear()
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(len(runtime.call_retention.pending_tokens), 1)
        recorder.events[0].ready = True
        runtime.close()
        gc.collect()
        self.assertIsNone(reference())

    def test_results_cannot_be_replaced_or_bound_after_event_or_release(self):
        registry = AttentionOperatorCallRetention()
        call, _ = masked_call()
        token = registry.retain(call)
        registry.retain_result(token, None)
        with self.assertRaisesRegex(AttentionCallRetentionError, "bind once"):
            registry.retain_result(token, Output())
        second = registry.retain(call)
        registry.bind_event(second, Event(second, ready=True))
        with self.assertRaisesRegex(AttentionCallRetentionError, "before its event"):
            registry.retain_result(second, Output())
        registry.poll(second)
        with self.assertRaisesRegex(AttentionCallRetentionError, "unknown"):
            registry.retain_result(second, Output())

    def test_independent_returns_release_only_after_their_own_events(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        references, tokens = [], []
        for _ in range(2):
            value = Output()
            references.append(weakref.ref(value))
            actual, token = execute_attention_retained_call(registry, call, lambda _: value, recorder)
            tokens.append(token)
            del actual, value
        recorder.events[1].ready = True
        registry.collect_completed()
        gc.collect()
        self.assertIsNotNone(references[0]())
        self.assertIsNone(references[1]())
        self.assertEqual(registry.pending_tokens, (tokens[0],))
        recorder.events[0].ready = True
        registry.collect_completed()
        gc.collect()
        self.assertIsNone(references[0]())

    def test_query_failure_preserves_returned_output_until_later_completion(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        output = Output()
        reference = weakref.ref(output)
        result, token = execute_attention_retained_call(registry, call, lambda _: output, recorder)
        del result, output
        def fail():
            raise RuntimeError("synthetic query failure")
        recorder.events[0].action = fail
        report = registry.collect_completed()
        self.assertEqual(len(report.failures), 1)
        gc.collect()
        self.assertIsNotNone(reference())
        recorder.events[0].action = None
        self.assertFalse(registry.poll(token))
        recorder.events[0].ready = True
        registry.poll(token)
        gc.collect()
        self.assertIsNone(reference())
