import gc
import threading
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention.operator_retention import (
    AttentionCallRetentionError, AttentionOperatorCallRetention,
    AttentionRetainedInvocationError, execute_attention_retained_call,
)
from tests.test_checkpoint_128_mask_run_adapter import adapter_inputs


def masked_call():
    _, active, inspected, _, _, _, adapter, request = adapter_inputs(packed=True)
    return adapter.lower(active, request), weakref.ref(inspected.resource.owner)


class Event:
    def __init__(self, token, ready=False):
        self.token = token
        self.ready = ready
        self.queries = 0
        self.action = None

    def query(self):
        self.queries += 1
        return self.action() if self.action is not None else self.ready


class Recorder:
    def __init__(self):
        self.events = []
        self.failure = None

    def record(self, token):
        if self.failure is not None:
            raise self.failure
        event = Event(token)
        self.events.append(event)
        return event


class CallRetentionCheckpoint(unittest.TestCase):
    def test_retention_precedes_execution_and_survives_python_return(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, owner = masked_call()
        result = object()

        def invoke(actual):
            self.assertIs(actual, call)
            self.assertEqual(len(registry.pending_tokens), 1)
            self.assertEqual(recorder.events, [])
            return result

        returned, token = execute_attention_retained_call(registry, call, invoke, recorder)
        self.assertIs(returned, result)
        self.assertEqual(recorder.events[0].queries, 0)
        del call
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertFalse(registry.poll(token))
        self.assertIsNotNone(owner())
        recorder.events[0].ready = True
        self.assertTrue(registry.poll(token))
        gc.collect()
        self.assertIsNone(owner())
        self.assertEqual(registry.pending_tokens, ())

    def test_same_plan_runs_have_distinct_tokens_and_finish_out_of_order(self):
        registry = AttentionOperatorCallRetention()
        first_call, first_owner = masked_call()
        second_call, second_owner = masked_call()
        first, second = registry.retain(first_call), registry.retain(second_call)
        self.assertEqual(first.active_plan_fingerprint, second.active_plan_fingerprint)
        self.assertNotEqual(first.invocation_id, second.invocation_id)
        first_event, second_event = Event(first), Event(second, True)
        registry.bind_event(first, first_event)
        registry.bind_event(second, second_event)
        del first_call, second_call
        self.assertTrue(registry.poll(second))
        gc.collect()
        self.assertIsNone(second_owner())
        self.assertIsNotNone(first_owner())
        self.assertEqual(registry.pending_tokens, (first,))
        first_event.ready = True
        registry.poll(first)
        gc.collect()
        self.assertIsNone(first_owner())

    def test_replanned_calls_do_not_release_an_older_invocation(self):
        registry = AttentionOperatorCallRetention()
        call, owner = masked_call()
        first = registry.retain(call)
        second = registry.retain(replace(call, active_plan_fingerprint="d" * 64))
        registry.bind_event(first, Event(first))
        registry.bind_event(second, Event(second, True))
        del call
        registry.poll(second)
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertEqual(registry.pending_tokens, (first,))

    def test_execution_failure_still_records_completion_for_partial_work(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, owner = masked_call()
        failure = RuntimeError("submitted then failed")

        def invoke(actual):
            raise failure

        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            execute_attention_retained_call(registry, call, invoke, recorder)
        error = caught.exception
        self.assertIs(error.invocation_error, failure)
        self.assertIsNone(error.recording_error)
        self.assertEqual(registry.pending_tokens, (error.token,))
        self.assertIsNotNone(owner())
        self.assertFalse(registry.poll(error.token))
        recorder.events[0].ready = True
        self.assertTrue(registry.poll(error.token))

    def test_record_failure_requires_recovery_proof_and_cannot_close(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, owner = masked_call()
        recorder.failure = RuntimeError("event recording unavailable")
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            execute_attention_retained_call(registry, call, lambda actual: object(), recorder)
        error = caught.exception
        self.assertIsNone(error.invocation_error)
        self.assertIs(error.recording_error, recorder.failure)
        self.assertIsNotNone(owner())
        with self.assertRaisesRegex(AttentionCallRetentionError, "no completion event"):
            registry.poll(error.token)
        with self.assertRaisesRegex(AttentionCallRetentionError, "still retain"):
            registry.close()
        registry.bind_event(error.token, Event(error.token, True))
        registry.poll(error.token)
        registry.close()

    def test_execution_and_recording_errors_both_remain_available(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        recorder.failure = RuntimeError("recording failed")
        invocation_error = RuntimeError("execution failed")

        def invoke(actual):
            raise invocation_error

        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            execute_attention_retained_call(registry, call, invoke, recorder)
        self.assertIs(caught.exception.invocation_error, invocation_error)
        self.assertIs(caught.exception.recording_error, recorder.failure)
        self.assertEqual(registry.pending_tokens, (caught.exception.token,))

    def test_mismatched_reused_or_rebound_events_cannot_release_a_call(self):
        registry = AttentionOperatorCallRetention()
        call, _ = masked_call()
        first, second = registry.retain(call), registry.retain(call)
        event = Event(first, True)
        with self.assertRaisesRegex(AttentionCallRetentionError, "this invocation"):
            registry.bind_event(second, event)
        registry.bind_event(first, event)
        with self.assertRaisesRegex(AttentionCallRetentionError, "already has"):
            registry.bind_event(first, Event(first, True))
        event.token = second
        with self.assertRaisesRegex(AttentionCallRetentionError, "another invocation"):
            registry.bind_event(second, event)
        with self.assertRaisesRegex(AttentionCallRetentionError, "identity changed"):
            registry.poll(first)
        self.assertEqual(event.queries, 0)
        event.token = first
        registry.poll(first)
        with self.assertRaisesRegex(AttentionCallRetentionError, "unknown"):
            registry.poll(first)
        with self.assertRaisesRegex(AttentionCallRetentionError, "mismatched"):
            registry.poll(replace(second, operation_id="different"))
        self.assertEqual(registry.pending_tokens, (second,))

    def test_query_errors_non_boolean_and_identity_drift_preserve_owners(self):
        registry = AttentionOperatorCallRetention()
        call, owner = masked_call()
        token = registry.retain(call)
        event = Event(token)
        registry.bind_event(token, event)
        del call
        for value in (1, "complete", None):
            event.ready = value
            with self.assertRaisesRegex(AttentionCallRetentionError, "exact boolean"):
                registry.poll(token)
            self.assertIsNotNone(owner())

        def fail():
            raise RuntimeError("query unavailable")

        event.action = fail
        with self.assertRaisesRegex(RuntimeError, "query unavailable"):
            registry.poll(token)

        def drift():
            event.token = replace(token, invocation_id="different")
            return True

        event.action = drift
        with self.assertRaisesRegex(AttentionCallRetentionError, "identity changed"):
            registry.poll(token)
        self.assertIsNotNone(owner())
        event.token, event.action, event.ready = token, None, True
        registry.poll(token)
        gc.collect()
        self.assertIsNone(owner())

    def test_concurrent_queries_do_not_block_independent_completion(self):
        registry = AttentionOperatorCallRetention()
        call, _ = masked_call()
        first, second = registry.retain(call), registry.retain(call)
        entered, resume = threading.Event(), threading.Event()
        event = Event(first)

        def query():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError("test query was not resumed")
            return True

        event.action = query
        registry.bind_event(first, event)
        registry.bind_event(second, Event(second, True))
        results, failures = [], []

        def poll():
            try:
                results.append(registry.poll(first))
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=poll)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(AttentionCallRetentionError, "already in progress"):
                registry.poll(first)
            self.assertTrue(registry.poll(second))
            self.assertEqual(registry.pending_tokens, (first,))
        finally:
            resume.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results, [True])
        self.assertEqual(registry.pending_tokens, ())

    def test_closed_registry_rejects_execution_before_invocation_or_recording(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        registry.close()
        calls = []
        with self.assertRaisesRegex(AttentionCallRetentionError, "closed"):
            execute_attention_retained_call(registry, call, calls.append, recorder)
        self.assertEqual(calls, [])
        self.assertEqual(recorder.events, [])
