import gc
import unittest
import weakref

from flashinfer_npu.attention.operator_retention import (
    AttentionOperatorCallRetention, AttentionRetainedInvocationError,
    execute_attention_retained_call,
)
from tests.test_checkpoint_129_call_retention import Event, Recorder, masked_call


class Executor:
    def execute(self, call):
        return None

    def __call__(self, call):
        return self.execute(call)

    def __repr__(self):
        raise AssertionError("opaque execution owners must not be formatted")


class InvocationOwnerCheckpoint(unittest.TestCase):
    def test_bound_executor_survives_replacement_until_its_event(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        executor = Executor()
        reference = weakref.ref(executor)
        _, token = execute_attention_retained_call(registry, call, executor.execute, recorder)
        executor = Executor()  # The active runtime may now select another executor.
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertFalse(registry.poll(token))
        recorder.events[0].ready = True
        registry.poll(token)
        gc.collect()
        self.assertIsNone(reference())

    def test_callable_objects_are_retained_independently_and_not_in_reports(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        references = []
        for _ in range(2):
            executor = Executor()
            references.append(weakref.ref(executor))
            execute_attention_retained_call(registry, call, executor, recorder)
            del executor
        recorder.events[1].ready = True
        report = registry.collect_completed()
        repr(report)
        gc.collect()
        self.assertIsNotNone(references[0]())
        self.assertIsNone(references[1]())
        recorder.events[0].ready = True
        registry.collect_completed()
        gc.collect()
        self.assertIsNone(references[0]())

    def test_record_failure_retains_executor_without_exception_traceback(self):
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        recorder.failure = RuntimeError("synthetic record fault")
        executor = Executor()
        reference = weakref.ref(executor)
        try:
            execute_attention_retained_call(registry, call, executor.execute, recorder)
        except AttentionRetainedInvocationError as error:
            token = error.token
        else:
            self.fail("recording must fail")
        recorder.failure = None
        del executor
        gc.collect()
        self.assertIsNotNone(reference())
        registry.bind_event(token, Event(token, ready=True))
        registry.poll(token)
        gc.collect()
        self.assertIsNone(reference())

    def test_partial_invocation_failure_retains_executor_until_completion(self):
        class FailingExecutor(Executor):
            def execute(self, call):
                raise RuntimeError("synthetic partial submission")
        registry, recorder = AttentionOperatorCallRetention(), Recorder()
        call, _ = masked_call()
        executor = FailingExecutor()
        reference = weakref.ref(executor)
        try:
            execute_attention_retained_call(registry, call, executor.execute, recorder)
        except AttentionRetainedInvocationError as error:
            token = error.token
        else:
            self.fail("invocation must fail")
        del executor
        gc.collect()
        self.assertIsNotNone(reference())
        recorder.events[0].ready = True
        registry.poll(token)
        gc.collect()
        self.assertIsNone(reference())
