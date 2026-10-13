import gc
import unittest
import weakref

from flashinfer_npu.attention import AttentionMode, AttentionStateError
from flashinfer_npu.attention.operator_resolver import AttentionOperatorRuntime
from flashinfer_npu.attention.operator_single_owner import (
    AttentionSingleRuntimeOwner, AttentionSingleOwnerClosePending,
)
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_043_provider_workspace_reset import runtime_registry
from tests.test_checkpoint_129_call_retention import Recorder, Event, masked_call


class SingleOwnerCheckpoint(unittest.TestCase):
    def setUp(self):
        self.owner = AttentionSingleRuntimeOwner()
        self.values = build_components()

    def runtime(self, mode=AttentionMode.SINGLE_PREFILL, tracked=True):
        return AttentionOperatorRuntime(
            "npu:0", runtime_registry(self.values), self.values["catalog"], mode=mode,
            completion_event_recorder=Recorder() if tracked else None)

    def pending(self, runtime, ready=False):
        # This layer tests ownership, not lowering or execution admission.
        call, _ = masked_call()
        token = runtime.call_retention.retain(call)
        event = Event(token, ready)
        runtime.call_retention.bind_event(token, event)
        return token, event

    def test_preparing_empty_runtime_is_not_closed_then_sealed_runtime_is_released(self):
        for mode in (AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE):
            runtime = self.runtime(mode)
            key = self.owner.adopt(runtime)
            reference = weakref.ref(runtime)
            self.assertEqual(self.owner.collect_completed().preparing, (key,))
            self.assertFalse(runtime.is_closing)
            self.owner.seal(key)
            self.owner.seal(key)
            del runtime
            report = self.owner.collect_completed()
            gc.collect()
            self.assertEqual(report.released, (key,))
            self.assertIsNone(reference())
            self.assertFalse(self.owner.is_closed)

    def test_pending_and_failed_queries_do_not_block_independent_release(self):
        first, second = self.runtime(), self.runtime()
        first_id, second_id = self.owner.adopt(first), self.owner.adopt(second)
        _, event = self.pending(first)
        self.pending(second, True)
        self.owner.seal(first_id)
        self.owner.seal(second_id)
        def fail():
            raise RuntimeError("synthetic query fault")
        event.action = fail
        report = self.owner.collect_completed()
        self.assertEqual(report.released, (second_id,))
        self.assertEqual(report.pending, (first_id,))
        self.assertEqual(report.failures[0][1], "AttentionCallCompletionCollectionError")
        self.assertTrue(all(isinstance(value, str) for value in report.failures[0]))
        event.action, event.ready = None, True
        self.assertEqual(self.owner.collect_completed().released, (first_id,))

    def test_shutdown_waits_for_preparation_and_allows_finally_sealing(self):
        runtime = self.runtime()
        key = self.owner.adopt(runtime)
        with self.assertRaises(AttentionSingleOwnerClosePending) as caught:
            self.owner.close()
        self.assertEqual(caught.exception.report.preparing, (key,))
        self.assertFalse(runtime.is_closing)
        with self.assertRaises(AttentionStateError):
            self.owner.adopt(self.runtime())
        with self.assertRaises(AttentionStateError):
            self.owner.collect_completed()
        _, event = self.pending(runtime)
        self.owner.seal(key)
        with self.assertRaises(AttentionSingleOwnerClosePending):
            self.owner.close()
        event.ready = True
        self.owner.close()
        self.owner.close()
        self.assertTrue(self.owner.is_closed)

    def test_unrecorded_call_retains_recovery_access(self):
        runtime = self.runtime()
        key = self.owner.adopt(runtime)
        call, _ = masked_call()
        token = runtime.call_retention.retain(call)
        self.owner.seal(key)
        self.assertEqual(self.owner.collect_completed().pending, (key,))
        self.owner.get_runtime(key).call_retention.bind_event(token, Event(token, True))
        self.assertEqual(self.owner.collect_completed().released, (key,))
        with self.assertRaises(KeyError):
            self.owner.seal(key)

    def test_adoption_validation_does_not_change_owner(self):
        for runtime in (self.runtime(AttentionMode.BATCH_DECODE_PAGED), self.runtime(tracked=False)):
            with self.assertRaises(AttentionStateError):
                self.owner.adopt(runtime)
        with self.assertRaises(TypeError):
            self.owner.adopt(object())
        runtime = self.runtime()
        key = self.owner.adopt(runtime)
        with self.assertRaises(AttentionStateError):
            self.owner.adopt(runtime)
        self.assertEqual(self.owner.owned_runtime_ids, (key,))
        self.owner.seal(key)
        self.owner.close()

    def test_reentrant_close_is_rejected_without_sealing_preparing_requests(self):
        first, second = self.runtime(), self.runtime()
        first_id, second_id = self.owner.adopt(first), self.owner.adopt(second)
        _, event = self.pending(first)
        self.owner.seal(first_id)
        event.action = self.owner.close
        report = self.owner.collect_completed()
        self.assertEqual(report.preparing, (second_id,))
        self.assertEqual(report.pending, (first_id,))
        self.assertTrue(report.failures)
        # Reentrant shutdown did not close the service to new requests.
        self.owner.adopt(self.runtime())
        event.action, event.ready = None, True
        self.owner.collect_completed()

    def test_sealing_during_collection_waits_for_next_pass(self):
        first, second = self.runtime(), self.runtime()
        first_id, second_id = self.owner.adopt(first), self.owner.adopt(second)
        _, event = self.pending(first)
        self.owner.seal(first_id)
        def seal_next():
            self.owner.seal(second_id)
            return True
        event.action = seal_next
        report = self.owner.collect_completed()
        self.assertEqual(report.released, (first_id,))
        self.assertEqual(report.pending, (second_id,))
        self.assertFalse(second.is_closed)
        self.assertEqual(self.owner.collect_completed().released, (second_id,))

    def test_interrupted_collection_retains_runtime_and_clears_busy_guard(self):
        runtime = self.runtime()
        key = self.owner.adopt(runtime)
        _, event = self.pending(runtime)
        self.owner.seal(key)
        def interrupt():
            raise KeyboardInterrupt()
        event.action = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.owner.collect_completed()
        self.assertEqual(self.owner.owned_runtime_ids, (key,))
        event.action, event.ready = None, True
        self.assertEqual(self.owner.collect_completed().released, (key,))
