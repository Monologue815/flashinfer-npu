import gc
import unittest
import weakref

from flashinfer_npu.attention import AttentionStateError
from flashinfer_npu.attention.operator_runtime_owner import (
    AttentionBatchRuntimeOwner, AttentionRuntimeOwnerClosePending,
)
from tests.test_checkpoint_129_call_retention import Event, masked_call
from tests.test_checkpoint_135_quantized_mask_composition import components
from tests.test_checkpoint_142_returned_output_retention import Output


class OwnerCollectionCheckpoint(unittest.TestCase):
    def setUp(self):
        self.owner = AttentionBatchRuntimeOwner()

    def runtime(self):
        runtime, _, _, _, _ = components()
        runtime_id = self.owner.adopt(runtime)
        return runtime_id, runtime

    def pending(self, runtime, ready=False):
        call, _ = masked_call()
        registry = runtime.call_retention
        token = registry.retain(call)
        output = Output()
        reference = weakref.ref(output)
        registry.retain_result(token, output)
        event = Event(token, ready=ready)
        registry.bind_event(token, event)
        return token, event, reference

    def test_idle_output_release_preserves_runtime_and_active_plan(self):
        runtime_id, runtime = self.runtime()
        plan = runtime.plan_state
        token, event, reference = self.pending(runtime)
        report = self.owner.collect_completed()
        self.assertEqual(report.collections[0][1].pending, (token,))
        self.assertIsNotNone(reference())
        event.ready = True
        report = self.owner.collect_completed()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(report.collections[0][0], runtime_id)
        self.assertEqual(report.collections[0][1].released, (token,))
        self.assertIs(runtime.plan_state, plan)
        self.assertFalse(runtime.is_closing)
        self.assertFalse(runtime.is_closed)
        self.assertEqual(self.owner.owned_runtime_ids, (runtime_id,))
        # Collection does not seal either the runtime registry or the owner.
        self.pending(runtime, ready=True)
        self.runtime()
        self.owner.collect_completed()
        self.owner.close()

    def test_query_failure_does_not_prevent_independent_collection(self):
        first_id, first = self.runtime()
        second_id, second = self.runtime()
        first_token, first_event, first_ref = self.pending(first)
        second_token, _, second_ref = self.pending(second, ready=True)
        def fail():
            raise RuntimeError("synthetic event failure")
        first_event.action = fail
        report = self.owner.collect_completed()
        by_id = dict(report.collections)
        self.assertEqual(by_id[first_id].failures[0].message, "synthetic event failure")
        self.assertEqual(by_id[first_id].pending, (first_token,))
        self.assertEqual(by_id[second_id].released, (second_token,))
        gc.collect()
        self.assertIsNotNone(first_ref())
        self.assertIsNone(second_ref())
        first_event.action, first_event.ready = None, True
        self.owner.collect_completed()
        self.owner.close()

    def test_unrecorded_call_requires_explicit_event_recovery(self):
        _, runtime = self.runtime()
        call, _ = masked_call()
        token = runtime.call_retention.retain(call)
        report = self.owner.collect_completed()
        self.assertEqual(report.collections[0][1].unrecorded, (token,))
        runtime.call_retention.bind_event(token, Event(token, ready=True))
        self.owner.collect_completed()
        self.owner.close()

    def test_reentrant_collection_and_close_cannot_change_lifecycle(self):
        _, runtime = self.runtime()
        token, event, _ = self.pending(runtime)
        for action in (self.owner.collect_completed, self.owner.close):
            event.action = action
            report = self.owner.collect_completed()
            self.assertEqual(report.collections[0][1].failures[0].error_type, "AttentionStateError")
            self.assertEqual(runtime.call_retention.pending_tokens, (token,))
            self.assertFalse(runtime.is_closing)
        event.action, event.ready = None, True
        self.owner.collect_completed()
        self.owner.close()

    def test_newly_adopted_runtime_waits_for_next_pass(self):
        first_id, first = self.runtime()
        _, event, _ = self.pending(first, ready=True)
        added = []
        def adopt():
            runtime_id, runtime = self.runtime()
            token, next_event, _ = self.pending(runtime, ready=True)
            added.append((runtime_id, token, next_event))
            return True
        event.action = adopt
        report = self.owner.collect_completed()
        self.assertEqual(tuple(key for key, _ in report.collections), (first_id,))
        self.assertEqual(added[0][2].queries, 0)
        report = self.owner.collect_completed()
        self.assertEqual(dict(report.collections)[added[0][0]].released, (added[0][1],))
        self.owner.close()

    def test_collection_during_or_after_shutdown_is_rejected(self):
        _, runtime = self.runtime()
        _, event, _ = self.pending(runtime)
        with self.assertRaises(AttentionRuntimeOwnerClosePending):
            self.owner.close()
        with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
            self.owner.collect_completed()
        event.ready = True
        self.owner.close()
        with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
            self.owner.collect_completed()
