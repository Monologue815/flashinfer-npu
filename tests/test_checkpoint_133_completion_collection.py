import gc
import unittest
import weakref

from flashinfer_npu.attention import AttentionStateError
from flashinfer_npu.attention.operator_retention import (
    AttentionCallCompletionCollectionError, AttentionOperatorCallRetention,
)
from tests.test_checkpoint_019_package_runtime_integration import framework_inputs, package_attention
from tests.test_checkpoint_129_call_retention import Event, Recorder, masked_call
from tests.test_checkpoint_130_runtime_call_retention import owning_runtime


def retain(registry, ready=None):
    call, owner = masked_call()
    token = registry.retain(call)
    event = None if ready is None else Event(token, ready)
    if event is not None:
        registry.bind_event(token, event)
    return token, event, owner


class CompletionCollectionCheckpoint(unittest.TestCase):
    def setUp(self):
        package_attention.calls[:] = []

    def test_collection_continues_after_error_and_keeps_only_pending_owners(self):
        registry = AttentionOperatorCallRetention()
        failed, bad_event, bad_owner = retain(registry, False)
        completed, _, completed_owner = retain(registry, True)
        pending, pending_event, pending_owner = retain(registry, False)
        unrecorded, _, unrecorded_owner = retain(registry)

        def fail():
            raise RuntimeError("query unavailable")

        bad_event.action = fail
        report = registry.collect_completed()
        self.assertEqual(report.released, (completed,))
        self.assertEqual(report.pending, (failed, pending, unrecorded))
        self.assertEqual(report.unrecorded, (unrecorded,))
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].token, failed)
        self.assertEqual(report.failures[0].error_type, "RuntimeError")
        self.assertEqual(report.failures[0].message, "query unavailable")
        gc.collect()
        self.assertIsNone(completed_owner())
        self.assertTrue(all(owner() is not None for owner in (bad_owner, pending_owner, unrecorded_owner)))
        bad_event.action, bad_event.ready, pending_event.ready = None, True, True
        registry.bind_event(unrecorded, Event(unrecorded, True))
        recovered = registry.collect_completed()
        self.assertEqual(recovered.released, (failed, pending, unrecorded))
        self.assertEqual(recovered.pending, ())
        # Holding the old failure report cannot keep completed resources alive.
        gc.collect()
        self.assertTrue(all(owner() is None for owner in (bad_owner, pending_owner, unrecorded_owner)))

    def test_failed_exception_tracebacks_are_not_kept_in_collection_reports(self):
        registry = AttentionOperatorCallRetention()
        token, event, _ = retain(registry, False)
        diagnostic_owners = []

        class DiagnosticOwner:
            pass

        class BrokenMessageError(RuntimeError):
            def __str__(self):
                raise RuntimeError("message cannot be rendered")

        def fail():
            owner = DiagnosticOwner()
            diagnostic_owners.append(weakref.ref(owner))
            error = BrokenMessageError()
            error.owner = owner
            raise error

        event.action = fail
        report = registry.collect_completed()
        self.assertEqual(report.failures[0].token, token)
        self.assertEqual(report.failures[0].error_type, "BrokenMessageError")
        self.assertIn("message unavailable", report.failures[0].message)
        gc.collect()
        self.assertIsNone(diagnostic_owners[0]())

    def test_non_boolean_completion_remains_a_failure_not_a_release(self):
        registry = AttentionOperatorCallRetention()
        token, event, owner = retain(registry, True)
        event.ready = 1
        report = registry.collect_completed()
        self.assertEqual(report.released, ())
        self.assertEqual(report.pending, (token,))
        self.assertIn("exact boolean", report.failures[0].message)
        self.assertIsNotNone(owner())

    def test_reentrant_collection_skips_busy_and_already_removed_tokens(self):
        registry = AttentionOperatorCallRetention()
        first, event, _ = retain(registry, True)
        second, second_event, _ = retain(registry, True)
        inner_reports = []

        def query():
            inner_reports.append(registry.collect_completed())
            return True

        event.action = query
        report = registry.collect_completed()
        self.assertEqual(inner_reports[0].released, (second,))
        self.assertEqual(inner_reports[0].pending, (first,))
        self.assertEqual(report.released, (first,))
        self.assertEqual(report.pending, ())
        self.assertEqual(report.failures, ())
        self.assertEqual((event.queries, second_event.queries), (1, 1))

    def test_calls_added_during_collection_wait_for_the_next_pass(self):
        registry = AttentionOperatorCallRetention()
        first, event, _ = retain(registry, True)
        added = []

        def query():
            added.append(retain(registry, True))
            return True

        event.action = query
        report = registry.collect_completed()
        new_token, new_event, _ = added[0]
        self.assertEqual(report.released, (first,))
        self.assertEqual(report.pending, (new_token,))
        self.assertEqual(new_event.queries, 0)
        self.assertEqual(registry.collect_completed().released, (new_token,))

    def test_next_run_reclaims_previous_completed_call_without_exposing_a_token(self):
        recorder = Recorder()
        _, adapter, runtime = owning_runtime(recorder)
        self.assertEqual(runtime.run("first", ("k", "v"), return_lse=False), "package-output:first")
        old = runtime.call_retention.pending_tokens[0]
        recorder.events[0].ready = True
        self.assertEqual(runtime.run("second", ("k", "v"), return_lse=True),
                         ("package-output:second", "package-lse:0.25"))
        gc.collect()
        self.assertIsNone(adapter.owners[0]())
        self.assertIsNotNone(adapter.owners[1]())
        self.assertNotIn(old, runtime.call_retention.pending_tokens)
        self.assertEqual(len(runtime.call_retention.pending_tokens), 1)
        self.assertEqual(recorder.events[0].queries, 1)

    def test_next_plan_reclaims_completed_but_preserves_incomplete_calls(self):
        recorder = Recorder()
        _, adapter, runtime = owning_runtime(recorder)
        runtime.run("first", ("k", "v"), return_lse=False)
        runtime.run("second", ("k", "v"), return_lse=False)
        second = runtime.call_retention.pending_tokens[1]
        recorder.events[0].ready = True
        runtime.plan(*framework_inputs(kv_length=64))
        gc.collect()
        self.assertIsNone(adapter.owners[0]())
        self.assertIsNotNone(adapter.owners[1]())
        self.assertEqual(runtime.call_retention.pending_tokens, (second,))

    def test_query_failure_blocks_new_work_and_preserves_the_active_plan(self):
        recorder = Recorder()
        values, _, runtime = owning_runtime(recorder)
        runtime.run("first", ("k", "v"), return_lse=False)
        active = runtime.operator_session
        previous_call = runtime.last_lowered_call
        token = runtime.call_retention.pending_tokens[0]
        events = list(values["events"])

        def fail():
            raise RuntimeError("event failed")

        recorder.events[0].action = fail
        with self.assertRaises(AttentionCallCompletionCollectionError) as caught:
            runtime.plan(*framework_inputs(kv_length=64))
        self.assertEqual(caught.exception.report.failures[0].token, token)
        self.assertIs(runtime.operator_session, active)
        self.assertIs(runtime.last_lowered_call, previous_call)
        self.assertEqual(values["events"], events)
        with self.assertRaises(AttentionCallCompletionCollectionError):
            runtime.run("blocked", ("k", "v"), return_lse=False)
        self.assertEqual(len(package_attention.calls), 1)
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_lowered_call
        self.assertEqual(runtime.call_retention.pending_tokens, (token,))
        recorder.events[0].action, recorder.events[0].ready = None, True
        self.assertEqual(runtime.run("recovered", ("k", "v"), return_lse=False), "package-output:recovered")
        self.assertNotIn(token, runtime.call_retention.pending_tokens)
