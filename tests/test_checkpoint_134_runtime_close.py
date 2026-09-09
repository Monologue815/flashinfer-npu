import gc
import unittest
import weakref
from unittest.mock import patch

from flashinfer_npu.attention import AttentionStateError
from flashinfer_npu.attention.operator_retention import (
    AttentionCallCompletionCollectionError, AttentionCallRetentionClosePending,
    AttentionRetainedInvocationError,
)
from tests.test_checkpoint_019_package_runtime_integration import (
    batch_runtime, build_components, framework_inputs, package_attention,
)
from tests.test_checkpoint_129_call_retention import Event, Recorder
from tests.test_checkpoint_130_runtime_call_retention import owning_runtime
from tests.test_checkpoint_131_transactional_mask_plan import mask_calls, mask_request, mask_runtime


class RuntimeCloseCheckpoint(unittest.TestCase):
    def setUp(self):
        package_attention.calls[:] = []
        mask_calls[:] = []

    def test_completed_final_mask_call_and_active_adapter_are_released(self):
        recorder = Recorder()
        _, operation, runtime = mask_runtime(recorder)
        plan, binder, _, _, owner = mask_request(operation)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        runtime.run("q", ("k", "v"), return_lse=False)
        executor = weakref.ref(runtime._executor)
        del binder
        recorder.events[0].ready = True
        self.assertIsNone(runtime.close())
        gc.collect()
        self.assertIsNone(owner())
        self.assertIsNone(executor())
        self.assertTrue(runtime.is_closed)
        self.assertFalse(runtime.is_closing)
        self.assertFalse(runtime.is_planned)
        self.assertEqual(runtime.call_retention.pending_tokens, ())
        with self.assertRaises(AttentionStateError):
            _ = runtime.plan_state
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_lowered_call
        queries = recorder.events[0].queries
        self.assertIsNone(runtime.close())
        self.assertEqual(recorder.events[0].queries, queries)

    def test_pending_close_stops_new_work_and_is_retryable(self):
        recorder = Recorder()
        values, adapter, runtime = owning_runtime(recorder)
        runtime.run("first", ("k", "v"), return_lse=False)
        runtime.run("second", ("k", "v"), return_lse=False)
        tokens = runtime.call_retention.pending_tokens
        recorder.events[1].ready = True
        events = list(values["events"])
        with self.assertRaises(AttentionCallRetentionClosePending) as caught:
            runtime.close()
        self.assertEqual(caught.exception.report.released, (tokens[1],))
        self.assertEqual(caught.exception.report.pending, (tokens[0],))
        self.assertTrue(runtime.is_closing)
        self.assertFalse(runtime.is_closed)
        self.assertTrue(runtime.is_planned)
        self.assertIsNotNone(adapter.owners[0]())
        for action in (
            lambda: runtime.plan(*framework_inputs()),
            lambda: runtime.run("blocked", ("k", "v"), return_lse=False),
            runtime.fork_unplanned,
            lambda: runtime.rebind_workspace_contract(None),
        ):
            with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
                action()
        self.assertEqual(values["events"], events)
        self.assertEqual(len(package_attention.calls), 2)
        recorder.events[0].ready = True
        runtime.close()
        gc.collect()
        self.assertTrue(all(owner() is None for owner in adapter.owners))
        with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
            runtime.run("closed", ("k", "v"), return_lse=False)

    def test_query_failure_remains_closing_until_recovered(self):
        recorder = Recorder()
        _, _, runtime = owning_runtime(recorder)
        runtime.run("q", ("k", "v"), return_lse=False)
        token = runtime.call_retention.pending_tokens[0]

        def fail():
            raise RuntimeError("event query failed")

        recorder.events[0].action = fail
        with self.assertRaises(AttentionCallCompletionCollectionError) as caught:
            runtime.close()
        self.assertEqual(caught.exception.report.failures[0].token, token)
        self.assertTrue(runtime.is_closing)
        self.assertEqual(runtime.call_retention.pending_tokens, (token,))
        recorder.events[0].action, recorder.events[0].ready = None, True
        runtime.close()
        self.assertTrue(runtime.is_closed)

    def test_unrecorded_work_requires_recovery_event_before_close(self):
        recorder = Recorder()
        _, _, runtime = owning_runtime(recorder)
        recorder.failure = RuntimeError("event recording failed")
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            runtime.run("q", ("k", "v"), return_lse=False)
        token = caught.exception.token
        with self.assertRaises(AttentionCallRetentionClosePending) as pending:
            runtime.close()
        self.assertEqual(pending.exception.report.unrecorded, (token,))
        self.assertTrue(runtime.is_closing)
        runtime.call_retention.bind_event(token, Event(token, True))
        runtime.close()
        self.assertTrue(runtime.is_closed)

    def test_untracked_legacy_runtime_cannot_claim_safe_close(self):
        values = build_components()
        runtime = batch_runtime(values)
        runtime.plan(*framework_inputs())
        runtime.run("before", ("k", "v"), return_lse=False)
        with self.assertRaisesRegex(AttentionStateError, "requires a completion event recorder"):
            runtime.close()
        self.assertFalse(runtime.is_closing)
        self.assertFalse(runtime.is_closed)
        self.assertTrue(runtime.is_planned)
        self.assertEqual(runtime.run("after", ("k", "v"), return_lse=False), "package-output:after")

    def test_empty_configured_runtime_can_close_without_observing_packages(self):
        values, _, runtime = mask_runtime(Recorder())
        runtime.close()
        self.assertTrue(runtime.is_closed)
        self.assertEqual(values["events"], [])
        with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
            runtime.fork_unplanned()

    def test_close_during_candidate_binding_prevents_late_plan_publication(self):
        _, operation, runtime = mask_runtime(Recorder())
        plan, binder, _, _, _ = mask_request(operation)
        original = binder.bind

        def bind_then_close(*args):
            adapter = original(*args)
            runtime.close()
            return adapter

        with patch.object(binder, "bind", side_effect=bind_then_close):
            with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
                runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertTrue(runtime.is_closed)
        self.assertFalse(runtime.is_planned)
        self.assertEqual(mask_calls, [])

    def test_close_during_lowering_prevents_submission_after_teardown(self):
        _, operation, runtime = mask_runtime(Recorder())
        plan, binder, _, _, _ = mask_request(operation)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        adapter = runtime.operator_session._run_adapter
        original = adapter.lower

        def lower_then_close(*args):
            call = original(*args)
            runtime.close()
            return call

        with patch.object(adapter, "lower", side_effect=lower_then_close):
            with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
                runtime.run("q", ("k", "v"), return_lse=False)
        self.assertTrue(runtime.is_closed)
        self.assertEqual(mask_calls, [])
        self.assertEqual(runtime.call_retention.pending_tokens, ())
