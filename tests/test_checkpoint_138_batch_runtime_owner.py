import gc
import unittest
import weakref

from flashinfer_npu.attention import (
    AttentionStateError, BatchAttention, attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.holistic import _install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.operator_runtime_owner import (
    AttentionBatchRuntimeOwner, AttentionRuntimeOwnerClosePending,
)
from flashinfer_npu.attention.operator_retention import AttentionRetainedInvocationError
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import build_components, package_attention
from tests.test_checkpoint_043_provider_workspace_reset import FakeNpuWorkspace, plan_wrapper, runtime_registry
from tests.test_checkpoint_129_call_retention import Event
from tests.test_checkpoint_137_batch_recorder_bootstrap import Factory


class Query:
    def __str__(self):
        return "query"


class BatchRuntimeOwnerCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values = build_components()
        self.factory, self.owner = Factory(), AttentionBatchRuntimeOwner()
        package_attention.calls.clear()

    def tearDown(self):
        old = self.original
        _install_attention_operator_runtime_resolvers(
            old.registry, operation_catalog=old.operation_catalog,
            runtime_declarations=old.runtime_declarations,
            plan_scoring_manifest_binding=old.plan_scoring_manifest_binding,
            provider_integration_bundle_binding=old.provider_integration_bundle_binding,
            batch_completion_event_recorder_factory=old.batch_completion_event_recorder_factory,
            batch_runtime_owner=old.batch_runtime_owner,
            batch_mask_integration=old.batch_mask_integration)
        package_attention.calls.clear()

    def install(self, owner, factory):
        return install_attention_operator_runtime_resolvers(
            runtime_registry(self.values), operation_catalog=self.values["catalog"],
            batch_completion_event_recorder_factory=factory, batch_runtime_owner=owner)

    def wrapper(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="HND")
        plan_wrapper(wrapper)
        return wrapper

    def ready(self):
        for recorder in self.factory.recorders:
            for event in recorder.events:
                event.ready = True

    def test_owner_preserves_runtime_and_inputs_after_wrapper_is_dropped(self):
        self.install(self.owner, self.factory)
        wrapper = self.wrapper()
        query = Query()
        runtime_ref, wrapper_ref, query_ref = weakref.ref(wrapper._operator_runtime), weakref.ref(wrapper), weakref.ref(query)
        wrapper.run(query, ("k", "v"))
        package_attention.calls.clear()
        del wrapper, query
        gc.collect()
        self.assertIsNone(wrapper_ref())
        self.assertIsNotNone(runtime_ref())
        self.assertIsNotNone(query_ref())
        with self.assertRaises(AttentionRuntimeOwnerClosePending):
            self.owner.close()
        self.ready()
        self.owner.close()
        gc.collect()
        self.assertIsNone(runtime_ref())
        self.assertIsNone(query_ref())

    def test_all_batch_wrapper_constructors_adopt_without_exposing_runtime_ids(self):
        snapshot = self.install(self.owner, self.factory)
        self.assertIs(snapshot.batch_runtime_owner, self.owner)
        wrappers = [BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace()),
                    BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace()),
                    BatchPrefillWithRaggedKVCacheWrapper(FakeNpuWorkspace()),
                    BatchAttention(device="npu:0")]
        self.assertEqual(len(self.owner.owned_runtime_ids), 4)
        for runtime_id, wrapper in zip(self.owner.owned_runtime_ids, wrappers):
            self.assertIs(self.owner.get_runtime(runtime_id), wrapper._operator_runtime)
        self.assertEqual(self.values["events"], [])
        self.owner.close()
        self.assertTrue(self.owner.is_closed)
        self.assertEqual(self.owner.owned_runtime_ids, ())

    def test_close_releases_independent_completed_runtimes_and_retries_pending_ones(self):
        self.install(self.owner, self.factory)
        first, second = self.wrapper(), self.wrapper()
        first.run("first", ("k", "v"))
        second.run("second", ("k", "v"))
        ids = self.owner.owned_runtime_ids
        self.factory.recorders[1].events[0].ready = True
        with self.assertRaises(AttentionRuntimeOwnerClosePending) as caught:
            self.owner.close()
        self.assertEqual(caught.exception.report.remaining_runtime_ids, (ids[0],))
        self.assertEqual(caught.exception.report.failures[0][:2], (ids[0], "AttentionCallRetentionClosePending"))
        self.assertTrue(second._operator_runtime.is_closed)
        self.assertTrue(first._operator_runtime.is_closing)
        with self.assertRaisesRegex(AttentionStateError, "closing or closed"):
            self.wrapper()
        self.ready()
        self.owner.close()
        self.owner.close()
        self.assertTrue(self.owner.is_closed)

    def test_query_failure_is_reported_as_metadata_and_keeps_recovery_access(self):
        self.install(self.owner, self.factory)
        wrapper = self.wrapper()
        wrapper.run("q", ("k", "v"))
        event = self.factory.recorders[0].events[0]
        def fail():
            raise RuntimeError("synthetic query failure")
        event.action = fail
        with self.assertRaises(AttentionRuntimeOwnerClosePending) as caught:
            self.owner.close()
        failure = caught.exception.report.failures[0]
        self.assertTrue(all(isinstance(value, str) for value in failure))
        self.assertEqual(failure[1], "AttentionCallCompletionCollectionError")
        runtime = self.owner.get_runtime(failure[0])
        self.assertIs(runtime, wrapper._operator_runtime)
        event.action, event.ready = None, True
        self.owner.close()

    def test_unrecorded_call_requires_explicit_recovery_before_owner_close(self):
        self.install(self.owner, self.factory)
        wrapper = self.wrapper()
        self.factory.recorders[0].failure = RuntimeError("synthetic recording failure")
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            wrapper.run("q", ("k", "v"))
        token = caught.exception.token
        with self.assertRaises(AttentionRuntimeOwnerClosePending):
            self.owner.close()
        runtime = self.owner.get_runtime(self.owner.owned_runtime_ids[0])
        runtime.call_retention.bind_event(token, Event(token, ready=True))
        self.owner.close()

    def test_registry_replacement_does_not_drain_old_owner(self):
        self.install(self.owner, self.factory)
        old = self.wrapper()
        old.run("q", ("k", "v"))
        new_owner, new_factory = AttentionBatchRuntimeOwner(), Factory()
        self.install(new_owner, new_factory)
        new = self.wrapper()
        self.assertEqual(len(self.owner.owned_runtime_ids), 1)
        self.assertEqual(len(new_owner.owned_runtime_ids), 1)
        self.assertIsNot(old._operator_runtime, new._operator_runtime)
        new_owner.close()
        self.assertEqual(len(self.owner.owned_runtime_ids), 1)
        self.ready()
        self.owner.close()

    def test_owner_configuration_and_adoption_fail_without_mutating_registry(self):
        baseline = attention_operator_runtime_registry_snapshot()
        for owner, factory, error in ((self.owner, None, SchemaError), (object(), self.factory, TypeError)):
            with self.assertRaises(error):
                self.install(owner, factory)
            self.assertEqual(attention_operator_runtime_registry_snapshot().generation, baseline.generation)
        self.install(None, None)
        untracked = self.wrapper()
        with self.assertRaisesRegex(AttentionStateError, "completion-tracked"):
            self.owner.adopt(untracked._operator_runtime)
        self.install(self.owner, self.factory)
        tracked = self.wrapper()
        with self.assertRaisesRegex(AttentionStateError, "already owned"):
            self.owner.adopt(tracked._operator_runtime)
        self.owner.close()

    def test_reentrant_close_cannot_drop_pending_runtime(self):
        self.install(self.owner, self.factory)
        wrapper = self.wrapper()
        wrapper.run("q", ("k", "v"))
        event = self.factory.recorders[0].events[0]
        event.action = self.owner.close
        with self.assertRaises(AttentionRuntimeOwnerClosePending):
            self.owner.close()
        self.assertEqual(len(self.owner.owned_runtime_ids), 1)
        self.assertEqual(len(wrapper._operator_runtime.call_retention.pending_tokens), 1)
        event.action, event.ready = None, True
        self.owner.close()
