import unittest

from flashinfer_npu.attention import (
    AttentionMode, BatchAttention, attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.holistic import _install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.operator_retention import (
    AttentionCallRetentionClosePending, AttentionRetainedInvocationError,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import build_components, package_attention
from tests.test_checkpoint_043_provider_workspace_reset import FakeNpuWorkspace, plan_wrapper, runtime_registry
from tests.test_checkpoint_129_call_retention import Event, Recorder


class Factory:
    def __init__(self):
        self.calls = []
        self.recorders = []

    def create(self, *, device, mode):
        self.calls.append((device, mode))
        recorder = Recorder()
        self.recorders.append(recorder)
        return recorder

    def __repr__(self):
        raise AssertionError("opaque factory must not be represented")


class BrokenFactory:
    def __init__(self, result=None, failure=None):
        self.result, self.failure = result, failure

    def create(self, *, device, mode):
        if self.failure is not None:
            raise self.failure
        return self.result


class BatchRecorderBootstrapCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values = build_components()
        self.factory = Factory()
        package_attention.calls.clear()

    def tearDown(self):
        old = self.original
        _install_attention_operator_runtime_resolvers(
            old.registry, operation_catalog=old.operation_catalog,
            runtime_declarations=old.runtime_declarations,
            plan_scoring_manifest_binding=old.plan_scoring_manifest_binding,
            provider_integration_bundle_binding=old.provider_integration_bundle_binding,
            batch_completion_event_recorder_factory=old.batch_completion_event_recorder_factory)

    def install(self, factory):
        return install_attention_operator_runtime_resolvers(
            runtime_registry(self.values), operation_catalog=self.values["catalog"],
            batch_completion_event_recorder_factory=factory)

    def decode(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="HND")
        plan_wrapper(wrapper)
        return wrapper

    def finish(self, wrapper, recorder):
        for event in recorder.events:
            event.ready = True
        wrapper._operator_runtime.close()

    def test_install_and_snapshot_do_not_create_recorders_or_probe_packages(self):
        installed = self.install(self.factory)
        captured = attention_operator_runtime_registry_snapshot()
        self.assertIs(captured.batch_completion_event_recorder_factory, self.factory)
        self.assertEqual(captured.generation, installed.generation)
        repr(captured)
        self.assertEqual(self.factory.calls, [])
        self.assertEqual(self.values["events"], [])

    def test_all_batch_facades_receive_device_and_mode_without_new_caller_arguments(self):
        self.install(self.factory)
        wrappers = (
            (self.decode(), AttentionMode.BATCH_DECODE_PAGED),
            (BatchPrefillWithPagedKVCacheWrapper(FakeNpuWorkspace()), AttentionMode.BATCH_PREFILL_PAGED),
            (BatchPrefillWithRaggedKVCacheWrapper(FakeNpuWorkspace()), AttentionMode.BATCH_PREFILL_RAGGED),
            (BatchAttention(device="npu:0"), AttentionMode.BATCH_MIXED_PAGED),
        )
        for index, (wrapper, mode) in enumerate(wrappers):
            with self.subTest(mode=mode):
                if mode == AttentionMode.BATCH_PREFILL_PAGED:
                    wrapper.plan([0, 1], [0, 1], [7], [64], 8, 2, 128, 128, q_data_type="bfloat16")
                elif mode == AttentionMode.BATCH_PREFILL_RAGGED:
                    wrapper.plan([0, 1], [0, 64], 8, 2, 128, q_data_type="bfloat16")
                elif mode == AttentionMode.BATCH_MIXED_PAGED:
                    wrapper.plan([0, 1], [0, 1], [7], [64], 8, 2, 128, 128, 128)
                result = wrapper.run("q", "k", "v") if mode == AttentionMode.BATCH_PREFILL_RAGGED else wrapper.run("q", ("k", "v"))
                expected = ("package-output:q", "package-lse:0.25") if mode == AttentionMode.BATCH_MIXED_PAGED else "package-output:q"
                self.assertEqual(result, expected)
                self.assertEqual(self.factory.calls[index], ("npu:0", mode))
                self.assertEqual(len(self.factory.recorders[index].events), 1)
                self.assertEqual(len(wrapper._operator_runtime.call_retention.pending_tokens), 1)
                self.finish(wrapper, self.factory.recorders[index])
        self.assertEqual(len(self.factory.calls), 4)
        self.assertEqual(len({id(recorder) for recorder in self.factory.recorders}), 4)

    def test_replanning_keeps_recorder_and_new_snapshot_does_not_change_existing_wrapper(self):
        self.install(self.factory)
        old = self.decode()
        old.run("first", ("k", "v"))
        other = Factory()
        self.install(other)
        new = self.decode()
        plan_wrapper(old)
        old.run("second", ("k", "v"))
        new.run("third", ("k", "v"))
        self.assertEqual(len(self.factory.calls), 1)
        self.assertEqual(len(self.factory.recorders[0].events), 2)
        self.assertEqual(len(other.recorders[0].events), 1)
        self.assertIsNot(old._operator_runtime.call_retention, new._operator_runtime.call_retention)
        self.finish(old, self.factory.recorders[0])
        self.finish(new, other.recorders[0])

    def test_default_install_clears_factory_for_future_wrappers_only(self):
        self.install(self.factory)
        old = self.decode()
        install_attention_operator_runtime_resolvers(runtime_registry(self.values), operation_catalog=self.values["catalog"])
        new = self.decode()
        old.run("old", ("k", "v"))
        new.run("new", ("k", "v"))
        self.assertEqual(len(self.factory.calls), 1)
        self.assertEqual(new._operator_runtime.call_retention.pending_tokens, ())
        self.assertIsNone(attention_operator_runtime_registry_snapshot().batch_completion_event_recorder_factory)
        self.finish(old, self.factory.recorders[0])

    def test_invalid_factory_install_is_atomic(self):
        installed = self.install(self.factory)
        for factory in (object(), type("NonCallable", (), {"create": 1})()):
            with self.assertRaisesRegex(TypeError, "must implement create"):
                self.install(factory)
            current = attention_operator_runtime_registry_snapshot()
            self.assertEqual(current.generation, installed.generation)
            self.assertIs(current.batch_completion_event_recorder_factory, self.factory)
        self.assertEqual(self.factory.calls, [])

    def test_factory_failure_or_invalid_recorder_precedes_provider_probe(self):
        for factory, error in (
            (BrokenFactory(), TypeError),
            (BrokenFactory(type("BadRecorder", (), {"record": 1})()), TypeError),
            (BrokenFactory(failure=RuntimeError("factory failure")), RuntimeError),
        ):
            self.install(factory)
            with self.assertRaises(error):
                self.decode()
            self.assertEqual(self.values["events"], [])
            self.assertEqual(package_attention.calls, [])

    def test_nonbatch_modes_and_host_reference_do_not_create_recorders(self):
        snapshot = self.install(self.factory)
        for mode, device in ((AttentionMode.SINGLE_PREFILL, "npu:0"),
                             (AttentionMode.SINGLE_DECODE, "npu:0"),
                             (AttentionMode.BATCH_DECODE_PAGED, "cpu")):
            with self.assertRaises(SchemaError):
                snapshot.create_batch_completion_event_recorder(device, mode)
        BatchAttention(device="cpu")
        self.assertEqual(self.factory.calls, [])

    def test_internal_teardown_still_waits_for_exact_completion(self):
        self.install(self.factory)
        wrapper = self.decode()
        wrapper.run("q", ("k", "v"))
        with self.assertRaises(AttentionCallRetentionClosePending):
            wrapper._operator_runtime.close()
        self.assertEqual(len(wrapper._operator_runtime.call_retention.pending_tokens), 1)
        self.finish(wrapper, self.factory.recorders[0])
        self.assertTrue(wrapper._operator_runtime.is_closed)

    def test_recording_failure_preserves_unrecorded_call_for_explicit_recovery(self):
        self.install(self.factory)
        wrapper = self.decode()
        recorder = self.factory.recorders[0]
        recorder.failure = RuntimeError("synthetic record failure")
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            wrapper.run("q", ("k", "v"))
        token = caught.exception.token
        self.assertEqual(len(package_attention.calls), 1)
        self.assertEqual(wrapper._operator_runtime.call_retention.pending_tokens, (token,))
        wrapper._operator_runtime.call_retention.bind_event(token, Event(token, ready=True))
        wrapper._operator_runtime.close()
