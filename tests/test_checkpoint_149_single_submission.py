import unittest

from flashinfer_npu.attention import AttentionMode, AttentionStateError
from flashinfer_npu.attention.holistic import AttentionOperatorRuntimeRegistrySnapshot
from flashinfer_npu.attention.operator_single_owner import (
    AttentionSingleRuntimeOwner, AttentionSingleOwnerClosePending,
)
from flashinfer_npu.attention.operator_single_resources import AttentionSingleRuntimeResources
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_043_provider_workspace_reset import runtime_registry
from tests.test_checkpoint_129_call_retention import masked_call, Event
from tests.test_checkpoint_137_batch_recorder_bootstrap import Factory


class SingleSubmissionCheckpoint(unittest.TestCase):
    def setUp(self):
        values = build_components()
        self.snapshot = AttentionOperatorRuntimeRegistrySnapshot(
            0, ("npu",), runtime_registry(values), values["catalog"])
        self.owner, self.factory = AttentionSingleRuntimeOwner(), Factory()
        self.resources = AttentionSingleRuntimeResources(self.factory, self.owner)

    def scope(self, **options):
        return self.resources.submission(self.snapshot, device=options.get("device", "npu:0"),
                                         mode=options.get("mode", AttentionMode.SINGLE_PREFILL))

    def test_adoption_precedes_submission_and_exit_seals_without_polling(self):
        self.assertEqual(self.factory.calls, [])
        repr(self.resources)  # Factory deliberately rejects repr().
        for mode in (AttentionMode.SINGLE_PREFILL, AttentionMode.SINGLE_DECODE):
            with self.scope(mode=mode) as runtime:
                key = self.owner.owned_runtime_ids[0]
                self.assertIs(self.owner.get_runtime(key), runtime)
                self.assertIs(runtime._resolver_registry, self.snapshot.registry)
                self.assertEqual(self.owner.collect_completed().preparing, (key,))
                call, _ = masked_call()
                token = runtime.call_retention.retain(call)
                event = Event(token, True)
                runtime.call_retention.bind_event(token, event)
            self.assertEqual(event.queries, 0)
            self.assertEqual(self.owner.collect_completed().released, (key,))

    def test_all_submission_exceptions_propagate_and_seal(self):
        for error in (ValueError("synthetic planning failure"), KeyboardInterrupt()):
            with self.assertRaises(type(error)) as caught:
                with self.scope():
                    raise error
            self.assertIs(caught.exception, error)
            report = self.owner.collect_completed()
            self.assertEqual(len(report.released), 1)
            self.assertEqual(report.preparing, ())

    def test_shutdown_during_submission_waits_for_scope_exit(self):
        with self.scope() as runtime:
            with self.assertRaises(AttentionSingleOwnerClosePending):
                self.owner.close()
            self.assertFalse(runtime.is_closing)
        self.owner.close()
        self.assertTrue(self.owner.is_closed)
        with self.assertRaises(AttentionStateError):
            with self.scope():
                self.fail("closed owner must not permit execution")

    def test_invalid_inputs_do_not_create_recorders_or_owned_runtimes(self):
        for options in ({"device": "cpu"}, {"mode": AttentionMode.BATCH_DECODE_PAGED}):
            with self.assertRaises(ValueError):
                with self.scope(**options):
                    self.fail("invalid scope")
        with self.assertRaises(TypeError):
            with self.resources.submission(object(), device="npu:0", mode=AttentionMode.SINGLE_PREFILL):
                self.fail("invalid snapshot")
        self.assertEqual(self.factory.calls, [])
        self.assertEqual(self.owner.owned_runtime_ids, ())

    def test_configuration_and_recorder_failures_never_enter_submission(self):
        with self.assertRaises(TypeError):
            AttentionSingleRuntimeResources(None, self.owner)
        with self.assertRaises(TypeError):
            AttentionSingleRuntimeResources(self.factory, None)
        class InvalidFactory:
            def create(self, **kwargs):
                return object()
        invalid = AttentionSingleRuntimeResources(InvalidFactory(), self.owner)
        with self.assertRaises(TypeError):
            with invalid.submission(self.snapshot, device="npu:0", mode=AttentionMode.SINGLE_PREFILL):
                self.fail("invalid recorder")
        self.assertEqual(self.owner.owned_runtime_ids, ())

    def test_failed_submission_with_unrecorded_call_retains_recovery(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic partial"):
            with self.scope() as runtime:
                key = self.owner.owned_runtime_ids[0]
                call, _ = masked_call()
                token = runtime.call_retention.retain(call)
                raise RuntimeError("synthetic partial submission")
        report = self.owner.collect_completed()
        self.assertEqual(report.pending, (key,))
        self.assertEqual(report.preparing, ())
        self.owner.get_runtime(key).call_retention.bind_event(token, Event(token, True))
        self.assertEqual(self.owner.collect_completed().released, (key,))
