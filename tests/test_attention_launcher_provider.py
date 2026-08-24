import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event

from flashinfer_npu.attention import (
    ATTENTION_KERNEL_ERROR_ABI,
    AttentionArtifactLoadEvidence,
    AttentionArtifactVerificationKind,
    AttentionHostBufferRegistry,
    AttentionLaunchSession,
    AttentionLaunchSessionState,
    AttentionLaunchCoordinator,
    AttentionLauncherProviderError,
    AttentionLeaseRegistry,
    AttentionProviderCompletionResult,
    AttentionProviderRecoveryResult,
    AttentionProviderProbe,
    AttentionProviderSubmitResult,
    AttentionProtocolRoute,
    AttentionProtocolState,
    AttentionResolvedLauncher,
    AttentionResolvedLauncherRegistry,
    AttentionResolvedLauncherState,
    AttentionRuntimeTeardownEvidence,
    AttentionStorageLeaseError,
    AttentionUnknownSubmitStatus,
    attention_kernel_binary_abi,
    capture_attention_protocol,
    resolve_attention_launcher,
)
from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactVerificationError,
    Backend,
    SchemaError,
)
from tests.test_attention_capability import (
    aclnn_attention_artifact,
    bound_kernel,
    functional_profile,
)
from tests.test_attention_launch_packet import packet_fixture


def resolution_fixture():
    packet, _, _, _ = packet_fixture()
    profile = functional_profile()
    kernel = bound_kernel(profile)
    assert kernel.artifact is not None
    assert kernel.binary_abi is not None
    probe = AttentionProviderProbe(
        "fake-ascend-loader",
        "1.0-test",
        kernel.backend,
        packet.dispatch_receipt.environment_fingerprint,
        (kernel.artifact.format,),
        kernel.binary_abi.fingerprint,
        kernel.binary_abi.error_abi.fingerprint,
    )
    payload = ("synthetic:" + kernel.artifact.locator).encode("utf-8")
    evidence = AttentionArtifactLoadEvidence.verify_bytes(
        kernel.artifact,
        payload,
        probe,
        loader_instance_id="fake-loader-instance",
        loader_generation=1,
        artifact_handle=0xA000,
    )
    resolved = resolve_attention_launcher(
        kernel,
        packet.dispatch_receipt,
        probe,
        evidence,
        symbol_address=0xB000,
        symbol_generation=1,
    )
    return packet, kernel, probe, evidence, resolved


def registered_launcher(resolved, evidence):
    registry = AttentionResolvedLauncherRegistry()
    registry.register(resolved, evidence)
    return registry


class FakeProvider:
    def __init__(self, probe, submit_codes=(0,), completion_code=0):
        self.probe_value = probe
        self.submit_codes = list(submit_codes)
        self.completion_code = completion_code
        self.stale_packet_result = False
        self.wrong_completion_event = False
        self.raise_on_submit = False
        self.recovery_status = AttentionUnknownSubmitStatus.INDETERMINATE
        self.recovery_completion_code = None
        self.stale_recovery_attempt = False

    def probe(self):
        return self.probe_value

    def submit(self, resolved, packet):
        if self.raise_on_submit:
            raise RuntimeError("synthetic provider transport failure")
        code = self.submit_codes.pop(0)
        return AttentionProviderSubmitResult(
            self.probe_value.fingerprint,
            resolved.fingerprint,
            "f" * 64 if self.stale_packet_result else packet.fingerprint,
            code,
            submission_id="submission-1" if code == 0 else None,
            completion_event_id="event-1" if code == 0 else None,
        )

    def query_completion(self, resolved, packet, completion_event_id):
        return AttentionProviderCompletionResult(
            self.probe_value.fingerprint,
            resolved.fingerprint,
            packet.fingerprint,
            "wrong-event" if self.wrong_completion_event else completion_event_id,
            self.completion_code,
            (
                hashlib.sha256(b"synthetic async failure").hexdigest()
                if self.completion_code != 0
                else None
            ),
        )

    def recover_submit(self, resolved, packet, attempt_number):
        submitted = self.recovery_status in {
            AttentionUnknownSubmitStatus.SUBMITTED,
            AttentionUnknownSubmitStatus.COMPLETED,
        }
        completed = self.recovery_status == AttentionUnknownSubmitStatus.COMPLETED
        code = self.recovery_completion_code if completed else None
        return AttentionProviderRecoveryResult(
            self.probe_value.fingerprint,
            resolved.fingerprint,
            packet.fingerprint,
            attempt_number + 1 if self.stale_recovery_attempt else attempt_number,
            self.recovery_status,
            submission_id="recovered-submission" if submitted else None,
            completion_event_id="recovered-event" if submitted else None,
            completion_code=code,
            error_detail_digest=(
                hashlib.sha256(b"recovered async failure").hexdigest()
                if completed and code not in (None, 0)
                else None
            ),
        )


class BlockingCompletionProvider(FakeProvider):
    def __init__(self, probe):
        super().__init__(probe)
        self.completion_entered = Event()
        self.allow_completion = Event()

    def query_completion(self, resolved, packet, completion_event_id):
        self.completion_entered.set()
        if not self.allow_completion.wait(2.0):
            raise RuntimeError("synthetic completion gate timed out")
        return super().query_completion(resolved, packet, completion_event_id)


class AttentionProviderEvidenceTests(unittest.TestCase):
    def test_probe_load_evidence_and_resolved_symbol_round_trip(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        self.assertEqual(AttentionProviderProbe.from_dict(probe.to_dict()), probe)
        self.assertEqual(
            AttentionArtifactLoadEvidence.from_dict(evidence.to_dict()), evidence
        )
        self.assertEqual(
            AttentionResolvedLauncher.from_dict(resolved.to_dict()), resolved
        )
        resolved.validate_packet(packet)
        self.assertEqual(evidence.verification_kind, AttentionArtifactVerificationKind.BYTES)
        self.assertEqual(resolved.entry_point, "attention_paged_decode_entry")

    def test_byte_and_builtin_evidence_have_distinct_verification_paths(self):
        _, kernel, probe, _, _ = resolution_fixture()
        assert kernel.artifact is not None
        with self.assertRaises(ArtifactVerificationError):
            AttentionArtifactLoadEvidence.verify_bytes(
                kernel.artifact,
                b"wrong",
                probe,
                loader_instance_id="loader",
                loader_generation=1,
                artifact_handle=1,
            )

        builtin = aclnn_attention_artifact()
        builtin_probe = AttentionProviderProbe(
            "fake-aclnn-provider",
            "1",
            Backend.ACLNN,
            probe.environment_fingerprint,
            (ArtifactFormat.ACLNN_BUILTIN,),
            attention_kernel_binary_abi().fingerprint,
            ATTENTION_KERNEL_ERROR_ABI.fingerprint,
        )
        evidence = AttentionArtifactLoadEvidence.verify_builtin_contract(
            builtin,
            builtin.digest,
            builtin_probe,
            loader_instance_id="builtin-provider",
            loader_generation=1,
            artifact_handle=2,
        )
        evidence.validate(builtin, builtin_probe)
        self.assertIsNone(evidence.verified_size_bytes)
        with self.assertRaisesRegex(SchemaError, "digest"):
            AttentionArtifactLoadEvidence.verify_builtin_contract(
                builtin,
                "1" * 64,
                builtin_probe,
                loader_instance_id="builtin-provider",
                loader_generation=1,
                artifact_handle=2,
            )

    def test_resolution_rejects_environment_format_and_receipt_drift(self):
        packet, kernel, probe, evidence, _ = resolution_fixture()
        with self.assertRaisesRegex(SchemaError, "cannot resolve"):
            resolve_attention_launcher(
                kernel,
                packet.dispatch_receipt,
                replace(probe, environment_fingerprint="1" * 64),
                evidence,
                symbol_address=1,
                symbol_generation=1,
            )
        with self.assertRaisesRegex(SchemaError, "descriptor is stale"):
            resolve_attention_launcher(
                replace(kernel, kernel_id="another-kernel"),
                packet.dispatch_receipt,
                probe,
                evidence,
                symbol_address=1,
                symbol_generation=1,
            )

    def test_recovery_and_runtime_teardown_evidence_are_strict_wire_records(self):
        packet, _, probe, _, resolved = resolution_fixture()
        recovery = AttentionProviderRecoveryResult(
            probe.fingerprint,
            resolved.fingerprint,
            packet.fingerprint,
            1,
            AttentionUnknownSubmitStatus.COMPLETED,
            "submission",
            "event",
            0,
        )
        self.assertEqual(
            AttentionProviderRecoveryResult.from_dict(recovery.to_dict()), recovery
        )
        teardown = AttentionRuntimeTeardownEvidence(
            probe.fingerprint,
            packet.dispatch_receipt.environment_fingerprint,
            packet.stream_binding.runtime_id,
            packet.stream_binding.runtime_generation,
            "runtime-reset",
            1,
            hashlib.sha256(b"quiesced").hexdigest(),
        )
        self.assertEqual(
            AttentionRuntimeTeardownEvidence.from_dict(teardown.to_dict()), teardown
        )
        with self.assertRaisesRegex(SchemaError, "cannot claim launch evidence"):
            replace(
                recovery,
                status=AttentionUnknownSubmitStatus.NOT_SUBMITTED,
            )
        with self.assertRaisesRegex(SchemaError, "asynchronous"):
            replace(
                recovery,
                completion_code=6,
                error_detail_digest=hashlib.sha256(b"sync").hexdigest(),
            )


class AttentionLaunchSessionTests(unittest.TestCase):
    def test_success_keeps_lease_until_matching_completion_and_release(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        launchers = registered_launcher(resolved, evidence)
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            launchers,
        )
        self.assertEqual(session.state, AttentionLaunchSessionState.PREPARED)
        result = session.submit(provider)
        self.assertEqual(result.code_name, "success")
        self.assertEqual(session.state, AttentionLaunchSessionState.SUBMITTED)
        with self.assertRaisesRegex(Exception, "before completion"):
            session.release()
        completion = session.poll_completion(provider)
        self.assertEqual(completion.code_name, "success")
        self.assertEqual(session.state, AttentionLaunchSessionState.COMPLETED)
        session.release()
        self.assertEqual(session.state, AttentionLaunchSessionState.RELEASED)
        self.assertEqual(launchers.active_count(resolved), 0)

    def test_opt_in_capture_is_automatically_bound_to_session_evidence(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        with capture_attention_protocol("provider-auto") as capture:
            session = AttentionLaunchSession(
                packet,
                resolved,
                AttentionLeaseRegistry(),
                AttentionHostBufferRegistry(),
                registered_launcher(resolved, evidence),
            )
            session.submit(provider)
            session.poll_completion(provider)
            session.release()

        self.assertEqual(len(capture.traces), 1)
        trace = capture.traces[0]
        self.assertIs(session.protocol_trace, trace)
        self.assertEqual(trace.route, AttentionProtocolRoute.PROVIDER)
        self.assertEqual(trace.subject_fingerprint, packet.fingerprint)
        self.assertEqual(
            trace.stream_context_fingerprint,
            packet.execution_identity.stream_context_fingerprint,
        )
        self.assertEqual(
            tuple(event.state for event in trace.events),
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.SUBMIT_UNKNOWN,
                AttentionProtocolState.SUBMITTED,
                AttentionProtocolState.COMPLETED,
                AttentionProtocolState.RELEASED,
            ),
        )

    def test_unknown_submit_teardown_capture_requires_quiesced_event(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        teardown = AttentionRuntimeTeardownEvidence(
            probe.fingerprint,
            packet.dispatch_receipt.environment_fingerprint,
            packet.stream_binding.runtime_id,
            packet.stream_binding.runtime_generation,
            "capture-runtime-reset",
            1,
            hashlib.sha256(b"capture-quiesced").hexdigest(),
        )
        with capture_attention_protocol("provider-teardown") as capture:
            session = AttentionLaunchSession(
                packet,
                resolved,
                AttentionLeaseRegistry(),
                AttentionHostBufferRegistry(),
                registered_launcher(resolved, evidence),
            )
            with self.assertRaises(RuntimeError):
                session.submit(provider)
            self.assertEqual(capture.traces, ())
            self.assertEqual(capture.incomplete_count, 1)
            with self.assertRaisesRegex(SchemaError, "incomplete recorders"):
                capture.to_corpus("must-not-publish")
            session.release_after_runtime_teardown(teardown)

        self.assertEqual(capture.incomplete_count, 0)
        self.assertEqual(
            tuple(event.state for event in capture.traces[0].events),
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.SUBMIT_UNKNOWN,
                AttentionProtocolState.RUNTIME_QUIESCED,
                AttentionProtocolState.RELEASED,
            ),
        )

    def test_resource_busy_retries_but_terminal_sync_error_requires_release(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe, submit_codes=(7, 0))
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        busy = session.submit(provider)
        self.assertTrue(busy.retryable)
        self.assertEqual(session.state, AttentionLaunchSessionState.PREPARED)
        session.submit(provider)
        self.assertEqual(session.attempt_count, 2)
        session.poll_completion(provider)
        session.release()

        failed = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        failure = failed.submit(FakeProvider(probe, submit_codes=(6,)))
        self.assertEqual(failure.code_name, "launch_failure")
        self.assertEqual(failed.state, AttentionLaunchSessionState.FAILED_SYNC)
        failed.release()

    def test_async_failure_is_owned_only_by_completion_event(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe, completion_code=8)
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        session.submit(provider)
        completion = session.poll_completion(provider)
        self.assertEqual(completion.code_name, "async_failure")
        self.assertEqual(session.state, AttentionLaunchSessionState.FAILED_ASYNC)
        session.release()
        with self.assertRaisesRegex(SchemaError, "asynchronous error code"):
            AttentionProviderSubmitResult(
                probe.fingerprint,
                resolved.fingerprint,
                packet.fingerprint,
                8,
            )
        with self.assertRaisesRegex(SchemaError, "asynchronous"):
            AttentionProviderCompletionResult(
                probe.fingerprint,
                resolved.fingerprint,
                packet.fingerprint,
                "event",
                6,
                hashlib.sha256(b"detail").hexdigest(),
            )

    def test_stale_result_and_wrong_event_do_not_advance_session(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        stale = FakeProvider(probe)
        stale.stale_packet_result = True
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        with self.assertRaisesRegex(AttentionLauncherProviderError, "stale"):
            session.submit(stale)
        self.assertEqual(session.state, AttentionLaunchSessionState.SUBMIT_UNKNOWN)
        with self.assertRaisesRegex(AttentionLauncherProviderError, "outcome is unknown"):
            session.release()

        provider = FakeProvider(probe)
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        session.submit(provider)
        provider.wrong_completion_event = True
        with self.assertRaisesRegex(AttentionLauncherProviderError, "event ownership"):
            session.poll_completion(provider)
        self.assertEqual(session.state, AttentionLaunchSessionState.SUBMITTED)
        provider.wrong_completion_event = False
        session.poll_completion(provider)
        session.release()

        raises = FakeProvider(probe)
        raises.raise_on_submit = True
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            session.submit(raises)
        self.assertEqual(session.state, AttentionLaunchSessionState.SUBMIT_UNKNOWN)

    def test_resolved_symbol_cannot_unload_while_session_owns_it(self):
        packet, _, _, evidence, resolved = resolution_fixture()
        launchers = registered_launcher(resolved, evidence)
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            launchers,
        )
        self.assertEqual(launchers.active_count(resolved), 1)
        with self.assertRaisesRegex(AttentionLauncherProviderError, "sessions are active"):
            launchers.unload(resolved)
        session.release()
        launchers.unload(resolved)
        self.assertEqual(
            launchers.state(resolved), AttentionResolvedLauncherState.UNLOADED
        )
        with self.assertRaisesRegex(AttentionLauncherProviderError, "unloaded"):
            AttentionLaunchSession(
                packet,
                resolved,
                AttentionLeaseRegistry(),
                AttentionHostBufferRegistry(),
                launchers,
            )

    def test_host_descriptor_arena_cannot_be_shared_by_active_sessions(self):
        packet, _, _, evidence, resolved = resolution_fixture()
        host_buffers = AttentionHostBufferRegistry()
        launchers = registered_launcher(resolved, evidence)
        first = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            host_buffers,
            launchers,
        )
        with self.assertRaisesRegex(AttentionLauncherProviderError, "Host descriptor"):
            AttentionLaunchSession(
                packet,
                resolved,
                AttentionLeaseRegistry(),
                host_buffers,
                launchers,
            )
        first.release()
        second = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            host_buffers,
            launchers,
        )
        second.release()

    def test_unknown_submit_can_recover_as_not_submitted_then_retry(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        with self.assertRaises(RuntimeError):
            session.submit(provider)
        self.assertEqual(session.attempt_count, 1)
        provider.raise_on_submit = False
        provider.recovery_status = AttentionUnknownSubmitStatus.NOT_SUBMITTED
        recovered = session.recover_unknown(provider)
        self.assertEqual(recovered.status, AttentionUnknownSubmitStatus.NOT_SUBMITTED)
        self.assertEqual(session.state, AttentionLaunchSessionState.PREPARED)
        session.submit(provider)
        self.assertEqual(session.attempt_count, 2)
        session.poll_completion(provider)
        session.release()

    def test_unknown_submit_can_recover_as_submitted_or_completed(self):
        packet, _, probe, evidence, resolved = resolution_fixture()

        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        submitted = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        with self.assertRaises(RuntimeError):
            submitted.submit(provider)
        provider.raise_on_submit = False
        provider.recovery_status = AttentionUnknownSubmitStatus.SUBMITTED
        submitted.recover_unknown(provider)
        self.assertEqual(submitted.state, AttentionLaunchSessionState.SUBMITTED)
        submitted.poll_completion(provider)
        submitted.release()

        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        completed = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        with self.assertRaises(RuntimeError):
            completed.submit(provider)
        provider.raise_on_submit = False
        provider.recovery_status = AttentionUnknownSubmitStatus.COMPLETED
        provider.recovery_completion_code = 8
        completed.recover_unknown(provider)
        self.assertEqual(completed.state, AttentionLaunchSessionState.FAILED_ASYNC)
        completed.release()

    def test_indeterminate_or_stale_recovery_retains_all_ownership(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        host_registry = AttentionHostBufferRegistry()
        launcher_registry = registered_launcher(resolved, evidence)
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            host_registry,
            launcher_registry,
        )
        with self.assertRaises(RuntimeError):
            session.submit(provider)
        provider.raise_on_submit = False
        result = session.recover_unknown(provider)
        self.assertEqual(result.status, AttentionUnknownSubmitStatus.INDETERMINATE)
        self.assertEqual(session.state, AttentionLaunchSessionState.SUBMIT_UNKNOWN)
        provider.stale_recovery_attempt = True
        with self.assertRaisesRegex(AttentionLauncherProviderError, "attempt is stale"):
            session.recover_unknown(provider)
        self.assertEqual(host_registry.state(session.host_buffer_token).value, "acquired")
        self.assertEqual(launcher_registry.active_count(resolved), 1)

    def test_not_submitted_recovery_cannot_contradict_registry_evidence(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, evidence),
        )
        with self.assertRaises(RuntimeError):
            session.submit(provider)
        session.registry.submit(session.lease_token, "observed-event")
        provider.raise_on_submit = False
        provider.recovery_status = AttentionUnknownSubmitStatus.NOT_SUBMITTED
        with self.assertRaisesRegex(
            AttentionLauncherProviderError, "contradicts registered submission"
        ):
            session.recover_unknown(provider)
        self.assertEqual(session.state, AttentionLaunchSessionState.SUBMIT_UNKNOWN)

    def test_partial_registry_event_drift_cannot_be_recovered_as_submitted(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        teardown = AttentionRuntimeTeardownEvidence(
            probe.fingerprint,
            packet.dispatch_receipt.environment_fingerprint,
            packet.stream_binding.runtime_id,
            packet.stream_binding.runtime_generation,
            "fault-injection-reset",
            1,
            hashlib.sha256(b"quiesced after partial registration").hexdigest(),
        )

        for side in ("device", "host"):
            provider = FakeProvider(probe)
            provider.raise_on_submit = True
            session = AttentionLaunchSession(
                packet,
                resolved,
                AttentionLeaseRegistry(),
                AttentionHostBufferRegistry(),
                registered_launcher(resolved, evidence),
            )
            with self.assertRaises(RuntimeError):
                session.submit(provider)
            if side == "device":
                session.registry.submit(session.lease_token, "drift-event")
            else:
                session.host_buffer_registry.submit(
                    session.host_buffer_token, "drift-event"
                )
            provider.raise_on_submit = False
            provider.recovery_status = AttentionUnknownSubmitStatus.SUBMITTED
            with self.subTest(side=side):
                with self.assertRaisesRegex(
                    AttentionLauncherProviderError,
                    "%s.*event contradicts" % ("device lease" if side == "device" else "Host buffer"),
                ):
                    session.recover_unknown(provider)
                self.assertEqual(
                    session.state, AttentionLaunchSessionState.SUBMIT_UNKNOWN
                )
            session.release_after_runtime_teardown(teardown)

    def test_runtime_teardown_evidence_is_generation_scoped(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = FakeProvider(probe)
        provider.raise_on_submit = True
        coordinator = AttentionLaunchCoordinator()
        coordinator.register_launcher(resolved, evidence)
        session = coordinator.prepare(packet, resolved)
        with self.assertRaises(RuntimeError):
            session.submit(provider)
        base = AttentionRuntimeTeardownEvidence(
            probe.fingerprint,
            packet.dispatch_receipt.environment_fingerprint,
            packet.stream_binding.runtime_id,
            packet.stream_binding.runtime_generation,
            "synthetic-runtime-reset",
            1,
            hashlib.sha256(b"all fake work quiesced").hexdigest(),
        )
        self.assertEqual(
            AttentionRuntimeTeardownEvidence.from_dict(base.to_dict()), base
        )
        with self.assertRaisesRegex(AttentionLauncherProviderError, "does not own"):
            session.release_after_runtime_teardown(
                replace(base, runtime_generation=base.runtime_generation + 1)
            )
        session.release_after_runtime_teardown(base)
        self.assertEqual(session.state, AttentionLaunchSessionState.RELEASED)
        coordinator.unload_launcher(resolved)

    def test_coordinator_shares_all_registries_across_sessions(self):
        packet, _, _, evidence, resolved = resolution_fixture()
        coordinator = AttentionLaunchCoordinator()
        coordinator.register_launcher(resolved, evidence)
        first = coordinator.prepare(packet, resolved)
        self.assertIs(first.registry, coordinator.device_leases)
        self.assertIs(first.host_buffer_registry, coordinator.host_buffers)
        self.assertIs(first.launcher_registry, coordinator.launchers)
        with self.assertRaises(AttentionStorageLeaseError):
            coordinator.prepare(packet, resolved)
        first.release()
        second = coordinator.prepare(packet, resolved)
        second.release()

    def test_concurrent_prepare_admits_exactly_one_conflicting_session(self):
        packet, _, _, evidence, resolved = resolution_fixture()
        coordinator = AttentionLaunchCoordinator()
        coordinator.register_launcher(resolved, evidence)
        worker_count = 8
        barrier = Barrier(worker_count)

        def prepare_once():
            barrier.wait()
            try:
                return coordinator.prepare(packet, resolved)
            except AttentionStorageLeaseError:
                return None

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            sessions = list(pool.map(lambda _: prepare_once(), range(worker_count)))
        admitted = [session for session in sessions if session is not None]
        self.assertEqual(len(admitted), 1)
        admitted[0].release()
        coordinator.unload_launcher(resolved)

    def test_poll_serializes_release_and_teardown_races(self):
        packet, _, probe, evidence, resolved = resolution_fixture()
        provider = BlockingCompletionProvider(probe)
        coordinator = AttentionLaunchCoordinator()
        coordinator.register_launcher(resolved, evidence)
        session = coordinator.prepare(packet, resolved)
        session.submit(provider)

        release_started = Event()

        def release_session():
            release_started.set()
            session.release()

        with ThreadPoolExecutor(max_workers=2) as pool:
            poll = pool.submit(session.poll_completion, provider)
            self.assertTrue(provider.completion_entered.wait(1.0))
            release = pool.submit(release_session)
            self.assertTrue(release_started.wait(1.0))
            self.assertFalse(release.done())
            provider.allow_completion.set()
            self.assertEqual(poll.result().return_code, 0)
            release.result()
        self.assertEqual(session.state, AttentionLaunchSessionState.RELEASED)

        provider = BlockingCompletionProvider(probe)
        session = coordinator.prepare(packet, resolved)
        session.submit(provider)
        teardown = AttentionRuntimeTeardownEvidence(
            probe.fingerprint,
            packet.dispatch_receipt.environment_fingerprint,
            packet.stream_binding.runtime_id,
            packet.stream_binding.runtime_generation,
            "race-reset",
            1,
            hashlib.sha256(b"runtime quiesced").hexdigest(),
        )
        teardown_started = Event()

        def teardown_session():
            teardown_started.set()
            session.release_after_runtime_teardown(teardown)

        with ThreadPoolExecutor(max_workers=2) as pool:
            poll = pool.submit(session.poll_completion, provider)
            self.assertTrue(provider.completion_entered.wait(1.0))
            force_release = pool.submit(teardown_session)
            self.assertTrue(teardown_started.wait(1.0))
            self.assertFalse(force_release.done())
            provider.allow_completion.set()
            poll.result()
            with self.assertRaisesRegex(
                AttentionLauncherProviderError, "requires unknown/submitted"
            ):
                force_release.result()
        session.release()
        coordinator.unload_launcher(resolved)


if __name__ == "__main__":
    unittest.main()
