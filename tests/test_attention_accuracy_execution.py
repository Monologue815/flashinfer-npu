import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionAccuracyExecutionBinding,
    AttentionAccuracyExecutionBindingError,
    AttentionHostBufferRegistry,
    AttentionLaunchSession,
    AttentionLeaseRegistry,
    ReferenceAttentionResult,
    ReferenceTensor,
    bind_attention_accuracy_dispatch,
    bind_attention_accuracy_execution,
    capture_attention_protocol,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_accuracy_evidence import accuracy_dispatch_fixture
from tests.test_attention_launcher_provider import (
    FakeProvider,
    registered_launcher,
    resolution_fixture,
)


def execution_binding_fixture(completion_code=0):
    (
        conformance_corpus,
        case,
        accuracy_corpus,
        candidate,
        report,
        profile,
        descriptor,
        receipt,
    ) = accuracy_dispatch_fixture()
    dispatch_binding = bind_attention_accuracy_dispatch(
        binding_id="synthetic-host-binding-v1",
        runner="host-contract-test",
        accuracy_case=case,
        accuracy_corpus=accuracy_corpus,
        accuracy_report=report,
        candidate=candidate,
        dispatch_receipt=receipt,
        profile=profile,
        descriptor=descriptor,
        environment=profile.environment,
        conformance_corpus=conformance_corpus,
    )
    packet, _kernel, probe, load_evidence, resolved = resolution_fixture()
    provider = FakeProvider(probe, completion_code=completion_code)
    with capture_attention_protocol("accuracy-execution") as capture:
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, load_evidence),
        )
        submit = session.submit(provider)
        completion = session.poll_completion(provider)
        session.release()
    return (
        dispatch_binding,
        candidate,
        packet,
        resolved,
        submit,
        completion,
        capture.traces[0],
    )


class AttentionAccuracyExecutionBindingTests(unittest.TestCase):
    def test_successful_provider_lifecycle_round_trips_and_revalidates(self):
        values = execution_binding_fixture()
        binding = bind_attention_accuracy_execution(
            binding_id="synthetic-execution-binding-v1",
            runner="host-contract-provider",
            accuracy_dispatch_binding=values[0],
            candidate=values[1],
            launch_packet=values[2],
            resolved_launcher=values[3],
            submit_result=values[4],
            completion_result=values[5],
            protocol_trace=values[6],
        )
        restored = AttentionAccuracyExecutionBinding.from_dict(binding.to_dict())
        self.assertEqual(restored, binding)
        self.assertEqual(restored.fingerprint, binding.fingerprint)
        restored.validate(*values)
        self.assertEqual(binding.launch_packet_fingerprint, values[2].fingerprint)
        self.assertEqual(binding.protocol_trace_fingerprint, values[6].fingerprint)
        self.assertEqual(binding.result_origin, "runner_declared_post_completion")

    def test_async_failure_cannot_bind_accuracy_result(self):
        values = execution_binding_fixture(completion_code=8)
        with self.assertRaisesRegex(
            AttentionAccuracyExecutionBindingError, "successful provider completion"
        ):
            bind_attention_accuracy_execution(
                binding_id="failed-execution",
                runner="host-contract-provider",
                accuracy_dispatch_binding=values[0],
                candidate=values[1],
                launch_packet=values[2],
                resolved_launcher=values[3],
                submit_result=values[4],
                completion_result=values[5],
                protocol_trace=values[6],
            )

    def test_candidate_provider_and_protocol_identity_drift_are_rejected(self):
        values = execution_binding_fixture()
        changed_candidate = ReferenceAttentionResult(
            ReferenceTensor(
                values[1].output.shape,
                tuple(value + 1.0 for value in values[1].output.data),
                values[1].output.dtype,
                values[1].output.device,
            ),
            values[1].lse,
        )
        variants = (
            (
                dict(candidate=changed_candidate),
                "candidate result does not match",
            ),
            (
                dict(
                    completion_result=replace(
                        values[5], launch_packet_fingerprint="f" * 64
                    )
                ),
                "identity chain is inconsistent",
            ),
            (
                dict(
                    protocol_trace=replace(
                        values[6], subject_fingerprint="0" * 64
                    )
                ),
                "subject or stream",
            ),
        )
        names = (
            "accuracy_dispatch_binding",
            "candidate",
            "launch_packet",
            "resolved_launcher",
            "submit_result",
            "completion_result",
            "protocol_trace",
        )
        base = dict(zip(names, values))
        for changes, reason in variants:
            current = dict(base)
            current.update(changes)
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(
                    AttentionAccuracyExecutionBindingError, reason
                ):
                    bind_attention_accuracy_execution(
                        binding_id="drifted-execution",
                        runner="host-contract-provider",
                        **current,
                    )

    def test_stored_binding_tamper_and_schema_errors_are_rejected(self):
        values = execution_binding_fixture()
        binding = bind_attention_accuracy_execution(
            binding_id="synthetic-execution-binding-v1",
            runner="host-contract-provider",
            accuracy_dispatch_binding=values[0],
            candidate=values[1],
            launch_packet=values[2],
            resolved_launcher=values[3],
            submit_result=values[4],
            completion_result=values[5],
            protocol_trace=values[6],
        )
        stale = replace(binding, protocol_trace_fingerprint="1" * 64)
        with self.assertRaisesRegex(
            AttentionAccuracyExecutionBindingError, "stale or inconsistent"
        ):
            stale.validate(*values)
        with self.assertRaisesRegex(SchemaError, "result_origin"):
            replace(binding, result_origin="attested")
        with self.assertRaisesRegex(SchemaError, "runtime_generation"):
            replace(binding, runtime_generation=0)


if __name__ == "__main__":
    unittest.main()
