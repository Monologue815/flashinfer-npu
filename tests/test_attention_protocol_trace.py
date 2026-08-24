import hashlib
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionProtocolEvent,
    AttentionProtocolRoute,
    AttentionProtocolState,
    AttentionProtocolTrace,
    AttentionProtocolTraceCase,
    AttentionProtocolTraceCorpus,
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
)
from flashinfer_npu.runtime import SchemaError


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def events(states, *, stream="stream-0", owner="owner-0"):
    return tuple(
        AttentionProtocolEvent(
            sequence=index,
            state=state,
            stream_context_fingerprint=digest(stream),
            ownership_fingerprint=digest(owner),
            evidence_fingerprint=digest("evidence-%d-%s" % (index, state.value)),
        )
        for index, state in enumerate(states)
    )


def jit_trace(trace_id="jit-trace"):
    return AttentionProtocolTrace(
        trace_id,
        AttentionProtocolRoute.SINGLE_JIT,
        digest("jit-call-abi-" + trace_id),
        events(
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.INVOKED,
                AttentionProtocolState.COMPLETED,
                AttentionProtocolState.RELEASED,
            ),
            stream="stream-" + trace_id,
            owner="owner-" + trace_id,
        ),
    )


class AttentionProtocolTraceTests(unittest.TestCase):
    def test_single_jit_success_round_trips_with_stable_fingerprint(self):
        trace = AttentionProtocolTrace(
            "jit-success",
            AttentionProtocolRoute.SINGLE_JIT,
            digest("jit-call-abi"),
            events(
                (
                    AttentionProtocolState.PREPARED,
                    AttentionProtocolState.INVOKED,
                    AttentionProtocolState.COMPLETED,
                    AttentionProtocolState.RELEASED,
                )
            ),
        )
        restored = AttentionProtocolTrace.from_json(trace.to_json(indent=2))
        self.assertEqual(restored, trace)
        self.assertEqual(restored.fingerprint, trace.fingerprint)

    def test_provider_unknown_recovery_and_teardown_paths_are_explicit(self):
        recovered = AttentionProtocolTrace(
            "provider-recovered",
            AttentionProtocolRoute.PROVIDER,
            digest("launch-packet"),
            events(
                (
                    AttentionProtocolState.PREPARED,
                    AttentionProtocolState.SUBMIT_UNKNOWN,
                    AttentionProtocolState.SUBMIT_UNKNOWN,
                    AttentionProtocolState.SUBMITTED,
                    AttentionProtocolState.COMPLETED,
                    AttentionProtocolState.RELEASED,
                )
            ),
        )
        quiesced = AttentionProtocolTrace(
            "provider-quiesced",
            AttentionProtocolRoute.PROVIDER,
            digest("launch-packet-2"),
            events(
                (
                    AttentionProtocolState.PREPARED,
                    AttentionProtocolState.SUBMITTED,
                    AttentionProtocolState.RUNTIME_QUIESCED,
                    AttentionProtocolState.RELEASED,
                ),
                stream="stream-1",
                owner="owner-1",
            ),
        )
        self.assertEqual(recovered.events[-1].state, AttentionProtocolState.RELEASED)
        self.assertEqual(
            quiesced.events[-2].state, AttentionProtocolState.RUNTIME_QUIESCED
        )

    def test_stream_and_resource_ownership_cannot_drift(self):
        base = events(
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.INVOKED,
                AttentionProtocolState.COMPLETED,
                AttentionProtocolState.RELEASED,
            )
        )
        stream_drift = base[:2] + (
            replace(base[2], stream_context_fingerprint=digest("other-stream")),
        ) + base[3:]
        with self.assertRaisesRegex(SchemaError, "stream ownership drifted"):
            AttentionProtocolTrace(
                "stream-drift",
                AttentionProtocolRoute.SINGLE_JIT,
                digest("subject"),
                stream_drift,
            )

        owner_drift = base[:1] + (
            replace(base[1], ownership_fingerprint=digest("other-owner")),
        ) + base[2:]
        with self.assertRaisesRegex(SchemaError, "resource ownership drifted"):
            AttentionProtocolTrace(
                "owner-drift",
                AttentionProtocolRoute.SINGLE_JIT,
                digest("subject"),
                owner_drift,
            )

    def test_route_specific_state_machine_rejects_cross_protocol_transitions(self):
        with self.assertRaisesRegex(SchemaError, "invalid single_jit"):
            AttentionProtocolTrace(
                "jit-async",
                AttentionProtocolRoute.SINGLE_JIT,
                digest("subject"),
                events(
                    (
                        AttentionProtocolState.PREPARED,
                        AttentionProtocolState.SUBMITTED,
                        AttentionProtocolState.COMPLETED,
                        AttentionProtocolState.RELEASED,
                    )
                ),
            )
        with self.assertRaisesRegex(SchemaError, "invalid provider"):
            AttentionProtocolTrace(
                "provider-invoked",
                AttentionProtocolRoute.PROVIDER,
                digest("subject"),
                events(
                    (
                        AttentionProtocolState.PREPARED,
                        AttentionProtocolState.INVOKED,
                        AttentionProtocolState.COMPLETED,
                        AttentionProtocolState.RELEASED,
                    )
                ),
            )

    def test_trace_requires_contiguous_sequence_and_terminal_release(self):
        not_released = events(
            (
                AttentionProtocolState.PREPARED,
                AttentionProtocolState.INVOKED,
                AttentionProtocolState.COMPLETED,
            )
        )
        with self.assertRaisesRegex(SchemaError, "end released"):
            AttentionProtocolTrace(
                "not-released",
                AttentionProtocolRoute.SINGLE_JIT,
                digest("subject"),
                not_released,
            )
        noncontiguous = (
            replace(not_released[0], sequence=1),
            not_released[1],
            replace(not_released[2], sequence=2, state=AttentionProtocolState.RELEASED),
        )
        with self.assertRaisesRegex(SchemaError, "not contiguous"):
            AttentionProtocolTrace(
                "noncontiguous",
                AttentionProtocolRoute.SINGLE_JIT,
                digest("subject"),
                noncontiguous,
            )

    def test_protocol_corpus_round_trip_and_numerical_cross_binding(self):
        bound = AttentionProtocolTraceCase(
            "bound-jit",
            jit_trace("jit-bound"),
            numerical_case_id="single-prefill-int8",
            numerical_input_fingerprint=digest("numerical-input"),
        )
        unbound = AttentionProtocolTraceCase(
            "unbound-jit", jit_trace("jit-unbound")
        )
        corpus = AttentionProtocolTraceCorpus(
            "protocol-smoke-v1", (bound, unbound)
        )
        restored = AttentionProtocolTraceCorpus.from_json(corpus.to_json(indent=2))
        self.assertEqual(restored, corpus)
        self.assertEqual(restored.fingerprint, corpus.fingerprint)
        self.assertEqual(restored.route_counts, {"single_jit": 2, "provider": 0})
        with self.assertRaisesRegex(SchemaError, "corpus cases exceed limit 1"):
            AttentionProtocolTraceCorpus.from_json(
                corpus.to_json(),
                limits=replace(
                    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS, max_cases=1
                ),
            )

    def test_protocol_corpus_rejects_partial_binding_and_duplicate_trace(self):
        trace = jit_trace("jit-duplicate")
        with self.assertRaisesRegex(SchemaError, "provided together"):
            AttentionProtocolTraceCase(
                "partial", trace, numerical_case_id="missing-fingerprint"
            )
        first = AttentionProtocolTraceCase("first", trace)
        duplicate = AttentionProtocolTraceCase("second", trace)
        with self.assertRaisesRegex(SchemaError, "trace ids must be unique"):
            AttentionProtocolTraceCorpus("duplicates", (first, duplicate))


if __name__ == "__main__":
    unittest.main()
