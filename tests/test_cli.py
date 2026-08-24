import io
import hashlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from flashinfer_npu.attention import (
    AttentionAccuracyCorpus,
    AttentionMode,
    AttentionPlanSpec,
    AttentionProtocolEvent,
    AttentionProtocolRoute,
    AttentionProtocolState,
    AttentionProtocolTrace,
    AttentionProtocolTraceCase,
    AttentionProtocolTraceCorpus,
    AttentionTrace,
    AttentionTraceCorpus,
    RaggedKVCacheSpec,
    ReferenceKVData,
    ReferenceTensor,
    SingleAttentionMetadata,
)
from flashinfer_npu.cli import main


def tensor(value):
    return ReferenceTensor.from_nested(value, dtype="float32", device="cpu")


def trace_json():
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_DECODE,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
    )
    cache_spec = RaggedKVCacheSpec(
        1, 1, 1, 1, "float32", device="cpu"
    )
    trace = AttentionTrace.capture(
        spec=spec,
        metadata=SingleAttentionMetadata(1, 1),
        q=tensor([[1.0]]),
        kv_data=ReferenceKVData(
            cache_spec, (tensor([[[1.0]]]), tensor([[[2.0]]]))
        ),
    )
    return trace.to_json(indent=2)


def protocol_corpus_json():
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    states = (
        AttentionProtocolState.PREPARED,
        AttentionProtocolState.INVOKED,
        AttentionProtocolState.COMPLETED,
        AttentionProtocolState.RELEASED,
    )
    trace = AttentionProtocolTrace(
        "cli-jit",
        AttentionProtocolRoute.SINGLE_JIT,
        digest("subject"),
        tuple(
            AttentionProtocolEvent(
                index,
                state,
                digest("stream"),
                digest("owner"),
                digest("evidence-%d" % index),
            )
            for index, state in enumerate(states)
        ),
    )
    return AttentionProtocolTraceCorpus(
        "cli-protocol-v1", (AttentionProtocolTraceCase("cli-case", trace),)
    ).to_json(indent=2)


class AttentionReplayCliTests(unittest.TestCase):
    def test_attention_replay_reports_identity_shape_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(trace_json(), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(["attention-replay", str(path)])
        self.assertEqual(status, 0)
        self.assertIn("trace_fingerprint", stdout.getvalue())
        self.assertIn("single_decode", stdout.getvalue())
        self.assertIn("shape=(1, 1)", stdout.getvalue())
        self.assertIn("validation        : passed", stdout.getvalue())

    def test_attention_replay_rejects_invalid_or_oversized_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{}", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                invalid_status = main(["attention-replay", str(path)])
            self.assertEqual(invalid_status, 2)
            self.assertIn("fields", stderr.getvalue())

            path.write_text(trace_json(), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                oversized_status = main(
                    ["attention-replay", str(path), "--max-bytes", "1"]
                )
            self.assertEqual(oversized_status, 2)
            self.assertIn("max-bytes", stderr.getvalue())


class AttentionCorpusCliTests(unittest.TestCase):
    def test_builtin_accuracy_corpus_is_replayable_and_exportable(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["attention-accuracy"])
        self.assertEqual(status, 0)
        output = stdout.getvalue()
        self.assertIn("cases:    4", output)
        self.assertIn("accepted: 3", output)
        self.assertIn("rejected: 1", output)
        self.assertIn("replay:   passed", output)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["attention-accuracy-corpus"])
        self.assertEqual(status, 0)
        corpus = AttentionAccuracyCorpus.from_json(stdout.getvalue())
        self.assertEqual(corpus.name, "attention-quantization-accuracy-v1")
        self.assertEqual(len(corpus.cases), 4)

    def test_builtin_coverage_is_complete_and_replayable(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(
                ["attention-coverage", "--replay", "--require-complete"]
            )
        self.assertEqual(status, 0)
        self.assertIn("cells:  51/51", stdout.getvalue())
        self.assertIn("complete: yes", stdout.getvalue())
        self.assertIn("replay: passed", stdout.getvalue())

    def test_builtin_corpus_command_emits_valid_versioned_json(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["attention-corpus"])
        self.assertEqual(status, 0)
        corpus = AttentionTraceCorpus.from_json(stdout.getvalue())
        self.assertEqual(corpus.name, "attention-framework-smoke-v4")
        self.assertEqual(len(corpus.cases), 14)

    def test_attention_capabilities_reports_honestly_empty_manifest(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["attention-capabilities"])
        self.assertEqual(exit_code, 0)
        self.assertIn("No Attention backend capability profiles", stdout.getvalue())

    def test_protocol_validate_accepts_corpus_and_rejects_invalid_wire(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(protocol_corpus_json(), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(["attention-protocol-validate", str(path)])
            self.assertEqual(status, 0)
            self.assertIn("kind        : corpus", stdout.getvalue())
            self.assertIn("single_jit=1 provider=0", stdout.getvalue())
            self.assertIn("validation  : passed", stdout.getvalue())

            path.write_text("{}", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                invalid = main(["attention-protocol-validate", str(path)])
            self.assertEqual(invalid, 2)
            self.assertIn("fields", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
