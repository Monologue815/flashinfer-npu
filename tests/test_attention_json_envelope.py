import json
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionJsonEnvelopeLimits,
    AttentionTrace,
    build_framework_attention_corpus,
    decode_attention_json,
)
from flashinfer_npu.runtime import SchemaError


class AttentionJsonEnvelopeTests(unittest.TestCase):
    def test_builtin_trace_is_measured_before_domain_construction(self):
        trace = build_framework_attention_corpus().cases[0].trace
        decoded, usage = decode_attention_json(trace.to_json())
        self.assertEqual(decoded["kind"], "attention_conformance")
        self.assertGreaterEqual(usage.tensors, 5)
        self.assertGreater(usage.total_tensor_elements, 0)
        restored = AttentionTrace.from_json(trace.to_json())
        self.assertEqual(restored.fingerprint, trace.fingerprint)

    def test_bytes_and_lexical_depth_are_bounded_before_json_loads(self):
        with self.assertRaisesRegex(SchemaError, "bytes exceed"):
            decode_attention_json(
                "{}", limits=AttentionJsonEnvelopeLimits(max_bytes=1)
            )
        nested = "[" * 5 + "0" + "]" * 5
        with self.assertRaisesRegex(SchemaError, "nesting depth"):
            decode_attention_json(
                nested,
                limits=AttentionJsonEnvelopeLimits(max_nesting_depth=4),
            )
        decoded, _ = decode_attention_json(
            json.dumps({"text": "[[[[[[ not structural nesting"}),
            limits=AttentionJsonEnvelopeLimits(max_nesting_depth=3),
        )
        self.assertIn("not structural", decoded["text"])

    def test_duplicate_keys_and_nonstandard_constants_are_rejected(self):
        with self.assertRaisesRegex(SchemaError, "duplicate JSON object key"):
            decode_attention_json('{"a":1,"a":2}')
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaisesRegex(SchemaError, "strict JSON"):
                    decode_attention_json('{"value":%s}' % constant)

    def test_generic_container_and_string_limits_are_independent(self):
        cases = (
            (
                '{"a":1,"b":2}',
                AttentionJsonEnvelopeLimits(max_object_fields=1),
                "object fields",
            ),
            (
                "[1,2]",
                AttentionJsonEnvelopeLimits(max_array_items=1),
                "array exceeds",
            ),
            (
                '{"a":"long"}',
                AttentionJsonEnvelopeLimits(max_string_bytes=3),
                "string exceeds",
            ),
            (
                "[1]",
                AttentionJsonEnvelopeLimits(max_nodes=1),
                "nodes exceed",
            ),
        )
        for payload, limits, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    decode_attention_json(payload, limits=limits)

    def test_tensor_rank_per_tensor_and_aggregate_elements_are_bounded(self):
        tensor = {
            "shape": [2, 2],
            "data": [0, 0, 0, 0],
            "dtype": "float32",
            "device": "cpu",
        }
        with self.assertRaisesRegex(SchemaError, "tensor rank"):
            decode_attention_json(
                json.dumps(tensor),
                limits=AttentionJsonEnvelopeLimits(max_tensor_rank=1),
            )
        with self.assertRaisesRegex(SchemaError, "tensor elements"):
            decode_attention_json(
                json.dumps(tensor),
                limits=AttentionJsonEnvelopeLimits(max_tensor_elements=3),
            )
        with self.assertRaisesRegex(SchemaError, "total tensor elements"):
            decode_attention_json(
                json.dumps([tensor, tensor]),
                limits=AttentionJsonEnvelopeLimits(
                    max_tensor_elements=4,
                    max_total_tensor_elements=7,
                ),
            )

    def test_tensor_count_and_corpus_case_count_are_bounded(self):
        tensor = {
            "shape": [1],
            "data": [0],
            "dtype": "float32",
            "device": "cpu",
        }
        with self.assertRaisesRegex(SchemaError, "tensors exceed"):
            decode_attention_json(
                json.dumps([tensor, tensor]),
                limits=AttentionJsonEnvelopeLimits(max_tensors=1),
            )
        corpus_envelope = {
            "kind": "attention_conformance_corpus",
            "cases": [{}, {}],
        }
        with self.assertRaisesRegex(SchemaError, "corpus cases"):
            decode_attention_json(
                json.dumps(corpus_envelope),
                limits=AttentionJsonEnvelopeLimits(max_cases=1),
            )

    def test_claimed_shape_is_charged_even_when_data_is_underfilled(self):
        tensor = {
            "shape": [100],
            "data": [],
            "dtype": "float32",
            "device": "cpu",
        }
        with self.assertRaisesRegex(SchemaError, "tensor elements"):
            decode_attention_json(
                json.dumps(tensor),
                limits=AttentionJsonEnvelopeLimits(max_tensor_elements=99),
            )

    def test_from_json_honors_caller_supplied_envelope(self):
        trace = build_framework_attention_corpus().cases[0].trace
        limits = replace(
            AttentionJsonEnvelopeLimits(), max_total_tensor_elements=1
        )
        with self.assertRaisesRegex(SchemaError, "total tensor elements"):
            AttentionTrace.from_json(trace.to_json(), limits=limits)

    def test_limits_reject_nonpositive_values(self):
        with self.assertRaisesRegex(SchemaError, "positive integer"):
            AttentionJsonEnvelopeLimits(max_cases=0)


if __name__ == "__main__":
    unittest.main()
