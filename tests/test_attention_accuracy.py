import math
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionAccuracyBudget,
    AttentionAccuracyCorpus,
    AttentionAccuracyExpectationError,
    AttentionAccuracyReport,
    AttentionErrorTolerance,
    AttentionMode,
    AttentionPlanSpec,
    AttentionTrace,
    ReferenceAttentionResult,
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    RaggedKVCacheSpec,
    SingleAttentionMetadata,
    compare_attention_results,
    build_attention_accuracy_corpus,
    evaluate_attention_accuracy,
)
from flashinfer_npu.runtime import QuantSpec, SchemaError


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def paired_traces(dense_values, storage_values, *, scale=1.0):
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
    )
    common = dict(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        q_dtype="float32",
        o_dtype="float32",
        sm_scale=1.0,
    )
    dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
    quantized_spec = AttentionPlanSpec(
        kv_dtype="int8", kv_quant_spec=quant, **common
    )
    metadata = SingleAttentionMetadata(1, 2)
    q = tensor([[[0.0]]])
    dense_cache = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
    quantized_cache = RaggedKVCacheSpec(
        2, 1, 1, 1, "int8", device="cpu", quant_spec=quant
    )
    dense = AttentionTrace.capture(
        spec=dense_spec,
        metadata=metadata,
        q=q,
        kv_data=ReferenceKVData(
            dense_cache,
            (
                tensor([[[0.0]], [[0.0]]]),
                tensor([[[dense_values[0]]], [[dense_values[1]]]]),
            ),
        ),
    )
    quantized = AttentionTrace.capture(
        spec=quantized_spec,
        metadata=metadata,
        q=q,
        kv_data=ReferenceQuantizedKVData(
            quantized_cache,
            ReferenceQuantizedTensor(
                (2, 1, 1), tensor([[[0]], [[0]]], "int8"), tensor(scale), quant
            ),
            ReferenceQuantizedTensor(
                (2, 1, 1),
                tensor([[[storage_values[0]]], [[storage_values[1]]]], "int8"),
                tensor(scale),
                quant,
            ),
        ),
    )
    return dense, quantized


class AttentionAccuracyBudgetTests(unittest.TestCase):
    def test_builtin_accuracy_corpus_round_trips_and_replays_declared_verdicts(self):
        corpus = build_attention_accuracy_corpus()
        self.assertEqual(corpus.name, "attention-quantization-accuracy-v1")
        self.assertEqual(len(corpus.cases), 4)
        restored = AttentionAccuracyCorpus.from_json(corpus.to_json(indent=2))
        self.assertEqual(restored.to_json(), corpus.to_json())
        self.assertEqual(restored.fingerprint, corpus.fingerprint)
        reports = dict(restored.replay_all())
        self.assertTrue(reports["exact_int8"].quantization_within_budget)
        self.assertTrue(
            reports["lossy_asymmetric_uint8"].quantization_within_budget
        )
        self.assertTrue(
            reports["lossy_packed_int4_odd_dimension"].quantization_within_budget
        )
        self.assertFalse(
            reports["int8_scale_overflow_rejected"].quantization_within_budget
        )
        tampered = restored.cases[0]
        with self.assertRaises(AttentionAccuracyExpectationError):
            replace(tampered, expect_quantization_pass=False).evaluate()

    def test_budget_round_trip_and_fingerprint_are_stable(self):
        budget = AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(0.25, 0.01),
            quantization_lse=AttentionErrorTolerance(0.1, 0.02),
            backend_output=AttentionErrorTolerance(1e-4, 1e-3),
            backend_lse=AttentionErrorTolerance(2e-4, 2e-3),
        )
        restored = AttentionAccuracyBudget.from_dict(budget.to_dict())
        self.assertEqual(restored, budget)
        self.assertEqual(restored.fingerprint, budget.fingerprint)
        with self.assertRaisesRegex(SchemaError, "finite and non-negative"):
            AttentionErrorTolerance(-1.0, 0.0)
        invalid = budget.to_dict()
        invalid["schema_version"] = 2
        with self.assertRaisesRegex(SchemaError, "budget version"):
            AttentionAccuracyBudget.from_dict(invalid)

    def test_exact_int8_has_zero_quantization_and_backend_error(self):
        dense, quantized = paired_traces((1.0, 3.0), (1, 3))
        report = evaluate_attention_accuracy(
            dense, quantized, budget=AttentionAccuracyBudget()
        )
        self.assertTrue(report.passes)
        self.assertEqual(report.quantization.output.max_abs_error, 0.0)
        self.assertEqual(report.backend.output.max_abs_error, 0.0)
        self.assertEqual(len(report.fingerprint), 64)
        restored = AttentionAccuracyReport.from_dict(report.to_dict())
        self.assertEqual(restored, report)
        self.assertEqual(restored.fingerprint, report.fingerprint)
        tampered = report.to_dict()
        tampered["passes"] = False
        with self.assertRaisesRegex(SchemaError, "passes is inconsistent"):
            AttentionAccuracyReport.from_dict(tampered)
        malformed = report.to_dict()
        malformed["candidate_result_fingerprint"] = 7
        with self.assertRaisesRegex(SchemaError, "SHA-256"):
            AttentionAccuracyReport.from_dict(malformed)

    def test_lossy_quantization_uses_its_own_budget(self):
        dense, quantized = paired_traces((1.1, 2.1), (1, 2))
        budget = AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(atol=0.11),
        )
        report = evaluate_attention_accuracy(dense, quantized, budget=budget)
        self.assertTrue(report.quantization_within_budget)
        self.assertTrue(report.backend_within_budget)
        self.assertTrue(report.passes)
        self.assertAlmostEqual(report.quantization.output.max_abs_error, 0.1)

    def test_backend_error_does_not_get_charged_to_quantization(self):
        dense, quantized = paired_traces((1.1, 2.1), (1, 2))
        reference = quantized.replay()
        candidate = ReferenceAttentionResult(
            ReferenceTensor(
                reference.output.shape,
                (reference.output.data[0] + 0.2,),
                reference.output.dtype,
                "npu:0",
            ),
            reference.lse,
        )
        report = evaluate_attention_accuracy(
            dense,
            quantized,
            budget=AttentionAccuracyBudget(
                quantization_output=AttentionErrorTolerance(atol=0.11),
                backend_output=AttentionErrorTolerance(atol=0.01),
            ),
            candidate=candidate,
        )
        self.assertTrue(report.quantization_within_budget)
        self.assertFalse(report.backend_within_budget)
        self.assertFalse(report.passes)
        self.assertEqual(report.backend.output.tolerance_violations, 1)

    def test_scale_overflow_is_quantization_failure_not_backend_failure(self):
        dense, quantized = paired_traces(
            (1e308, 1e308), (127, 127), scale=1.5e306
        )
        report = evaluate_attention_accuracy(
            dense,
            quantized,
            budget=AttentionAccuracyBudget(
                quantization_output=AttentionErrorTolerance(atol=1e308)
            ),
        )
        self.assertFalse(report.quantization_within_budget)
        self.assertTrue(report.backend_within_budget)
        self.assertEqual(report.quantization.output.nonfinite_mismatches, 1)

    def test_nonfinite_semantics_and_trace_pair_identity_are_strict(self):
        expected = ReferenceAttentionResult(
            ReferenceTensor((3,), (math.nan, math.inf, -math.inf)), None
        )
        matching = ReferenceAttentionResult(
            ReferenceTensor(
                (3,), (math.nan, math.inf, -math.inf), device="npu:0"
            ),
            None,
        )
        metrics = compare_attention_results(expected, matching)
        self.assertTrue(metrics.within_budget)
        self.assertEqual(metrics.output.nonfinite_matches, 3)
        mismatching = ReferenceAttentionResult(
            ReferenceTensor((3,), (0.0, math.inf, math.inf)), None
        )
        metrics = compare_attention_results(expected, mismatching)
        self.assertFalse(metrics.within_budget)
        self.assertEqual(metrics.output.nonfinite_mismatches, 2)

        dense, quantized = paired_traces((1.0, 3.0), (1, 3))
        incompatible = replace(
            quantized, spec=replace(quantized.spec, window_left=0)
        )
        with self.assertRaisesRegex(SchemaError, "plan semantics"):
            evaluate_attention_accuracy(
                dense, incompatible, budget=AttentionAccuracyBudget()
            )


if __name__ == "__main__":
    unittest.main()
