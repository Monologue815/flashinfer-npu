import unittest

from flashinfer_npu.attention import (
    AttentionCoverageCell,
    AttentionCoveragePolicy,
    AttentionTraceCase,
    AttentionTraceCorpus,
    attention_trace_features,
    build_framework_attention_corpus,
    framework_attention_coverage_policy,
)
from flashinfer_npu.runtime import SchemaError


class AttentionCorpusTests(unittest.TestCase):
    def test_builtin_corpus_round_trips_replays_and_has_stable_identity(self):
        corpus = build_framework_attention_corpus()
        self.assertEqual(len(corpus.cases), 14)
        self.assertEqual(
            corpus.fingerprint,
            "7577420ea94a1c97352ea2265a76939b8c566aeaf792daf083a123d3b38da536",
        )
        restored = AttentionTraceCorpus.from_json(corpus.to_json(indent=2))
        # Dataclass equality is unsuitable for a corpus containing NaN because
        # IEEE NaN is not equal to itself; canonical JSON/fingerprint is stable.
        self.assertEqual(restored.to_json(), corpus.to_json())
        self.assertEqual(restored.fingerprint, corpus.fingerprint)
        self.assertEqual(len(restored.replay_all()), 14)

    def test_framework_policy_is_complete_and_exposes_case_matches(self):
        corpus = build_framework_attention_corpus()
        report = framework_attention_coverage_policy().evaluate(corpus)
        self.assertTrue(report.is_complete)
        self.assertEqual(report.covered_cells, 51)
        self.assertEqual(report.missing_cells, ())
        self.assertIn("cells:  51/51", report.format())
        mode_cell = next(
            cell
            for cell in report.to_dict()["cells"]
            if cell["name"] == "mode=batch_mixed_paged"
        )
        self.assertEqual(
            mode_cell["case_ids"],
            [
                "mixed_dense_distinct_dims",
                "mixed_int4_window_softcap_per_head_scale",
                "mixed_asymmetric_uint8_channel_per_head_scale",
            ],
        )
        joint_cells = {
            cell["name"]: cell["case_ids"]
            for cell in report.to_dict()["cells"]
            if cell["name"].startswith("mixed-")
        }
        self.assertEqual(
            joint_cells["mixed-int4-window-softcap-per-head-scale"],
            ["mixed_int4_window_softcap_per_head_scale"],
        )
        self.assertEqual(
            joint_cells["mixed-asymmetric-uint8-channel-per-head-scale"],
            ["mixed_asymmetric_uint8_channel_per_head_scale"],
        )

    def test_paged_int4_combination_cell_requires_one_joint_case(self):
        corpus = build_framework_attention_corpus()
        policy = AttentionCoveragePolicy(
            "combination-check",
            (
                AttentionCoverageCell(
                    "paged-int4",
                    (("cache_kind", "paged"), ("kv_storage", "int4_packed")),
                ),
            ),
        )
        report = policy.evaluate(corpus)
        self.assertTrue(report.is_complete)
        self.assertEqual(
            report.matches[0][1],
            (
                "paged_prefill_int4_multi_request_shared_page",
                "mixed_int4_window_softcap_per_head_scale",
            ),
        )

    def test_features_are_derived_from_trace_not_manual_tags(self):
        corpus = build_framework_attention_corpus()
        cases = {case.case_id: case for case in corpus.cases}
        features = dict(
            attention_trace_features(
                cases["ragged_prefill_uint8_group_packed"].trace
            )
        )
        self.assertEqual(features["mode"], "batch_prefill_ragged")
        self.assertEqual(features["kv_layout"], "HND")
        self.assertEqual(features["kv_storage"], "uint8")
        self.assertEqual(features["quant_scheme"], "asymmetric")
        self.assertEqual(features["quant_granularity"], "group")
        self.assertEqual(features["mask"], "packed")
        self.assertEqual(features["head_mapping"], "gqa")
        mixed_int4 = dict(
            attention_trace_features(
                cases["mixed_int4_window_softcap_per_head_scale"].trace
            )
        )
        self.assertEqual(mixed_int4["window"], "sliding")
        self.assertEqual(mixed_int4["soft_cap"], "runtime")
        self.assertEqual(mixed_int4["kv_runtime_scale"], "per_head")
        mixed_uint8 = dict(
            attention_trace_features(
                cases[
                    "mixed_asymmetric_uint8_channel_per_head_scale"
                ].trace
            )
        )
        self.assertEqual(mixed_uint8["kv_layout"], "HND")
        self.assertEqual(mixed_uint8["quant_scheme"], "asymmetric")
        self.assertEqual(mixed_uint8["quant_granularity"], "channel")
        edge_cases = {
            case_id: dict(attention_trace_features(case.trace))["numerical_edge"]
            for case_id, case in cases.items()
        }
        self.assertEqual(edge_cases["single_prefill_nan_logit"], "nan")
        self.assertEqual(
            edge_cases["single_prefill_positive_infinity_logits"],
            "positive_infinity",
        )
        self.assertEqual(
            edge_cases["single_prefill_negative_infinity_row"],
            "negative_infinity",
        )
        self.assertEqual(
            edge_cases["single_prefill_all_mask"], "zero_support_mask"
        )

    def test_corpus_rejects_duplicate_case_or_input_identity(self):
        corpus = build_framework_attention_corpus()
        first = corpus.cases[0]
        with self.assertRaisesRegex(SchemaError, "case_id"):
            AttentionTraceCorpus("duplicate-id", (first, first))
        duplicate_input = AttentionTraceCase(
            "another_id", first.trace, "same input under another name"
        )
        with self.assertRaisesRegex(SchemaError, "inputs"):
            AttentionTraceCorpus("duplicate-input", (first, duplicate_input))

    def test_corpus_schema_rejects_unknown_fields(self):
        value = build_framework_attention_corpus().to_dict()
        value["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "fields"):
            AttentionTraceCorpus.from_dict(value)


if __name__ == "__main__":
    unittest.main()
