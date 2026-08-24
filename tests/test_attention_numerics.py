import math
import unittest

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionNumericsPolicy,
    AttentionPlanSpec,
    ReferenceAttentionExecutor,
    ReferenceKVData,
    ReferenceTensor,
    RaggedKVCacheSpec,
    SingleAttentionMetadata,
    normalize_attention_logits,
)
from flashinfer_npu.runtime import SchemaError


def tensor(value):
    return ReferenceTensor.from_nested(value, dtype="float32", device="cpu")


def execute_row(keys, values, *, query=1.0):
    metadata = SingleAttentionMetadata(1, len(keys))
    spec = AttentionPlanSpec(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        q_dtype="float32",
        kv_dtype="float32",
        sm_scale=1.0,
    )
    plan = AttentionFrameworkSession(spec.mode).plan(spec, metadata)
    cache = RaggedKVCacheSpec(
        len(keys), 1, 1, 1, "float32", device="cpu"
    )
    kv = ReferenceKVData(
        cache,
        (
            tensor([[[value]] for value in keys]),
            tensor([[[value]] for value in values]),
        ),
    )
    return ReferenceAttentionExecutor().execute(
        plan, tensor([[[query]]]), kv, return_lse=True
    )


class AttentionNumericsPolicyTests(unittest.TestCase):
    def test_policy_round_trip_and_fingerprint_are_versioned(self):
        policy = AttentionNumericsPolicy()
        restored = AttentionNumericsPolicy.from_dict(policy.to_dict())
        self.assertEqual(restored, policy)
        self.assertEqual(restored.fingerprint, policy.fingerprint)
        changed = policy.to_dict()
        changed["schema_version"] = 2
        with self.assertRaisesRegex(SchemaError, "policy version"):
            AttentionNumericsPolicy.from_dict(changed)

    def test_finite_softmax_is_stable_for_large_logits(self):
        probabilities, lse = normalize_attention_logits((1000.0, 999.0))
        self.assertAlmostEqual(probabilities[0], 1.0 / (1.0 + math.exp(-1.0)))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(lse, 1000.0 + math.log1p(math.exp(-1.0)))

    def test_nan_taints_row_independent_of_logit_position(self):
        for logits in ((float("nan"), 1.0), (1.0, float("nan"))):
            probabilities, lse = normalize_attention_logits(logits)
            self.assertTrue(all(math.isnan(value) for value in probabilities))
            self.assertTrue(math.isnan(lse))

    def test_positive_infinities_share_probability_mass(self):
        probabilities, lse = normalize_attention_logits(
            (float("inf"), 4.0, float("inf"), float("-inf"))
        )
        self.assertEqual(probabilities, (0.5, 0.0, 0.5, 0.0))
        self.assertEqual(lse, float("inf"))

    def test_empty_and_all_negative_infinity_rows_have_empty_support(self):
        for logits in ((), (float("-inf"), float("-inf"))):
            probabilities, lse = normalize_attention_logits(logits)
            self.assertEqual(probabilities, ())
            self.assertEqual(lse, float("-inf"))


class ReferenceExceptionalRowTests(unittest.TestCase):
    def test_positive_infinite_logits_average_only_positive_infinite_keys(self):
        result = execute_row(
            (1e308, 1.0, 1e308),
            (2.0, float("inf"), 6.0),
            query=1e308,
        )
        self.assertEqual(result.output.data, (4.0,))
        self.assertEqual(result.lse.data, (float("inf"),))

    def test_all_negative_infinite_logits_match_zero_support_policy(self):
        result = execute_row((-1e308, -1e308), (2.0, 6.0), query=1e308)
        self.assertEqual(result.output.data, (0.0,))
        self.assertEqual(result.lse.data, (float("-inf"),))

    def test_nan_logit_deterministically_taints_output_and_lse(self):
        for keys in ((float("nan"), 1.0), (1.0, float("nan"))):
            result = execute_row(keys, (2.0, 6.0))
            self.assertTrue(math.isnan(result.output.data[0]))
            self.assertTrue(math.isnan(result.lse.data[0]))


if __name__ == "__main__":
    unittest.main()
