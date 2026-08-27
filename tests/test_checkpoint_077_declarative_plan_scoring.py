from dataclasses import replace
import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionMode,
    AttentionOperatorPlanScoreRule,
    AttentionOperatorPlanScoringError,
    AttentionOperatorPlanScoringPolicy,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionDeclaredOperatorPackageRuntimeSpec,
    BatchAttention,
    KVLayout,
    build_attention_operator_package_runtime,
    build_declared_attention_operator_runtime_resolvers,
    describe_attention_operator_package_runtime,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_001_quant_spec_int8_shape import int8_tensor_spec
from tests.test_checkpoint_015_runtime_implementation_auto_selection import (
    FakeImplementation,
    authorized_runtime,
    framework_plan,
)
from tests.test_checkpoint_020_provider_plan_gates import (
    framework_plan as provider_framework_plan,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_076_plan_specific_runtime_scoring import (
    SyntheticPlanFactory,
    plan_public,
)


class PolicyImplementation(FakeImplementation):
    def __init__(self, policy, priority=20):
        super().__init__(
            policy.provider_id,
            policy.operation_id,
            priority,
        )
        self.policy = policy

    def plan_score(self, plan, device):
        return self.policy.score(plan, device)


class PublicPolicyImplementation(PolicyImplementation):
    def resolve(self, plan, device):
        self.resolve_calls += 1
        resolved = authorized_runtime(plan, self.provider_id)
        return replace(
            resolved,
            factory=SyntheticPlanFactory(self.provider_id, self.operation_id),
        )


def policy(provider_id, operation_id, rules, *, policy_id=None, default_score=0):
    return AttentionOperatorPlanScoringPolicy(
        policy_id=policy_id or "%s.attention.policy.v1" % provider_id,
        provider_id=provider_id,
        operation_id=operation_id,
        rules=tuple(rules),
        default_score=default_score,
        default_reason="provider default preference",
    )


def page_rule(rule_id, page_size, score, *, precedence=10):
    return AttentionOperatorPlanScoreRule(
        rule_id=rule_id,
        precedence=precedence,
        score=score,
        reason="page-size preference %d" % page_size,
        modes=(AttentionMode.BATCH_MIXED_PAGED,),
        page_sizes=(page_size,),
    )


class DeclarativePlanScoringCheckpoint(unittest.TestCase):
    """Checkpoint 077: provider preference can be reviewed as versioned data."""

    def test_exact_workload_rule_precedes_general_bucket(self):
        plan = framework_plan(page_size=128)
        exact = AttentionOperatorPlanScoreRule(
            rule_id="exact_tuning_record_v1",
            precedence=100,
            score=97,
            reason="offline tuning record for this exact workload",
            workload_fingerprints=(plan.workload.fingerprint,),
        )
        bucket = AttentionOperatorPlanScoreRule(
            rule_id="paged_128_heuristic_v1",
            precedence=10,
            score=25,
            reason="deterministic page-size heuristic",
            modes=(plan.spec.mode,),
            page_sizes=(128,),
        )
        scorer = policy(
            "cann",
            CANN_V2_OPERATION_ID,
            (bucket, exact),
        )

        score = scorer.score(plan, "npu:0")

        self.assertEqual(score.value, 97)
        self.assertEqual(score.reason, exact.reason)
        self.assertIn(exact.rule_id, score.source)
        self.assertIn(scorer.fingerprint, score.source)

    def test_quantization_kind_and_exact_quantspec_are_matchable(self):
        quant = int8_tensor_spec()
        quantized = provider_framework_plan(
            layout=KVLayout.HND,
            page_size=128,
            kv_dtype="int8",
            quant=quant,
        )
        dense = provider_framework_plan(layout=KVLayout.HND, page_size=128)
        quant_rule = AttentionOperatorPlanScoreRule(
            rule_id="int8_tensor_kv_v1",
            precedence=20,
            score=80,
            reason="INT8 tensor-scale KV path is tuned",
            quantization="quantized",
            quant_spec_fingerprints=(quant.fingerprint,),
        )
        dense_rule = AttentionOperatorPlanScoreRule(
            rule_id="dense_kv_v1",
            precedence=20,
            score=30,
            reason="dense fallback preference",
            quantization="dense",
        )
        scorer = policy(
            "cann",
            CANN_V2_OPERATION_ID,
            (dense_rule, quant_rule),
        )

        self.assertEqual(scorer.score(quantized, "npu:0").value, 80)
        self.assertEqual(scorer.score(dense, "npu:0").value, 30)

    def test_ranges_and_semantic_fields_form_a_deterministic_bucket(self):
        plan = framework_plan(page_size=64)
        matching = AttentionOperatorPlanScoreRule(
            rule_id="small_mixed_bucket_v1",
            precedence=10,
            score=42,
            reason="small mixed workload bucket",
            modes=(plan.spec.mode,),
            kv_layouts=(plan.spec.kv_layout,),
            dtype_signatures=(
                (plan.spec.q_dtype, plan.spec.kv_dtype, plan.spec.o_dtype),
            ),
            page_sizes=(64,),
            head_dim_qk_values=(128,),
            head_dim_vo_values=(128,),
            gqa_group_sizes=(4,),
            causal_values=(True,),
            min_batch_size=2,
            max_batch_size=2,
            min_total_qo_tokens=3,
            max_total_qo_tokens=3,
            min_total_kv_tokens=160,
            max_total_kv_tokens=160,
        )

        self.assertTrue(matching.matches(plan))
        self.assertFalse(
            replace(
                matching,
                min_total_kv_tokens=150,
                max_total_kv_tokens=159,
            ).matches(plan)
        )

    def test_same_precedence_overlaps_fail_closed(self):
        plan = framework_plan(page_size=128)
        scorer = policy(
            "cann",
            CANN_V2_OPERATION_ID,
            (
                page_rule("page_rule_v1", 128, 70, precedence=20),
                AttentionOperatorPlanScoreRule(
                    rule_id="mode_rule_v1",
                    precedence=20,
                    score=60,
                    reason="overlapping mode rule",
                    modes=(plan.spec.mode,),
                ),
            ),
        )

        with self.assertRaisesRegex(
            AttentionOperatorPlanScoringError,
            "ambiguous.*precedence 20",
        ):
            scorer.score(plan, "npu:0")

    def test_policy_round_trip_and_fingerprint_ignore_declaration_order(self):
        first = page_rule("page_64_v1", 64, 80, precedence=20)
        second = AttentionOperatorPlanScoreRule(
            rule_id="dtype_v1",
            precedence=10,
            score=40,
            reason="dense dtype bucket",
            modes=(AttentionMode.BATCH_MIXED_PAGED, AttentionMode.BATCH_DECODE_PAGED),
            kv_layouts=(KVLayout.NHD, KVLayout.HND),
            dtype_signatures=(
                ("float16", "float16", "float16"),
                ("bfloat16", "bfloat16", "bfloat16"),
            ),
            page_sizes=(128, 64),
        )
        left = policy("cann", CANN_V2_OPERATION_ID, (first, second))
        right = policy(
            "cann",
            CANN_V2_OPERATION_ID,
            (replace(second, page_sizes=(64, 128)), first),
        )
        restored = AttentionOperatorPlanScoringPolicy.from_dict(left.to_dict())

        self.assertEqual(left, restored)
        self.assertEqual(left.fingerprint, restored.fingerprint)
        self.assertEqual(left.fingerprint, right.fingerprint)

    def test_registry_consumes_declarative_policies_without_custom_selector(self):
        plan = framework_plan(page_size=128)
        cann_policy = policy(
            "cann",
            CANN_V2_OPERATION_ID,
            (page_rule("cann_page_128_v1", 128, 90),),
        )
        flash_policy = policy(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            (page_rule("flash_page_128_v1", 128, 30),),
        )
        resolver = AttentionOperatorRuntimeImplementationRegistry(
            (
                PolicyImplementation(flash_policy),
                PolicyImplementation(cann_policy),
            )
        )

        report = resolver.explain(plan, "npu:0")

        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(report.top_plan_score, 90)
        self.assertTrue(report.selected.plan_score.source.startswith("declarative:"))

    def test_public_plan_uses_declarative_policy_without_provider_argument(self):
        cann_policy = policy(
            "cann",
            CANN_V2_OPERATION_ID,
            (
                page_rule("cann_page_128_public_v1", 128, 90),
                page_rule("cann_page_64_public_v1", 64, 10),
            ),
        )
        flash_policy = policy(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            (
                page_rule("flash_page_128_public_v1", 128, 10),
                page_rule("flash_page_64_public_v1", 64, 90),
            ),
        )
        implementations = AttentionOperatorRuntimeImplementationRegistry(
            (
                PublicPolicyImplementation(flash_policy),
                PublicPolicyImplementation(cann_policy),
            )
        )
        registry = AttentionOperatorRuntimeResolverRegistry(
            (("npu", implementations),)
        )
        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            registry,
        ):
            page_128 = BatchAttention(kv_layout="HND", device="npu:0")
            page_64 = BatchAttention(kv_layout="HND", device="npu:0")

        self.assertIsNone(plan_public(page_128, 128))
        self.assertEqual(page_128.plan_selection.provider_id, "cann")
        self.assertIn(
            cann_policy.fingerprint,
            page_128.plan_selection.plan_score_source,
        )
        self.assertIsNone(plan_public(page_64, 64))
        self.assertEqual(
            page_64.plan_selection.provider_id,
            "flash_attention_npu",
        )
        self.assertIn(
            flash_policy.fingerprint,
            page_64.plan_selection.plan_score_source,
        )

    def test_bootstrap_accepts_policy_without_loading_provider_package(self):
        values = bootstrap_components()
        plan = framework_plan()
        scorer = policy(
            "cann",
            values["operation"].operation_id,
            (
                AttentionOperatorPlanScoreRule(
                    rule_id="mixed_mode_v1",
                    precedence=10,
                    score=55,
                    reason="mixed-mode package preference",
                    modes=(plan.spec.mode,),
                ),
            ),
        )
        spec = replace(values["spec"], plan_scorer=scorer)

        implementation = build_attention_operator_package_runtime(
            spec,
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        score = implementation.plan_score(plan, "npu:0")

        self.assertEqual(score.value, 55)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_runtime_declaration_binds_policy_fingerprint_and_rejects_drift(self):
        values = bootstrap_components()
        scorer = policy(
            "cann",
            values["operation"].operation_id,
            (page_rule("declaration_page_128_v1", 128, 70),),
        )
        spec = replace(values["spec"], plan_scorer=scorer)
        declaration = describe_attention_operator_package_runtime(
            spec,
            operation_catalog=values["catalog"],
        )
        scorer_component = next(
            item for item in declaration.components if item.role == "plan_scorer"
        )

        self.assertIn(
            ("policy_fingerprint", scorer.fingerprint),
            scorer_component.identities,
        )
        drifted = replace(spec, plan_scorer=replace(scorer, default_score=1))
        registration = AttentionDeclaredOperatorPackageRuntimeSpec(
            declaration=declaration,
            runtime_spec=drifted,
        )
        with self.assertRaisesRegex(SchemaError, "declaration is stale"):
            build_declared_attention_operator_runtime_resolvers(
                (registration,),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_invalid_or_ambiguous_rule_data_is_rejected(self):
        cases = (
            ({}, "match constraint"),
            (
                {
                    "quantization": "dense",
                    "quant_spec_fingerprints": ("1" * 64,),
                },
                "require quantization='quantized'",
            ),
            (
                {"min_batch_size": 4, "max_batch_size": 3},
                "range is inverted",
            ),
            ({"causal_values": (1,)}, "must contain booleans"),
        )
        base = {
            "rule_id": "invalid_case_v1",
            "precedence": 1,
            "score": 1,
            "reason": "invalid fixture",
        }
        for overrides, message in cases:
            with self.subTest(message=message):
                fields = dict(base)
                fields.update(overrides)
                with self.assertRaisesRegex(SchemaError, message):
                    AttentionOperatorPlanScoreRule(**fields)


if __name__ == "__main__":
    unittest.main()
