from dataclasses import replace
import unittest
from unittest.mock import patch

from flashinfer_npu.attention import (
    ATTENTION_OPERATOR_PLAN_SCORE_MAX,
    ATTENTION_OPERATOR_PLAN_SCORE_MIN,
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimePlanScore,
    AttentionOperatorRuntimeResolutionError,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionPreparedOperatorPlan,
    BatchAttention,
    build_attention_operator_package_runtime,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_015_runtime_implementation_auto_selection import (
    FakeImplementation,
    authorized_runtime,
    framework_plan,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components


class ScoredImplementation(FakeImplementation):
    def __init__(
        self,
        provider_id,
        operation_id,
        score_value,
        *,
        priority=20,
        reasons=(),
    ):
        super().__init__(provider_id, operation_id, priority, reasons)
        self._score_value = score_value
        self.score_calls = []

    def plan_score(self, plan, device):
        self.score_calls.append((plan.fingerprint, device))
        value = (
            self._score_value(plan)
            if callable(self._score_value)
            else self._score_value
        )
        return AttentionOperatorRuntimePlanScore(
            value=value,
            source="checkpoint-076-policy",
            reason="provider preference for this canonical plan",
        )


class SyntheticPlanFactory:
    """Framework-only plan preparation; it never imports or calls a provider."""

    def __init__(self, provider_id, operation_id):
        self.provider_id = provider_id
        self.operation_id = operation_id

    def prepare(self, plan, receipt, selection):
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="checkpoint-076-synthetic-plan",
            opaque_state=None,
        )


class PublicScoredImplementation(ScoredImplementation):
    def resolve(self, plan, device):
        self.resolve_calls += 1
        resolved = authorized_runtime(plan, self.provider_id)
        return replace(
            resolved,
            factory=SyntheticPlanFactory(self.provider_id, self.operation_id),
        )


class ExplodingScoreImplementation(FakeImplementation):
    def __init__(self, provider_id, operation_id, *, priority=10, reasons=()):
        super().__init__(provider_id, operation_id, priority, reasons)
        self.score_calls = 0

    def plan_score(self, plan, device):
        self.score_calls += 1
        raise AssertionError("non-finalist scorer must not be called")


class InvalidScoreImplementation(FakeImplementation):
    def plan_score(self, plan, device):
        return 100


class FixedPlanScorer:
    def __init__(self, provider_id, operation_id, value=17):
        self.provider_id = provider_id
        self.operation_id = operation_id
        self.value = value
        self.calls = []

    def score(self, plan, device):
        self.calls.append((plan.fingerprint, device))
        return AttentionOperatorRuntimePlanScore(
            self.value,
            "bootstrap-test-policy",
            "fixed framework-only preference",
        )


def scored_cann(score_value, *, priority=20, reasons=()):
    return ScoredImplementation(
        "cann",
        CANN_V2_OPERATION_ID,
        score_value,
        priority=priority,
        reasons=reasons,
    )


def scored_flash(score_value, *, priority=20, reasons=()):
    return ScoredImplementation(
        "flash_attention_npu",
        FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
        score_value,
        priority=priority,
        reasons=reasons,
    )


def public_scored(provider_id, operation_id, score_value):
    return PublicScoredImplementation(
        provider_id,
        operation_id,
        score_value,
    )


def plan_public(wrapper, page_size):
    return wrapper.plan(
        (0, 2, 3),
        (0, 2, 3),
        (7, 3, 9),
        (page_size + page_size // 2, page_size),
        8,
        2,
        128,
        128,
        page_size,
        causal=True,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
    )


class PlanSpecificRuntimeScoringCheckpoint(unittest.TestCase):
    """Checkpoint 076: canonical plans choose among equally ranked providers."""

    def test_unique_score_selects_independent_of_registration_order(self):
        plan = framework_plan(page_size=128)
        fingerprints = []
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                cann = scored_cann(90)
                flash = scored_flash(30)
                values = (flash, cann) if reverse else (cann, flash)
                resolver = AttentionOperatorRuntimeImplementationRegistry(values)

                report = resolver.explain(plan, "npu:0")

                self.assertEqual(report.top_priority, 20)
                self.assertEqual(report.top_plan_score, 90)
                self.assertEqual(report.selected.provider_id, "cann")
                scores = {
                    item.provider_id: item.plan_score.to_dict()
                    for item in report.candidates
                }
                self.assertEqual(scores["cann"]["value"], 90)
                self.assertEqual(scores["flash_attention_npu"]["value"], 30)
                self.assertEqual(len(report.fingerprint), 64)
                fingerprints.append(report.fingerprint)
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_public_plan_automatically_switches_provider_by_page_size(self):
        cann = public_scored(
            "cann",
            CANN_V2_OPERATION_ID,
            lambda plan: 100 if plan.metadata.page_size == 128 else 10
        )
        flash = public_scored(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            lambda plan: 100 if plan.metadata.page_size == 64 else 10
        )
        implementation_registry = AttentionOperatorRuntimeImplementationRegistry(
            (flash, cann)
        )
        device_registry = AttentionOperatorRuntimeResolverRegistry(
            (("npu", implementation_registry),)
        )

        with patch(
            "flashinfer_npu.attention.holistic._operator_runtime_resolvers",
            device_registry,
        ):
            page_128 = BatchAttention(kv_layout="HND", device="npu:0")
            page_64 = BatchAttention(kv_layout="HND", device="npu:0")

        self.assertIsNone(plan_public(page_128, 128))
        self.assertEqual(page_128.plan_selection.provider_id, "cann")
        self.assertEqual(page_128.plan_selection.plan_score, 100)
        self.assertEqual(
            page_128.plan_selection.plan_score_source,
            "checkpoint-076-policy",
        )
        self.assertIn(
            "canonical plan", page_128.plan_selection.plan_score_reason
        )
        self.assertEqual(
            len(page_128.plan_selection.runtime_resolution_fingerprint), 64
        )

        self.assertIsNone(plan_public(page_64, 64))
        self.assertEqual(page_64.plan_selection.provider_id, "flash_attention_npu")
        self.assertEqual(page_64.plan_selection.plan_score, 100)
        self.assertEqual(cann.resolve_calls, 1)
        self.assertEqual(flash.resolve_calls, 1)

    def test_static_priority_dominates_and_lower_scorer_is_not_called(self):
        plan = framework_plan()
        high = scored_cann(-100, priority=30)
        low = ExplodingScoreImplementation(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            priority=20,
        )
        resolver = AttentionOperatorRuntimeImplementationRegistry((low, high))

        report = resolver.explain(plan, "npu:0")

        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(low.score_calls, 0)
        by_provider = {item.provider_id: item for item in report.candidates}
        self.assertIsNone(by_provider["flash_attention_npu"].plan_score)

    def test_rejected_candidate_is_never_scored(self):
        plan = framework_plan()
        accepted = scored_cann(1, priority=20)
        rejected = ExplodingScoreImplementation(
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            priority=100,
            reasons=("page size unsupported",),
        )
        resolver = AttentionOperatorRuntimeImplementationRegistry(
            (rejected, accepted)
        )

        report = resolver.explain(plan, "npu:0")

        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(rejected.score_calls, 0)
        self.assertIsNone(report.candidates[0].plan_score)

    def test_equal_scores_remain_ambiguous_and_fail_closed(self):
        resolver = AttentionOperatorRuntimeImplementationRegistry(
            (scored_cann(50), scored_flash(50))
        )

        with self.assertRaisesRegex(
            AttentionOperatorRuntimeResolutionError,
            "priority 20 and plan score 50",
        ) as captured:
            resolver.resolve(framework_plan(), "npu:0")

        self.assertIsNone(captured.exception.report.selected)
        self.assertEqual(
            tuple(item.provider_id for item in captured.exception.report.finalists),
            ("cann", "flash_attention_npu"),
        )

    def test_score_schema_rejects_ambiguous_or_unbounded_values(self):
        invalid = (
            (True, "source", "reason"),
            (ATTENTION_OPERATOR_PLAN_SCORE_MIN - 1, "source", "reason"),
            (ATTENTION_OPERATOR_PLAN_SCORE_MAX + 1, "source", "reason"),
            (0, " ", "reason"),
            (0, "source", " "),
        )
        for value, source, reason in invalid:
            with self.subTest(value=value, source=source, reason=reason):
                with self.assertRaises(SchemaError):
                    AttentionOperatorRuntimePlanScore(value, source, reason)

    def test_invalid_top_candidate_score_return_is_rejected(self):
        invalid = InvalidScoreImplementation(
            "cann",
            CANN_V2_OPERATION_ID,
            20,
        )
        resolver = AttentionOperatorRuntimeImplementationRegistry((invalid,))

        with self.assertRaisesRegex(TypeError, "invalid plan score"):
            resolver.explain(framework_plan(), "npu:0")

    def test_bootstrap_injects_identity_bound_pure_scorer(self):
        values = bootstrap_components()
        scorer = FixedPlanScorer("cann", values["operation"].operation_id)
        spec = replace(values["spec"], plan_scorer=scorer)

        implementation = build_attention_operator_package_runtime(
            spec,
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        score = implementation.plan_score(group_plan(), "npu:3")

        self.assertEqual(score.value, 17)
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_bootstrap_rejects_scorer_with_foreign_identity(self):
        values = bootstrap_components()
        scorer = FixedPlanScorer(
            "flash_attention_npu", values["operation"].operation_id
        )

        with self.assertRaisesRegex(SchemaError, "plan scorer identity differs"):
            replace(values["spec"], plan_scorer=scorer)


if __name__ == "__main__":
    unittest.main()
