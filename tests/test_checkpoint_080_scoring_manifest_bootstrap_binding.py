from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorPlanScoreRule,
    AttentionOperatorPlanScoringManifest,
    AttentionOperatorPlanScoringPolicy,
    AttentionOperatorRuntimePlanScore,
    bind_attention_operator_plan_scoring_manifest,
    build_attention_operator_runtime_resolvers,
    describe_attention_operator_package_runtime,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)


def scoring_policy(spec, *, score=73, policy_id="cann.bootstrap.policy.v1"):
    return AttentionOperatorPlanScoringPolicy(
        policy_id=policy_id,
        provider_id=spec.provider_id,
        operation_id=spec.operation_id,
        rules=(
            AttentionOperatorPlanScoreRule(
                rule_id="mixed_paged_bootstrap_v1",
                precedence=10,
                score=score,
                reason="reviewed mixed paged bootstrap preference",
                modes=(
                    AttentionMode.BATCH_MIXED_PAGED,
                    AttentionMode.BATCH_DECODE_PAGED,
                ),
            ),
        ),
        default_score=0,
        default_reason="no reviewed bootstrap preference",
    )


def scoring_manifest(spec, *, score=73):
    return AttentionOperatorPlanScoringManifest(
        manifest_id="checkpoint.080.bootstrap.scoring.v1",
        policies=(scoring_policy(spec, score=score),),
    )


class CustomPlanScorer:
    def __init__(self, spec):
        self.provider_id = spec.provider_id
        self.operation_id = spec.operation_id

    def score(self, plan, device):
        return AttentionOperatorRuntimePlanScore(
            1,
            "custom:checkpoint-080",
            "custom scorer must not be overwritten",
        )


class ScoringManifestBootstrapBindingCheckpoint(unittest.TestCase):
    """Checkpoint 080: one manifest binds an entire bootstrap spec set."""

    def test_binding_is_exact_immutable_and_side_effect_free(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])

        bound = bind_attention_operator_plan_scoring_manifest(
            (values["spec"],),
            manifest,
        )

        self.assertIsNone(values["spec"].plan_scorer)
        self.assertEqual(len(bound), 1)
        self.assertIs(bound[0].plan_scorer, manifest.policies[0])
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_resolver_builder_consumes_manifest_without_manual_replace(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])

        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
        )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        report = registry.resolvers[0][1].explain(group_plan(), "npu:0")
        self.assertEqual(report.selected.plan_score.value, 73)
        self.assertIn(
            manifest.policies[0].fingerprint,
            report.selected.plan_score.source,
        )
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_identity_set_mismatch_fails_before_package_probe(self):
        values = bootstrap_components()
        expected = scoring_policy(values["spec"])
        foreign = replace(
            expected,
            policy_id="cann.foreign.policy.v1",
            operation_id="cann.foreign_attention@v1",
        )
        manifest = AttentionOperatorPlanScoringManifest(
            "checkpoint.080.foreign.scoring.v1",
            (foreign,),
        )

        with self.assertRaisesRegex(SchemaError, "identity set differs"):
            build_attention_operator_runtime_resolvers(
                (values["spec"],),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
                plan_scoring_manifest=manifest,
            )

        self.assertIsNone(values["spec"].plan_scorer)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_binding_is_idempotent_but_rejects_policy_drift(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])
        bound = bind_attention_operator_plan_scoring_manifest(
            (values["spec"],), manifest
        )

        rebound = bind_attention_operator_plan_scoring_manifest(bound, manifest)
        self.assertIs(rebound[0].plan_scorer, manifest.policies[0])

        drifted = scoring_manifest(values["spec"], score=74)
        with self.assertRaisesRegex(SchemaError, "differs from manifest policy"):
            bind_attention_operator_plan_scoring_manifest(bound, drifted)

    def test_binding_never_overwrites_custom_scorer(self):
        values = bootstrap_components()
        custom = replace(
            values["spec"],
            plan_scorer=CustomPlanScorer(values["spec"]),
        )

        with self.assertRaisesRegex(SchemaError, "non-manifest plan scorer"):
            bind_attention_operator_plan_scoring_manifest(
                (custom,),
                scoring_manifest(values["spec"]),
            )

    def test_bound_policy_is_part_of_reviewed_runtime_declaration(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])
        bound = bind_attention_operator_plan_scoring_manifest(
            (values["spec"],), manifest
        )[0]

        declaration = describe_attention_operator_package_runtime(
            bound,
            operation_catalog=values["catalog"],
        )

        scorer = next(
            item for item in declaration.components if item.role == "plan_scorer"
        )
        self.assertIn(
            ("policy_fingerprint", manifest.policies[0].fingerprint),
            scorer.identities,
        )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_duplicate_specs_fail_before_replacement(self):
        values = bootstrap_components()
        with self.assertRaisesRegex(SchemaError, "duplicate.*spec identity"):
            bind_attention_operator_plan_scoring_manifest(
                (values["spec"], values["spec"]),
                scoring_manifest(values["spec"]),
            )
        self.assertIsNone(values["spec"].plan_scorer)


if __name__ == "__main__":
    unittest.main()
