from dataclasses import replace
import json
import unittest

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionJsonEnvelopeLimits,
    AttentionMode,
    AttentionOperatorPlanScoreRule,
    AttentionOperatorPlanScoringManifest,
    AttentionOperatorPlanScoringManifestLimits,
    AttentionOperatorRuntimeImplementationRegistry,
    build_attention_operator_package_runtime,
    describe_attention_operator_package_runtime,
    load_attention_operator_plan_scoring_manifest,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_015_runtime_implementation_auto_selection import (
    framework_plan,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_077_declarative_plan_scoring import (
    PolicyImplementation,
    page_rule,
    policy,
)


def manifest_fixture():
    cann = policy(
        "cann",
        CANN_V2_OPERATION_ID,
        (
            page_rule("cann_page_128_manifest_v1", 128, 90),
            page_rule("cann_page_64_manifest_v1", 64, 10),
        ),
    )
    flash = policy(
        "flash_attention_npu",
        FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
        (
            page_rule("flash_page_128_manifest_v1", 128, 30),
            page_rule("flash_page_64_manifest_v1", 64, 80),
        ),
    )
    return AttentionOperatorPlanScoringManifest(
        manifest_id="ascend.attention.provider_preferences.v1",
        policies=(flash, cann),
    )


class BoundedScoringManifestCheckpoint(unittest.TestCase):
    """Checkpoint 079: declarative provider policies load from bounded JSON."""

    def test_manifest_json_round_trip_is_canonical_and_measured(self):
        manifest = manifest_fixture()
        encoded = manifest.to_json()

        restored, usage = load_attention_operator_plan_scoring_manifest(encoded)

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.fingerprint, manifest.fingerprint)
        self.assertEqual(usage.encoded_bytes, len(encoded.encode("utf-8")))
        self.assertGreater(usage.nodes, 1)
        self.assertEqual(
            restored.policy_ids,
            (
                "cann.attention.policy.v1",
                "flash_attention_npu.attention.policy.v1",
            ),
        )

    def test_loader_rejects_duplicate_unknown_and_oversized_json(self):
        manifest = manifest_fixture()
        encoded = manifest.to_json()
        duplicate = encoded[:-1] + ',"manifest_id":"duplicate"}'
        with self.assertRaisesRegex(SchemaError, "duplicate JSON object key"):
            load_attention_operator_plan_scoring_manifest(duplicate)

        unknown = encoded[:-1] + ',"unknown":1}'
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            load_attention_operator_plan_scoring_manifest(unknown)

        with self.assertRaisesRegex(SchemaError, "bytes exceed limit"):
            load_attention_operator_plan_scoring_manifest(
                encoded,
                limits=AttentionJsonEnvelopeLimits(max_bytes=len(encoded) - 1),
            )

    def test_policy_rule_and_predicate_limits_apply_before_construction(self):
        manifest = manifest_fixture()
        encoded = manifest.to_json()
        wide_predicate = manifest.to_dict()
        wide_predicate["policies"][0]["rules"][0]["page_sizes"] = [64, 128]
        wide_predicate_encoded = json.dumps(wide_predicate)
        cases = (
            (
                encoded,
                AttentionOperatorPlanScoringManifestLimits(max_policies=1),
                "policies exceed limit",
            ),
            (
                encoded,
                AttentionOperatorPlanScoringManifestLimits(
                    max_rules_per_policy=1
                ),
                "per-policy limit",
            ),
            (
                encoded,
                AttentionOperatorPlanScoringManifestLimits(max_total_rules=3),
                "total limit",
            ),
            (
                wide_predicate_encoded,
                AttentionOperatorPlanScoringManifestLimits(
                    max_values_per_predicate=1
                ),
                "predicate page_sizes exceeds limit",
            ),
            (
                encoded,
                AttentionOperatorPlanScoringManifestLimits(
                    max_total_predicate_values=7
                ),
                "predicate values exceed total limit",
            ),
        )
        for payload, limits, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    load_attention_operator_plan_scoring_manifest(
                        payload,
                        manifest_limits=limits,
                    )

    def test_loader_rejects_non_array_manifest_policy_and_rule_shapes(self):
        base = manifest_fixture().to_dict()
        cases = (
            ({**base, "policies": {}}, "policies must be an array"),
            (
                {
                    **base,
                    "policies": [
                        {**base["policies"][0], "rules": {}}
                    ],
                },
                "policy rules must be an array",
            ),
            (
                {
                    **base,
                    "policies": [
                        {
                            **base["policies"][0],
                            "rules": [
                                {
                                    **base["policies"][0]["rules"][0],
                                    "page_sizes": 128,
                                }
                            ],
                        }
                    ],
                },
                "page_sizes must be an array",
            ),
        )
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    load_attention_operator_plan_scoring_manifest(
                        json.dumps(value)
                    )

    def test_manifest_identity_lookup_is_exact_and_unique(self):
        manifest = manifest_fixture()

        self.assertEqual(
            manifest.get("cann", CANN_V2_OPERATION_ID).provider_id,
            "cann",
        )
        with self.assertRaisesRegex(SchemaError, "unknown.*identity"):
            manifest.get("cann", "unknown.operation")

        cann = manifest.get("cann", CANN_V2_OPERATION_ID)
        with self.assertRaisesRegex(SchemaError, "identities duplicate"):
            replace(manifest, policies=(cann, replace(cann, policy_id="other.v1")))

    def test_loaded_manifest_drives_registry_selection(self):
        loaded, _ = load_attention_operator_plan_scoring_manifest(
            manifest_fixture().to_json()
        )
        resolver = AttentionOperatorRuntimeImplementationRegistry(
            tuple(PolicyImplementation(item) for item in loaded.policies)
        )

        report = resolver.explain(framework_plan(page_size=128), "npu:0")

        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(report.top_plan_score, 90)
        self.assertIn(
            loaded.get("cann", CANN_V2_OPERATION_ID).fingerprint,
            report.selected.plan_score.source,
        )

    def test_loaded_policy_injects_bootstrap_without_package_probe(self):
        values = bootstrap_components()
        plan = framework_plan()
        runtime_policy = policy(
            "cann",
            values["operation"].operation_id,
            (
                AttentionOperatorPlanScoreRule(
                    rule_id="bootstrap_manifest_mode_v1",
                    precedence=10,
                    score=66,
                    reason="loaded manifest mode preference",
                    modes=(AttentionMode.BATCH_MIXED_PAGED,),
                ),
            ),
            policy_id="cann.bootstrap.manifest.v1",
        )
        source = AttentionOperatorPlanScoringManifest(
            "bootstrap.attention.scoring.v1",
            (runtime_policy,),
        )
        loaded, _ = load_attention_operator_plan_scoring_manifest(
            source.to_json()
        )
        selected = loaded.get("cann", values["operation"].operation_id)
        spec = replace(values["spec"], plan_scorer=selected)

        implementation = build_attention_operator_package_runtime(
            spec,
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        declaration = describe_attention_operator_package_runtime(
            spec,
            operation_catalog=values["catalog"],
        )

        self.assertEqual(implementation.plan_score(plan, "npu:0").value, 66)
        scorer = next(
            item for item in declaration.components if item.role == "plan_scorer"
        )
        self.assertIn(("policy_fingerprint", selected.fingerprint), scorer.identities)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])


if __name__ == "__main__":
    unittest.main()
