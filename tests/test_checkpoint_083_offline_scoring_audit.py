from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeDeclarationBinding,
    AttentionOperatorRuntimeRegistrySnapshot,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionPlanSelection,
    AttentionPlanScoringAuditError,
    AttentionPlanScoringAuditReport,
    build_provider_plan_selection,
    build_reference_plan_selection,
    load_packaged_attention_operator_catalog,
    verify_attention_plan_scoring_chain,
)
from tests.test_checkpoint_015_runtime_implementation_auto_selection import (
    framework_plan,
)
from tests.test_checkpoint_062_runtime_completion_publication import (
    run_runtime,
    strict_results,
    valid_result,
)
from tests.test_checkpoint_080_scoring_manifest_bootstrap_binding import (
    scoring_manifest,
)
from tests.test_checkpoint_077_declarative_plan_scoring import (
    PolicyImplementation,
)
from tests.test_checkpoint_079_bounded_scoring_manifest import manifest_fixture
from tests.test_checkpoint_082_scoring_manifest_plan_run_chain import (
    scored_strict_runtime,
)


def audit_fixture(*, with_run=False):
    values, manifest, declaration, plan, runtime = scored_strict_runtime()
    resolver = runtime._resolver_registry.resolvers[0][1]
    resolution = resolver.explain(plan, "npu:0")
    runtime.plan(plan.spec, plan.metadata)
    binding = runtime.runtime_plan_scoring_binding
    score = runtime.runtime_plan_score
    snapshot = AttentionOperatorRuntimeRegistrySnapshot(
        generation=23,
        device_types=("npu",),
        registry=runtime._resolver_registry,
        operation_catalog=values["catalog"],
        runtime_declarations=(
            AttentionOperatorRuntimeDeclarationBinding(
                declaration.provider_id,
                declaration.operation_id,
                declaration.fingerprint,
            ),
        ),
        plan_scoring_manifest_binding=manifest.binding,
    )
    selection = build_provider_plan_selection(
        runtime.plan_state,
        runtime.operator_session.active_plan,
        registry_generation=snapshot.generation,
        runtime_declaration_fingerprint=declaration.fingerprint,
        plan_scoring_manifest_id=binding[0],
        plan_scoring_manifest_fingerprint=binding[1],
        plan_scoring_policy_id=binding[2],
        plan_scoring_policy_fingerprint=binding[3],
        plan_score=score.value,
        plan_score_source=score.source,
        plan_score_reason=score.reason,
        runtime_resolution_fingerprint=runtime.runtime_resolution_fingerprint,
    )
    receipt = None
    if with_run:
        strict_results.append(valid_result(runtime))
        run_runtime(runtime)
        receipt = runtime.last_run_receipt
    return {
        "values": values,
        "manifest": manifest,
        "declaration": declaration,
        "plan": runtime.plan_state,
        "runtime": runtime,
        "resolution": resolution,
        "snapshot": snapshot,
        "selection": selection,
        "receipt": receipt,
    }


def verify(values, **overrides):
    arguments = {
        "plan": values["plan"],
        "resolution_report": values["resolution"],
        "registry_snapshot": values["snapshot"],
        "scoring_manifest": values["manifest"],
        "plan_selection": values["selection"],
        "run_receipt": values["receipt"],
    }
    arguments.update(overrides)
    return verify_attention_plan_scoring_chain(**arguments)


class OfflineScoringAuditCheckpoint(unittest.TestCase):
    """Checkpoint 083: replay scoring evidence without a provider or device."""

    def setUp(self):
        strict_results[:] = []

    def test_plan_only_audit_replays_selected_policy(self):
        values = audit_fixture()

        report = verify(values)

        self.assertIsInstance(report, AttentionPlanScoringAuditReport)
        self.assertEqual(report.provider_id, "cann")
        self.assertEqual(report.operation_id, values["declaration"].operation_id)
        self.assertEqual(report.plan_score, 73)
        self.assertEqual(report.manifest_fingerprint, values["manifest"].fingerprint)
        self.assertEqual(
            report.policy_fingerprint,
            values["manifest"].policies[0].fingerprint,
        )
        self.assertEqual(len(report.replayed_candidates), 1)
        self.assertIsNone(report.run_receipt_fingerprint)
        self.assertEqual(len(report.fingerprint), 64)
        self.assertEqual(
            report.to_dict()["kind"],
            "attention_plan_scoring_audit",
        )

    def test_run_audit_closes_selection_and_receipt(self):
        values = audit_fixture(with_run=True)

        report = verify(values)

        self.assertEqual(
            report.run_receipt_fingerprint,
            values["receipt"].fingerprint,
        )
        self.assertEqual(
            report.active_plan_fingerprint,
            values["receipt"].active_plan_fingerprint,
        )
        self.assertEqual(
            report.runtime_declaration_fingerprint,
            values["receipt"].runtime_declaration_fingerprint,
        )

    def test_audit_replays_every_scored_top_priority_provider(self):
        plan = framework_plan(page_size=128)
        manifest = manifest_fixture()
        resolver = AttentionOperatorRuntimeImplementationRegistry(
            tuple(PolicyImplementation(policy) for policy in manifest.policies)
        )
        resolution = resolver.explain(plan, "npu:0")
        registry = AttentionOperatorRuntimeResolverRegistry(
            (("npu", resolver),)
        )
        declaration_fingerprints = {
            "cann": "c" * 64,
            "flash_attention_npu": "d" * 64,
        }
        snapshot = AttentionOperatorRuntimeRegistrySnapshot(
            generation=9,
            device_types=("npu",),
            registry=registry,
            operation_catalog=load_packaged_attention_operator_catalog(),
            runtime_declarations=tuple(
                AttentionOperatorRuntimeDeclarationBinding(
                    policy.provider_id,
                    policy.operation_id,
                    declaration_fingerprints[policy.provider_id],
                )
                for policy in manifest.policies
            ),
            plan_scoring_manifest_binding=manifest.binding,
        )
        selected = resolution.selected
        policy = manifest.get(selected.provider_id, selected.operation_id)
        selection = AttentionPlanSelection(
            mode=plan.spec.mode,
            route="provider",
            backend="aclnn",
            plan_generation=plan.generation,
            framework_plan_fingerprint=plan.fingerprint,
            provider_id=selected.provider_id,
            operation_id=selected.operation_id,
            active_plan_fingerprint="e" * 64,
            registry_generation=snapshot.generation,
            runtime_declaration_fingerprint=(
                declaration_fingerprints[selected.provider_id]
            ),
            plan_scoring_manifest_id=manifest.manifest_id,
            plan_scoring_manifest_fingerprint=manifest.fingerprint,
            plan_scoring_policy_id=policy.policy_id,
            plan_scoring_policy_fingerprint=policy.fingerprint,
            plan_score=selected.plan_score.value,
            plan_score_source=selected.plan_score.source,
            plan_score_reason=selected.plan_score.reason,
            runtime_resolution_fingerprint=resolution.fingerprint,
        )

        report = verify_attention_plan_scoring_chain(
            plan=plan,
            resolution_report=resolution,
            registry_snapshot=snapshot,
            scoring_manifest=manifest,
            plan_selection=selection,
        )

        self.assertEqual(
            tuple(item.provider_id for item in report.replayed_candidates),
            ("cann", "flash_attention_npu"),
        )
        self.assertEqual(
            tuple(item.score for item in report.replayed_candidates),
            (90, 30),
        )

    def test_audit_rejects_resolution_and_selection_drift(self):
        values = audit_fixture()
        selection = replace(
            values["selection"],
            runtime_resolution_fingerprint="a" * 64,
        )
        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "resolution fingerprint differs"
        ):
            verify(values, plan_selection=selection)

        selection = replace(
            values["selection"],
            plan_score=values["selection"].plan_score + 1,
        )
        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "selected score evidence differs"
        ):
            verify(values, plan_selection=selection)

    def test_audit_rejects_registry_generation_and_declaration_drift(self):
        values = audit_fixture()
        selection = replace(
            values["selection"],
            registry_generation=values["snapshot"].generation + 1,
        )
        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "registry generation differs"
        ):
            verify(values, plan_selection=selection)

        selection = replace(
            values["selection"],
            runtime_declaration_fingerprint="b" * 64,
        )
        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "runtime declaration differs"
        ):
            verify(values, plan_selection=selection)

    def test_audit_replays_policy_instead_of_trusting_manifest_fields(self):
        values = audit_fixture()
        drifted = scoring_manifest(values["values"]["spec"], score=74)
        snapshot = replace(
            values["snapshot"],
            plan_scoring_manifest_binding=drifted.binding,
        )
        selection = replace(
            values["selection"],
            plan_scoring_manifest_id=drifted.manifest_id,
            plan_scoring_manifest_fingerprint=drifted.fingerprint,
            plan_scoring_policy_id=drifted.policies[0].policy_id,
            plan_scoring_policy_fingerprint=drifted.policies[0].fingerprint,
        )

        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "replay differs"
        ):
            verify(
                values,
                registry_snapshot=snapshot,
                scoring_manifest=drifted,
                plan_selection=selection,
            )

    def test_audit_rejects_tampered_run_receipt(self):
        values = audit_fixture(with_run=True)
        receipt = replace(
            values["receipt"],
            plan_scoring_manifest_id="tampered.manifest.v1",
        )

        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "run receipt differs"
        ):
            verify(values, run_receipt=receipt)

    def test_reference_selection_is_outside_manifest_audit(self):
        values = audit_fixture()
        reference = build_reference_plan_selection(values["plan"])

        with self.assertRaisesRegex(
            AttentionPlanScoringAuditError, "requires a provider"
        ):
            verify(values, plan_selection=reference)

    def test_audit_has_no_package_or_callable_side_effect(self):
        values = audit_fixture()
        loader = values["values"]["loader"]
        before = (
            loader.version_calls,
            loader.resolve_calls,
            tuple(values["values"]["events"]),
        )

        first = verify(values)
        second = verify(values)

        after = (
            loader.version_calls,
            loader.resolve_calls,
            tuple(values["values"]["events"]),
        )
        self.assertEqual(after, before)
        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)


if __name__ == "__main__":
    unittest.main()
