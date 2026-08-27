from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorRuntime,
    AttentionStateError,
    attention_operator_runtime_registry_snapshot,
    bind_attention_operator_plan_scoring_manifest,
    build_attention_operator_runtime_resolvers,
    build_provider_plan_selection,
    describe_attention_operator_package_runtime,
    install_attention_operator_runtime_resolvers,
    install_declared_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)
from tests.test_checkpoint_062_runtime_completion_publication import (
    StrictResultPackageLoader,
    run_runtime,
    strict_results,
    valid_result,
)
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_080_scoring_manifest_bootstrap_binding import (
    scoring_manifest,
)
from tests.test_checkpoint_081_declared_scoring_manifest_snapshot import (
    scored_registration,
)


def scored_strict_runtime(*, runtime_manifest_score=73):
    values = bootstrap_components()
    values["loader"] = StrictResultPackageLoader(values["events"])
    values["spec"] = replace(values["spec"], validate_provider_results=True)
    declared_manifest = scoring_manifest(values["spec"])
    spec = bind_attention_operator_plan_scoring_manifest(
        (values["spec"],), declared_manifest
    )[0]
    declaration = describe_attention_operator_package_runtime(
        spec,
        operation_catalog=values["catalog"],
    )
    registry = build_attention_operator_runtime_resolvers(
        (spec,),
        operation_catalog=values["catalog"],
        package_loader=values["loader"],
        plan_scoring_manifest=declared_manifest,
    )
    plan = group_plan()
    runtime_manifest = scoring_manifest(
        values["spec"], score=runtime_manifest_score
    )
    runtime = AttentionOperatorRuntime(
        "npu:0",
        registry,
        values["catalog"],
        mode=plan.spec.mode,
        runtime_declaration_bindings=(
            (
                declaration.provider_id,
                declaration.operation_id,
                declaration.fingerprint,
            ),
        ),
        plan_scoring_manifest_binding=runtime_manifest.binding,
    )
    return values, declared_manifest, declaration, plan, runtime


class ScoringManifestPlanRunChainCheckpoint(unittest.TestCase):
    """Checkpoint 082: selected policy authority reaches plan and run audit."""

    def setUp(self):
        strict_results[:] = []
        self.original = attention_operator_runtime_registry_snapshot()

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
            expected_generation=current.generation,
        )

    def test_declarative_score_carries_structured_policy_identity(self):
        values = bootstrap_components()
        manifest = scoring_manifest(values["spec"])
        score = manifest.policies[0].score(group_plan(), "npu:0")

        self.assertEqual(score.policy_id, manifest.policies[0].policy_id)
        self.assertEqual(
            score.policy_fingerprint,
            manifest.policies[0].fingerprint,
        )
        self.assertEqual(
            score.to_dict()["policy_fingerprint"],
            manifest.policies[0].fingerprint,
        )

    def test_runtime_rejects_score_from_a_different_manifest_policy(self):
        _, _, _, plan, runtime = scored_strict_runtime(
            runtime_manifest_score=74
        )

        with self.assertRaisesRegex(
            SchemaError, "plan score differs from scoring manifest"
        ):
            runtime.plan(plan.spec, plan.metadata)

        self.assertFalse(runtime.is_planned)
        with self.assertRaisesRegex(AttentionStateError, "has not been planned"):
            _ = runtime.runtime_plan_scoring_binding

    def test_public_plan_selection_exposes_installed_manifest_identity(self):
        values = bootstrap_components()
        manifest, registration = scored_registration(values)
        installed = install_declared_attention_operator_runtime_resolvers(
            (registration,),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
            plan_scoring_manifest=manifest,
            expected_generation=self.original.generation,
        )
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="NHD",
            backend="auto",
        )
        plan = group_plan()

        self.assertIsNone(
            wrapper.plan(
                plan.metadata.indptr,
                plan.metadata.indices,
                plan.metadata.last_page_len,
                plan.spec.num_qo_heads,
                plan.spec.num_kv_heads,
                plan.spec.head_dim_vo,
                plan.metadata.page_size,
                q_data_type=plan.spec.q_dtype,
                kv_data_type=plan.spec.kv_quant_spec,
                o_data_type=plan.spec.o_dtype,
            )
        )
        selection = wrapper.plan_selection

        self.assertEqual(selection.registry_generation, installed.generation)
        self.assertEqual(
            selection.plan_scoring_manifest_id,
            manifest.manifest_id,
        )
        self.assertEqual(
            selection.plan_scoring_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertEqual(
            selection.plan_scoring_policy_id,
            manifest.policies[0].policy_id,
        )
        self.assertEqual(
            selection.plan_scoring_policy_fingerprint,
            manifest.policies[0].fingerprint,
        )
        self.assertEqual(selection.plan_score, 73)
        self.assertEqual(
            selection.to_dict()["plan_scoring_manifest_fingerprint"],
            manifest.fingerprint,
        )

    def test_successful_run_receipt_binds_same_manifest_and_policy(self):
        _, manifest, declaration, plan, runtime = scored_strict_runtime()
        runtime.plan(plan.spec, plan.metadata)
        strict_results.append(valid_result(runtime))

        run_runtime(runtime)

        receipt = runtime.last_run_receipt
        scoring_binding = runtime.runtime_plan_scoring_binding
        self.assertEqual(
            scoring_binding,
            (
                manifest.manifest_id,
                manifest.fingerprint,
                manifest.policies[0].policy_id,
                manifest.policies[0].fingerprint,
            ),
        )
        self.assertEqual(
            receipt.runtime_declaration_fingerprint,
            declaration.fingerprint,
        )
        self.assertEqual(
            receipt.plan_scoring_manifest_fingerprint,
            manifest.fingerprint,
        )
        self.assertEqual(
            receipt.plan_scoring_policy_fingerprint,
            manifest.policies[0].fingerprint,
        )
        self.assertEqual(
            receipt.to_dict()["plan_scoring_manifest_id"],
            manifest.manifest_id,
        )

    def test_plan_selection_and_run_receipt_share_exact_audit_identity(self):
        _, manifest, declaration, plan, runtime = scored_strict_runtime()
        runtime.plan(plan.spec, plan.metadata)
        active = runtime.operator_session.active_plan
        score = runtime.runtime_plan_score
        binding = runtime.runtime_plan_scoring_binding
        selection = build_provider_plan_selection(
            runtime.plan_state,
            active,
            registry_generation=12,
            runtime_declaration_fingerprint=declaration.fingerprint,
            plan_scoring_manifest_id=binding[0],
            plan_scoring_manifest_fingerprint=binding[1],
            plan_scoring_policy_id=binding[2],
            plan_scoring_policy_fingerprint=binding[3],
            plan_score=score.value,
            plan_score_source=score.source,
            plan_score_reason=score.reason,
            runtime_resolution_fingerprint=(
                runtime.runtime_resolution_fingerprint
            ),
        )
        strict_results.append(valid_result(runtime))
        run_runtime(runtime)
        receipt = runtime.last_run_receipt

        self.assertEqual(
            selection.plan_scoring_manifest_fingerprint,
            receipt.plan_scoring_manifest_fingerprint,
        )
        self.assertEqual(
            selection.plan_scoring_policy_fingerprint,
            receipt.plan_scoring_policy_fingerprint,
        )
        self.assertEqual(
            selection.runtime_declaration_fingerprint,
            receipt.runtime_declaration_fingerprint,
        )
        self.assertEqual(selection.plan_scoring_manifest_id, manifest.manifest_id)

    def test_failed_completion_publishes_no_scoring_receipt(self):
        _, _, _, plan, runtime = scored_strict_runtime()
        runtime.plan(plan.spec, plan.metadata)
        strict_results.append(None)

        with self.assertRaisesRegex(SchemaError, "multiple values"):
            run_runtime(runtime)
        with self.assertRaisesRegex(AttentionStateError, "no successful atomic"):
            _ = runtime.last_run_receipt

    def test_unplanned_fork_preserves_manifest_authority(self):
        _, manifest, _, plan, runtime = scored_strict_runtime()
        runtime.plan(plan.spec, plan.metadata)
        forked = runtime.fork_unplanned()
        forked.plan(plan.spec, plan.metadata)

        self.assertEqual(
            forked.runtime_plan_scoring_binding,
            (
                manifest.manifest_id,
                manifest.fingerprint,
                manifest.policies[0].policy_id,
                manifest.policies[0].fingerprint,
            ),
        )

    def test_incomplete_plan_or_run_manifest_identity_fails_closed(self):
        _, _, declaration, plan, runtime = scored_strict_runtime()
        runtime.plan(plan.spec, plan.metadata)
        active = runtime.operator_session.active_plan
        score = runtime.runtime_plan_score
        binding = runtime.runtime_plan_scoring_binding
        selection = build_provider_plan_selection(
            runtime.plan_state,
            active,
            registry_generation=12,
            runtime_declaration_fingerprint=declaration.fingerprint,
            plan_scoring_manifest_id=binding[0],
            plan_scoring_manifest_fingerprint=binding[1],
            plan_scoring_policy_id=binding[2],
            plan_scoring_policy_fingerprint=binding[3],
            plan_score=score.value,
            plan_score_source=score.source,
            plan_score_reason=score.reason,
            runtime_resolution_fingerprint=(
                runtime.runtime_resolution_fingerprint
            ),
        )
        with self.assertRaisesRegex(SchemaError, "identity is incomplete"):
            replace(selection, plan_scoring_policy_fingerprint=None)

        strict_results.append(valid_result(runtime))
        run_runtime(runtime)
        with self.assertRaisesRegex(SchemaError, "identity is incomplete"):
            replace(
                runtime.last_run_receipt,
                plan_scoring_policy_fingerprint=None,
            )


if __name__ == "__main__":
    unittest.main()
