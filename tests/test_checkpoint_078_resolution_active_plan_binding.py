from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    AttentionOperatorRuntime,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    attention_operator_runtime_registry_snapshot,
    build_provider_plan_selection,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_015_runtime_implementation_auto_selection import (
    framework_plan,
)
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_047_public_plan_selection import (
    plan_wrapper,
    provider_wrappers,
    runtime_registry,
)
from tests.test_checkpoint_062_runtime_completion_publication import (
    run_runtime,
    strict_results,
    strict_runtime,
    valid_result,
)
from tests.test_checkpoint_076_plan_specific_runtime_scoring import (
    PublicScoredImplementation,
)


def runtime_with_score(value):
    implementation = PublicScoredImplementation(
        "cann",
        CANN_V2_OPERATION_ID,
        value,
    )
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (implementation,)
    )
    registry = AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )
    plan = framework_plan()
    runtime = AttentionOperatorRuntime(
        "npu:0",
        registry,
        mode=plan.spec.mode,
    )
    runtime.plan(plan.spec, plan.metadata)
    return runtime


class ResolutionActivePlanBindingCheckpoint(unittest.TestCase):
    """Checkpoint 078: execution identity transitively binds plan scoring."""

    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        strict_results[:] = []

    def tearDown(self):
        current = attention_operator_runtime_registry_snapshot()
        if current is not self.original:
            install_attention_operator_runtime_resolvers(
                self.original.registry,
                operation_catalog=self.original.operation_catalog,
                expected_generation=current.generation,
            )

    def test_every_batch_wrapper_publishes_same_resolution_as_active_plan(self):
        components = build_components()
        install_attention_operator_runtime_resolvers(
            runtime_registry(components),
            operation_catalog=components["catalog"],
            expected_generation=self.original.generation,
        )

        for mode, wrapper in provider_wrappers():
            with self.subTest(mode=mode.value):
                self.assertIsNone(plan_wrapper(mode, wrapper))
                selection = wrapper.plan_selection
                active = wrapper._operator_runtime.operator_session.active_plan

                self.assertEqual(selection.plan_score, 0)
                self.assertEqual(selection.plan_score_source, "default")
                self.assertEqual(
                    selection.runtime_resolution_fingerprint,
                    active.runtime_resolution_fingerprint,
                )
                self.assertEqual(
                    selection.active_plan_fingerprint,
                    active.fingerprint,
                )

    def test_changing_only_score_report_changes_active_plan_identity(self):
        low = runtime_with_score(10)
        high = runtime_with_score(90)
        low_active = low.operator_session.active_plan
        high_active = high.operator_session.active_plan

        self.assertNotEqual(
            low_active.runtime_resolution_fingerprint,
            high_active.runtime_resolution_fingerprint,
        )
        self.assertNotEqual(low_active.fingerprint, high_active.fingerprint)
        self.assertEqual(
            replace(
                low_active,
                runtime_resolution_fingerprint=(
                    high_active.runtime_resolution_fingerprint
                ),
            ).fingerprint,
            high_active.fingerprint,
        )

    def test_successful_run_receipt_binds_scored_active_plan(self):
        _, runtime = strict_runtime()
        strict_results.append(valid_result(runtime))
        run_runtime(runtime)
        active = runtime.operator_session.active_plan
        score = runtime.runtime_plan_score
        selection = build_provider_plan_selection(
            runtime.plan_state,
            active,
            registry_generation=0,
            plan_score=score.value,
            plan_score_source=score.source,
            plan_score_reason=score.reason,
            runtime_resolution_fingerprint=(
                runtime.runtime_resolution_fingerprint
            ),
        )

        self.assertEqual(
            runtime.last_run_receipt.active_plan_fingerprint,
            active.fingerprint,
        )
        self.assertEqual(
            selection.active_plan_fingerprint,
            runtime.last_run_receipt.active_plan_fingerprint,
        )
        self.assertEqual(
            selection.runtime_resolution_fingerprint,
            active.runtime_resolution_fingerprint,
        )

    def test_plan_selection_rejects_resolution_not_bound_by_active_plan(self):
        runtime = runtime_with_score(50)
        active = runtime.operator_session.active_plan
        score = runtime.runtime_plan_score

        with self.assertRaisesRegex(
            SchemaError,
            "runtime resolution differs from active plan",
        ):
            build_provider_plan_selection(
                runtime.plan_state,
                active,
                registry_generation=0,
                plan_score=score.value,
                plan_score_source=score.source,
                plan_score_reason=score.reason,
                runtime_resolution_fingerprint="f" * 64,
            )

    def test_active_plan_rejects_invalid_resolution_fingerprint(self):
        runtime = runtime_with_score(50)

        with self.assertRaisesRegex(
            SchemaError,
            "runtime_resolution_fingerprint",
        ):
            replace(
                runtime.operator_session.active_plan,
                runtime_resolution_fingerprint="not-a-hash",
            )


if __name__ == "__main__":
    unittest.main()
