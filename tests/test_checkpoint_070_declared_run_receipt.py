import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorRuntime,
    AttentionStateError,
    build_attention_operator_runtime_resolvers,
    describe_attention_operator_package_runtime,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_062_runtime_completion_publication import (
    StrictResultPackageLoader,
    run_runtime,
    strict_results,
    strict_runtime,
    valid_result,
)


def declared_strict_runtime():
    values = bootstrap_components()
    values["loader"] = StrictResultPackageLoader(values["events"])
    values["spec"] = replace(values["spec"], validate_provider_results=True)
    declaration = describe_attention_operator_package_runtime(
        values["spec"], operation_catalog=values["catalog"]
    )
    registry = build_attention_operator_runtime_resolvers(
        (values["spec"],),
        operation_catalog=values["catalog"],
        package_loader=values["loader"],
    )
    plan = group_plan()
    binding = (
        declaration.provider_id,
        declaration.operation_id,
        declaration.fingerprint,
    )
    runtime = AttentionOperatorRuntime(
        "npu:0",
        registry,
        values["catalog"],
        mode=plan.spec.mode,
        runtime_declaration_bindings=(binding,),
    )
    runtime.plan(plan.spec, plan.metadata)
    return values, declaration, runtime


class DeclaredRunReceiptCheckpoint(unittest.TestCase):
    """Atomic run evidence inherits the selected runtime declaration identity."""

    def setUp(self):
        strict_results[:] = []

    def test_successful_run_receipt_binds_exact_declaration(self):
        _, declaration, runtime = declared_strict_runtime()
        expected = valid_result(runtime)
        strict_results.append(expected)

        self.assertIs(run_runtime(runtime), expected)

        receipt = runtime.last_run_receipt
        self.assertEqual(
            receipt.runtime_declaration_fingerprint,
            declaration.fingerprint,
        )
        self.assertEqual(
            receipt.to_dict()["runtime_declaration_fingerprint"],
            declaration.fingerprint,
        )

    def test_legacy_runtime_receipt_is_explicitly_unbound(self):
        _, runtime = strict_runtime()
        strict_results.append(valid_result(runtime))

        run_runtime(runtime)

        self.assertIsNone(runtime.last_run_receipt.runtime_declaration_fingerprint)

    def test_failed_completion_publishes_no_declared_receipt(self):
        _, _, runtime = declared_strict_runtime()
        strict_results.append(None)

        with self.assertRaisesRegex(SchemaError, "multiple values"):
            run_runtime(runtime)
        with self.assertRaisesRegex(AttentionStateError, "no successful atomic"):
            _ = runtime.last_run_receipt

    def test_replan_keeps_declaration_but_rebinds_active_plan(self):
        _, declaration, runtime = declared_strict_runtime()
        strict_results.append(valid_result(runtime, "-first"))
        run_runtime(runtime)
        first = runtime.last_run_receipt
        plan = runtime.plan_state

        runtime.plan(plan.spec, plan.metadata)
        strict_results.append(valid_result(runtime, "-second"))
        run_runtime(runtime)
        second = runtime.last_run_receipt

        self.assertNotEqual(
            first.active_plan_fingerprint, second.active_plan_fingerprint
        )
        self.assertEqual(
            first.runtime_declaration_fingerprint, declaration.fingerprint
        )
        self.assertEqual(
            second.runtime_declaration_fingerprint, declaration.fingerprint
        )

    def test_unplanned_fork_preserves_declaration_authority(self):
        _, declaration, runtime = declared_strict_runtime()
        forked = runtime.fork_unplanned()
        plan = runtime.plan_state
        forked.plan(plan.spec, plan.metadata)
        strict_results.append(valid_result(forked))

        run_runtime(forked)

        self.assertEqual(
            forked.last_run_receipt.runtime_declaration_fingerprint,
            declaration.fingerprint,
        )

    def test_runtime_rejects_invalid_or_duplicate_declaration_bindings(self):
        values = bootstrap_components()
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        plan = group_plan()
        valid = ("cann", values["operation"].operation_id, "a" * 64)
        cases = (
            (("cann", values["operation"].operation_id, "short"),),
            (valid, valid),
        )
        for bindings in cases:
            with self.subTest(bindings=bindings):
                with self.assertRaisesRegex(
                    SchemaError, "runtime declaration bindings|binding identity"
                ):
                    AttentionOperatorRuntime(
                        "npu:0",
                        registry,
                        values["catalog"],
                        mode=plan.spec.mode,
                        runtime_declaration_bindings=bindings,
                    )


if __name__ == "__main__":
    unittest.main()
