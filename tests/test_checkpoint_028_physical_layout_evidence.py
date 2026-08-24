import hashlib
import inspect
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDispatchError,
    AttentionOperatorPhysicalLayoutEvidence,
    BatchAttention,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_025_quantized_provider_run_lowering import active_session
from tests.test_checkpoint_027_provider_physical_layout_binding import (
    physical_input,
    physical_runtime,
)


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def physical_evidence(spec, operation, quant_spec, descriptor, **overrides):
    profile = spec.profiles[0]
    kernel = spec.descriptors[0]
    values = {
        "provider_id": operation.provider_id,
        "operation_id": operation.operation_id,
        "evidence_id": "checkpoint-028-physical-functional-v1",
        "runner": "synthetic-physical-layout-harness",
        "suite_id": "checkpoint-028-provider-layout-suite-v1",
        "suite_fingerprint": digest("synthetic-suite-definition"),
        "passed_case_ids": (
            "paged-decode-int8-group-blocked-key",
            "paged-decode-int8-group-blocked-value",
        ),
        "result_digest": digest("synthetic-external-result-record"),
        "quant_spec_fingerprint": quant_spec.fingerprint,
        "descriptor_fingerprint": descriptor.fingerprint,
        "catalog_fingerprint": spec.quant_physical_layout_catalog.fingerprint,
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "rule_id": profile.rules[0].rule_id,
        "environment_fingerprint": profile.environment.fingerprint,
        "kernel_id": kernel.kernel_id,
        "kernel_fingerprint": kernel.fingerprint,
        "artifact_fingerprint": kernel.artifact.fingerprint,
        "launch_abi_fingerprint": kernel.launch_abi.fingerprint,
        "binary_abi_fingerprint": kernel.binary_abi.fingerprint,
    }
    values.update(overrides)
    return AttentionOperatorPhysicalLayoutEvidence(**values)


class PhysicalLayoutEvidenceCheckpoint(unittest.TestCase):
    """Checkpoint 028: only exact external physical evidence can authorize."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_evidence_is_canonical_versioned_and_round_trippable(self):
        values, _, spec, quant_spec, descriptor = physical_runtime()
        evidence = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )

        restored = AttentionOperatorPhysicalLayoutEvidence.from_dict(
            evidence.to_dict()
        )

        self.assertEqual(restored, evidence)
        self.assertEqual(restored.fingerprint, evidence.fingerprint)
        self.assertEqual(
            restored.passed_case_ids, tuple(sorted(restored.passed_case_ids))
        )

    def test_schema_requires_real_case_and_digest_identities(self):
        values, _, spec, quant_spec, descriptor = physical_runtime()
        with self.assertRaisesRegex(SchemaError, "passed_case_ids"):
            physical_evidence(
                spec,
                values["operation"],
                quant_spec,
                descriptor,
                passed_case_ids=(),
            )
        with self.assertRaisesRegex(SchemaError, "result_digest"):
            physical_evidence(
                spec,
                values["operation"],
                quant_spec,
                descriptor,
                result_digest="not-a-digest",
            )

    def test_missing_evidence_remains_fail_closed_before_callable_import(self):
        values, plan, spec, _, _ = physical_runtime()

        with self.assertRaisesRegex(
            AttentionDispatchError, "requires exactly one physical-layout evidence"
        ):
            active_session(values, spec=spec, plan=plan)

        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(package_attention.calls, [])

    def test_exact_evidence_authorizes_nonlogical_plan_and_metadata_lowering(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        evidence = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        authorized = replace(spec, physical_layout_evidence=(evidence,))
        active_plan, session = active_session(
            values, spec=authorized, plan=plan
        )
        kv_input = physical_input(quant_spec, descriptor)

        lowered = session.run("query", kv_input)

        receipt = session.active_plan.dispatch_receipt
        self.assertEqual(receipt.evidence_id, evidence.evidence_id)
        self.assertEqual(receipt.evidence_result_digest, evidence.result_digest)
        self.assertEqual(active_plan.fingerprint, plan.fingerprint)
        self.assertIs(dict(lowered.positional_arguments)["key"], kv_input.key_storage)
        self.assertEqual(package_attention.calls, [])

    def test_descriptor_catalog_or_quant_identity_drift_is_rejected(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        base = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        cases = (
            replace(base, descriptor_fingerprint=digest("other-descriptor")),
            replace(base, catalog_fingerprint=digest("other-catalog")),
            replace(base, quant_spec_fingerprint=digest("other-quant")),
        )
        for evidence in cases:
            with self.subTest(field=evidence.fingerprint):
                with self.assertRaises(AttentionDispatchError):
                    active_session(
                        values,
                        spec=replace(spec, physical_layout_evidence=(evidence,)),
                        plan=plan,
                    )
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_profile_environment_kernel_and_abi_drift_is_rejected(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        base = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        cases = (
            replace(base, profile_fingerprint=digest("other-profile")),
            replace(base, environment_fingerprint=digest("other-environment")),
            replace(base, kernel_fingerprint=digest("other-kernel")),
            replace(base, binary_abi_fingerprint=digest("other-abi")),
        )
        for evidence in cases:
            with self.subTest(field=evidence.fingerprint):
                with self.assertRaises(AttentionDispatchError):
                    active_session(
                        values,
                        spec=replace(spec, physical_layout_evidence=(evidence,)),
                        plan=plan,
                    )
        self.assertEqual(package_attention.calls, [])

    def test_operation_identity_and_duplicate_evidence_are_rejected(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        base = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        foreign = replace(base, operation_id="foreign.operation")
        with self.assertRaises(AttentionDispatchError):
            active_session(
                values,
                spec=replace(spec, physical_layout_evidence=(foreign,)),
                plan=plan,
            )
        with self.assertRaisesRegex(SchemaError, "evidence ids must be unique"):
            replace(spec, physical_layout_evidence=(base, base))

    def test_public_flashinfer_plan_run_surface_is_unchanged(self):
        for method in (BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("physical_layout_evidence", parameters)
            self.assertNotIn("provider", parameters)


if __name__ == "__main__":
    unittest.main()
