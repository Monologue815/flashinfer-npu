import sys
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDispatchError,
    AttentionEvidenceOperatorRuntimeAuthorityResolver,
    AttentionOperatorOperationSpec,
    AttentionOperatorProviderProbe,
    AttentionOperatorRuntimeAuthorityResolver,
)
from flashinfer_npu.runtime import Backend, SchemaError
from tests.test_attention_capability import (
    attention_artifact,
    bound_kernel,
    functional_profile,
    group_plan,
)


def operation(provider_id="cann"):
    plan = group_plan()
    return AttentionOperatorOperationSpec(
        operation_id="%s.checkpoint_021_attention@v1" % provider_id,
        provider_id=provider_id,
        package_name="checkpoint-021-package",
        callable_path="checkpoint_021_package.attention",
        api_version="v1",
        candidate_modes=(plan.spec.mode,),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=("scale",),
        return_names=("output",),
        source_url="https://example.com/checkpoint-021-attention-v1",
    )


def probe(provider_id="cann", version="1.0.0"):
    return AttentionOperatorProviderProbe(
        provider_id=provider_id,
        adapter_version="checkpoint-021-adapter-v1",
        available=True,
        package_versions=(("checkpoint-021-package", version),),
    )


def resolver(*, environment=None, backend="auto", descriptors=None):
    profile = functional_profile()
    return AttentionEvidenceOperatorRuntimeAuthorityResolver(
        operation(),
        (profile,),
        (bound_kernel(profile),) if descriptors is None else descriptors,
        profile.environment if environment is None else environment,
        backend=backend,
    )


class EvidenceAuthorityResolverCheckpoint(unittest.TestCase):
    """Checkpoint 021: package authority uses existing evidence dispatch."""

    def test_resolver_implements_authority_protocol_without_package_import(self):
        imported_before = set(sys.modules)
        authority_resolver = resolver()

        self.assertIsInstance(
            authority_resolver, AttentionOperatorRuntimeAuthorityResolver
        )
        self.assertEqual(authority_resolver.provider_id, "cann")
        self.assertEqual(
            authority_resolver.operation_id, operation().operation_id
        )
        self.assertEqual(len(authority_resolver.profile_ids), 1)
        self.assertEqual(len(authority_resolver.kernel_ids), 1)
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_authorize_closes_plan_evidence_kernel_provider_and_device(self):
        profile = functional_profile()
        descriptor = bound_kernel(profile)
        authority_resolver = AttentionEvidenceOperatorRuntimeAuthorityResolver(
            operation(), (profile,), (descriptor,), profile.environment
        )
        plan = group_plan()
        provider_probe = probe()

        authority = authority_resolver.authorize(
            plan, "npu:3", operation(), provider_probe
        )

        self.assertEqual(authority.framework_plan_fingerprint, plan.fingerprint)
        self.assertEqual(authority.device, "npu:3")
        self.assertEqual(
            authority.provider_probe_fingerprint, provider_probe.fingerprint
        )
        self.assertEqual(authority.operation_fingerprint, operation().fingerprint)
        self.assertEqual(authority.receipt.profile_id, profile.profile_id)
        self.assertEqual(authority.receipt.kernel_id, descriptor.kernel_id)
        self.assertEqual(
            authority.selection.dispatch_receipt_fingerprint,
            authority.receipt.fingerprint,
        )
        self.assertEqual(authority.selection.requested_provider, "cann")
        self.assertEqual(len(authority.fingerprint), 64)
        authority.receipt.validate(
            plan, profile, descriptor, profile.environment
        )

    def test_environment_drift_is_an_evidence_dispatch_rejection(self):
        profile = functional_profile()
        changed = replace(profile.environment, firmware_version="changed")
        authority_resolver = resolver(environment=changed)

        with self.assertRaisesRegex(
            AttentionDispatchError, "environment fingerprint mismatch"
        ):
            authority_resolver.authorize(
                group_plan(), "npu:0", operation(), probe()
            )

    def test_backend_policy_cannot_relabel_an_incompatible_descriptor(self):
        authority_resolver = resolver(backend=Backend.ACLNN)

        with self.assertRaisesRegex(AttentionDispatchError, "backend policy"):
            authority_resolver.authorize(
                group_plan(), "npu", operation(), probe()
            )

    def test_provider_operation_and_device_identity_mismatches_are_rejected(self):
        authority_resolver = resolver()
        plan = group_plan()
        cases = (
            ("cpu", operation(), probe(), "device npu"),
            ("npu:", operation(), probe(), "device npu"),
            ("npu:abc", operation(), probe(), "device npu"),
            (
                "npu:0",
                operation("flash_attention_npu"),
                probe(),
                "different operation",
            ),
            (
                "npu:0",
                operation(),
                probe("flash_attention_npu"),
                "available provider probe",
            ),
        )
        for device, candidate_operation, candidate_probe, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    authority_resolver.authorize(
                        plan, device, candidate_operation, candidate_probe
                    )

    def test_unknown_or_reference_backend_is_rejected_as_schema_input(self):
        for backend, message in (
            ("unknown", "unknown operator authority backend"),
            (Backend.REFERENCE, "cannot request reference"),
        ):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(SchemaError, message):
                    resolver(backend=backend)

    def test_tuned_kernel_can_only_select_an_evidence_accepted_descriptor(self):
        profile = functional_profile()
        default = bound_kernel(profile, priority=20)
        tuned = bound_kernel(
            profile,
            kernel_id="checkpoint_021_tuned_kernel",
            artifact=attention_artifact(
                "artifacts/ascend910b/checkpoint_021_tuned.o"
            ),
            priority=1,
        )
        authority_resolver = AttentionEvidenceOperatorRuntimeAuthorityResolver(
            operation(),
            (profile,),
            (default, tuned),
            profile.environment,
            tuned_kernel_ids=("unknown-kernel", tuned.kernel_id),
        )

        authority = authority_resolver.authorize(
            group_plan(), "npu:0", operation(), probe()
        )

        self.assertEqual(authority.receipt.kernel_id, tuned.kernel_id)
        self.assertEqual(authority.receipt.selection_source, "tuning")


if __name__ == "__main__":
    unittest.main()
