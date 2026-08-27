from dataclasses import replace
import unittest

from flashinfer_npu.attention import (
    AttentionOperatorProviderIntegrationBundleBinding,
    AttentionOperatorRuntime,
    AttentionOperatorRuntimeDeclarationBinding,
    AttentionOperatorRuntimeRegistrySnapshot,
    AttentionStateError,
    BatchAttention,
    attention_operator_runtime_registry_snapshot,
    build_provider_plan_selection,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
    verify_attention_plan_scoring_chain,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_062_runtime_completion_publication import (
    run_runtime,
    strict_results,
    valid_result,
)
from tests.test_checkpoint_082_scoring_manifest_plan_run_chain import (
    scored_strict_runtime,
)
from tests.test_checkpoint_084_provider_integration_bundle import (
    provider_bundle,
)


def stable_type_name(value):
    value_type = type(value)
    return "%s.%s" % (value_type.__module__, value_type.__qualname__)


def synthetic_bundle_binding(values, manifest, declaration):
    return AttentionOperatorProviderIntegrationBundleBinding(
        bundle_id="checkpoint.085.strict.bundle.v1",
        bundle_fingerprint="b" * 64,
        catalog_name=values["catalog"].name,
        catalog_fingerprint=values["catalog"].fingerprint,
        scoring_manifest_id=manifest.manifest_id,
        scoring_manifest_fingerprint=manifest.fingerprint,
        package_loader_id=values["loader"].loader_id,
        package_loader_type=stable_type_name(values["loader"]),
        registration_bindings=(
            (
                declaration.provider_id,
                declaration.operation_id,
                declaration.fingerprint,
            ),
        ),
    )


def bundle_bound_strict_evidence(*, with_run=False):
    values, manifest, declaration, plan, source_runtime = scored_strict_runtime()
    bundle_binding = synthetic_bundle_binding(values, manifest, declaration)
    declaration_bindings = (
        (
            declaration.provider_id,
            declaration.operation_id,
            declaration.fingerprint,
        ),
    )
    resolver = source_runtime._resolver_registry.resolvers[0][1]
    resolution = resolver.explain(plan, "npu:0")
    runtime = AttentionOperatorRuntime(
        "npu:0",
        source_runtime._resolver_registry,
        values["catalog"],
        mode=plan.spec.mode,
        runtime_declaration_bindings=declaration_bindings,
        plan_scoring_manifest_binding=manifest.binding,
        provider_integration_bundle_binding=bundle_binding,
    )
    runtime.plan(plan.spec, plan.metadata)
    snapshot = AttentionOperatorRuntimeRegistrySnapshot(
        generation=31,
        device_types=("npu",),
        registry=source_runtime._resolver_registry,
        operation_catalog=values["catalog"],
        runtime_declarations=(
            AttentionOperatorRuntimeDeclarationBinding(
                declaration.provider_id,
                declaration.operation_id,
                declaration.fingerprint,
            ),
        ),
        plan_scoring_manifest_binding=manifest.binding,
        provider_integration_bundle_binding=bundle_binding,
    )
    scoring_binding = runtime.runtime_plan_scoring_binding
    runtime_bundle = runtime.runtime_provider_integration_bundle_binding
    score = runtime.runtime_plan_score
    selection = build_provider_plan_selection(
        runtime.plan_state,
        runtime.operator_session.active_plan,
        registry_generation=snapshot.generation,
        runtime_declaration_fingerprint=declaration.fingerprint,
        provider_integration_bundle_id=runtime_bundle[0],
        provider_integration_bundle_fingerprint=runtime_bundle[1],
        plan_scoring_manifest_id=scoring_binding[0],
        plan_scoring_manifest_fingerprint=scoring_binding[1],
        plan_scoring_policy_id=scoring_binding[2],
        plan_scoring_policy_fingerprint=scoring_binding[3],
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
        "runtime": runtime,
        "plan": runtime.plan_state,
        "resolution": resolution,
        "snapshot": snapshot,
        "bundle_binding": bundle_binding,
        "selection": selection,
        "receipt": receipt,
    }


class ProviderBundlePlanRunAuditCheckpoint(unittest.TestCase):
    """Checkpoint 085: bundle identity closes plan, run, and offline audit."""

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

    def test_public_plan_selection_exposes_installed_bundle_identity(self):
        values = bootstrap_components()
        bundle = provider_bundle(values)
        installed = install_attention_operator_provider_integration_bundle(
            bundle,
            expected_generation=self.original.generation,
        )
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="NHD",
            backend="auto",
        )
        plan = group_plan()

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
        selection = wrapper.plan_selection

        self.assertEqual(selection.registry_generation, installed.generation)
        self.assertEqual(
            selection.provider_integration_bundle_id,
            bundle.bundle_id,
        )
        self.assertEqual(
            selection.provider_integration_bundle_fingerprint,
            bundle.fingerprint,
        )
        self.assertEqual(
            selection.to_dict()["provider_integration_bundle_fingerprint"],
            bundle.fingerprint,
        )
        self.assertEqual(values["loader"].resolve_calls, 1)

        holistic = BatchAttention(kv_layout="HND", device="npu:0")
        self.assertEqual(
            holistic._operator_runtime._provider_integration_bundle_binding,
            bundle.binding,
        )
        self.assertEqual(values["loader"].resolve_calls, 1)

    def test_successful_run_receipt_carries_the_same_bundle_identity(self):
        evidence = bundle_bound_strict_evidence(with_run=True)
        runtime = evidence["runtime"]
        receipt = evidence["receipt"]
        selection = evidence["selection"]
        binding = evidence["bundle_binding"]

        self.assertEqual(
            runtime.runtime_provider_integration_bundle_binding,
            (binding.bundle_id, binding.bundle_fingerprint),
        )
        self.assertEqual(
            receipt.provider_integration_bundle_id,
            selection.provider_integration_bundle_id,
        )
        self.assertEqual(
            receipt.provider_integration_bundle_fingerprint,
            selection.provider_integration_bundle_fingerprint,
        )
        self.assertEqual(
            receipt.to_dict()["provider_integration_bundle_fingerprint"],
            binding.bundle_fingerprint,
        )

    def test_unplanned_fork_preserves_bundle_authority(self):
        evidence = bundle_bound_strict_evidence()
        forked = evidence["runtime"].fork_unplanned()
        plan = evidence["plan"]

        with self.assertRaisesRegex(AttentionStateError, "has not been planned"):
            _ = forked.runtime_provider_integration_bundle_binding
        forked.plan(plan.spec, plan.metadata)

        self.assertEqual(
            forked.runtime_provider_integration_bundle_binding,
            (
                evidence["bundle_binding"].bundle_id,
                evidence["bundle_binding"].bundle_fingerprint,
            ),
        )

    def test_runtime_rejects_catalog_declaration_or_manifest_bundle_drift(self):
        values, manifest, declaration, plan, runtime = scored_strict_runtime()
        binding = synthetic_bundle_binding(values, manifest, declaration)
        arguments = {
            "device": "npu:0",
            "resolver_registry": runtime._resolver_registry,
            "operation_catalog": values["catalog"],
            "mode": plan.spec.mode,
            "runtime_declaration_bindings": (
                (
                    declaration.provider_id,
                    declaration.operation_id,
                    declaration.fingerprint,
                ),
            ),
            "plan_scoring_manifest_binding": manifest.binding,
        }

        with self.assertRaisesRegex(SchemaError, "differs from operation catalog"):
            AttentionOperatorRuntime(
                **arguments,
                provider_integration_bundle_binding=replace(
                    binding,
                    catalog_fingerprint="c" * 64,
                ),
            )
        with self.assertRaisesRegex(SchemaError, "differs from declarations"):
            AttentionOperatorRuntime(
                **arguments,
                provider_integration_bundle_binding=replace(
                    binding,
                    registration_bindings=(
                        (
                            declaration.provider_id,
                            declaration.operation_id,
                            "d" * 64,
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(SchemaError, "differs from scoring manifest"):
            AttentionOperatorRuntime(
                **arguments,
                provider_integration_bundle_binding=replace(
                    binding,
                    scoring_manifest_fingerprint="e" * 64,
                ),
            )

    def test_bundle_identity_is_all_or_none_in_selection_and_receipt(self):
        evidence = bundle_bound_strict_evidence(with_run=True)

        with self.assertRaisesRegex(SchemaError, "identity is incomplete"):
            replace(
                evidence["selection"],
                provider_integration_bundle_fingerprint=None,
            )
        with self.assertRaisesRegex(SchemaError, "identity is incomplete"):
            replace(
                evidence["receipt"],
                provider_integration_bundle_id=None,
            )
        with self.assertRaisesRegex(SchemaError, "requires scoring manifest"):
            replace(
                evidence["selection"],
                plan_scoring_manifest_id=None,
                plan_scoring_manifest_fingerprint=None,
                plan_scoring_policy_id=None,
                plan_scoring_policy_fingerprint=None,
                plan_score=None,
                plan_score_source=None,
                plan_score_reason=None,
                runtime_resolution_fingerprint=None,
            )

    def test_offline_audit_binds_bundle_selection_and_run_receipt(self):
        evidence = bundle_bound_strict_evidence(with_run=True)

        report = verify_attention_plan_scoring_chain(
            plan=evidence["plan"],
            resolution_report=evidence["resolution"],
            registry_snapshot=evidence["snapshot"],
            scoring_manifest=evidence["manifest"],
            plan_selection=evidence["selection"],
            run_receipt=evidence["receipt"],
        )

        self.assertEqual(
            report.provider_integration_bundle_id,
            evidence["bundle_binding"].bundle_id,
        )
        self.assertEqual(
            report.provider_integration_bundle_fingerprint,
            evidence["bundle_binding"].bundle_fingerprint,
        )
        self.assertEqual(
            report.to_dict()["provider_integration_bundle_id"],
            evidence["bundle_binding"].bundle_id,
        )

    def test_offline_audit_rejects_plan_or_run_bundle_drift(self):
        evidence = bundle_bound_strict_evidence(with_run=True)
        arguments = {
            "plan": evidence["plan"],
            "resolution_report": evidence["resolution"],
            "registry_snapshot": evidence["snapshot"],
            "scoring_manifest": evidence["manifest"],
            "plan_selection": evidence["selection"],
            "run_receipt": evidence["receipt"],
        }

        with self.assertRaisesRegex(SchemaError, "plan provider bundle"):
            verify_attention_plan_scoring_chain(
                **{
                    **arguments,
                    "plan_selection": replace(
                        evidence["selection"],
                        provider_integration_bundle_fingerprint="f" * 64,
                    ),
                }
            )
        with self.assertRaisesRegex(SchemaError, "run receipt differs"):
            verify_attention_plan_scoring_chain(
                **{
                    **arguments,
                    "run_receipt": replace(
                        evidence["receipt"],
                        provider_integration_bundle_fingerprint="f" * 64,
                    ),
                }
            )


if __name__ == "__main__":
    unittest.main()
