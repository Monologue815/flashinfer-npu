import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog, AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageResolver, AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorRuntime, AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimePlanScore, AttentionOperatorRuntimeResolverRegistry,
    AttentionOperatorRuntimeResolutionError, AttentionStateError,
)
from flashinfer_npu.attention.operator_mask_binding import (
    AttentionMaskPlanRunAdapterBinder, AttentionOperatorMaskArgumentSpec,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeAuthorityResolver, FakeLogicalRunAdapter,
)
from tests.test_checkpoint_125_mask_tensor_inspection import inspection_inputs
from tests.test_checkpoint_129_call_retention import Recorder
from tests.test_checkpoint_131_transactional_mask_plan import MaskLoader, mask_runtime


def candidate(provider, operation_id, priority):
    values, original, _ = mask_runtime(Recorder())
    operation = replace(original, provider_id=provider, operation_id=operation_id,
                        callable_path=provider + ".synthetic_mask_attention")
    catalog = AttentionOperatorOperationCatalog(provider + "_synthetic", (operation,))
    resolver = AttentionOperatorPackageResolver(
        catalog, AttentionOperatorPackageCompatibility(provider, operation_id, "synthetic-v1", ("1.0.0",)),
        MaskLoader(values["events"]))

    class Authority(FakeAuthorityResolver):
        def authorize(self, *args):
            resolved = super().authorize(*args)
            return replace(resolved, selection=replace(resolved.selection, provider_id=provider))

    class Adapter(FakeLogicalRunAdapter):
        def lower(self, active, request):
            return replace(super().lower(active, request), operation_id=operation_id)

    authority = Authority(values["events"])
    authority.provider_id, authority.operation_id = provider, operation_id
    adapter = Adapter()
    adapter.provider_id = provider
    for item in (values["gate"], values["factory"]):
        item.provider_id, item.operation_id = provider, operation_id
    values["materializer"].provider_id = provider
    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=priority, package_resolver=resolver, plan_gate=values["gate"],
        authority_resolver=authority, logical_factory=values["factory"],
        logical_run_adapter=adapter, tensor_materializer=values["materializer"])

    class Observed:
        provider_id = provider

        def rejection_reasons(self, plan, device):
            values["events"].append("candidate_probe")
            return implementation.rejection_reasons(plan, device)

        def plan_score(self, plan, device):
            values["events"].append("candidate_score")
            return AttentionOperatorRuntimePlanScore(0, "synthetic", "test tie score")

        def resolve(self, plan, device):
            return implementation.resolve(plan, device)

    observed = Observed()
    observed.operation_id, observed.priority = operation_id, priority
    return operation, observed, values["events"]


def selection_inputs():
    low, low_impl, low_events = candidate("cann", "synthetic.bool@v1", 100)
    high, high_impl, high_events = candidate("flash_attention_npu", "synthetic.packed@v1", 200)
    catalog = AttentionOperatorOperationCatalog("synthetic_mask_selection", (low, high))
    implementations = AttentionOperatorRuntimeImplementationRegistry((low_impl, high_impl))
    registry = AttentionOperatorRuntimeResolverRegistry((("npu", implementations),))
    _, plan, _, _ = inspection_inputs()
    runtime = AttentionOperatorRuntime("npu:0", registry, catalog, mode=plan.spec.mode,
                                       completion_event_recorder=Recorder())
    mappings = (
        AttentionOperatorMaskArgumentSpec(low.fingerprint, "bool_allow_flat", "mask", "logical_offsets"),
        AttentionOperatorMaskArgumentSpec(high.fingerprint, "packed_allow_little_segments",
                                         "mask", "logical_offsets", "byte_offsets"),
    )
    return runtime, implementations, (low, high), mappings, (low_events, high_events)


def request(mappings, packed=False, device="npu:0"):
    _, plan, resource, inspector = inspection_inputs(packed)
    binder = AttentionMaskPlanRunAdapterBinder(resource.payload, resource.owner, inspector, mappings, device)
    return plan, binder, inspector


class MaskCandidateAdmissionCheckpoint(unittest.TestCase):
    def test_format_selects_compatible_provider_before_probe_and_priority(self):
        runtime, _, operations, mappings, events = selection_inputs()
        plan, binder, inspector = request(mappings)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(runtime.operator_session.active_plan.prepared_plan.implementation_id,
                         operations[0].operation_id)
        self.assertEqual(events[1], [])
        self.assertEqual(events[0].count("candidate_score"), 1)
        self.assertEqual(len(inspector.calls), 1)
        self.assertEqual(runtime.run("q", ("k", "v"), return_lse=False), "output")
        low_events = list(events[0])
        packed, packed_binder, packed_inspector = request(mappings, packed=True)
        runtime.plan(packed.spec, packed.metadata, run_adapter_plan_binder=packed_binder)
        self.assertEqual(runtime.operator_session.active_plan.prepared_plan.implementation_id,
                         operations[1].operation_id)
        self.assertEqual(events[0], low_events)
        self.assertEqual(events[1].count("candidate_score"), 1)
        self.assertEqual(len(packed_inspector.calls), 1)
        self.assertEqual(runtime.run("q", ("k", "v"), return_lse=True), ("output", "lse"))

    def test_all_rejected_report_preserves_old_plan_without_package_or_tensor_observation(self):
        runtime, _, _, mappings, events = selection_inputs()
        plan, binder, _ = request(mappings)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        old = runtime.operator_session
        snapshots = tuple(list(items) for items in events)
        plan, unsupported, inspector = request(mappings[1:])
        with self.assertRaises(AttentionOperatorRuntimeResolutionError) as caught:
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=unsupported)
        report = caught.exception.report
        self.assertEqual(report.accepted, ())
        self.assertEqual(len(report.candidates), 2)
        self.assertTrue(all(item.reasons and item.plan_score is None for item in report.candidates))
        self.assertIs(runtime.operator_session, old)
        self.assertEqual(tuple(events), snapshots)
        self.assertEqual(inspector.calls, [])
        self.assertEqual(runtime.run("q", ("k", "v"), return_lse=False), "output")

    def test_admission_does_not_modify_registry_or_subsequent_unmasked_selection(self):
        runtime, implementations, operations, mappings, _ = selection_inputs()
        identities = implementations.implementation_ids
        plan, binder, _ = request(mappings)
        runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        runtime.plan(replace(plan.spec, custom_mask=None), plan.metadata)
        self.assertEqual(runtime.operator_session.active_plan.prepared_plan.implementation_id,
                         operations[1].operation_id)
        self.assertEqual(implementations.implementation_ids, identities)

    def test_device_mismatch_excludes_every_candidate_without_inspection(self):
        runtime, _, _, mappings, events = selection_inputs()
        plan, binder, inspector = request(mappings, device="npu:1")
        with self.assertRaisesRegex(AttentionOperatorRuntimeResolutionError, "planned device"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(events, ([], []))
        self.assertEqual(inspector.calls, [])

    def test_reserved_argument_roles_are_rejected_during_admission(self):
        runtime, _, operations, mappings, events = selection_inputs()
        invalid = AttentionOperatorMaskArgumentSpec(
            operations[0].fingerprint, "bool_allow_flat", "key_scale", "logical_offsets")
        plan, binder, inspector = request((invalid, mappings[1]))
        with self.assertRaisesRegex(AttentionOperatorRuntimeResolutionError, "another operation role"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(events, ([], []))
        self.assertEqual(inspector.calls, [])

    def test_legacy_resolver_cannot_silently_ignore_resource_admission(self):
        runtime, _, _, mappings, _ = selection_inputs()

        class LegacyResolver:
            def resolve(self, plan, device):
                raise AssertionError("legacy resolver must not be invoked")

        legacy = AttentionOperatorRuntime(
            "npu:0", AttentionOperatorRuntimeResolverRegistry((("npu", LegacyResolver()),)),
            runtime._operation_catalog, mode=runtime.mode, completion_event_recorder=Recorder())
        plan, binder, inspector = request(mappings)
        with self.assertRaisesRegex(AttentionStateError, "pre-probe candidate admission"):
            legacy.plan(plan.spec, plan.metadata, run_adapter_plan_binder=binder)
        self.assertEqual(inspector.calls, [])

    def test_mask_binder_without_admission_cannot_reach_package_resolution(self):
        runtime, _, _, mappings, events = selection_inputs()

        class LateOnlyBinder:
            requires_call_retention = True

            def bind(self, base, active, operation):
                raise AssertionError("late-only mask binder must not be used")

        plan, _, _ = request(mappings)
        with self.assertRaisesRegex(AttentionStateError, "require pre-probe"):
            runtime.plan(plan.spec, plan.metadata, run_adapter_plan_binder=LateOnlyBinder())
        self.assertEqual(events, ([], []))

    def test_malformed_admission_and_duplicate_mappings_fail_closed(self):
        _, implementations, _, mappings, events = selection_inputs()
        plan, _, _ = request(mappings)
        for reasons in ("unsupported", ("",), (1,), ("duplicate", "duplicate")):
            with self.subTest(reasons=reasons):
                with self.assertRaises((TypeError, SchemaError)):
                    implementations.explain(plan, "npu:0", candidate_admission=lambda *args: reasons)
        self.assertEqual(events, ([], []))
        with self.assertRaisesRegex(SchemaError, "unique operation/encoding"):
            request((mappings[0], mappings[0]))
