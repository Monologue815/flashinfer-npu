"""Synthetic package routing only; these profiles do not certify real operators."""

import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorQuantizedKVInput,
    AttentionOperatorRuntimeResolutionError,
    AttentionStateError,
    AttentionTraceCorpus,
    ReferenceKVData,
    attention_operator_runtime_registry_snapshot,
    infer_quant_scale_shape,
    framework_attention_coverage_policy,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_case, group_plan
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_025_quantized_provider_run_lowering import metadata_tensor
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_087_provider_bundle_assembly import (
    FLASH_OPERATION_ID, FlashLogicalRunAdapter, assemble_two_provider,
    two_provider_assembly_inputs,
)


class DenseLogicalAdapter(FlashLogicalRunAdapter):
    def lower(self, active_plan, request):
        # The shared synthetic adapter's default catalog operation is CANN.
        return replace(super().lower(active_plan, request), operation_id=FLASH_OPERATION_ID)


def switching_bundle():
    values = two_provider_assembly_inputs()
    quant_spec, dense_spec = values["specs"]
    profile = dense_spec.profiles[0]
    rule = profile.rules[0]
    q_dtype, _, o_dtype = rule.dtype_signatures[0]
    dense_rule = replace(rule, supports_dense_kv=True, quant_specs=(),
                         dtype_signatures=((q_dtype, q_dtype, o_dtype),))
    _, case = group_case()
    kv_data = case.trace.kv_data
    dense_case = replace(case, case_id="synthetic-dense-paged-decode", trace=replace(
        case.trace, spec=replace(case.trace.spec, kv_dtype=q_dtype, kv_quant_spec=None),
        kv_data=ReferenceKVData(
            replace(kv_data.spec, dtype=q_dtype, quant_spec=None),
            (kv_data.key_data.dequantize(), kv_data.value_data.dequantize()),
        ),
    ))
    corpus = AttentionTraceCorpus("synthetic-dense-routing", (dense_case,),
                                  "Host-only fixture; not real provider evidence")
    policy = framework_attention_coverage_policy()
    coverage = policy.evaluate(corpus)
    evidence = replace(profile.evidence[0], corpus_fingerprint=corpus.fingerprint,
                       covered_cells=coverage.covered_cells,
                       total_cells=len(coverage.requirements),
                       passed_case_ids=(dense_case.case_id,))
    dense_profile = replace(profile, rules=(dense_rule,), evidence=(evidence,))
    descriptor = dense_spec.descriptors[0]
    dense_descriptor = replace(
        descriptor,
        constraints=replace(descriptor.constraints,
                            dtype_signatures=dense_rule.dtype_signatures,
                            quant_storage_dtypes=()),
        capability_binding=replace(descriptor.capability_binding,
                                   profile_fingerprint=dense_profile.fingerprint),
    )
    dense_spec = replace(dense_spec, profiles=(dense_profile,),
                         descriptors=(dense_descriptor,), quantization_bindings=(),
                         corpus=corpus, coverage_policy=policy,
                         logical_run_adapter=DenseLogicalAdapter())
    values["specs"] = (quant_spec, dense_spec)
    return values, assemble_two_provider(values)


def plan(wrapper, kv_dtype):
    reference = group_plan()
    return wrapper.plan(
        reference.metadata.indptr, reference.metadata.indices,
        reference.metadata.last_page_len, reference.spec.num_qo_heads,
        reference.spec.num_kv_heads, reference.spec.head_dim_vo,
        reference.metadata.page_size, q_data_type=reference.spec.q_dtype,
        kv_data_type=kv_dtype, o_data_type=reference.spec.o_dtype,
    )


def inputs(wrapper):
    active = wrapper.plan_state
    q = metadata_tensor("query", active.expected_query_shape, active.spec.q_dtype)
    # This bounded fixture has two NHD pages and equal QK/VO dimensions.
    shape = (2, 2, 1, 2)
    quant = active.spec.kv_quant_spec
    key = metadata_tensor("key", shape, active.spec.kv_dtype)
    value = metadata_tensor("value", shape, active.spec.kv_dtype)
    if quant is None:
        return q, (key, value)
    return q, AttentionOperatorQuantizedKVInput(
        quant_spec=quant, key_storage=key, value_storage=value,
        key_scale=metadata_tensor("key-scale", infer_quant_scale_shape(shape, quant),
                                  quant.scale_dtype),
        value_scale=metadata_tensor("value-scale", infer_quant_scale_shape(shape, quant),
                                    quant.scale_dtype),
    )


class DenseQuantizedPlanSwitchCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        package_attention.calls[:] = []
        self.values, self.bundle = switching_bundle()
        self.installed = install_attention_operator_provider_integration_bundle(self.bundle)
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="NHD")

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry, operation_catalog=self.original.operation_catalog,
        )

    def test_one_wrapper_switches_exact_operations_without_run_time_resolution(self):
        quant = group_plan().spec.kv_quant_spec
        dense_dtype = group_plan().spec.q_dtype
        old_input = None
        for kv_dtype, provider in ((dense_dtype, "flash_attention_npu"),
                                   (quant, "cann"), (dense_dtype, "flash_attention_npu")):
            with self.subTest(provider=provider, kv_dtype=kv_dtype):
                self.assertIsNone(plan(self.wrapper, kv_dtype))
                selection = self.wrapper.plan_selection
                self.assertEqual(selection.provider_id, provider)
                expected_operation = (FLASH_OPERATION_ID if provider == "flash_attention_npu"
                                      else self.values["cann"]["operation"].operation_id)
                self.assertEqual(selection.operation_id, expected_operation)
                self.assertEqual(selection.provider_integration_bundle_fingerprint,
                                 self.bundle.fingerprint)
                with self.assertRaises(AttentionStateError):
                    self.wrapper.last_run_receipt
                q, kv = inputs(self.wrapper)
                calls_before = len(package_attention.calls)
                if old_input is not None:
                    with self.assertRaises((SchemaError, TypeError)):
                        self.wrapper.run(q, old_input)
                    self.assertEqual(len(package_attention.calls), calls_before)
                events = list(self.values["events"])
                active = self.wrapper.plan_state
                for _ in range(2):
                    self.wrapper.run(q, kv)
                    self.assertIs(self.wrapper.plan_state, active)
                    self.assertEqual(self.values["events"], events)
                    lowered = self.wrapper._operator_runtime.last_lowered_call
                    self.assertEqual(lowered.provider_id, provider)
                    self.assertEqual(lowered.operation_id, expected_operation)
                    call = package_attention.calls[-1]
                    if isinstance(kv, AttentionOperatorQuantizedKVInput):
                        self.assertIs(call[1], kv.key_storage)
                        self.assertIs(call[6], kv.key_scale)
                        self.assertIs(call[7], kv.value_scale)
                    else:
                        self.assertIs(call[1], kv[0])
                        self.assertIsNone(call[6])
                        self.assertIsNone(call[7])
                self.assertEqual(len(package_attention.calls), calls_before + 2)
                old_input = kv
        self.assertEqual(self.values["cann"]["loader"].resolve_calls, 1)
        self.assertEqual(self.values["flash_loader"].resolve_calls, 2)

    def test_unsupported_quant_plan_keeps_dense_runtime_executable(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        q, kv = inputs(self.wrapper)
        self.wrapper.run(q, kv)
        active = self.wrapper.plan_state
        selection = self.wrapper.plan_selection
        workspace = self.wrapper.workspace_contract
        unsupported = replace(group_plan().spec.kv_quant_spec, granularity="tensor",
                              group_size=None, axis=None)
        with self.assertRaises(AttentionOperatorRuntimeResolutionError):
            plan(self.wrapper, unsupported)
        self.assertIs(self.wrapper.plan_state, active)
        self.assertEqual(self.wrapper.plan_selection, selection)
        self.assertIs(self.wrapper.workspace_contract, workspace)
        self.wrapper.run(q, kv)
        self.assertEqual(len(package_attention.calls), 2)
        self.assertEqual(self.values["cann"]["loader"].resolve_calls, 0)
        self.assertEqual(self.values["flash_loader"].resolve_calls, 1)

    def test_incompatible_higher_scored_candidate_is_excluded_before_scoring(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        resolver = self.installed.registry.resolvers[0][1]
        report = resolver.explain(self.wrapper.plan_state, "npu:0")
        candidates = {item.provider_id: item for item in report.candidates}
        rejected = candidates["cann"]
        self.assertFalse(rejected.accepted)
        self.assertIsNone(rejected.plan_score)
        self.assertTrue(any("dense KV is unsupported" in reason for reason in rejected.reasons))
        self.assertEqual(report.selected.provider_id, "flash_attention_npu")
        self.assertEqual(self.values["cann"]["loader"].resolve_calls, 0)
        self.assertEqual(package_attention.calls, [])
