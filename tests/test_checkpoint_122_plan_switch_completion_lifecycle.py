"""Strict result validation with synthetic providers, never real NPU kernels."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from flashinfer_npu.attention import (
    AttentionOperatorPackageLoaderRoute, AttentionStateError,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor
from tests.test_checkpoint_062_runtime_completion_publication import (
    StrictResultPackageLoader, strict_calls, strict_results, valid_result,
)
from tests.test_checkpoint_087_provider_bundle_assembly import assemble_two_provider
from tests.test_checkpoint_119_dense_quantized_plan_switch import inputs, plan, switching_bundle


class PlanSwitchCompletionLifecycleCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        strict_calls[:] = []
        strict_results[:] = []
        values, _ = switching_bundle()
        values["specs"] = tuple(replace(spec, validate_provider_results=True)
                                for spec in values["specs"])
        loaders = tuple(StrictResultPackageLoader(values["events"]) for _ in range(2))
        values["routes"] = tuple(AttentionOperatorPackageLoaderRoute.from_catalog_operation(op, loader)
                                 for op, loader in zip(values["operations"], loaders))
        self.values = values
        self.loaders = loaders
        self.bundle = assemble_two_provider(values)
        install_attention_operator_provider_integration_bundle(self.bundle)
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="NHD")

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry, operation_catalog=self.original.operation_catalog)
        strict_calls[:] = []
        strict_results[:] = []

    def run_success(self):
        expected = valid_result(self.wrapper._operator_runtime, "-%d" % len(strict_calls))
        strict_results.append(expected)
        result = self.wrapper.run(*inputs(self.wrapper), return_lse=True)
        self.assertIs(result, expected)
        return self.wrapper.last_run_receipt

    def assert_no_run_evidence(self):
        with self.assertRaises(AttentionStateError):
            self.wrapper.last_run_receipt
        with self.assertRaises(AttentionStateError):
            self.wrapper._operator_runtime.last_completion_receipt
        with self.assertRaises(AttentionStateError):
            self.wrapper._operator_runtime.last_lowered_call

    def test_strict_receipts_follow_dense_quant_dense_plan_switches(self):
        baseline = group_plan().spec
        receipts = []
        for kv_dtype, provider in ((baseline.q_dtype, "flash_attention_npu"),
                                   (baseline.kv_quant_spec, "cann"),
                                   (baseline.q_dtype, "flash_attention_npu")):
            plan(self.wrapper, kv_dtype)
            self.assert_no_run_evidence()
            receipt = self.run_success()
            selection = self.wrapper.plan_selection
            self.assertEqual(receipt.provider_id, provider)
            self.assertEqual(receipt.operation_id, selection.operation_id)
            self.assertEqual(receipt.active_plan_fingerprint,
                             self.wrapper._operator_runtime.operator_session.active_plan.fingerprint)
            self.assertEqual(receipt.completion.framework_plan_fingerprint,
                             self.wrapper.plan_state.fingerprint)
            self.assertEqual(receipt.provider_integration_bundle_fingerprint,
                             self.bundle.fingerprint)
            self.assertEqual(receipt.runtime_declaration_fingerprint,
                             selection.runtime_declaration_fingerprint)
            receipts.append(receipt)
        self.assertEqual(len({r.active_plan_fingerprint for r in receipts}), 3)
        self.assertEqual(len(strict_calls), 3)
        self.assertEqual(tuple(loader.resolve_calls for loader in self.loaders), (1, 2))

    def test_failed_input_validation_clears_previous_success_without_execution(self):
        plan(self.wrapper, group_plan().spec.kv_quant_spec)
        self.run_success()
        active = self.wrapper.plan_state
        q, _ = inputs(self.wrapper)
        with self.assertRaises(SchemaError):
            self.wrapper.run(q, ("dense-k", "dense-v"), return_lse=True)
        self.assertEqual(len(strict_calls), 1)
        self.assert_no_run_evidence()
        self.assertIs(self.wrapper.plan_state, active)
        self.run_success()

    def test_failed_frontend_options_clear_previous_success_without_execution(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        self.run_success()
        with self.assertRaises(NotImplementedError):
            self.wrapper.run(*inputs(self.wrapper), enable_pdl=True)
        self.assertEqual(len(strict_calls), 1)
        self.assert_no_run_evidence()
        self.run_success()

    def test_invalid_second_result_clears_all_previous_publication(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        self.run_success()
        active = self.wrapper.plan_state
        strict_results.append((buffer_tensor("bad-output", (1,), active.spec.o_dtype),
                               buffer_tensor("lse", active.expected_lse_shape, "float32")))
        with self.assertRaisesRegex(SchemaError, "output result shape"):
            self.wrapper.run(*inputs(self.wrapper), return_lse=True)
        self.assertEqual(len(strict_calls), 2)
        self.assert_no_run_evidence()
        self.assertIs(self.wrapper.plan_state, active)
        self.run_success()

    def test_receipt_assembly_failure_cannot_publish_partial_completion(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        self.run_success()
        strict_results.append(valid_result(self.wrapper._operator_runtime, "-second"))
        with patch("flashinfer_npu.attention.operator_resolver.AttentionOperatorRunReceipt",
                   side_effect=SchemaError("synthetic receipt assembly failure")):
            with self.assertRaisesRegex(SchemaError, "receipt assembly failure"):
                self.wrapper.run(*inputs(self.wrapper), return_lse=True)
        self.assertEqual(len(strict_calls), 2)
        self.assert_no_run_evidence()
        self.run_success()
