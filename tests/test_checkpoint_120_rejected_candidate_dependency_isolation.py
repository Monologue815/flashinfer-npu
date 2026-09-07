"""Rejected synthetic candidates must not observe their optional packages."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from flashinfer_npu.attention import (
    AttentionOperatorRuntimeResolutionError,
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_provider_integration_bundle,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_047_public_plan_selection import FakeNpuWorkspace
from tests.test_checkpoint_119_dense_quantized_plan_switch import (
    inputs, plan, switching_bundle,
)


class RejectedCandidateDependencyIsolationCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.bundle = switching_bundle()
        install_attention_operator_provider_integration_bundle(self.bundle)
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="NHD")
        package_attention.calls[:] = []

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry, operation_catalog=self.original.operation_catalog,
        )

    def test_dense_plan_never_observes_incompatible_quant_package(self):
        loader = self.values["cann"]["loader"]
        with patch.object(loader, "package_version", side_effect=RuntimeError("unused package")) as probe:
            plan(self.wrapper, group_plan().spec.q_dtype)
            self.wrapper.run(*inputs(self.wrapper))
            probe.assert_not_called()
        self.assertEqual(self.wrapper.plan_selection.provider_id, "flash_attention_npu")
        self.assertEqual(loader.resolve_calls, 0)
        self.assertEqual(len(package_attention.calls), 1)

    def test_quant_plan_never_observes_incompatible_dense_package(self):
        loader = self.values["flash_loader"]
        with patch.object(loader, "package_version", side_effect=RuntimeError("unused package")) as probe:
            plan(self.wrapper, group_plan().spec.kv_quant_spec)
            self.wrapper.run(*inputs(self.wrapper))
            probe.assert_not_called()
        self.assertEqual(self.wrapper.plan_selection.provider_id, "cann")
        self.assertEqual(loader.resolve_calls, 0)
        self.assertEqual(len(package_attention.calls), 1)

    def test_all_rejected_plans_keep_old_runtime_without_observing_packages(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        q, kv = inputs(self.wrapper)
        active = self.wrapper.plan_state
        unsupported = replace(group_plan().spec.kv_quant_spec, granularity="tensor",
                              group_size=None, axis=None)
        with patch.object(self.values["cann"]["loader"], "package_version",
                          side_effect=RuntimeError("unused quant package")) as quant_probe:
            with patch.object(self.values["flash_loader"], "package_version",
                              side_effect=RuntimeError("unused dense package")) as dense_probe:
                with self.assertRaises(AttentionOperatorRuntimeResolutionError) as caught:
                    plan(self.wrapper, unsupported)
                self.assertIs(self.wrapper.plan_state, active)
                self.assertEqual(len(caught.exception.report.candidates), 2)
                self.assertTrue(all(not item.accepted for item in caught.exception.report.candidates))
                self.wrapper.run(q, kv)
                quant_probe.assert_not_called()
                dense_probe.assert_not_called()
        self.assertEqual(len(package_attention.calls), 1)

    def test_relevant_package_errors_are_not_silently_ignored(self):
        plan(self.wrapper, group_plan().spec.q_dtype)
        active = self.wrapper.plan_state
        with patch.object(self.values["flash_loader"], "package_version",
                          side_effect=RuntimeError("relevant package failed")) as probe:
            with self.assertRaisesRegex(RuntimeError, "relevant package failed"):
                plan(self.wrapper, group_plan().spec.q_dtype)
            probe.assert_called_once()
        self.assertIs(self.wrapper.plan_state, active)
        self.assertEqual(self.values["cann"]["loader"].resolve_calls, 0)
        self.assertEqual(self.values["flash_loader"].resolve_calls, 1)
        self.assertEqual(package_attention.calls, [])
