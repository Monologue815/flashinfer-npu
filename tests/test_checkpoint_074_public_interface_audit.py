import io
import unittest
from contextlib import redirect_stdout

from flashinfer_npu.attention import (
    AttentionPublicInterfaceAuditError,
    audit_attention_public_interface,
)
from flashinfer_npu.attention.interface_audit import _audit_callable
from flashinfer_npu.cli import main


class AttentionPublicInterfaceAuditTests(unittest.TestCase):
    def test_live_callables_are_bound_to_the_packaged_interface_contract(self):
        report = audit_attention_public_interface()
        self.assertEqual(report.model_facing_dispatch, "private_auto")
        self.assertEqual(len(report.model_facing), 14)
        self.assertEqual(len(report.advanced_injected_module), 2)
        roles = [item.role for item in report.model_facing]
        self.assertEqual(roles.count("one_shot"), 2)
        self.assertEqual(roles.count("constructor"), 4)
        self.assertEqual(roles.count("plan"), 4)
        self.assertEqual(roles.count("run"), 4)
        self.assertEqual(
            {item.parameters[0].name for item in report.advanced_injected_module},
            {"jit_module"},
        )
        paged_constructor = next(
            item
            for item in report.model_facing
            if item.local
            == "flashinfer_npu.prefill.BatchPrefillWithPagedKVCacheWrapper"
        )
        self.assertTrue(
            {"jit_args", "jit_kwargs"}
            <= {item.name for item in paged_constructor.parameters}
        )

    def test_report_serializes_exact_parameter_contract_and_stable_identity(self):
        report = audit_attention_public_interface()
        payload = report.to_dict()
        self.assertFalse(payload["run_accepts_plan_handle"])
        self.assertFalse(payload["caller_selects_provider"])
        batch_run = next(
            item
            for item in payload["model_facing"]
            if item["local"] == "flashinfer_npu.attention.BatchAttention.run"
        )
        self.assertEqual(
            [item["name"] for item in batch_run["parameters"]],
            [
                "self",
                "q",
                "kv_cache",
                "out",
                "lse",
                "k_scale",
                "v_scale",
                "logits_soft_cap",
                "profiler_buffer",
                "kv_cache_sf",
            ],
        )
        self.assertEqual(
            report.fingerprint,
            "fe73f710eab89577445e64b03e52beb53f0a1a6ce9fae9264b337f3e18edd3af",
        )

    def test_model_facing_internal_control_parameters_fail_closed(self):
        def invalid_run(q, plan_handle=None):
            return q

        with self.assertRaisesRegex(
            AttentionPublicInterfaceAuditError, "plan_handle"
        ):
            _audit_callable(
                "flashinfer_npu.attention.Invalid.run",
                "run",
                invalid_run,
                model_facing=True,
            )

    def test_advanced_entry_requires_explicit_jit_module_first(self):
        def invalid_advanced(q, module):
            return q

        with self.assertRaisesRegex(
            AttentionPublicInterfaceAuditError, "jit_module first"
        ):
            _audit_callable(
                "flashinfer_npu.prefill.invalid_with_jit_module",
                "advanced_injected_module",
                invalid_advanced,
                model_facing=False,
            )

    def test_cli_exposes_a_read_only_interface_audit(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["attention-interface"])
        self.assertEqual(status, 0)
        output = stdout.getvalue()
        self.assertIn("model-facing-dispatch: private_auto", output)
        self.assertIn("run-accepts-plan-handle: no", output)
        self.assertIn("caller-selects-provider: no", output)
        self.assertIn("fingerprint:", output)


if __name__ == "__main__":
    unittest.main()
