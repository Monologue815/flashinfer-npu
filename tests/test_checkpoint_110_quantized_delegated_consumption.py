import unittest
from dataclasses import replace

from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalRunAdapter,
    package_attention,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    quantized_input,
    query_input,
)


class ReportingAdapter(FakeLogicalRunAdapter):
    def __init__(self, omitted=None, extra=None):
        self.omitted = omitted
        self.extra = extra
        self.calls = 0

    def lower(self, active, request):
        self.calls += 1
        lowered = super().lower(active, request)
        fields = tuple(
            name for name in lowered.consumed_request_fields
            if name != self.omitted
        )
        if self.extra is not None:
            fields += (self.extra,)
        return replace(lowered, consumed_request_fields=fields)


class QuantizedDelegatedConsumptionCheckpoint(unittest.TestCase):
    def setUp(self):
        package_attention.calls[:] = []

    def session(self, base):
        values = bootstrap_components()
        spec = replace(values["spec"], logical_run_adapter=base)
        plan, session = active_session(values, spec=spec)
        return plan, session, quantized_input(plan.spec.kv_quant_spec)

    def test_missing_base_receipt_is_rejected_without_losing_active_plan(self):
        for field in ("query", "kv_cache", "return_lse", "logits_soft_cap"):
            with self.subTest(field=field):
                base = ReportingAdapter(omitted=field)
                plan, session, kv_input = self.session(base)
                active = session.active_plan
                with self.assertRaisesRegex(SchemaError, "every run field"):
                    session.run(query_input(plan), kv_input)
                self.assertIs(session.active_plan, active)
                self.assertEqual(base.calls, 1)
                self.assertEqual(package_attention.calls, [])

                base.omitted = None
                lowered = session.run(query_input(plan), kv_input)
                self.assertIs(session.active_plan, active)
                self.assertIs(dict(lowered.keyword_arguments)["key_scale"],
                              kv_input.key_scale)
                self.assertIs(dict(lowered.keyword_arguments)["value_scale"],
                              kv_input.value_scale)
                self.assertEqual(base.calls, 2)
                self.assertEqual(package_attention.calls, [])

    def test_base_cannot_claim_fields_outside_its_delegated_request(self):
        base = ReportingAdapter(extra="v_scale")
        plan, session, kv_input = self.session(base)
        with self.assertRaisesRegex(SchemaError, "every run field"):
            session.run(query_input(plan), kv_input)
        self.assertEqual(package_attention.calls, [])


if __name__ == "__main__":
    unittest.main()
