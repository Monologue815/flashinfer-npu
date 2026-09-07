import unittest
from dataclasses import replace

from flashinfer_npu.attention import AttentionWorkspaceContract
from flashinfer_npu.runtime import SchemaError


class WorkspaceIntegerContractCheckpoint(unittest.TestCase):
    def contract(self):
        return AttentionWorkspaceContract(
            backend="reference", device="cpu", float_capacity_bytes=8,
            int_capacity_bytes=8, required_float_bytes=0, required_int_bytes=0,
            plan_generation=1,
        )

    def test_resource_fields_reject_lossy_or_boolean_numbers(self):
        original = self.contract()
        for field in ("float_capacity_bytes", "int_capacity_bytes",
                      "required_float_bytes", "required_int_bytes",
                      "binding_generation", "plan_generation"):
            for value in (True, 1.5, "1"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SchemaError):
                        replace(original, **{field: value})

    def test_fractional_requirements_cannot_be_rounded_down_to_fit(self):
        with self.assertRaises(SchemaError):
            AttentionWorkspaceContract(
                backend="test", device="cpu", float_capacity_bytes=0,
                int_capacity_bytes=0, required_float_bytes=0.5,
                required_int_bytes=0,
            )

    def test_binding_entrypoints_preserve_integer_identity_and_original_state(self):
        original = self.contract()
        fingerprint = original.fingerprint
        for value in (True, 1.5, "1"):
            actions = (
                lambda: original.bind_plan(value),
                lambda: original.bind_requirements(
                    required_float_bytes=0, required_int_bytes=0,
                    plan_generation=value),
                lambda: original.rebind(
                    device="cpu", float_capacity_bytes=value,
                    int_capacity_bytes=8, allow_device_change=False),
                lambda: original.rebind(
                    device="cpu", float_capacity_bytes=8,
                    int_capacity_bytes=value, allow_device_change=False),
                lambda: original.validate_run(device="cpu", plan_generation=value),
            )
            for index, action in enumerate(actions):
                with self.subTest(value=value, action=index):
                    with self.assertRaises(SchemaError):
                        action()
                    self.assertEqual(original.fingerprint, fingerprint)
        rebound = original.rebind(device="cpu", float_capacity_bytes=16,
                                  int_capacity_bytes=8, allow_device_change=False)
        self.assertEqual(rebound.binding_generation, 2)
        self.assertEqual(rebound.plan_generation, 1)
        original.validate_run(device="cpu", plan_generation=1)


if __name__ == "__main__":
    unittest.main()
