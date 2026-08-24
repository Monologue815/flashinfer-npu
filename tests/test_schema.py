import unittest

from flashinfer_npu.runtime import QuantSpec, SchemaError, WorkloadSpec, WorkspaceFormula


class QuantSpecTests(unittest.TestCase):
    def test_valid_groupwise_int4_is_stable(self):
        spec = QuantSpec(
            scheme="symmetric",
            storage_dtype="int4_packed",
            compute_dtype="float16",
            accumulator_dtype="float32",
            scale_dtype="float16",
            granularity="group",
            group_size=[128],
            axis=[1],
            physical_layout="nk_interleaved_v1",
            packing_order="low_nibble_first",
        )
        reconstructed = QuantSpec.from_dict(spec.to_dict())
        self.assertEqual(spec, reconstructed)
        self.assertEqual(spec.fingerprint, reconstructed.fingerprint)

    def test_packed_int4_requires_packing_order(self):
        with self.assertRaisesRegex(SchemaError, "packing_order"):
            QuantSpec(
                scheme="symmetric",
                storage_dtype="int4_packed",
                compute_dtype="float16",
                accumulator_dtype="float32",
            )

    def test_asymmetric_requires_zero_point(self):
        with self.assertRaisesRegex(SchemaError, "zero point"):
            QuantSpec(
                scheme="asymmetric",
                storage_dtype="int8",
                compute_dtype="int8",
                accumulator_dtype="int32",
            )


class WorkloadSpecTests(unittest.TestCase):
    def test_attribute_order_does_not_change_fingerprint(self):
        first = WorkloadSpec(
            op="gemm",
            dtypes=("int8", "int8", "float16"),
            attributes=(("n", "4096"), ("k", "4096")),
        )
        second = WorkloadSpec(
            op="gemm",
            dtypes=("int8", "int8", "float16"),
            attributes=(("k", "4096"), ("n", "4096")),
        )
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_workspace_formula_is_aligned(self):
        workload = WorkloadSpec(
            op="attention",
            dtypes=("float16",),
            dynamic_bounds=(3, 17),
        )
        formula = WorkspaceFormula(
            constant_bytes=7,
            dynamic_coefficients=(10, 2),
            alignment=32,
        )
        self.assertEqual(formula.size_for(workload), 96)


if __name__ == "__main__":
    unittest.main()

