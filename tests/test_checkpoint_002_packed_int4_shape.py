import unittest

from flashinfer_npu.attention import infer_logical_quant_storage_shape
from flashinfer_npu.runtime import QuantSpec, SchemaError


def packed_int4_spec(storage_dtype, packing_order="low_nibble_first"):
    unsigned = storage_dtype == "uint4_packed"
    return QuantSpec(
        scheme="asymmetric" if unsigned else "symmetric",
        storage_dtype=storage_dtype,
        compute_dtype="float16",
        accumulator_dtype="float32",
        granularity="tensor",
        has_zero_point=unsigned,
        physical_layout="logical",
        packing_order=packing_order,
    )


class PackedInt4StorageShapeCheckpoint(unittest.TestCase):
    """Checkpoint 002: packed INT4/UINT4 logical storage shape only."""

    def test_odd_last_dimension_rounds_up_to_a_complete_byte(self):
        logical_shape = (2, 3, 4, 33)

        for storage_dtype in ("int4_packed", "uint4_packed"):
            with self.subTest(storage_dtype=storage_dtype):
                self.assertEqual(
                    infer_logical_quant_storage_shape(
                        logical_shape, packed_int4_spec(storage_dtype)
                    ),
                    (2, 3, 4, 17),
                )

    def test_even_single_and_empty_last_dimensions_have_canonical_sizes(self):
        spec = packed_int4_spec("int4_packed")
        cases = (
            ((2, 3, 4, 32), (2, 3, 4, 16)),
            ((2, 3, 4, 1), (2, 3, 4, 1)),
            ((2, 3, 4, 0), (2, 3, 4, 0)),
        )

        for logical_shape, storage_shape in cases:
            with self.subTest(logical_shape=logical_shape):
                self.assertEqual(
                    infer_logical_quant_storage_shape(logical_shape, spec),
                    storage_shape,
                )

    def test_packing_order_is_required_and_identity_bound_but_shape_neutral(self):
        low = packed_int4_spec("int4_packed", "low_nibble_first")
        high = packed_int4_spec("int4_packed", "high_nibble_first")

        self.assertNotEqual(low.fingerprint, high.fingerprint)
        self.assertEqual(
            infer_logical_quant_storage_shape((1, 1, 7), low),
            infer_logical_quant_storage_shape((1, 1, 7), high),
        )
        for storage_dtype in ("int4_packed", "uint4_packed"):
            with self.subTest(storage_dtype=storage_dtype):
                with self.assertRaisesRegex(SchemaError, "requires packing_order"):
                    packed_int4_spec(storage_dtype, packing_order=None)


if __name__ == "__main__":
    unittest.main()
