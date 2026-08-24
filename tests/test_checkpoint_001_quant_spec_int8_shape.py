import unittest
from dataclasses import replace

from flashinfer_npu.attention import infer_logical_quant_storage_shape
from flashinfer_npu.runtime import QuantSpec, SchemaError


def int8_tensor_spec():
    return QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float16",
        accumulator_dtype="float32",
        granularity="tensor",
        physical_layout="logical",
    )


class QuantSpecInt8StorageShapeCheckpoint(unittest.TestCase):
    """Checkpoint 001: QuantSpec identity and INT8 logical storage only."""

    def test_quant_spec_round_trip_and_fingerprint_are_stable(self):
        spec = int8_tensor_spec()
        restored = QuantSpec.from_dict(spec.to_dict())

        self.assertEqual(restored, spec)
        self.assertEqual(restored.fingerprint, spec.fingerprint)
        self.assertEqual(len(spec.fingerprint), 64)
        self.assertNotEqual(
            replace(spec, accumulator_dtype="float16").fingerprint,
            spec.fingerprint,
        )

    def test_int8_logical_storage_shape_equals_logical_tensor_shape(self):
        spec = int8_tensor_spec()

        for logical_shape in ((2, 3, 4, 64), (1, 1, 1), (0, 3, 4, 64)):
            with self.subTest(logical_shape=logical_shape):
                self.assertEqual(
                    infer_logical_quant_storage_shape(logical_shape, spec),
                    logical_shape,
                )

    def test_invalid_quant_spec_and_invalid_logical_shape_are_rejected(self):
        with self.assertRaisesRegex(SchemaError, "asymmetric.*zero point"):
            replace(int8_tensor_spec(), scheme="asymmetric")
        with self.assertRaisesRegex(SchemaError, "requires group_size"):
            replace(int8_tensor_spec(), granularity="group", axis=(3,))
        with self.assertRaisesRegex(SchemaError, "non-empty"):
            infer_logical_quant_storage_shape((), int8_tensor_spec())
        with self.assertRaisesRegex(SchemaError, "non-negative"):
            infer_logical_quant_storage_shape((2, -1, 4), int8_tensor_spec())


if __name__ == "__main__":
    unittest.main()
