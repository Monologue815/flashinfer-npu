import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Nvfp4CapabilitySurfaceCheckpoint(unittest.TestCase):
    """Public NVFP4 framework coverage never implies an installed NPU op."""

    def test_batch_plan_manifest_describes_private_nvfp4_canonicalization(self):
        manifest = json.loads(
            (ROOT / "flashinfer_npu/data/attention_api_parity.json").read_text(
                encoding="utf-8"
            )
        )
        entry = next(
            item
            for item in manifest["entries"]
            if item["local"] == "flashinfer_npu.attention.BatchAttention.plan"
        )

        self.assertEqual(entry["semantic_status"], "compatible")
        self.assertEqual(entry["implementation_status"], "reference")
        self.assertIn("bare uint8", entry["notes"])
        self.assertIn("NVFP4 QuantSpec", entry["notes"])
        self.assertIn("plan remains internal", entry["notes"])
        self.assertIn("No production NPU implementation", entry["notes"])

    def test_support_matrix_separates_framework_and_production_status(self):
        matrix = (ROOT / "docs/support_matrix.md").read_text(encoding="utf-8")

        self.assertIn(
            "| NVFP4 | `framework` | `framework` | `integration-required` |",
            matrix,
        )
        self.assertIn(
            "| FP8 | `reference` | `framework` | `integration-required` |",
            matrix,
        )
        self.assertIn("| MX | `planned` | `planned` | `planned` |", matrix)
        self.assertNotIn("| FP8/NVFP4/MX |", matrix)
        self.assertIn("Host oracle does not execute packed NVFP4", matrix)


if __name__ == "__main__":
    unittest.main()
