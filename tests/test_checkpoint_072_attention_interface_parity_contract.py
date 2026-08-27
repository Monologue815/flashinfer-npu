import json
import tempfile
import unittest
from pathlib import Path

from flashinfer_npu.parity import ParityManifest, load_packaged_manifest
from flashinfer_npu.runtime import SchemaError


class AttentionInterfaceParityContractTests(unittest.TestCase):
    def setUp(self):
        self.path = (
            Path(__file__).resolve().parents[1]
            / "flashinfer_npu"
            / "data"
            / "attention_api_parity.json"
        )
        self.source = json.loads(self.path.read_text(encoding="utf-8"))

    def load_modified(self, mutate):
        value = json.loads(json.dumps(self.source))
        mutate(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention_api_parity.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return ParityManifest.load(path)

    def test_packaged_contract_separates_normal_and_advanced_interfaces(self):
        manifest = load_packaged_manifest("attention")
        contract = manifest.attention_interface_contract
        self.assertIsNotNone(contract)
        self.assertEqual(contract.model_facing_dispatch, "private_auto")
        self.assertFalse(contract.run_accepts_plan_handle)
        self.assertFalse(contract.caller_selects_provider)
        self.assertEqual(
            set(contract.advanced_injected_module_symbols),
            {
                "flashinfer_npu.prefill.single_prefill_with_kv_cache_with_jit_module",
                "flashinfer_npu.decode.single_decode_with_kv_cache_with_jit_module",
            },
        )
        surface_locals = {item.local for item in manifest.attention_surfaces}
        self.assertTrue(
            set(contract.advanced_injected_module_symbols).isdisjoint(surface_locals)
        )

    def test_plan_handle_and_provider_selection_are_fail_closed(self):
        for field, message in (
            ("run_accepts_plan_handle", "cannot accept a plan handle"),
            ("caller_selects_provider", "cannot expose provider selection"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(SchemaError, message):
                    self.load_modified(
                        lambda value, field=field: value[
                            "attention_interface_contract"
                        ].__setitem__(field, True)
                    )

    def test_advanced_symbol_list_must_match_inventory_exactly(self):
        with self.assertRaisesRegex(SchemaError, "does not match inventory"):
            self.load_modified(
                lambda value: value["attention_interface_contract"][
                    "advanced_injected_module_symbols"
                ].pop()
            )

    def test_advanced_symbol_cannot_be_registered_as_model_facing_surface(self):
        advanced = self.source["attention_interface_contract"][
            "advanced_injected_module_symbols"
        ][0]

        def expose_advanced_symbol(value):
            value["attention_surfaces"][0]["local"] = advanced
            for entry in value["entries"]:
                if entry["local"] == advanced:
                    entry["implementation_status"] = "reference"
                    break

        with self.assertRaisesRegex(SchemaError, "model-facing surface"):
            self.load_modified(expose_advanced_symbol)

    def test_report_exposes_interface_contract_without_handles(self):
        report = load_packaged_manifest("attention").report()
        self.assertIn("attention-dispatch: private_auto", report)
        self.assertIn("run-accepts-plan-handle: no", report)
        self.assertIn("caller-selects-provider: no", report)


if __name__ == "__main__":
    unittest.main()
