import json
import tempfile
import unittest
from pathlib import Path

from flashinfer_npu.cli import (
    packaged_attention_capability_manifest_path,
    packaged_kernel_manifest_path,
)
from flashinfer_npu.attention import load_attention_capability_manifest
from flashinfer_npu.parity import (
    ParityManifest,
    load_packaged_manifest,
    packaged_manifest_path,
)
from flashinfer_npu.runtime import SchemaError, load_kernel_manifest


class ManifestTests(unittest.TestCase):
    def test_packaged_kernel_manifest_is_valid_and_honestly_empty(self):
        self.assertEqual(load_kernel_manifest(packaged_kernel_manifest_path()), ())

    def test_packaged_attention_capabilities_are_valid_and_honestly_empty(self):
        self.assertEqual(
            load_attention_capability_manifest(
                packaged_attention_capability_manifest_path()
            ),
            (),
        )

    def test_packaged_parity_manifest_is_valid(self):
        manifest = load_packaged_manifest("all")
        self.assertEqual(manifest.schema_version, 2)
        self.assertEqual(manifest.inventory_status, "bootstrap")
        self.assertGreaterEqual(len(manifest.entries), 20)
        self.assertFalse(manifest.is_complete)
        self.assertEqual(manifest.counts()["missing"], len(manifest.entries))

    def test_attention_parity_inventory_tracks_single_reference_facades(self):
        manifest = load_packaged_manifest("attention")
        self.assertEqual(manifest.schema_version, 3)
        self.assertEqual(manifest.scope, "attention_core")
        self.assertEqual(
            manifest.upstream_ref,
            "919a24e5b1d971d50c97a3cd38862f801527eab5",
        )
        self.assertEqual(manifest.inventory_status, "complete")
        self.assertGreaterEqual(len(manifest.entries), 18)
        self.assertEqual(manifest.counts()["reference"], 32)
        self.assertEqual(manifest.counts()["framework"], 2)
        self.assertEqual(manifest.counts()["missing"], 0)
        statuses = {
            entry.upstream: entry.implementation_status
            for entry in manifest.entries
        }
        for symbol in (
            "flashinfer.attention.BatchAttention",
            "flashinfer.attention.BatchAttention.plan",
            "flashinfer.attention.BatchAttention.run",
        ):
            self.assertEqual(statuses[symbol], "reference")
        self.assertEqual(
            statuses["flashinfer.prefill.single_prefill_with_kv_cache"],
            "reference",
        )
        self.assertEqual(
            statuses["flashinfer.decode.single_decode_with_kv_cache"],
            "reference",
        )
        for symbol in (
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.workspace_size",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.plan",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.plan",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.workspace_size",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.forward",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.forward",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.forward",
        ):
            self.assertEqual(statuses[symbol], "reference")
        for symbol in (
            "flashinfer.prefill.single_prefill_with_kv_cache_with_jit_module",
            "flashinfer.decode.single_decode_with_kv_cache_with_jit_module",
        ):
            self.assertEqual(statuses[symbol], "framework")
        self.assertFalse(manifest.is_complete)

    def test_attention_parity_requires_immutable_upstream_commit(self):
        source = json.loads(
            packaged_manifest_path("attention").read_text(encoding="utf-8")
        )
        source["upstream"]["ref"] = "main snapshot 2026-08-27"
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "attention_api_parity.json"
            manifest_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "full commit SHA"):
                ParityManifest.load(manifest_path)

    def test_attention_parity_tracks_the_six_public_provider_routes(self):
        manifest = load_packaged_manifest("attention")
        surfaces = {item.attention_mode: item for item in manifest.attention_surfaces}

        self.assertEqual(
            set(surfaces),
            {
                "single_prefill",
                "single_decode",
                "batch_mixed_paged",
                "batch_prefill_paged",
                "batch_prefill_ragged",
                "batch_decode_paged",
            },
        )
        for mode in ("single_prefill", "single_decode"):
            self.assertEqual(surfaces[mode].public_lifecycle, "one_shot")
            self.assertEqual(surfaces[mode].provider_routing, "ephemeral")
        for mode in (
            "batch_mixed_paged",
            "batch_prefill_paged",
            "batch_prefill_ragged",
            "batch_decode_paged",
        ):
            self.assertEqual(surfaces[mode].public_lifecycle, "plan_run")
            self.assertEqual(surfaces[mode].provider_routing, "mode_bound")
        self.assertTrue(all(item.host_reference for item in surfaces.values()))
        self.assertTrue(
            all(
                item.npu_execution == "integration_required"
                for item in surfaces.values()
            )
        )
        self.assertEqual(
            manifest.attention_surface_counts(),
            {
                "host_reference": 6,
                "provider_routed": 6,
                "npu_callable": 0,
            },
        )
        report = manifest.report()
        self.assertIn("attention-surfaces: 6", report)
        self.assertIn("provider-routed-surfaces: 6", report)
        self.assertIn("npu-callable-surfaces: 0", report)

    def test_attention_surface_lifecycle_and_inventory_links_are_enforced(self):
        source = json.loads(
            packaged_manifest_path("attention").read_text(encoding="utf-8")
        )
        invalid_values = []

        invalid_lifecycle = json.loads(json.dumps(source))
        invalid_lifecycle["attention_surfaces"][1]["provider_routing"] = (
            "mode_bound"
        )
        invalid_values.append((invalid_lifecycle, "one-shot"))

        unknown_local = json.loads(json.dumps(source))
        unknown_local["attention_surfaces"][0]["local"] = (
            "flashinfer_npu.attention.Unknown"
        )
        invalid_values.append((unknown_local, "not in the inventory"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.json"
            for value, message in invalid_values:
                with self.subTest(message=message):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(SchemaError, message):
                        ParityManifest.load(path)

    def test_duplicate_kernel_ids_are_rejected(self):
        kernel = {
            "kernel_id": "duplicate",
            "op": "rmsnorm",
            "backend": "reference",
        }
        value = {
            "schema_version": 3,
            "generated_at": "2026-08-19",
            "kernels": [kernel, kernel],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "duplicate"):
                load_kernel_manifest(path)


if __name__ == "__main__":
    unittest.main()
