import json
import tempfile
import unittest
from pathlib import Path

from flashinfer_npu.cli import (
    packaged_attention_capability_manifest_path,
    packaged_kernel_manifest_path,
)
from flashinfer_npu.attention import load_attention_capability_manifest
from flashinfer_npu.parity import load_packaged_manifest
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
        self.assertEqual(manifest.inventory_status, "bootstrap")
        self.assertGreaterEqual(len(manifest.entries), 20)
        self.assertFalse(manifest.is_complete)
        self.assertEqual(manifest.counts()["missing"], len(manifest.entries))

    def test_attention_parity_inventory_tracks_single_reference_facades(self):
        manifest = load_packaged_manifest("attention")
        self.assertEqual(manifest.scope, "attention_core")
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
