import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    ArtifactVerificationError,
    Backend,
    KernelArgumentABI,
    KernelArgumentDirection,
    KernelArgumentPassing,
    KernelBinaryABI,
    KernelDescriptor,
    KernelErrorABI,
    KernelErrorCodeABI,
    KernelLaunchABI,
    SchemaError,
    load_kernel_manifest,
)


def artifact(payload=b"synthetic-ascend-object", locator="kernels/test.o"):
    return ArtifactRef(
        kind=ArtifactKind.FILE,
        format=ArtifactFormat.ASCENDC_OBJECT,
        locator=locator,
        digest=hashlib.sha256(payload).hexdigest(),
        target_soc="Ascend910B",
        build_id="test-build-id",
        size_bytes=len(payload),
    )


def launch_abi():
    return KernelLaunchABI(
        abi_name="flashinfer_npu.test.v1",
        entry_point="test_entry",
        argument_names=("x", "out", "workspace", "stream"),
        mutable_arguments=("out", "workspace"),
        stream_argument="stream",
    )


def binary_abi():
    return KernelBinaryABI(
        abi_name="flashinfer_npu.test.binary.v1",
        arguments=(
            KernelArgumentABI(
                "x",
                KernelArgumentPassing.POINTER,
                KernelArgumentDirection.INPUT,
            ),
            KernelArgumentABI(
                "out",
                KernelArgumentPassing.POINTER,
                KernelArgumentDirection.INOUT,
            ),
            KernelArgumentABI(
                "workspace",
                KernelArgumentPassing.POINTER,
                KernelArgumentDirection.INOUT,
                nullable=True,
            ),
            KernelArgumentABI(
                "stream",
                KernelArgumentPassing.OPAQUE_HANDLE,
                KernelArgumentDirection.INPUT,
            ),
        ),
        error_abi=KernelErrorABI(
            "flashinfer_npu.test_error.v1",
            (KernelErrorCodeABI("success", 0),),
        ),
    )


class ArtifactRefTests(unittest.TestCase):
    def test_round_trip_fingerprint_and_payload_verification(self):
        payload = b"synthetic-ascend-object"
        original = artifact(payload)
        restored = ArtifactRef.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)
        restored.verify_bytes(payload)
        with self.assertRaisesRegex(ArtifactVerificationError, "size mismatch"):
            restored.verify_bytes(payload + b"x")
        with self.assertRaisesRegex(ArtifactVerificationError, "digest mismatch"):
            artifact(bytes([payload[0] ^ 1]) + payload[1:]).verify_bytes(payload)

    def test_file_verification_is_package_root_bounded(self):
        payload = b"verified-object"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "kernels" / "verified.o"
            target.parent.mkdir()
            target.write_bytes(payload)
            reference = artifact(payload, "kernels/verified.o")
            self.assertEqual(reference.verify_file(root), target.resolve())

            outside = root.parent / (root.name + "-outside.o")
            outside.write_bytes(payload)
            link = root / "kernels" / "escape.o"
            try:
                link.symlink_to(outside)
                escaping = artifact(payload, "kernels/escape.o")
                with self.assertRaisesRegex(
                    ArtifactVerificationError, "escapes package root"
                ):
                    escaping.verify_file(root)
            finally:
                outside.unlink()

    def test_locator_kind_format_and_size_are_strict(self):
        base = artifact()
        invalid = (
            (dict(locator="../escape.o"), "normalized relative"),
            (dict(locator="./kernels/test.o"), "normalized relative"),
            (dict(locator="kernels\\test.o"), "normalized relative"),
            (dict(kind=ArtifactKind.BUILTIN), "kind does not match"),
            (dict(size_bytes=None), "requires non-negative size"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    replace(base, **changes)

    def test_builtin_has_provider_digest_but_no_file_bytes(self):
        locator = "builtin:aclnn:attention"
        builtin = ArtifactRef(
            kind=ArtifactKind.BUILTIN,
            format=ArtifactFormat.ACLNN_BUILTIN,
            locator=locator,
            digest=hashlib.sha256(locator.encode("utf-8")).hexdigest(),
            target_soc="Ascend910B",
            build_id="provider-contract-v1",
        )
        with self.assertRaisesRegex(ArtifactVerificationError, "builtin provider"):
            builtin.verify_bytes(b"")


class KernelLaunchABITests(unittest.TestCase):
    def test_round_trip_and_fingerprint(self):
        original = launch_abi()
        restored = KernelLaunchABI.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)

    def test_argument_mutation_stream_and_pointer_contracts(self):
        base = launch_abi()
        invalid = (
            (dict(argument_names=("x", "x", "stream")), "unique"),
            (dict(mutable_arguments=("missing",)), "argument subset"),
            (dict(stream_argument="missing"), "must occur"),
            (dict(pointer_width_bits=16), "32 or 64"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    replace(base, **changes)


class ArtifactManifestTests(unittest.TestCase):
    def test_non_reference_descriptor_requires_provenance_and_abi(self):
        with self.assertRaisesRegex(SchemaError, "ArtifactRef"):
            KernelDescriptor(
                "missing-provenance", "test", Backend.ASCENDC_AOT,
                artifact="kernels/test.o",
            )
        with self.assertRaisesRegex(SchemaError, "KernelLaunchABI"):
            KernelDescriptor(
                "missing-abi", "test", Backend.ASCENDC_AOT,
                artifact=artifact(),
            )
        with self.assertRaisesRegex(SchemaError, "KernelBinaryABI"):
            KernelDescriptor(
                "missing-binary-abi", "test", Backend.ASCENDC_AOT,
                artifact=artifact(), launch_abi=launch_abi(),
            )
        with self.assertRaisesRegex(SchemaError, "format does not match"):
            KernelDescriptor(
                "wrong-format", "test", Backend.ACLNN,
                artifact=artifact(), launch_abi=launch_abi(),
                binary_abi=binary_abi(),
            )

    def test_manifest_round_trip_and_bare_string_rejection(self):
        descriptor = KernelDescriptor(
            "structured", "test", Backend.ASCENDC_AOT,
            artifact=artifact(), launch_abi=launch_abi(),
            binary_abi=binary_abi(),
        )
        payload = {
            "schema_version": 3,
            "generated_at": "2026-08-19",
            "kernels": [descriptor.to_dict()],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_kernel_manifest(path), (descriptor,))
            payload["kernels"][0]["artifact"] = "kernels/test.o"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "artifact must be an object"):
                load_kernel_manifest(path)
            payload["schema_version"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "schema_version"):
                load_kernel_manifest(path)


if __name__ == "__main__":
    unittest.main()
