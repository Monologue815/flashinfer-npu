"""Bounded, data-only manifests for package-shipped physical evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    AttentionJsonEnvelopeUsage,
    decode_attention_json,
)
from .operation_catalog import AttentionOperatorOperationSpec
from .operator_physical_evidence import AttentionOperatorPhysicalLayoutEvidence
from .quant_physical_layout import QuantPhysicalLayoutCatalog


ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION = 1
ATTENTION_OPERATOR_EVIDENCE_MAX_RESULT_BYTES = 64 * 1024 * 1024
ATTENTION_OPERATOR_EVIDENCE_MAX_RECORDS = 1024


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _safe_resource_locator(value: str) -> str:
    locator = str(value)
    if not locator or "\\" in locator or "\x00" in locator:
        raise SchemaError("evidence result locator must be a safe relative path")
    path = PurePosixPath(locator)
    if (
        path.is_absolute()
        or path.as_posix() != locator
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SchemaError("evidence result locator must be a safe relative path")
    if path.parts[0] != "evidence":
        raise SchemaError("evidence result locator must be inside evidence/")
    return path.as_posix()


@dataclass(frozen=True)
class AttentionOperatorEvidenceResultArtifact:
    evidence_id: str
    locator: str
    digest: str
    size_bytes: int
    schema_version: int = ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION:
            raise SchemaError("unsupported evidence result artifact version")
        if not str(self.evidence_id):
            raise SchemaError("evidence result artifact id must be non-empty")
        object.__setattr__(self, "evidence_id", str(self.evidence_id))
        object.__setattr__(self, "locator", _safe_resource_locator(self.locator))
        _require_hash("evidence result artifact digest", self.digest)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
            or self.size_bytes > ATTENTION_OPERATOR_EVIDENCE_MAX_RESULT_BYTES
        ):
            raise SchemaError("evidence result artifact size is outside limits")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("evidence result artifact fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("evidence result artifact fields are invalid") from error


@dataclass(frozen=True)
class AttentionOperatorPhysicalEvidenceManifest:
    name: str
    provider_id: str
    operation_id: str
    package_name: str
    adapter_version: str
    supported_package_versions: Tuple[str, ...]
    catalog_fingerprint: str
    evidences: Tuple[AttentionOperatorPhysicalLayoutEvidence, ...]
    result_artifacts: Tuple[AttentionOperatorEvidenceResultArtifact, ...]
    schema_version: int = ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION
    kind: str = field(
        default="attention_operator_physical_evidence_manifest", init=False
    )

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION:
            raise SchemaError("unsupported physical evidence manifest version")
        if self.kind != "attention_operator_physical_evidence_manifest":
            raise SchemaError("physical evidence manifest kind is invalid")
        for name in (
            "name",
            "provider_id",
            "operation_id",
            "package_name",
            "adapter_version",
        ):
            text = str(getattr(self, name))
            if not text:
                raise SchemaError("physical evidence manifest %s is empty" % name)
            object.__setattr__(self, name, text)
        versions = tuple(str(item) for item in self.supported_package_versions)
        if not versions or any(not item for item in versions) or len(set(versions)) != len(
            versions
        ):
            raise SchemaError("manifest package versions must be non-empty and unique")
        object.__setattr__(self, "supported_package_versions", tuple(sorted(versions)))
        _require_hash("manifest catalog_fingerprint", self.catalog_fingerprint)
        evidences = tuple(self.evidences)
        artifacts = tuple(self.result_artifacts)
        if not evidences or any(
            not isinstance(item, AttentionOperatorPhysicalLayoutEvidence)
            for item in evidences
        ):
            raise TypeError("manifest evidences must contain physical evidence records")
        if not artifacts or any(
            not isinstance(item, AttentionOperatorEvidenceResultArtifact)
            for item in artifacts
        ):
            raise TypeError("manifest result_artifacts must contain result artifacts")
        if len(evidences) > ATTENTION_OPERATOR_EVIDENCE_MAX_RECORDS:
            raise SchemaError("manifest evidence count exceeds limit")
        if sum(item.size_bytes for item in artifacts) > (
            ATTENTION_OPERATOR_EVIDENCE_MAX_RESULT_BYTES
        ):
            raise SchemaError("manifest total result bytes exceed limit")
        evidence_ids = tuple(item.evidence_id for item in evidences)
        artifact_ids = tuple(item.evidence_id for item in artifacts)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise SchemaError("manifest evidence ids must be unique")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise SchemaError("manifest artifact evidence ids must be unique")
        if set(evidence_ids) != set(artifact_ids):
            raise SchemaError("manifest evidence and result artifact sets differ")
        artifact_by_id = {item.evidence_id: item for item in artifacts}
        for evidence in evidences:
            artifact = artifact_by_id[evidence.evidence_id]
            if (
                evidence.provider_id != self.provider_id
                or evidence.operation_id != self.operation_id
                or evidence.catalog_fingerprint != self.catalog_fingerprint
                or evidence.result_digest != artifact.digest
            ):
                raise SchemaError("manifest evidence identity differs from its envelope")
        object.__setattr__(
            self, "evidences", tuple(sorted(evidences, key=lambda item: item.evidence_id))
        )
        object.__setattr__(
            self,
            "result_artifacts",
            tuple(sorted(artifacts, key=lambda item: item.evidence_id)),
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "package_name": self.package_name,
            "adapter_version": self.adapter_version,
            "supported_package_versions": list(self.supported_package_versions),
            "catalog_fingerprint": self.catalog_fingerprint,
            "evidences": [item.to_dict() for item in self.evidences],
            "result_artifacts": [item.to_dict() for item in self.result_artifacts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        data = dict(value)
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise SchemaError("physical evidence manifest fields are invalid")
        if data.pop("kind") != "attention_operator_physical_evidence_manifest":
            raise SchemaError("physical evidence manifest kind is invalid")
        for name, item_type in (
            ("evidences", AttentionOperatorPhysicalLayoutEvidence),
            ("result_artifacts", AttentionOperatorEvidenceResultArtifact),
        ):
            items = data.get(name)
            if not isinstance(items, (list, tuple)):
                raise SchemaError("physical evidence manifest %s must be an array" % name)
            if any(not isinstance(item, Mapping) for item in items):
                raise SchemaError(
                    "physical evidence manifest %s entries must be objects" % name
                )
            data[name] = tuple(item_type.from_dict(item) for item in items)
        versions = data.get("supported_package_versions")
        if not isinstance(versions, (list, tuple)):
            raise SchemaError("manifest package versions must be an array")
        data["supported_package_versions"] = tuple(versions)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("physical evidence manifest fields are invalid") from error

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate_runtime_spec(
        self,
        operation: AttentionOperatorOperationSpec,
        adapter_version: str,
        supported_package_versions: Tuple[str, ...],
        catalog: QuantPhysicalLayoutCatalog,
    ) -> None:
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(catalog, QuantPhysicalLayoutCatalog):
            raise TypeError("catalog must be QuantPhysicalLayoutCatalog")
        if (
            self.provider_id != operation.provider_id
            or self.operation_id != operation.operation_id
            or self.package_name != operation.package_name
            or self.adapter_version != str(adapter_version)
            or self.supported_package_versions
            != tuple(sorted(str(item) for item in supported_package_versions))
            or self.catalog_fingerprint != catalog.fingerprint
        ):
            raise SchemaError("physical evidence manifest runtime identity is stale")


@dataclass(frozen=True)
class AttentionVerifiedOperatorPhysicalEvidenceBundle:
    manifest: AttentionOperatorPhysicalEvidenceManifest
    verified_result_digests: Tuple[Tuple[str, str], ...]
    schema_version: int = ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION:
            raise SchemaError("unsupported verified physical evidence bundle version")
        if not isinstance(self.manifest, AttentionOperatorPhysicalEvidenceManifest):
            raise TypeError("bundle manifest has the wrong type")
        values = tuple((str(name), str(digest)) for name, digest in self.verified_result_digests)
        expected = tuple(
            (item.locator, item.digest) for item in self.manifest.result_artifacts
        )
        if tuple(sorted(values)) != tuple(sorted(expected)):
            raise SchemaError("verified result digests do not match manifest")
        object.__setattr__(self, "verified_result_digests", tuple(sorted(values)))

    @property
    def evidences(self) -> Tuple[AttentionOperatorPhysicalLayoutEvidence, ...]:
        return self.manifest.evidences

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "manifest_fingerprint": self.manifest.fingerprint,
                "verified_result_digests": [
                    list(item) for item in self.verified_result_digests
                ],
            }
        )


def load_attention_operator_physical_evidence_manifest(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
) -> Tuple[
    AttentionOperatorPhysicalEvidenceManifest, AttentionJsonEnvelopeUsage
]:
    decoded, usage = decode_attention_json(value, limits=limits)
    if not isinstance(decoded, Mapping):
        raise SchemaError("physical evidence manifest root must be an object")
    return AttentionOperatorPhysicalEvidenceManifest.from_dict(decoded), usage


def verify_attention_operator_physical_evidence_results(
    manifest: AttentionOperatorPhysicalEvidenceManifest,
    result_payloads: Mapping[str, bytes],
) -> AttentionVerifiedOperatorPhysicalEvidenceBundle:
    """Verify already-read package resource bytes without filesystem access."""

    if not isinstance(manifest, AttentionOperatorPhysicalEvidenceManifest):
        raise TypeError("manifest must be AttentionOperatorPhysicalEvidenceManifest")
    if not isinstance(result_payloads, Mapping):
        raise TypeError("result_payloads must be a mapping")
    expected = {item.locator: item for item in manifest.result_artifacts}
    if set(result_payloads) != set(expected):
        raise SchemaError("physical evidence result payload set differs from manifest")
    verified = []
    for locator in sorted(expected):
        payload = result_payloads[locator]
        if not isinstance(payload, bytes):
            raise TypeError("physical evidence result payloads must be bytes")
        artifact = expected[locator]
        if len(payload) != artifact.size_bytes:
            raise SchemaError("physical evidence result payload size mismatch")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact.digest:
            raise SchemaError("physical evidence result payload digest mismatch")
        verified.append((locator, digest))
    return AttentionVerifiedOperatorPhysicalEvidenceBundle(
        manifest, tuple(verified)
    )


__all__ = [
    "ATTENTION_OPERATOR_EVIDENCE_MANIFEST_VERSION",
    "ATTENTION_OPERATOR_EVIDENCE_MAX_RESULT_BYTES",
    "ATTENTION_OPERATOR_EVIDENCE_MAX_RECORDS",
    "AttentionOperatorEvidenceResultArtifact",
    "AttentionOperatorPhysicalEvidenceManifest",
    "AttentionVerifiedOperatorPhysicalEvidenceBundle",
    "load_attention_operator_physical_evidence_manifest",
    "verify_attention_operator_physical_evidence_results",
]
