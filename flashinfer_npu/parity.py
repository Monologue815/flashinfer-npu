"""Machine-readable FlashInfer API parity tracking."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from .runtime.schema import SchemaError


PARITY_SCHEMA_VERSION = 1
UPSTREAM_KINDS = {"stable", "experimental"}
SEMANTIC_STATUSES = {"unassessed", "exact", "compatible", "intentionally_different"}
IMPLEMENTATION_STATUSES = {
    "missing",
    "framework",
    "reference",
    "functional",
    "optimized",
}


@dataclass(frozen=True)
class ParityEntry:
    upstream: str
    local: str
    upstream_kind: str
    semantic_status: str
    implementation_status: str
    priority: str
    notes: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParityEntry":
        entry = cls(
            upstream=str(value["upstream"]),
            local=str(value["local"]),
            upstream_kind=str(value["upstream_kind"]),
            semantic_status=str(value["semantic_status"]),
            implementation_status=str(value["implementation_status"]),
            priority=str(value["priority"]),
            notes=str(value.get("notes", "")),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if not self.upstream.startswith("flashinfer."):
            raise SchemaError("invalid upstream symbol: %s" % self.upstream)
        if not self.local.startswith("flashinfer_npu."):
            raise SchemaError("invalid local symbol: %s" % self.local)
        if self.upstream_kind not in UPSTREAM_KINDS:
            raise SchemaError("invalid upstream_kind: %s" % self.upstream_kind)
        if self.semantic_status not in SEMANTIC_STATUSES:
            raise SchemaError("invalid semantic_status: %s" % self.semantic_status)
        if self.implementation_status not in IMPLEMENTATION_STATUSES:
            raise SchemaError(
                "invalid implementation_status: %s" % self.implementation_status
            )
        if self.priority not in {"P0", "P1", "P2", "P3"}:
            raise SchemaError("invalid priority: %s" % self.priority)


@dataclass(frozen=True)
class ParityManifest:
    upstream_project: str
    upstream_ref: str
    upstream_url: str
    scope: str
    scope_definition: str
    inventory_status: str
    generated_at: str
    entries: Tuple[ParityEntry, ...]

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParityManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            value: Dict[str, Any] = json.load(handle)
        if value.get("schema_version") != PARITY_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported parity schema_version %r" % value.get("schema_version")
            )
        upstream = value.get("upstream", {})
        manifest = cls(
            upstream_project=str(upstream["project"]),
            upstream_ref=str(upstream["ref"]),
            upstream_url=str(upstream["url"]),
            scope=str(value.get("scope", "all")),
            scope_definition=str(value.get("scope_definition", "All tracked APIs")),
            inventory_status=str(value["inventory_status"]),
            generated_at=str(value["generated_at"]),
            entries=tuple(ParityEntry.from_dict(item) for item in value["entries"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.inventory_status not in {"bootstrap", "complete"}:
            raise SchemaError("invalid inventory_status: %s" % self.inventory_status)
        upstream_symbols = tuple(entry.upstream for entry in self.entries)
        local_symbols = tuple(entry.local for entry in self.entries)
        if len(upstream_symbols) != len(set(upstream_symbols)):
            raise SchemaError("parity manifest contains duplicate upstream symbols")
        if len(local_symbols) != len(set(local_symbols)):
            raise SchemaError("parity manifest contains duplicate local symbols")

    @property
    def is_complete(self) -> bool:
        return self.inventory_status == "complete" and all(
            entry.implementation_status in {"functional", "optimized"}
            and entry.semantic_status != "unassessed"
            for entry in self.entries
            if entry.upstream_kind == "stable"
        )

    def counts(self) -> Counter:
        return Counter(entry.implementation_status for entry in self.entries)

    def report(self) -> str:
        counts = self.counts()
        lines = [
            "FlashInfer API parity",
            "  upstream: %s (%s)" % (self.upstream_project, self.upstream_ref),
            "  scope: %s" % self.scope,
            "  inventory: %s" % self.inventory_status,
            "  symbols: %d" % len(self.entries),
        ]
        for status in (
            "optimized",
            "functional",
            "reference",
            "framework",
            "missing",
        ):
            lines.append("  %-10s %d" % (status + ":", counts[status]))
        lines.append("  release-compatible: %s" % ("yes" if self.is_complete else "no"))
        return "\n".join(lines)


def packaged_manifest_path(scope: str = "all") -> Path:
    filename = {
        "all": "api_parity.json",
        "attention": "attention_api_parity.json",
    }.get(scope)
    if filename is None:
        raise ValueError("unknown parity scope: %s" % scope)
    return Path(__file__).resolve().parent / "data" / filename


def load_packaged_manifest(scope: str = "all") -> ParityManifest:
    return ParityManifest.load(packaged_manifest_path(scope))
