"""Machine-readable FlashInfer API parity tracking."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from .runtime.schema import SchemaError


PARITY_SCHEMA_VERSION = 2
SUPPORTED_PARITY_SCHEMA_VERSIONS = {1, PARITY_SCHEMA_VERSION}
UPSTREAM_KINDS = {"stable", "experimental"}
SEMANTIC_STATUSES = {"unassessed", "exact", "compatible", "intentionally_different"}
IMPLEMENTATION_STATUSES = {
    "missing",
    "framework",
    "reference",
    "functional",
    "optimized",
}
ATTENTION_MODES = {
    "single_prefill",
    "single_decode",
    "batch_mixed_paged",
    "batch_prefill_paged",
    "batch_prefill_ragged",
    "batch_decode_paged",
}
PUBLIC_LIFECYCLES = {"one_shot", "plan_run"}
PROVIDER_ROUTING_KINDS = {"none", "ephemeral", "mode_bound"}
NPU_EXECUTION_STATUSES = {
    "none",
    "integration_required",
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
class AttentionParitySurface:
    """One user-facing Attention mode tracked independently of symbol counts."""

    surface_id: str
    local: str
    attention_mode: str
    public_lifecycle: str
    host_reference: bool
    provider_routing: str
    npu_execution: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionParitySurface":
        surface = cls(
            surface_id=str(value["surface_id"]),
            local=str(value["local"]),
            attention_mode=str(value["attention_mode"]),
            public_lifecycle=str(value["public_lifecycle"]),
            host_reference=value["host_reference"],
            provider_routing=str(value["provider_routing"]),
            npu_execution=str(value["npu_execution"]),
        )
        surface.validate()
        return surface

    def validate(self) -> None:
        if not self.surface_id or any(
            item not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for item in self.surface_id
        ):
            raise SchemaError("invalid Attention parity surface_id")
        if not self.local.startswith("flashinfer_npu."):
            raise SchemaError("invalid Attention parity surface local symbol")
        if self.attention_mode not in ATTENTION_MODES:
            raise SchemaError("invalid Attention parity surface mode")
        if self.public_lifecycle not in PUBLIC_LIFECYCLES:
            raise SchemaError("invalid Attention parity public lifecycle")
        if not isinstance(self.host_reference, bool):
            raise SchemaError("Attention parity host_reference must be boolean")
        if self.provider_routing not in PROVIDER_ROUTING_KINDS:
            raise SchemaError("invalid Attention parity provider routing")
        if self.npu_execution not in NPU_EXECUTION_STATUSES:
            raise SchemaError("invalid Attention parity NPU execution status")
        if self.public_lifecycle == "one_shot" and self.provider_routing == "mode_bound":
            raise SchemaError("one-shot Attention cannot own a reusable mode-bound route")
        if self.public_lifecycle == "plan_run" and self.provider_routing == "ephemeral":
            raise SchemaError("plan/run Attention cannot use an ephemeral route")
        if self.npu_execution in {"functional", "optimized"} and (
            self.provider_routing == "none"
        ):
            raise SchemaError("callable NPU execution requires provider routing")


@dataclass(frozen=True)
class ParityManifest:
    schema_version: int
    upstream_project: str
    upstream_ref: str
    upstream_url: str
    scope: str
    scope_definition: str
    inventory_status: str
    generated_at: str
    entries: Tuple[ParityEntry, ...]
    attention_surfaces: Tuple[AttentionParitySurface, ...] = ()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParityManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            value: Dict[str, Any] = json.load(handle)
        schema_version = value.get("schema_version")
        if schema_version not in SUPPORTED_PARITY_SCHEMA_VERSIONS:
            raise SchemaError(
                "unsupported parity schema_version %r" % value.get("schema_version")
            )
        upstream = value.get("upstream", {})
        manifest = cls(
            schema_version=int(schema_version),
            upstream_project=str(upstream["project"]),
            upstream_ref=str(upstream["ref"]),
            upstream_url=str(upstream["url"]),
            scope=str(value.get("scope", "all")),
            scope_definition=str(value.get("scope_definition", "All tracked APIs")),
            inventory_status=str(value["inventory_status"]),
            generated_at=str(value["generated_at"]),
            entries=tuple(ParityEntry.from_dict(item) for item in value["entries"]),
            attention_surfaces=tuple(
                AttentionParitySurface.from_dict(item)
                for item in value.get("attention_surfaces", ())
            ),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.inventory_status not in {"bootstrap", "complete"}:
            raise SchemaError("invalid inventory_status: %s" % self.inventory_status)
        if self.scope == "attention_core" and not re.fullmatch(
            r"[0-9a-f]{40}", self.upstream_ref
        ):
            raise SchemaError(
                "attention parity upstream_ref must pin a full commit SHA"
            )
        upstream_symbols = tuple(entry.upstream for entry in self.entries)
        local_symbols = tuple(entry.local for entry in self.entries)
        if len(upstream_symbols) != len(set(upstream_symbols)):
            raise SchemaError("parity manifest contains duplicate upstream symbols")
        if len(local_symbols) != len(set(local_symbols)):
            raise SchemaError("parity manifest contains duplicate local symbols")
        surface_ids = tuple(item.surface_id for item in self.attention_surfaces)
        surface_locals = tuple(item.local for item in self.attention_surfaces)
        surface_modes = tuple(item.attention_mode for item in self.attention_surfaces)
        if len(surface_ids) != len(set(surface_ids)):
            raise SchemaError("parity manifest contains duplicate Attention surfaces")
        if len(surface_locals) != len(set(surface_locals)):
            raise SchemaError("parity manifest maps multiple surfaces to one local symbol")
        if len(surface_modes) != len(set(surface_modes)):
            raise SchemaError("parity manifest contains duplicate Attention modes")
        entry_statuses = {
            entry.local: entry.implementation_status for entry in self.entries
        }
        for surface in self.attention_surfaces:
            status = entry_statuses.get(surface.local)
            if status is None:
                raise SchemaError(
                    "Attention parity surface local symbol is not in the inventory"
                )
            if status == "missing":
                raise SchemaError("represented Attention surface cannot be missing")
            if surface.host_reference and status not in {
                "reference",
                "functional",
                "optimized",
            }:
                raise SchemaError(
                    "Attention surface Host reference is not reflected by its symbol"
                )
            if surface.npu_execution in {"functional", "optimized"} and status not in {
                "functional",
                "optimized",
            }:
                raise SchemaError(
                    "Attention surface NPU execution is not reflected by its symbol"
                )

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

    def attention_surface_counts(self) -> Counter:
        counts = Counter()
        for surface in self.attention_surfaces:
            counts["host_reference"] += int(surface.host_reference)
            counts["provider_routed"] += int(surface.provider_routing != "none")
            counts["npu_callable"] += int(
                surface.npu_execution in {"functional", "optimized"}
            )
        return counts

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
        if self.attention_surfaces:
            surface_counts = self.attention_surface_counts()
            lines.extend(
                (
                    "  attention-surfaces: %d" % len(self.attention_surfaces),
                    "  host-reference-surfaces: %d"
                    % surface_counts["host_reference"],
                    "  provider-routed-surfaces: %d"
                    % surface_counts["provider_routed"],
                    "  npu-callable-surfaces: %d" % surface_counts["npu_callable"],
                )
            )
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
