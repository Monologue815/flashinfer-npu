"""Machine-readable FlashInfer API parity tracking."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .runtime.schema import SchemaError


PARITY_SCHEMA_VERSION = 3
SUPPORTED_PARITY_SCHEMA_VERSIONS = {1, 2, PARITY_SCHEMA_VERSION}
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
MODEL_FACING_DISPATCH_KINDS = {"private_auto"}


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
class AttentionInterfaceContract:
    """Machine-readable boundary between model-facing and advanced APIs."""

    model_facing_dispatch: str
    run_accepts_plan_handle: bool
    caller_selects_provider: bool
    advanced_injected_module_symbols: Tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionInterfaceContract":
        symbols = value.get("advanced_injected_module_symbols")
        if not isinstance(symbols, list):
            raise SchemaError(
                "advanced_injected_module_symbols must be a JSON array"
            )
        contract = cls(
            model_facing_dispatch=str(value["model_facing_dispatch"]),
            run_accepts_plan_handle=value["run_accepts_plan_handle"],
            caller_selects_provider=value["caller_selects_provider"],
            advanced_injected_module_symbols=tuple(str(item) for item in symbols),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.model_facing_dispatch not in MODEL_FACING_DISPATCH_KINDS:
            raise SchemaError("invalid Attention model-facing dispatch contract")
        if not isinstance(self.run_accepts_plan_handle, bool):
            raise SchemaError("run_accepts_plan_handle must be boolean")
        if self.run_accepts_plan_handle:
            raise SchemaError("model-facing Attention run cannot accept a plan handle")
        if not isinstance(self.caller_selects_provider, bool):
            raise SchemaError("caller_selects_provider must be boolean")
        if self.caller_selects_provider:
            raise SchemaError("model-facing Attention cannot expose provider selection")
        if not self.advanced_injected_module_symbols:
            raise SchemaError("advanced injected-module symbols must be explicit")
        if len(self.advanced_injected_module_symbols) != len(
            set(self.advanced_injected_module_symbols)
        ):
            raise SchemaError("duplicate advanced injected-module symbol")
        for symbol in self.advanced_injected_module_symbols:
            if not symbol.startswith("flashinfer_npu.") or not symbol.endswith(
                "_with_jit_module"
            ):
                raise SchemaError("invalid advanced injected-module symbol")


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
    attention_interface_contract: Optional[AttentionInterfaceContract] = None

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
            attention_interface_contract=(
                AttentionInterfaceContract.from_dict(
                    value["attention_interface_contract"]
                )
                if "attention_interface_contract" in value
                else None
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
        if self.scope == "attention_core" and self.schema_version >= 3:
            if self.attention_interface_contract is None:
                raise SchemaError(
                    "Attention parity schema v3 requires an interface contract"
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
        contract = self.attention_interface_contract
        if contract is not None:
            entry_locals = {entry.local for entry in self.entries}
            declared_advanced = set(contract.advanced_injected_module_symbols)
            inventoried_advanced = {
                entry.local
                for entry in self.entries
                if entry.local.endswith("_with_jit_module")
            }
            if declared_advanced != inventoried_advanced:
                raise SchemaError(
                    "advanced injected-module contract does not match inventory"
                )
            if not declared_advanced <= entry_locals:
                raise SchemaError("advanced injected-module symbol is not inventoried")
            if declared_advanced & set(surface_locals):
                raise SchemaError(
                    "advanced injected-module API cannot be a model-facing surface"
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
        if self.attention_interface_contract is not None:
            contract = self.attention_interface_contract
            lines.extend(
                (
                    "  attention-dispatch: %s" % contract.model_facing_dispatch,
                    "  run-accepts-plan-handle: no",
                    "  caller-selects-provider: no",
                    "  advanced-injected-module-symbols: %d"
                    % len(contract.advanced_injected_module_symbols),
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
