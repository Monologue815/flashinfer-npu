"""Data-only audit snapshots for external Attention runtime specifications.

Creating a declaration never inspects package metadata, imports a provider,
touches a device, materializes a tensor, or invokes an operator.  It makes the
otherwise opaque Python component graph in ``AttentionOperatorPackageRuntimeSpec``
reviewable and drift-detectable before an integration is installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from flashinfer_npu.runtime import Backend, SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    AttentionJsonEnvelopeUsage,
    decode_attention_json,
)
from .operation_catalog import (
    AttentionOperatorOperationCatalog,
    load_packaged_attention_operator_catalog,
)
from .operator_bootstrap import (
    AttentionOperatorPackageRuntimeSpec,
    bind_attention_operator_plan_scoring_manifest,
)
from .operator_scoring import AttentionOperatorPlanScoringManifest


ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION = 3

_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: Optional[str], *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _type_name(value: object) -> str:
    value_type = type(value)
    result = "%s.%s" % (value_type.__module__, value_type.__qualname__)
    if not result or "<locals>" in result or "<lambda>" in result:
        raise SchemaError("runtime declaration component type is not stable")
    return result


def _coverage_policy_fingerprint(policy) -> Optional[str]:
    if policy is None:
        return None
    return _canonical_hash(
        {
            "name": policy.name,
            "cells": [
                {
                    "name": cell.name,
                    "selectors": [list(item) for item in cell.selectors],
                    "min_cases": cell.min_cases,
                }
                for cell in policy.cells
            ],
        }
    )


@dataclass(frozen=True)
class AttentionOperatorRuntimeComponentDeclaration:
    """Stable source identity for one injected framework component."""

    role: str
    type_name: str
    identities: Tuple[Tuple[str, str], ...] = ()
    schema_version: int = ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION:
            raise SchemaError("unsupported Attention runtime component version")
        if not _ROLE.fullmatch(str(self.role)):
            raise SchemaError("invalid Attention runtime component role")
        if not str(self.type_name) or "<locals>" in str(self.type_name):
            raise SchemaError("runtime component type_name must be stable")
        try:
            identities = tuple(
                (str(name), str(value)) for name, value in self.identities
            )
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "runtime component identities must contain name/value pairs"
            ) from error
        if any(not name or not value for name, value in identities):
            raise SchemaError("runtime component identities must be non-empty")
        names = tuple(name for name, _ in identities)
        if len(set(names)) != len(names):
            raise SchemaError("runtime component identity names must be unique")
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "type_name", str(self.type_name))
        object.__setattr__(self, "identities", tuple(sorted(identities)))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "type_name": self.type_name,
            "identities": [list(item) for item in self.identities],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("runtime component declaration fields are invalid")
        identities = data.get("identities")
        if not isinstance(identities, (list, tuple)):
            raise SchemaError("runtime component identities must be an array")
        data["identities"] = tuple(identities)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError(
                "runtime component declaration fields are invalid"
            ) from error


@dataclass(frozen=True)
class AttentionOperatorPackageRuntimeDeclaration:
    """Serializable identity of one runtime spec without its executable objects."""

    provider_id: str
    operation_id: str
    operation_fingerprint: str
    catalog_name: str
    catalog_fingerprint: str
    package_name: str
    callable_path: str
    api_version: str
    source_url: str
    priority: int
    adapter_version: str
    supported_package_versions: Tuple[str, ...]
    backend: str
    observed_environment_fingerprint: str
    profile_bindings: Tuple[Tuple[str, str, str, str], ...]
    descriptor_bindings: Tuple[Tuple[str, str, str], ...]
    components: Tuple[AttentionOperatorRuntimeComponentDeclaration, ...]
    tensor_access_policy_fingerprint: str
    quantization_binding_fingerprints: Tuple[str, ...]
    nvfp4_packed_kv_binding_fingerprints: Tuple[str, ...]
    quant_physical_layout_catalog_fingerprint: str
    physical_layout_evidence_bundle_fingerprint: Optional[str]
    numerics_policy_fingerprint: str
    corpus_fingerprint: Optional[str]
    coverage_policy_fingerprint: Optional[str]
    tuned_kernel_ids: Tuple[str, ...]
    replay_evidence: bool
    validate_provider_results: bool
    schema_version: int = ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION
    kind: str = field(
        default="attention_operator_package_runtime_declaration", init=False
    )

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION:
            raise SchemaError("unsupported Attention runtime declaration version")
        if self.kind != "attention_operator_package_runtime_declaration":
            raise SchemaError("Attention runtime declaration kind is invalid")
        for name in (
            "provider_id",
            "operation_id",
            "catalog_name",
            "package_name",
            "callable_path",
            "api_version",
            "source_url",
            "adapter_version",
            "backend",
        ):
            if not str(getattr(self, name)):
                raise SchemaError("runtime declaration %s must be non-empty" % name)
            object.__setattr__(self, name, str(getattr(self, name)))
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise SchemaError("runtime declaration priority must be an integer")
        for name in (
            "operation_fingerprint",
            "catalog_fingerprint",
            "observed_environment_fingerprint",
            "tensor_access_policy_fingerprint",
            "quant_physical_layout_catalog_fingerprint",
            "numerics_policy_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        for name in (
            "physical_layout_evidence_bundle_fingerprint",
            "corpus_fingerprint",
            "coverage_policy_fingerprint",
        ):
            _require_hash(name, getattr(self, name), optional=True)
        versions = tuple(str(item) for item in self.supported_package_versions)
        if not versions or any(not item for item in versions):
            raise SchemaError("runtime declaration package versions must be non-empty")
        if len(set(versions)) != len(versions):
            raise SchemaError("runtime declaration package versions must be unique")
        profiles = tuple(tuple(str(item) for item in row) for row in self.profile_bindings)
        if not profiles or any(len(row) != 4 for row in profiles):
            raise SchemaError("runtime declaration profile bindings are invalid")
        descriptors = tuple(
            tuple(str(item) for item in row) for row in self.descriptor_bindings
        )
        if not descriptors or any(len(row) != 3 for row in descriptors):
            raise SchemaError("runtime declaration descriptor bindings are invalid")
        for row in profiles:
            _require_hash("profile binding fingerprint", row[3])
        for row in descriptors:
            _require_hash("descriptor binding fingerprint", row[2])
        components = tuple(self.components)
        if not components or any(
            not isinstance(item, AttentionOperatorRuntimeComponentDeclaration)
            for item in components
        ):
            raise TypeError("runtime declaration components have the wrong type")
        roles = tuple(item.role for item in components)
        if len(set(roles)) != len(roles):
            raise SchemaError("runtime declaration component roles must be unique")
        quant = tuple(str(item) for item in self.quantization_binding_fingerprints)
        if any(not _HASH.fullmatch(item) for item in quant):
            raise SchemaError("quantization binding fingerprint must be SHA-256")
        nvfp4 = tuple(
            str(item) for item in self.nvfp4_packed_kv_binding_fingerprints
        )
        if any(not _HASH.fullmatch(item) for item in nvfp4):
            raise SchemaError("NVFP4 packed binding fingerprint must be SHA-256")
        tuned = tuple(str(item) for item in self.tuned_kernel_ids)
        if any(not item for item in tuned) or len(set(tuned)) != len(tuned):
            raise SchemaError("runtime declaration tuned kernel ids are invalid")
        if not isinstance(self.replay_evidence, bool) or not isinstance(
            self.validate_provider_results, bool
        ):
            raise SchemaError("runtime declaration flags must be boolean")
        object.__setattr__(self, "supported_package_versions", tuple(sorted(versions)))
        object.__setattr__(self, "profile_bindings", tuple(sorted(profiles)))
        object.__setattr__(self, "descriptor_bindings", tuple(sorted(descriptors)))
        object.__setattr__(self, "components", tuple(sorted(components, key=lambda x: x.role)))
        object.__setattr__(self, "quantization_binding_fingerprints", tuple(sorted(quant)))
        object.__setattr__(
            self,
            "nvfp4_packed_kv_binding_fingerprints",
            tuple(sorted(nvfp4)),
        )
        object.__setattr__(self, "tuned_kernel_ids", tuple(sorted(tuned)))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "operation_fingerprint": self.operation_fingerprint,
            "catalog_name": self.catalog_name,
            "catalog_fingerprint": self.catalog_fingerprint,
            "package_name": self.package_name,
            "callable_path": self.callable_path,
            "api_version": self.api_version,
            "source_url": self.source_url,
            "priority": self.priority,
            "adapter_version": self.adapter_version,
            "supported_package_versions": list(self.supported_package_versions),
            "backend": self.backend,
            "observed_environment_fingerprint": self.observed_environment_fingerprint,
            "profile_bindings": [list(item) for item in self.profile_bindings],
            "descriptor_bindings": [list(item) for item in self.descriptor_bindings],
            "components": [item.to_dict() for item in self.components],
            "tensor_access_policy_fingerprint": self.tensor_access_policy_fingerprint,
            "quantization_binding_fingerprints": list(
                self.quantization_binding_fingerprints
            ),
            "nvfp4_packed_kv_binding_fingerprints": list(
                self.nvfp4_packed_kv_binding_fingerprints
            ),
            "quant_physical_layout_catalog_fingerprint": (
                self.quant_physical_layout_catalog_fingerprint
            ),
            "physical_layout_evidence_bundle_fingerprint": (
                self.physical_layout_evidence_bundle_fingerprint
            ),
            "numerics_policy_fingerprint": self.numerics_policy_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "coverage_policy_fingerprint": self.coverage_policy_fingerprint,
            "tuned_kernel_ids": list(self.tuned_kernel_ids),
            "replay_evidence": self.replay_evidence,
            "validate_provider_results": self.validate_provider_results,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention runtime declaration fields are invalid")
        if data.pop("kind") != "attention_operator_package_runtime_declaration":
            raise SchemaError("Attention runtime declaration kind is invalid")
        for name in (
            "supported_package_versions",
            "profile_bindings",
            "descriptor_bindings",
            "quantization_binding_fingerprints",
            "nvfp4_packed_kv_binding_fingerprints",
            "tuned_kernel_ids",
        ):
            if not isinstance(data.get(name), (list, tuple)):
                raise SchemaError("Attention runtime declaration array is invalid")
            data[name] = tuple(data[name])
        components = data.get("components")
        if not isinstance(components, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in components
        ):
            raise SchemaError("Attention runtime declaration components are invalid")
        data["components"] = tuple(
            AttentionOperatorRuntimeComponentDeclaration.from_dict(item)
            for item in components
        )
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("Attention runtime declaration fields are invalid") from error

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
        spec: AttentionOperatorPackageRuntimeSpec,
        *,
        operation_catalog: Optional[AttentionOperatorOperationCatalog] = None,
    ) -> None:
        current = describe_attention_operator_package_runtime(
            spec, operation_catalog=operation_catalog
        )
        if current.fingerprint != self.fingerprint:
            raise SchemaError("Attention runtime declaration is stale")


@dataclass(frozen=True)
class AttentionDeclaredOperatorPackageRuntimeSpec:
    """One executable spec bound to its previously reviewed declaration."""

    declaration: AttentionOperatorPackageRuntimeDeclaration
    runtime_spec: AttentionOperatorPackageRuntimeSpec = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION:
            raise SchemaError("unsupported declared Attention runtime spec version")
        if not isinstance(
            self.declaration, AttentionOperatorPackageRuntimeDeclaration
        ):
            raise TypeError("declaration has the wrong type")
        if not isinstance(self.runtime_spec, AttentionOperatorPackageRuntimeSpec):
            raise TypeError("runtime_spec has the wrong type")
        spec = self.runtime_spec
        declaration = self.declaration
        if (
            declaration.provider_id != spec.provider_id
            or declaration.operation_id != spec.operation_id
            or declaration.adapter_version != spec.adapter_version
            or declaration.supported_package_versions
            != spec.supported_package_versions
        ):
            raise SchemaError("declared Attention runtime identity differs")

    def validate(
        self, operation_catalog: AttentionOperatorOperationCatalog
    ) -> None:
        if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
            raise TypeError("operation_catalog must be AttentionOperatorOperationCatalog")
        self.declaration.validate_runtime_spec(
            self.runtime_spec, operation_catalog=operation_catalog
        )

    @property
    def fingerprint(self) -> str:
        return self.declaration.fingerprint

    @property
    def binding(self) -> "AttentionOperatorRuntimeDeclarationBinding":
        return AttentionOperatorRuntimeDeclarationBinding(
            provider_id=self.declaration.provider_id,
            operation_id=self.declaration.operation_id,
            declaration_fingerprint=self.declaration.fingerprint,
        )


@dataclass(frozen=True)
class AttentionOperatorRuntimeDeclarationBinding:
    """Compact declaration identity stored in an installed registry snapshot."""

    provider_id: str
    operation_id: str
    declaration_fingerprint: str
    schema_version: int = ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION:
            raise SchemaError("unsupported runtime declaration binding version")
        if not str(self.provider_id) or not str(self.operation_id):
            raise SchemaError("runtime declaration binding identity is incomplete")
        _require_hash("runtime declaration fingerprint", self.declaration_fingerprint)
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "declaration_fingerprint": self.declaration_fingerprint,
        }


def _component(role: str, value: object, **identities: object):
    return AttentionOperatorRuntimeComponentDeclaration(
        role=role,
        type_name=_type_name(value),
        identities=tuple(
            (name, str(identity))
            for name, identity in identities.items()
            if identity is not None
        ),
    )


def describe_attention_operator_package_runtime(
    spec: AttentionOperatorPackageRuntimeSpec,
    *,
    operation_catalog: Optional[AttentionOperatorOperationCatalog] = None,
) -> AttentionOperatorPackageRuntimeDeclaration:
    """Describe one spec without loading or executing any external resource."""

    if not isinstance(spec, AttentionOperatorPackageRuntimeSpec):
        raise TypeError("spec must be AttentionOperatorPackageRuntimeSpec")
    if operation_catalog is None:
        operation_catalog = load_packaged_attention_operator_catalog()
    if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
        raise TypeError("operation_catalog must be AttentionOperatorOperationCatalog")
    operation = operation_catalog.get(spec.operation_id)
    if operation.provider_id != spec.provider_id:
        raise SchemaError("runtime declaration operation and provider differ")
    backend = spec.backend.value if isinstance(spec.backend, Backend) else str(spec.backend)
    components = [
        _component(
            "plan_gate",
            spec.plan_gate,
            provider_id=spec.plan_gate.provider_id,
            operation_id=spec.plan_gate.operation_id,
        ),
        _component(
            "logical_factory",
            spec.logical_factory,
            provider_id=spec.logical_factory.provider_id,
            operation_id=spec.logical_factory.operation_id,
        ),
        _component(
            "logical_run_adapter",
            spec.logical_run_adapter,
            provider_id=spec.logical_run_adapter.provider_id,
            operation_id=operation.operation_id,
        ),
        _component(
            "tensor_materializer",
            spec.tensor_materializer,
            provider_id=spec.tensor_materializer.provider_id,
            materializer_id=spec.tensor_materializer.materializer_id,
        ),
        _component("tensor_metadata_inspector", spec.tensor_metadata_inspector),
    ]
    if spec.plan_scorer is not None:
        components.append(
            _component(
                "plan_scorer",
                spec.plan_scorer,
                provider_id=spec.plan_scorer.provider_id,
                operation_id=spec.plan_scorer.operation_id,
                policy_fingerprint=getattr(spec.plan_scorer, "fingerprint", None),
            )
        )
    for role, value in (
        ("jit_plan_resolver", spec.jit_plan_resolver),
        ("jit_artifact_resolver", spec.jit_artifact_resolver),
        ("jit_module_resolver", spec.jit_module_resolver),
        ("jit_planner_binder", spec.jit_planner_binder),
        ("jit_executor_binder", spec.jit_executor_binder),
    ):
        if value is not None:
            identities = {}
            for name in (
                "provider_id",
                "operation_id",
                "binder_id",
                "binder_version",
            ):
                if hasattr(value, name):
                    identities[name] = getattr(value, name)
            components.append(_component(role, value, **identities))
    evidence = spec.physical_layout_evidence_bundle
    return AttentionOperatorPackageRuntimeDeclaration(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        operation_fingerprint=operation.fingerprint,
        catalog_name=operation_catalog.name,
        catalog_fingerprint=operation_catalog.fingerprint,
        package_name=operation.package_name,
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        source_url=operation.source_url,
        priority=spec.priority,
        adapter_version=spec.adapter_version,
        supported_package_versions=spec.supported_package_versions,
        backend=backend,
        observed_environment_fingerprint=spec.observed_environment.fingerprint,
        profile_bindings=tuple(
            (
                profile.profile_id,
                profile.backend.value,
                profile.status.value,
                profile.fingerprint,
            )
            for profile in spec.profiles
        ),
        descriptor_bindings=tuple(
            (descriptor.kernel_id, descriptor.backend.value, descriptor.fingerprint)
            for descriptor in spec.descriptors
        ),
        components=tuple(components),
        tensor_access_policy_fingerprint=spec.tensor_access_policy.fingerprint,
        quantization_binding_fingerprints=tuple(
            binding.fingerprint for binding in spec.quantization_bindings
        ),
        nvfp4_packed_kv_binding_fingerprints=tuple(
            binding.fingerprint for binding in spec.nvfp4_packed_kv_bindings
        ),
        quant_physical_layout_catalog_fingerprint=(
            spec.quant_physical_layout_catalog.fingerprint
        ),
        physical_layout_evidence_bundle_fingerprint=(
            evidence.fingerprint if evidence is not None else None
        ),
        numerics_policy_fingerprint=spec.numerics_policy.fingerprint,
        corpus_fingerprint=(spec.corpus.fingerprint if spec.corpus is not None else None),
        coverage_policy_fingerprint=_coverage_policy_fingerprint(
            spec.coverage_policy
        ),
        tuned_kernel_ids=spec.tuned_kernel_ids,
        replay_evidence=spec.replay_evidence,
        validate_provider_results=spec.validate_provider_results,
    )


def load_attention_operator_package_runtime_declaration(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
) -> Tuple[
    AttentionOperatorPackageRuntimeDeclaration, AttentionJsonEnvelopeUsage
]:
    """Decode a bounded declaration envelope without resolving its components."""

    decoded, usage = decode_attention_json(value, limits=limits)
    if not isinstance(decoded, Mapping):
        raise SchemaError("Attention runtime declaration root must be an object")
    return AttentionOperatorPackageRuntimeDeclaration.from_dict(decoded), usage


def build_declared_attention_operator_runtime_resolvers(
    registrations: Sequence[AttentionDeclaredOperatorPackageRuntimeSpec] = (),
    *,
    operation_catalog: Optional[AttentionOperatorOperationCatalog] = None,
    package_loader=None,
    plan_scoring_manifest: Optional[
        AttentionOperatorPlanScoringManifest
    ] = None,
):
    """Build a registry only after every declaration still matches its spec.

    The complete registration set is validated before the ordinary composition
    root is entered, so one stale item cannot publish a partial resolver tree.
    """

    values = tuple(registrations)
    if any(
        not isinstance(item, AttentionDeclaredOperatorPackageRuntimeSpec)
        for item in values
    ):
        raise TypeError(
            "registrations must contain AttentionDeclaredOperatorPackageRuntimeSpec"
        )
    if operation_catalog is None:
        operation_catalog = load_packaged_attention_operator_catalog()
    if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
        raise TypeError("operation_catalog must be AttentionOperatorOperationCatalog")
    fingerprints = tuple(item.fingerprint for item in values)
    if len(set(fingerprints)) != len(fingerprints):
        raise SchemaError("declared Attention runtime registrations are duplicated")
    if plan_scoring_manifest is not None:
        if not isinstance(
            plan_scoring_manifest, AttentionOperatorPlanScoringManifest
        ):
            raise TypeError(
                "plan_scoring_manifest must be "
                "AttentionOperatorPlanScoringManifest"
            )
        if any(item.runtime_spec.plan_scorer is None for item in values):
            raise SchemaError(
                "declared Attention runtime specs must bind the scoring "
                "manifest before declaration"
            )
        bind_attention_operator_plan_scoring_manifest(
            tuple(item.runtime_spec for item in values),
            plan_scoring_manifest,
        )
    for item in values:
        item.validate(operation_catalog)
    # Imported lazily to keep declaration parsing independent of registry
    # composition and to make the no-package-probe boundary explicit.
    from .operator_bootstrap import build_attention_operator_runtime_resolvers

    return build_attention_operator_runtime_resolvers(
        tuple(item.runtime_spec for item in values),
        operation_catalog=operation_catalog,
        package_loader=package_loader,
    )


__all__ = [
    "ATTENTION_OPERATOR_RUNTIME_DECLARATION_VERSION",
    "AttentionDeclaredOperatorPackageRuntimeSpec",
    "AttentionOperatorPackageRuntimeDeclaration",
    "AttentionOperatorRuntimeDeclarationBinding",
    "AttentionOperatorRuntimeComponentDeclaration",
    "build_declared_attention_operator_runtime_resolvers",
    "describe_attention_operator_package_runtime",
    "load_attention_operator_package_runtime_declaration",
]
