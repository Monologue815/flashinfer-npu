"""Explicit adapter sources for approved Attention provider contributions.

The registry never scans entry points, imports provider packages or discovers
factories by name.  Deployment code injects already constructed factories; the
registry freezes their stable identities and materializes them only when an
exact contribution approval manifest is supplied.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .provider_contribution import (
    AttentionOperatorProviderIntegrationContribution,
)
from .provider_contribution_manifest import (
    AttentionOperatorProviderContributionManifest,
)


ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION = 2

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stable_type_name(value: object) -> str:
    value_type = type(value)
    result = "%s.%s" % (value_type.__module__, value_type.__qualname__)
    if not result or "<locals>" in result or "<lambda>" in result:
        raise SchemaError("Attention contribution factory type is not stable")
    return result


@runtime_checkable
class AttentionOperatorProviderContributionFactory(Protocol):
    """Injected Host adapter factory; it must not probe a provider package."""

    @property
    def factory_id(self) -> str:
        ...

    def build_contribution(
        self,
    ) -> AttentionOperatorProviderIntegrationContribution:
        ...


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceOriginBinding:
    """Reviewed adapter-package origin for one explicitly loaded factory."""

    adapter_package_name: str
    observed_package_version: str
    factory_path: str
    factory_loader_id: str
    factory_loader_type: str
    declaration_fingerprint: str
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION:
            raise SchemaError("unsupported Attention contribution source origin")
        for name in (
            "adapter_package_name",
            "observed_package_version",
            "factory_path",
            "factory_loader_id",
            "factory_loader_type",
        ):
            value = str(getattr(self, name))
            if not value or any(item.isspace() for item in value):
                raise SchemaError("Attention contribution source origin is invalid")
            object.__setattr__(self, name, value)
        if not _HASH.fullmatch(str(self.declaration_fingerprint)):
            raise SchemaError(
                "Attention contribution source declaration fingerprint is invalid"
            )
        object.__setattr__(
            self, "declaration_fingerprint", str(self.declaration_fingerprint)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_package_name": self.adapter_package_name,
            "observed_package_version": self.observed_package_version,
            "factory_path": self.factory_path,
            "factory_loader_id": self.factory_loader_id,
            "factory_loader_type": self.factory_loader_type,
            "declaration_fingerprint": self.declaration_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceBinding:
    """Non-executable identity for one explicitly injected adapter source."""

    source_id: str
    source_version: str
    provider_id: str
    contribution_id: str
    factory_id: str
    factory_type: str
    origin_binding: Optional[
        AttentionOperatorProviderContributionSourceOriginBinding
    ] = None
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION:
            raise SchemaError("unsupported Attention provider contribution source")
        for name in ("source_id", "source_version", "contribution_id"):
            if not _IDENTIFIER.fullmatch(str(getattr(self, name))):
                raise SchemaError("Attention contribution source identity is invalid")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("Attention contribution source provider is invalid")
        for name in ("factory_id", "factory_type"):
            value = str(getattr(self, name))
            if not value or any(item.isspace() for item in value):
                raise SchemaError("Attention contribution factory identity is invalid")
        for name in (
            "source_id",
            "source_version",
            "provider_id",
            "contribution_id",
            "factory_id",
            "factory_type",
        ):
            object.__setattr__(self, name, str(getattr(self, name)))
        if self.origin_binding is not None and not isinstance(
            self.origin_binding,
            AttentionOperatorProviderContributionSourceOriginBinding,
        ):
            raise TypeError("origin_binding has the wrong type")

    @property
    def contribution_identity(self) -> Tuple[str, str]:
        return (self.provider_id, self.contribution_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "provider_id": self.provider_id,
            "contribution_id": self.contribution_id,
            "factory_id": self.factory_id,
            "factory_type": self.factory_type,
            "origin_binding": (
                None
                if self.origin_binding is None
                else self.origin_binding.to_dict()
            ),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSource:
    """One reviewed factory bound to one expected provider contribution."""

    source_id: str
    source_version: str
    provider_id: str
    contribution_id: str
    factory: AttentionOperatorProviderContributionFactory = field(
        repr=False, compare=False
    )
    origin_binding: Optional[
        AttentionOperatorProviderContributionSourceOriginBinding
    ] = None
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION
    _factory_id: str = field(init=False, repr=False, compare=False)
    _factory_type: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION:
            raise SchemaError("unsupported Attention provider contribution source")
        if not isinstance(
            self.factory,
            AttentionOperatorProviderContributionFactory,
        ):
            raise TypeError(
                "factory must implement AttentionOperatorProviderContributionFactory"
            )
        factory_id = str(self.factory.factory_id)
        factory_type = _stable_type_name(self.factory)
        binding = AttentionOperatorProviderContributionSourceBinding(
            source_id=self.source_id,
            source_version=self.source_version,
            provider_id=self.provider_id,
            contribution_id=self.contribution_id,
            factory_id=factory_id,
            factory_type=factory_type,
            origin_binding=self.origin_binding,
        )
        for name in (
            "source_id",
            "source_version",
            "provider_id",
            "contribution_id",
        ):
            object.__setattr__(self, name, getattr(binding, name))
        object.__setattr__(self, "_factory_id", binding.factory_id)
        object.__setattr__(self, "_factory_type", binding.factory_type)

    @property
    def factory_id(self) -> str:
        return self._factory_id

    @property
    def factory_type(self) -> str:
        return self._factory_type

    @property
    def binding(self) -> AttentionOperatorProviderContributionSourceBinding:
        return AttentionOperatorProviderContributionSourceBinding(
            source_id=self.source_id,
            source_version=self.source_version,
            provider_id=self.provider_id,
            contribution_id=self.contribution_id,
            factory_id=self.factory_id,
            factory_type=self.factory_type,
            origin_binding=self.origin_binding,
        )

    def validate_factory_identity(self) -> None:
        if (
            str(self.factory.factory_id) != self.factory_id
            or _stable_type_name(self.factory) != self.factory_type
        ):
            raise SchemaError("Attention contribution factory identity changed")

    def materialize(
        self,
    ) -> AttentionOperatorProviderIntegrationContribution:
        """Build and validate exactly one declared contribution."""

        self.validate_factory_identity()
        contribution = self.factory.build_contribution()
        self.validate_factory_identity()
        if not isinstance(
            contribution,
            AttentionOperatorProviderIntegrationContribution,
        ):
            raise TypeError(
                "contribution factory must return "
                "AttentionOperatorProviderIntegrationContribution"
            )
        if (
            contribution.provider_id != self.provider_id
            or contribution.contribution_id != self.contribution_id
        ):
            raise SchemaError(
                "Attention contribution factory result identity differs"
            )
        contribution.validate()
        return contribution


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceRegistryBinding:
    """Non-executable provenance for one complete adapter source registry."""

    registry_id: str
    registry_fingerprint: str
    source_bindings: Tuple[
        AttentionOperatorProviderContributionSourceBinding, ...
    ]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION:
            raise SchemaError("unsupported Attention contribution source registry")
        if not _IDENTIFIER.fullmatch(str(self.registry_id)):
            raise SchemaError("invalid Attention contribution source registry_id")
        if not _HASH.fullmatch(str(self.registry_fingerprint)):
            raise SchemaError(
                "Attention contribution source registry fingerprint is invalid"
            )
        bindings = tuple(self.source_bindings)
        if not bindings or any(
            not isinstance(item, AttentionOperatorProviderContributionSourceBinding)
            for item in bindings
        ):
            raise TypeError(
                "source_bindings must contain contribution source bindings"
            )
        source_ids = tuple(item.source_id for item in bindings)
        contribution_ids = tuple(item.contribution_id for item in bindings)
        contribution_identities = tuple(
            item.contribution_identity for item in bindings
        )
        if len(set(source_ids)) != len(source_ids):
            raise SchemaError("Attention contribution source ids duplicate")
        if (
            len(set(contribution_ids)) != len(contribution_ids)
            or len(set(contribution_identities)) != len(contribution_identities)
        ):
            raise SchemaError("Attention contribution source identities duplicate")
        object.__setattr__(self, "registry_id", str(self.registry_id))
        object.__setattr__(
            self, "registry_fingerprint", str(self.registry_fingerprint)
        )
        object.__setattr__(
            self,
            "source_bindings",
            tuple(sorted(bindings, key=lambda item: item.source_id)),
        )

    @property
    def contribution_identities(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(item.contribution_identity for item in self.source_bindings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "registry_fingerprint": self.registry_fingerprint,
            "source_bindings": [item.to_dict() for item in self.source_bindings],
        }


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceRegistry:
    """Explicit immutable set of provider adapter contribution factories."""

    registry_id: str
    sources: Tuple[AttentionOperatorProviderContributionSource, ...] = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION:
            raise SchemaError("unsupported Attention contribution source registry")
        if not _IDENTIFIER.fullmatch(str(self.registry_id)):
            raise SchemaError("invalid Attention contribution source registry_id")
        sources = tuple(self.sources)
        if not sources or any(
            not isinstance(item, AttentionOperatorProviderContributionSource)
            for item in sources
        ):
            raise TypeError("sources must contain provider contribution sources")
        source_ids = tuple(item.source_id for item in sources)
        contribution_ids = tuple(item.contribution_id for item in sources)
        contribution_identities = tuple(
            (item.provider_id, item.contribution_id) for item in sources
        )
        if len(set(source_ids)) != len(source_ids):
            raise SchemaError("Attention contribution source ids duplicate")
        if (
            len(set(contribution_ids)) != len(contribution_ids)
            or len(set(contribution_identities)) != len(contribution_identities)
        ):
            raise SchemaError("Attention contribution source identities duplicate")
        for source in sources:
            source.validate_factory_identity()
        object.__setattr__(self, "registry_id", str(self.registry_id))
        object.__setattr__(
            self,
            "sources",
            tuple(sorted(sources, key=lambda item: item.source_id)),
        )

    def validate_factory_identities(self) -> None:
        for source in self.sources:
            source.validate_factory_identity()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_operator_provider_contribution_source_registry",
            "registry_id": self.registry_id,
            "sources": [item.binding.to_dict() for item in self.sources],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def binding(self) -> AttentionOperatorProviderContributionSourceRegistryBinding:
        return AttentionOperatorProviderContributionSourceRegistryBinding(
            registry_id=self.registry_id,
            registry_fingerprint=self.fingerprint,
            source_bindings=tuple(item.binding for item in self.sources),
        )

    def materialize(
        self,
        approval_manifest: AttentionOperatorProviderContributionManifest,
    ) -> Tuple[AttentionOperatorProviderIntegrationContribution, ...]:
        """Materialize only an exact manifest-approved source set."""

        if not isinstance(
            approval_manifest,
            AttentionOperatorProviderContributionManifest,
        ):
            raise TypeError(
                "approval_manifest must be "
                "AttentionOperatorProviderContributionManifest"
            )
        self.validate_factory_identities()
        expected = tuple(
            (item.provider_id, item.contribution_id)
            for item in approval_manifest.contribution_bindings
        )
        observed = tuple(
            (item.provider_id, item.contribution_id) for item in self.sources
        )
        if set(expected) != set(observed):
            raise SchemaError(
                "Attention contribution source set differs from approval manifest"
            )
        contributions = tuple(source.materialize() for source in self.sources)
        return approval_manifest.validate_contributions(contributions)


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_VERSION",
    "AttentionOperatorProviderContributionFactory",
    "AttentionOperatorProviderContributionSource",
    "AttentionOperatorProviderContributionSourceBinding",
    "AttentionOperatorProviderContributionSourceOriginBinding",
    "AttentionOperatorProviderContributionSourceRegistry",
    "AttentionOperatorProviderContributionSourceRegistryBinding",
]
