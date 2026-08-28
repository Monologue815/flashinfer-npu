"""Explicit adapter-package loading for Attention contribution factories."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .provider_contribution_manifest import (
    AttentionOperatorProviderContributionManifest,
)
from .provider_contribution_source import (
    AttentionOperatorProviderContributionFactory,
    AttentionOperatorProviderContributionSource,
    AttentionOperatorProviderContributionSourceBinding,
    AttentionOperatorProviderContributionSourceOriginBinding,
    AttentionOperatorProviderContributionSourceRegistry,
)


ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION = 1

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_FACTORY_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


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
        raise SchemaError("Attention contribution loader type is not stable")
    return result


@runtime_checkable
class AttentionOperatorProviderContributionFactoryLoader(Protocol):
    """Injected loader for reviewed adapter packages, not provider operators."""

    @property
    def loader_id(self) -> str:
        ...

    def package_version(self, package_name: str) -> str:
        ...

    def resolve_factory(
        self, factory_path: str
    ) -> AttentionOperatorProviderContributionFactory:
        ...


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceDeclaration:
    """Exact adapter package and factory expected for one contribution source."""

    source_id: str
    source_version: str
    provider_id: str
    contribution_id: str
    adapter_package_name: str
    supported_package_versions: Tuple[str, ...]
    factory_path: str
    factory_id: str
    factory_type: str
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION:
            raise SchemaError("unsupported Attention contribution source declaration")
        source_binding = AttentionOperatorProviderContributionSourceBinding(
            source_id=self.source_id,
            source_version=self.source_version,
            provider_id=self.provider_id,
            contribution_id=self.contribution_id,
            factory_id=self.factory_id,
            factory_type=self.factory_type,
        )
        package_name = str(self.adapter_package_name)
        if not package_name or any(item.isspace() for item in package_name):
            raise SchemaError("Attention adapter package name is invalid")
        versions = tuple(str(item) for item in self.supported_package_versions)
        if (
            not versions
            or any(not item or any(char.isspace() for char in item) for item in versions)
            or len(set(versions)) != len(versions)
        ):
            raise SchemaError(
                "Attention adapter supported package versions are invalid"
            )
        factory_path = str(self.factory_path)
        if not _FACTORY_PATH.fullmatch(factory_path):
            raise SchemaError("Attention adapter factory path is invalid")
        for name in (
            "source_id",
            "source_version",
            "provider_id",
            "contribution_id",
            "factory_id",
            "factory_type",
        ):
            object.__setattr__(self, name, getattr(source_binding, name))
        object.__setattr__(self, "adapter_package_name", package_name)
        object.__setattr__(self, "supported_package_versions", tuple(sorted(versions)))
        object.__setattr__(self, "factory_path", factory_path)

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
            "adapter_package_name": self.adapter_package_name,
            "supported_package_versions": list(self.supported_package_versions),
            "factory_path": self.factory_path,
            "factory_id": self.factory_id,
            "factory_type": self.factory_type,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceDeclarationRegistry:
    """Manifest-gated explicit loader for a complete adapter source set."""

    registry_id: str
    declarations: Tuple[
        AttentionOperatorProviderContributionSourceDeclaration, ...
    ]
    factory_loader: AttentionOperatorProviderContributionFactoryLoader = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION
    _factory_loader_id: str = field(init=False, repr=False, compare=False)
    _factory_loader_type: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION:
            raise SchemaError("unsupported Attention source declaration registry")
        if not _IDENTIFIER.fullmatch(str(self.registry_id)):
            raise SchemaError("invalid Attention source declaration registry_id")
        if not isinstance(
            self.factory_loader,
            AttentionOperatorProviderContributionFactoryLoader,
        ):
            raise TypeError(
                "factory_loader must implement "
                "AttentionOperatorProviderContributionFactoryLoader"
            )
        declarations = tuple(self.declarations)
        if not declarations or any(
            not isinstance(
                item,
                AttentionOperatorProviderContributionSourceDeclaration,
            )
            for item in declarations
        ):
            raise TypeError("declarations must contain source declarations")
        source_ids = tuple(item.source_id for item in declarations)
        contribution_ids = tuple(item.contribution_id for item in declarations)
        contribution_identities = tuple(
            item.contribution_identity for item in declarations
        )
        factory_locations = tuple(
            (item.adapter_package_name, item.factory_path) for item in declarations
        )
        if len(set(source_ids)) != len(source_ids):
            raise SchemaError("Attention source declaration ids duplicate")
        if (
            len(set(contribution_ids)) != len(contribution_ids)
            or len(set(contribution_identities)) != len(contribution_identities)
        ):
            raise SchemaError("Attention source declaration identities duplicate")
        if len(set(factory_locations)) != len(factory_locations):
            raise SchemaError("Attention adapter factory locations duplicate")
        loader_id = str(self.factory_loader.loader_id)
        if not loader_id or any(item.isspace() for item in loader_id):
            raise SchemaError("Attention contribution factory loader_id is invalid")
        loader_type = _stable_type_name(self.factory_loader)
        object.__setattr__(self, "registry_id", str(self.registry_id))
        object.__setattr__(
            self,
            "declarations",
            tuple(sorted(declarations, key=lambda item: item.source_id)),
        )
        object.__setattr__(self, "_factory_loader_id", loader_id)
        object.__setattr__(self, "_factory_loader_type", loader_type)

    @property
    def factory_loader_id(self) -> str:
        return self._factory_loader_id

    @property
    def factory_loader_type(self) -> str:
        return self._factory_loader_type

    def validate_factory_loader_identity(self) -> None:
        if (
            str(self.factory_loader.loader_id) != self.factory_loader_id
            or _stable_type_name(self.factory_loader) != self.factory_loader_type
        ):
            raise SchemaError("Attention contribution factory loader identity changed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_operator_provider_contribution_source_declarations",
            "registry_id": self.registry_id,
            "factory_loader_id": self.factory_loader_id,
            "factory_loader_type": self.factory_loader_type,
            "declarations": [item.to_dict() for item in self.declarations],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def load_sources(
        self,
        approval_manifest: AttentionOperatorProviderContributionManifest,
    ) -> AttentionOperatorProviderContributionSourceRegistry:
        """Resolve factories only after declaration/manifest set equality."""

        if not isinstance(
            approval_manifest,
            AttentionOperatorProviderContributionManifest,
        ):
            raise TypeError(
                "approval_manifest must be "
                "AttentionOperatorProviderContributionManifest"
            )
        self.validate_factory_loader_identity()
        expected = {
            (item.provider_id, item.contribution_id)
            for item in approval_manifest.contribution_bindings
        }
        observed = {item.contribution_identity for item in self.declarations}
        if observed != expected:
            raise SchemaError(
                "Attention source declarations differ from approval manifest"
            )
        observed_versions = {}
        sources = []
        for declaration in self.declarations:
            package_name = declaration.adapter_package_name
            if package_name not in observed_versions:
                version = str(self.factory_loader.package_version(package_name))
                self.validate_factory_loader_identity()
                if (
                    not version
                    or any(item.isspace() for item in version)
                    or version not in declaration.supported_package_versions
                ):
                    raise SchemaError(
                        "Attention adapter package version is unsupported"
                    )
                observed_versions[package_name] = version
            version = observed_versions[package_name]
            if version not in declaration.supported_package_versions:
                raise SchemaError(
                    "Attention adapter package version differs across declarations"
                )
            factory = self.factory_loader.resolve_factory(
                declaration.factory_path
            )
            self.validate_factory_loader_identity()
            if not isinstance(
                factory,
                AttentionOperatorProviderContributionFactory,
            ):
                raise TypeError(
                    "resolved factory must implement "
                    "AttentionOperatorProviderContributionFactory"
                )
            if (
                str(factory.factory_id) != declaration.factory_id
                or _stable_type_name(factory) != declaration.factory_type
            ):
                raise SchemaError("resolved Attention factory identity differs")
            origin = AttentionOperatorProviderContributionSourceOriginBinding(
                adapter_package_name=package_name,
                observed_package_version=version,
                factory_path=declaration.factory_path,
                factory_loader_id=self.factory_loader_id,
                factory_loader_type=self.factory_loader_type,
                declaration_fingerprint=declaration.fingerprint,
            )
            sources.append(
                AttentionOperatorProviderContributionSource(
                    source_id=declaration.source_id,
                    source_version=declaration.source_version,
                    provider_id=declaration.provider_id,
                    contribution_id=declaration.contribution_id,
                    factory=factory,
                    origin_binding=origin,
                )
            )
        return AttentionOperatorProviderContributionSourceRegistry(
            registry_id=self.registry_id,
            sources=tuple(sources),
        )


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION",
    "AttentionOperatorProviderContributionFactoryLoader",
    "AttentionOperatorProviderContributionSourceDeclaration",
    "AttentionOperatorProviderContributionSourceDeclarationRegistry",
]
