"""Explicit adapter-package loading for Attention contribution factories."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    AttentionJsonEnvelopeUsage,
    decode_attention_json,
)
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
        raise SchemaError("Attention contribution loader type is not stable")
    return result


@runtime_checkable
class AttentionOperatorProviderContributionFactoryLoader(Protocol):
    """Injected loader for reviewed adapter packages, not provider operators."""

    @property
    def loader_id(self) -> str:
        ...

    def package_version(self, package_name: str) -> Optional[str]:
        ...

    def resolve_factory(
        self, factory_path: str
    ) -> AttentionOperatorProviderContributionFactory:
        ...


class ImportlibAttentionOperatorProviderContributionFactoryLoader:
    """Explicit Python adapter loader; never scans or registers entry points."""

    loader_id = "python.importlib.metadata.attention-contribution-factory.v1"

    def package_version(self, package_name: str) -> Optional[str]:
        try:
            return importlib.metadata.version(str(package_name))
        except importlib.metadata.PackageNotFoundError:
            return None

    def resolve_factory(
        self, factory_path: str
    ) -> AttentionOperatorProviderContributionFactory:
        path = str(factory_path)
        if not _FACTORY_PATH.fullmatch(path):
            raise SchemaError("Attention adapter factory path is invalid")
        module_name, separator, attribute_name = path.rpartition(".")
        if not separator or not module_name or not attribute_name:
            raise SchemaError(
                "Attention adapter factory path must include a module and name"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise SchemaError(
                "Attention adapter factory module %r is unavailable" % module_name
            ) from error
        try:
            return getattr(module, attribute_name)
        except AttributeError as error:
            raise SchemaError(
                "Attention adapter factory %r is absent" % path
            ) from error


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

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionOperatorProviderContributionSourceDeclaration":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "Attention contribution source declaration fields are invalid"
            )
        if not isinstance(data.get("supported_package_versions"), (list, tuple)):
            raise SchemaError(
                "Attention adapter supported package versions must be an array"
            )
        data["supported_package_versions"] = tuple(
            data["supported_package_versions"]
        )
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "Attention contribution source declaration fields are invalid"
            ) from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


def _normalize_declarations(values) -> Tuple[
    AttentionOperatorProviderContributionSourceDeclaration, ...
]:
    declarations = tuple(values)
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
    return tuple(sorted(declarations, key=lambda item: item.source_id))


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceDeclarationManifestLimits:
    """Semantic construction limits for serialized source declarations."""

    max_declarations: int = 64
    max_versions_per_declaration: int = 64
    max_total_versions: int = 2048
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION:
            raise SchemaError("unsupported Attention source manifest limits")
        for name in (
            "max_declarations",
            "max_versions_per_declaration",
            "max_total_versions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SchemaError("%s must be a positive integer" % name)


DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_DECLARATION_MANIFEST_LIMITS = (
    AttentionOperatorProviderContributionSourceDeclarationManifestLimits()
)


@dataclass(frozen=True)
class AttentionOperatorProviderContributionSourceDeclarationManifest:
    """Bounded data-only manifest for a complete adapter source set."""

    manifest_id: str
    declarations: Tuple[
        AttentionOperatorProviderContributionSourceDeclaration, ...
    ]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION
    kind: str = field(
        default="attention_operator_provider_contribution_source_declarations",
        init=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION:
            raise SchemaError("unsupported Attention source declaration manifest")
        if self.kind != "attention_operator_provider_contribution_source_declarations":
            raise SchemaError("Attention source declaration manifest kind is invalid")
        if not _IDENTIFIER.fullmatch(str(self.manifest_id)):
            raise SchemaError("invalid Attention source declaration manifest_id")
        object.__setattr__(self, "manifest_id", str(self.manifest_id))
        object.__setattr__(
            self,
            "declarations",
            _normalize_declarations(self.declarations),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "manifest_id": self.manifest_id,
            "declarations": [item.to_dict() for item in self.declarations],
        }

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

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        limits: AttentionOperatorProviderContributionSourceDeclarationManifestLimits = (
            DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_DECLARATION_MANIFEST_LIMITS
        ),
    ) -> "AttentionOperatorProviderContributionSourceDeclarationManifest":
        if not isinstance(
            limits,
            AttentionOperatorProviderContributionSourceDeclarationManifestLimits,
        ):
            raise TypeError(
                "limits must be "
                "AttentionOperatorProviderContributionSourceDeclarationManifestLimits"
            )
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "Attention source declaration manifest fields are invalid"
            )
        if data.pop("kind") != (
            "attention_operator_provider_contribution_source_declarations"
        ):
            raise SchemaError("Attention source declaration manifest kind is invalid")
        encoded = data.get("declarations")
        if not isinstance(encoded, list):
            raise SchemaError(
                "Attention source declaration manifest declarations must be an array"
            )
        if len(encoded) > limits.max_declarations:
            raise SchemaError(
                "Attention source declarations exceed limit %d"
                % limits.max_declarations
            )
        total_versions = 0
        declarations = []
        for item in encoded:
            if not isinstance(item, Mapping):
                raise SchemaError("Attention source declaration must be an object")
            versions = item.get("supported_package_versions")
            if not isinstance(versions, list):
                raise SchemaError(
                    "Attention adapter supported package versions must be an array"
                )
            if len(versions) > limits.max_versions_per_declaration:
                raise SchemaError(
                    "Attention adapter versions exceed per-declaration limit %d"
                    % limits.max_versions_per_declaration
                )
            total_versions += len(versions)
            if total_versions > limits.max_total_versions:
                raise SchemaError(
                    "Attention adapter versions exceed total limit %d"
                    % limits.max_total_versions
                )
            declarations.append(
                AttentionOperatorProviderContributionSourceDeclaration.from_dict(
                    item
                )
            )
        data["declarations"] = tuple(declarations)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "Attention source declaration manifest fields are invalid"
            ) from error


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
    declaration_manifest_id: Optional[str] = None
    declaration_manifest_fingerprint: Optional[str] = None
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
        declarations = _normalize_declarations(self.declarations)
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
        manifest_identity = (
            self.declaration_manifest_id,
            self.declaration_manifest_fingerprint,
        )
        if (manifest_identity[0] is None) != (manifest_identity[1] is None):
            raise SchemaError(
                "Attention source declaration manifest identity is incomplete"
            )
        if manifest_identity[0] is not None:
            if not _IDENTIFIER.fullmatch(str(manifest_identity[0])):
                raise SchemaError(
                    "Attention source declaration manifest id is invalid"
                )
            if not _HASH.fullmatch(str(manifest_identity[1])):
                raise SchemaError(
                    "Attention source declaration manifest fingerprint is invalid"
                )
            object.__setattr__(
                self, "declaration_manifest_id", str(manifest_identity[0])
            )
            object.__setattr__(
                self,
                "declaration_manifest_fingerprint",
                str(manifest_identity[1]),
            )

    @classmethod
    def from_manifest(
        cls,
        manifest: AttentionOperatorProviderContributionSourceDeclarationManifest,
        *,
        factory_loader: AttentionOperatorProviderContributionFactoryLoader,
    ) -> "AttentionOperatorProviderContributionSourceDeclarationRegistry":
        if not isinstance(
            manifest,
            AttentionOperatorProviderContributionSourceDeclarationManifest,
        ):
            raise TypeError(
                "manifest must be "
                "AttentionOperatorProviderContributionSourceDeclarationManifest"
            )
        return cls(
            registry_id=manifest.manifest_id,
            declarations=manifest.declarations,
            factory_loader=factory_loader,
            declaration_manifest_id=manifest.manifest_id,
            declaration_manifest_fingerprint=manifest.fingerprint,
        )

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
            "declaration_manifest_id": self.declaration_manifest_id,
            "declaration_manifest_fingerprint": (
                self.declaration_manifest_fingerprint
            ),
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
                observed_version = self.factory_loader.package_version(package_name)
                self.validate_factory_loader_identity()
                version = (
                    None
                    if observed_version is None
                    else str(observed_version)
                )
                if (
                    version is None
                    or not version
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
                declaration_manifest_id=self.declaration_manifest_id,
                declaration_manifest_fingerprint=(
                    self.declaration_manifest_fingerprint
                ),
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


def load_attention_operator_provider_contribution_source_declaration_manifest(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    manifest_limits: (
        AttentionOperatorProviderContributionSourceDeclarationManifestLimits
    ) = (
        DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_DECLARATION_MANIFEST_LIMITS
    ),
) -> Tuple[
    AttentionOperatorProviderContributionSourceDeclarationManifest,
    AttentionJsonEnvelopeUsage,
]:
    """Decode bounded JSON before constructing adapter source declarations."""

    if not isinstance(
        manifest_limits,
        AttentionOperatorProviderContributionSourceDeclarationManifestLimits,
    ):
        raise TypeError(
            "manifest_limits must be "
            "AttentionOperatorProviderContributionSourceDeclarationManifestLimits"
        )
    decoded, usage = decode_attention_json(value, limits=limits)
    if not isinstance(decoded, Mapping):
        raise SchemaError("Attention source declaration manifest root must be an object")
    return (
        AttentionOperatorProviderContributionSourceDeclarationManifest.from_dict(
            decoded,
            limits=manifest_limits,
        ),
        usage,
    )


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_LOADER_VERSION",
    "DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_SOURCE_DECLARATION_MANIFEST_LIMITS",
    "AttentionOperatorProviderContributionFactoryLoader",
    "AttentionOperatorProviderContributionSourceDeclaration",
    "AttentionOperatorProviderContributionSourceDeclarationManifest",
    "AttentionOperatorProviderContributionSourceDeclarationManifestLimits",
    "AttentionOperatorProviderContributionSourceDeclarationRegistry",
    "ImportlibAttentionOperatorProviderContributionFactoryLoader",
    "load_attention_operator_provider_contribution_source_declaration_manifest",
]
