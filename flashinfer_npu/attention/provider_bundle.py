"""One validated bootstrap unit for external Attention provider packages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import AttentionOperatorOperationCatalog
from .operator_bootstrap import bind_attention_operator_plan_scoring_manifest
from .operator_declaration import AttentionDeclaredOperatorPackageRuntimeSpec
from .operator_package import AttentionOperatorPackageLoader
from .operator_scoring import AttentionOperatorPlanScoringManifest


ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION = 1

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
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
        raise SchemaError("provider integration bundle loader type is not stable")
    return result


@dataclass(frozen=True)
class _IdentityBoundAttentionOperatorPackageLoader:
    """Keep the reviewed loader identity live for the registry lifetime."""

    delegate: AttentionOperatorPackageLoader = field(repr=False, compare=False)
    loader_id: str
    loader_type: str

    def _validate(self) -> None:
        if (
            str(self.delegate.loader_id) != self.loader_id
            or _stable_type_name(self.delegate) != self.loader_type
        ):
            raise SchemaError(
                "provider integration bundle package loader identity changed"
            )

    def package_version(self, package_name: str):
        self._validate()
        return self.delegate.package_version(package_name)

    def resolve_callable(self, callable_path: str):
        self._validate()
        return self.delegate.resolve_callable(callable_path)


@dataclass(frozen=True)
class AttentionOperatorProviderIntegrationBundleBinding:
    """Non-executable bundle authority stored with a registry generation."""

    bundle_id: str
    bundle_fingerprint: str
    catalog_name: str
    catalog_fingerprint: str
    scoring_manifest_id: str
    scoring_manifest_fingerprint: str
    package_loader_id: str
    package_loader_type: str
    registration_bindings: Tuple[Tuple[str, str, str], ...]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION:
            raise SchemaError("unsupported Attention provider bundle binding version")
        for name in ("bundle_id", "scoring_manifest_id"):
            if not _IDENTIFIER.fullmatch(str(getattr(self, name))):
                raise SchemaError("Attention provider bundle binding id is invalid")
        for name in (
            "bundle_fingerprint",
            "catalog_fingerprint",
            "scoring_manifest_fingerprint",
        ):
            if not _HASH.fullmatch(str(getattr(self, name))):
                raise SchemaError(
                    "Attention provider bundle binding fingerprint is invalid"
                )
        if not str(self.catalog_name):
            raise SchemaError(
                "Attention provider bundle binding catalog name is invalid"
            )
        for name in ("package_loader_id", "package_loader_type"):
            value = str(getattr(self, name))
            if not value or any(item.isspace() for item in value):
                raise SchemaError(
                    "Attention provider bundle binding component id is invalid"
                )
        bindings = tuple(tuple(item) for item in self.registration_bindings)
        if not bindings or any(len(item) != 3 for item in bindings):
            raise SchemaError(
                "Attention provider bundle registration bindings are invalid"
            )
        normalized = []
        for provider_id, operation_id, declaration_fingerprint in bindings:
            values = (
                str(provider_id),
                str(operation_id),
                str(declaration_fingerprint),
            )
            if not values[0] or not values[1] or not _HASH.fullmatch(values[2]):
                raise SchemaError(
                    "Attention provider bundle registration binding is invalid"
                )
            normalized.append(values)
        identities = tuple(item[:2] for item in normalized)
        if len(set(identities)) != len(identities):
            raise SchemaError(
                "Attention provider bundle registration identities duplicate"
            )
        object.__setattr__(
            self,
            "registration_bindings",
            tuple(sorted(normalized, key=lambda item: item[:2])),
        )
        for name in (
            "bundle_id",
            "bundle_fingerprint",
            "catalog_name",
            "catalog_fingerprint",
            "scoring_manifest_id",
            "scoring_manifest_fingerprint",
            "package_loader_id",
            "package_loader_type",
        ):
            object.__setattr__(self, name, str(getattr(self, name)))

    @property
    def identities(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(item[:2] for item in self.registration_bindings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "bundle_fingerprint": self.bundle_fingerprint,
            "catalog_name": self.catalog_name,
            "catalog_fingerprint": self.catalog_fingerprint,
            "scoring_manifest_id": self.scoring_manifest_id,
            "scoring_manifest_fingerprint": self.scoring_manifest_fingerprint,
            "package_loader_id": self.package_loader_id,
            "package_loader_type": self.package_loader_type,
            "registration_bindings": [
                list(item) for item in self.registration_bindings
            ],
        }


@dataclass(frozen=True)
class AttentionOperatorProviderIntegrationBundle:
    """Complete reviewed inputs for one atomic provider bootstrap install."""

    bundle_id: str
    operation_catalog: AttentionOperatorOperationCatalog = field(
        repr=False, compare=False
    )
    registrations: Tuple[AttentionDeclaredOperatorPackageRuntimeSpec, ...] = field(
        repr=False, compare=False
    )
    scoring_manifest: AttentionOperatorPlanScoringManifest = field(
        repr=False, compare=False
    )
    package_loader: AttentionOperatorPackageLoader = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION
    _package_loader_id: str = field(init=False, repr=False, compare=False)
    _package_loader_type: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION:
            raise SchemaError("unsupported Attention provider integration bundle")
        if not _IDENTIFIER.fullmatch(str(self.bundle_id)):
            raise SchemaError("invalid Attention provider integration bundle_id")
        if not isinstance(self.operation_catalog, AttentionOperatorOperationCatalog):
            raise TypeError("operation_catalog must be AttentionOperatorOperationCatalog")
        registrations = tuple(self.registrations)
        if not registrations or any(
            not isinstance(item, AttentionDeclaredOperatorPackageRuntimeSpec)
            for item in registrations
        ):
            raise TypeError("registrations must contain declared runtime specs")
        if not isinstance(
            self.scoring_manifest, AttentionOperatorPlanScoringManifest
        ):
            raise TypeError(
                "scoring_manifest must be AttentionOperatorPlanScoringManifest"
            )
        if not isinstance(self.package_loader, AttentionOperatorPackageLoader):
            raise TypeError(
                "package_loader must implement AttentionOperatorPackageLoader"
            )
        loader_id = str(self.package_loader.loader_id)
        if not loader_id or any(item.isspace() for item in loader_id):
            raise SchemaError("provider integration bundle loader_id is invalid")
        loader_type = _stable_type_name(self.package_loader)
        registration_identities = tuple(
            (item.declaration.provider_id, item.declaration.operation_id)
            for item in registrations
        )
        if len(set(registration_identities)) != len(registration_identities):
            raise SchemaError("provider integration bundle registrations duplicate")
        catalog_identities = tuple(
            (item.provider_id, item.operation_id)
            for item in self.operation_catalog.operations
        )
        if set(catalog_identities) != set(registration_identities):
            raise SchemaError(
                "provider integration bundle catalog identity set differs"
            )
        if set(self.scoring_manifest.binding.identities) != set(
            registration_identities
        ):
            raise SchemaError(
                "provider integration bundle scoring identity set differs"
            )
        if any(item.runtime_spec.plan_scorer is None for item in registrations):
            raise SchemaError(
                "provider integration bundle registrations must bind the "
                "scoring manifest before declaration"
            )
        declaration_fingerprints = tuple(
            item.declaration.fingerprint for item in registrations
        )
        if len(set(declaration_fingerprints)) != len(declaration_fingerprints):
            raise SchemaError(
                "provider integration bundle declarations duplicate"
            )
        bind_attention_operator_plan_scoring_manifest(
            tuple(item.runtime_spec for item in registrations),
            self.scoring_manifest,
        )
        for item in registrations:
            item.validate(self.operation_catalog)
        object.__setattr__(self, "bundle_id", str(self.bundle_id))
        object.__setattr__(self, "_package_loader_id", loader_id)
        object.__setattr__(self, "_package_loader_type", loader_type)
        object.__setattr__(
            self,
            "registrations",
            tuple(
                sorted(
                    registrations,
                    key=lambda item: (
                        item.declaration.provider_id,
                        item.declaration.operation_id,
                    ),
                )
            ),
        )

    @property
    def package_loader_id(self) -> str:
        return self._package_loader_id

    @property
    def package_loader_type(self) -> str:
        return self._package_loader_type

    def validate_package_loader_identity(self) -> None:
        """Reject loader identity drift before registry composition."""

        current_id = str(self.package_loader.loader_id)
        current_type = _stable_type_name(self.package_loader)
        if (
            current_id != self.package_loader_id
            or current_type != self.package_loader_type
        ):
            raise SchemaError(
                "provider integration bundle package loader identity changed"
            )

    def bind_package_loader(self) -> AttentionOperatorPackageLoader:
        """Return a delegate that revalidates identity on every observation."""

        self.validate_package_loader_identity()
        return _IdentityBoundAttentionOperatorPackageLoader(
            delegate=self.package_loader,
            loader_id=self.package_loader_id,
            loader_type=self.package_loader_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_operator_provider_integration_bundle",
            "bundle_id": self.bundle_id,
            "catalog_name": self.operation_catalog.name,
            "catalog_fingerprint": self.operation_catalog.fingerprint,
            "scoring_manifest_id": self.scoring_manifest.manifest_id,
            "scoring_manifest_fingerprint": self.scoring_manifest.fingerprint,
            "package_loader_id": self.package_loader_id,
            "package_loader_type": self.package_loader_type,
            "registrations": [
                {
                    "provider_id": item.declaration.provider_id,
                    "operation_id": item.declaration.operation_id,
                    "declaration_fingerprint": item.declaration.fingerprint,
                }
                for item in self.registrations
            ],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def binding(self) -> AttentionOperatorProviderIntegrationBundleBinding:
        return AttentionOperatorProviderIntegrationBundleBinding(
            bundle_id=self.bundle_id,
            bundle_fingerprint=self.fingerprint,
            catalog_name=self.operation_catalog.name,
            catalog_fingerprint=self.operation_catalog.fingerprint,
            scoring_manifest_id=self.scoring_manifest.manifest_id,
            scoring_manifest_fingerprint=self.scoring_manifest.fingerprint,
            package_loader_id=self.package_loader_id,
            package_loader_type=self.package_loader_type,
            registration_bindings=tuple(
                (
                    item.declaration.provider_id,
                    item.declaration.operation_id,
                    item.declaration.fingerprint,
                )
                for item in self.registrations
            ),
        )


def install_attention_operator_provider_integration_bundle(
    bundle: AttentionOperatorProviderIntegrationBundle,
    *,
    expected_generation=None,
):
    """Validate and atomically install one complete provider integration."""

    if not isinstance(bundle, AttentionOperatorProviderIntegrationBundle):
        raise TypeError("bundle must be AttentionOperatorProviderIntegrationBundle")
    package_loader = bundle.bind_package_loader()
    from .holistic import _install_declared_attention_operator_runtime_resolvers

    return _install_declared_attention_operator_runtime_resolvers(
        bundle.registrations,
        operation_catalog=bundle.operation_catalog,
        package_loader=package_loader,
        plan_scoring_manifest=bundle.scoring_manifest,
        provider_integration_bundle_binding=bundle.binding,
        expected_generation=expected_generation,
    )


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_INTEGRATION_BUNDLE_VERSION",
    "AttentionOperatorProviderIntegrationBundle",
    "AttentionOperatorProviderIntegrationBundleBinding",
    "install_attention_operator_provider_integration_bundle",
]
