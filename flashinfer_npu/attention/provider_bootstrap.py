"""Data-only authority for one complete Attention provider deployment."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .json_envelope import (
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    AttentionJsonEnvelopeLimits,
    AttentionJsonEnvelopeUsage,
    decode_attention_json,
)
from .provider_contribution_loader import (
    AttentionOperatorProviderContributionFactoryLoader,
    AttentionOperatorProviderContributionSourceDeclarationManifest,
)
from .provider_contribution_manifest import (
    AttentionOperatorProviderContributionManifest,
)


ATTENTION_OPERATOR_PROVIDER_BOOTSTRAP_VERSION = 1

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KIND = "attention_operator_provider_integration_bootstrap"


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
        raise SchemaError("Attention provider bootstrap loader type is not stable")
    return result


@dataclass(frozen=True)
class AttentionOperatorProviderIntegrationBootstrapManifest:
    """Reviewed identities needed to assemble and publish one provider set."""

    bootstrap_id: str
    bundle_id: str
    catalog_name: str
    scoring_manifest_id: str
    source_manifest_id: str
    source_manifest_fingerprint: str
    contribution_manifest_id: str
    contribution_manifest_fingerprint: str
    factory_loader_id: str
    factory_loader_type: str
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_BOOTSTRAP_VERSION
    kind: str = field(default=_KIND, init=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_BOOTSTRAP_VERSION:
            raise SchemaError("unsupported Attention provider bootstrap manifest")
        if self.kind != _KIND:
            raise SchemaError("Attention provider bootstrap manifest kind is invalid")
        for name in (
            "bootstrap_id",
            "bundle_id",
            "scoring_manifest_id",
            "source_manifest_id",
            "contribution_manifest_id",
        ):
            value = str(getattr(self, name))
            if not _IDENTIFIER.fullmatch(value):
                raise SchemaError("Attention provider bootstrap id is invalid")
            object.__setattr__(self, name, value)
        for name in (
            "source_manifest_fingerprint",
            "contribution_manifest_fingerprint",
        ):
            value = str(getattr(self, name))
            if not _HASH.fullmatch(value):
                raise SchemaError(
                    "Attention provider bootstrap fingerprint is invalid"
                )
            object.__setattr__(self, name, value)
        for name in ("catalog_name", "factory_loader_id", "factory_loader_type"):
            value = str(getattr(self, name))
            if not value or any(item.isspace() for item in value):
                raise SchemaError(
                    "Attention provider bootstrap component identity is invalid"
                )
            object.__setattr__(self, name, value)

    @classmethod
    def from_inputs(
        cls,
        *,
        bootstrap_id: str,
        bundle_id: str,
        catalog_name: str,
        scoring_manifest_id: str,
        source_manifest: (
            AttentionOperatorProviderContributionSourceDeclarationManifest
        ),
        approval_manifest: AttentionOperatorProviderContributionManifest,
        factory_loader: AttentionOperatorProviderContributionFactoryLoader,
    ) -> "AttentionOperatorProviderIntegrationBootstrapManifest":
        """Freeze reviewed input identities without observing adapter packages."""

        if not isinstance(
            source_manifest,
            AttentionOperatorProviderContributionSourceDeclarationManifest,
        ):
            raise TypeError(
                "source_manifest must be "
                "AttentionOperatorProviderContributionSourceDeclarationManifest"
            )
        if not isinstance(
            approval_manifest,
            AttentionOperatorProviderContributionManifest,
        ):
            raise TypeError(
                "approval_manifest must be "
                "AttentionOperatorProviderContributionManifest"
            )
        if not isinstance(
            factory_loader,
            AttentionOperatorProviderContributionFactoryLoader,
        ):
            raise TypeError(
                "factory_loader must implement "
                "AttentionOperatorProviderContributionFactoryLoader"
            )
        return cls(
            bootstrap_id=bootstrap_id,
            bundle_id=bundle_id,
            catalog_name=catalog_name,
            scoring_manifest_id=scoring_manifest_id,
            source_manifest_id=source_manifest.manifest_id,
            source_manifest_fingerprint=source_manifest.fingerprint,
            contribution_manifest_id=approval_manifest.manifest_id,
            contribution_manifest_fingerprint=approval_manifest.fingerprint,
            factory_loader_id=str(factory_loader.loader_id),
            factory_loader_type=_stable_type_name(factory_loader),
        )

    def validate_inputs(
        self,
        *,
        source_manifest: (
            AttentionOperatorProviderContributionSourceDeclarationManifest
        ),
        approval_manifest: AttentionOperatorProviderContributionManifest,
        factory_loader: AttentionOperatorProviderContributionFactoryLoader,
    ) -> None:
        """Fail closed on identity drift before any adapter package observation."""

        if not isinstance(
            source_manifest,
            AttentionOperatorProviderContributionSourceDeclarationManifest,
        ):
            raise TypeError(
                "source_manifest must be "
                "AttentionOperatorProviderContributionSourceDeclarationManifest"
            )
        if not isinstance(
            approval_manifest,
            AttentionOperatorProviderContributionManifest,
        ):
            raise TypeError(
                "approval_manifest must be "
                "AttentionOperatorProviderContributionManifest"
            )
        if (
            source_manifest.manifest_id != self.source_manifest_id
            or source_manifest.fingerprint != self.source_manifest_fingerprint
        ):
            raise SchemaError("Attention provider bootstrap source manifest differs")
        if (
            approval_manifest.manifest_id != self.contribution_manifest_id
            or approval_manifest.fingerprint
            != self.contribution_manifest_fingerprint
        ):
            raise SchemaError(
                "Attention provider bootstrap contribution manifest differs"
            )
        if not isinstance(
            factory_loader,
            AttentionOperatorProviderContributionFactoryLoader,
        ):
            raise TypeError(
                "factory_loader must implement "
                "AttentionOperatorProviderContributionFactoryLoader"
            )
        if (
            str(factory_loader.loader_id) != self.factory_loader_id
            or _stable_type_name(factory_loader) != self.factory_loader_type
        ):
            raise SchemaError("Attention provider bootstrap factory loader differs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "bootstrap_id": self.bootstrap_id,
            "bundle_id": self.bundle_id,
            "catalog_name": self.catalog_name,
            "scoring_manifest_id": self.scoring_manifest_id,
            "source_manifest_id": self.source_manifest_id,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "contribution_manifest_id": self.contribution_manifest_id,
            "contribution_manifest_fingerprint": (
                self.contribution_manifest_fingerprint
            ),
            "factory_loader_id": self.factory_loader_id,
            "factory_loader_type": self.factory_loader_type,
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
        cls, value: Mapping[str, Any]
    ) -> "AttentionOperatorProviderIntegrationBootstrapManifest":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention provider bootstrap fields are invalid")
        if data.pop("kind") != _KIND:
            raise SchemaError("Attention provider bootstrap manifest kind is invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention provider bootstrap fields are invalid") from error


def load_attention_operator_provider_integration_bootstrap_manifest(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
) -> Tuple[
    AttentionOperatorProviderIntegrationBootstrapManifest,
    AttentionJsonEnvelopeUsage,
]:
    """Decode bounded JSON before constructing a bootstrap authority."""

    decoded, usage = decode_attention_json(value, limits=limits)
    if not isinstance(decoded, Mapping):
        raise SchemaError("Attention provider bootstrap root must be an object")
    return (
        AttentionOperatorProviderIntegrationBootstrapManifest.from_dict(decoded),
        usage,
    )


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_BOOTSTRAP_VERSION",
    "AttentionOperatorProviderIntegrationBootstrapManifest",
    "load_attention_operator_provider_integration_bootstrap_manifest",
]
