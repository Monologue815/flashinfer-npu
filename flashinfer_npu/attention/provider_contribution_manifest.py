"""Deployment approval manifest for exact Attention provider contributions."""

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
from .provider_contribution import (
    AttentionOperatorProviderIntegrationContribution,
    AttentionOperatorProviderIntegrationContributionBinding,
)


ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_VERSION = 1

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttentionOperatorProviderContributionManifestLimits:
    """Semantic bounds applied after the generic JSON envelope."""

    max_contributions: int = 64
    max_operations: int = 1024
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_VERSION
        ):
            raise SchemaError(
                "unsupported Attention provider contribution manifest limits"
            )
        for name in ("max_contributions", "max_operations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SchemaError("%s must be a positive integer" % name)


DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_LIMITS = (
    AttentionOperatorProviderContributionManifestLimits()
)


@dataclass(frozen=True)
class AttentionOperatorProviderContributionManifest:
    """Exact deployment allow-list for independently supplied contributions."""

    manifest_id: str
    contribution_bindings: Tuple[
        AttentionOperatorProviderIntegrationContributionBinding, ...
    ]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_VERSION
    kind: str = field(
        default="attention_operator_provider_contribution_manifest",
        init=False,
    )

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_VERSION
        ):
            raise SchemaError("unsupported Attention provider contribution manifest")
        if self.kind != "attention_operator_provider_contribution_manifest":
            raise SchemaError("Attention provider contribution manifest kind is invalid")
        if not _IDENTIFIER.fullmatch(str(self.manifest_id)):
            raise SchemaError("invalid Attention provider contribution manifest_id")
        bindings = tuple(self.contribution_bindings)
        if not bindings or any(
            not isinstance(
                item,
                AttentionOperatorProviderIntegrationContributionBinding,
            )
            for item in bindings
        ):
            raise TypeError(
                "contribution_bindings must contain provider contribution bindings"
            )
        contribution_ids = tuple(item.contribution_id for item in bindings)
        if len(set(contribution_ids)) != len(contribution_ids):
            raise SchemaError(
                "Attention provider contribution manifest ids duplicate"
            )
        identities = tuple(
            identity for binding in bindings for identity in binding.identities
        )
        if len(set(identities)) != len(identities):
            raise SchemaError(
                "Attention provider contribution manifest identities overlap"
            )
        object.__setattr__(self, "manifest_id", str(self.manifest_id))
        object.__setattr__(
            self,
            "contribution_bindings",
            tuple(sorted(bindings, key=lambda item: item.contribution_id)),
        )

    @property
    def identities(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            identity
            for binding in self.contribution_bindings
            for identity in binding.identities
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "manifest_id": self.manifest_id,
            "contribution_bindings": [
                item.to_dict() for item in self.contribution_bindings
            ],
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
        limits: AttentionOperatorProviderContributionManifestLimits = (
            DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_LIMITS
        ),
    ) -> "AttentionOperatorProviderContributionManifest":
        if not isinstance(
            limits, AttentionOperatorProviderContributionManifestLimits
        ):
            raise TypeError(
                "limits must be AttentionOperatorProviderContributionManifestLimits"
            )
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "Attention provider contribution manifest fields are invalid"
            )
        if data.pop("kind") != "attention_operator_provider_contribution_manifest":
            raise SchemaError("Attention provider contribution manifest kind is invalid")
        encoded_bindings = data.get("contribution_bindings")
        if not isinstance(encoded_bindings, list):
            raise SchemaError(
                "Attention provider contribution manifest bindings must be an array"
            )
        if len(encoded_bindings) > limits.max_contributions:
            raise SchemaError(
                "Attention provider contributions exceed limit %d"
                % limits.max_contributions
            )
        operation_count = 0
        bindings = []
        for encoded in encoded_bindings:
            if not isinstance(encoded, Mapping):
                raise SchemaError(
                    "Attention provider contribution binding must be an object"
                )
            binding = (
                AttentionOperatorProviderIntegrationContributionBinding.from_dict(
                    encoded
                )
            )
            operation_count += len(binding.operation_bindings)
            if operation_count > limits.max_operations:
                raise SchemaError(
                    "Attention provider contribution operations exceed limit %d"
                    % limits.max_operations
                )
            bindings.append(binding)
        data["contribution_bindings"] = tuple(bindings)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "Attention provider contribution manifest fields are invalid"
            ) from error

    def validate_contributions(self, contributions) -> Tuple[
        AttentionOperatorProviderIntegrationContribution, ...
    ]:
        """Return canonical contributions only after an exact manifest match."""

        values = tuple(contributions)
        if not values or any(
            not isinstance(item, AttentionOperatorProviderIntegrationContribution)
            for item in values
        ):
            raise TypeError(
                "contributions must contain provider integration contributions"
            )
        contribution_ids = tuple(item.contribution_id for item in values)
        if len(set(contribution_ids)) != len(contribution_ids):
            raise SchemaError("Attention provider contribution ids duplicate")
        normalized = tuple(sorted(values, key=lambda item: item.contribution_id))
        for contribution in normalized:
            contribution.validate()
        observed = tuple(item.binding for item in normalized)
        if observed != self.contribution_bindings:
            expected_by_id = {
                item.contribution_id: item for item in self.contribution_bindings
            }
            observed_by_id = {item.contribution_id: item for item in observed}
            missing = tuple(sorted(set(expected_by_id).difference(observed_by_id)))
            orphan = tuple(sorted(set(observed_by_id).difference(expected_by_id)))
            drifted = tuple(
                sorted(
                    contribution_id
                    for contribution_id in set(expected_by_id).intersection(
                        observed_by_id
                    )
                    if expected_by_id[contribution_id]
                    != observed_by_id[contribution_id]
                )
            )
            raise SchemaError(
                "Attention provider contributions differ from approval manifest "
                "(missing=%r, orphan=%r, drifted=%r)"
                % (missing, orphan, drifted)
            )
        return normalized


def load_attention_operator_provider_contribution_manifest(
    value: str,
    *,
    limits: AttentionJsonEnvelopeLimits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    manifest_limits: AttentionOperatorProviderContributionManifestLimits = (
        DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_LIMITS
    ),
) -> Tuple[
    AttentionOperatorProviderContributionManifest,
    AttentionJsonEnvelopeUsage,
]:
    """Decode bounded JSON before constructing an approval manifest."""

    if not isinstance(
        manifest_limits, AttentionOperatorProviderContributionManifestLimits
    ):
        raise TypeError(
            "manifest_limits must be "
            "AttentionOperatorProviderContributionManifestLimits"
        )
    decoded, usage = decode_attention_json(value, limits=limits)
    if not isinstance(decoded, Mapping):
        raise SchemaError(
            "Attention provider contribution manifest root must be an object"
        )
    return (
        AttentionOperatorProviderContributionManifest.from_dict(
            decoded,
            limits=manifest_limits,
        ),
        usage,
    )


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_VERSION",
    "DEFAULT_ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_MANIFEST_LIMITS",
    "AttentionOperatorProviderContributionManifest",
    "AttentionOperatorProviderContributionManifestLimits",
    "load_attention_operator_provider_contribution_manifest",
]
