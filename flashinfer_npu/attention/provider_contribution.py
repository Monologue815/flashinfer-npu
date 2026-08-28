"""Provider-owned, reviewable inputs for one Attention integration slice.

A contribution is deliberately smaller than an installed provider bundle.  It
belongs to exactly one provider and contains the four exact inputs needed for
each operation: operation signature, runtime spec, scoring policy and loader
route.  Construction and validation are host-only and never probe a package or
device.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorOperationSpec,
)
from .operator_bootstrap import (
    AttentionOperatorPackageRuntimeSpec,
    bind_attention_operator_plan_scoring_manifest,
)
from .operator_declaration import (
    AttentionDeclaredOperatorPackageRuntimeSpec,
    describe_attention_operator_package_runtime,
)
from .operator_loader_routing import (
    AttentionOperatorPackageLoaderRoute,
    AttentionOperatorRoutedPackageLoader,
)
from .operator_scoring import (
    AttentionOperatorPlanScoringManifest,
    AttentionOperatorPlanScoringPolicy,
)


ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION = 1

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


def _local_identity(contribution_id: str, suffix: str) -> str:
    digest = hashlib.sha256(contribution_id.encode("utf-8")).hexdigest()
    return "flashinfer_npu.provider_contribution.%s.%s.v1" % (
        digest[:24],
        suffix,
    )


@dataclass(frozen=True)
class AttentionOperatorProviderIntegrationContributionBinding:
    """Non-executable identity of one independently reviewed contribution."""

    contribution_id: str
    contribution_fingerprint: str
    provider_id: str
    operation_bindings: Tuple[Tuple[str, str, str, str, str, str], ...]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION:
            raise SchemaError("unsupported Attention provider contribution binding")
        if not _IDENTIFIER.fullmatch(str(self.contribution_id)):
            raise SchemaError("invalid Attention provider contribution_id")
        if not _HASH.fullmatch(str(self.contribution_fingerprint)):
            raise SchemaError(
                "Attention provider contribution fingerprint is invalid"
            )
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid Attention contribution provider_id")
        bindings = tuple(tuple(item) for item in self.operation_bindings)
        if not bindings or any(len(item) != 6 for item in bindings):
            raise SchemaError("Attention contribution operation bindings are invalid")
        normalized = []
        for row in bindings:
            provider_id, operation_id = str(row[0]), str(row[1])
            fingerprints = tuple(str(item) for item in row[2:])
            if (
                provider_id != str(self.provider_id)
                or not operation_id
                or any(item.isspace() for item in operation_id)
                or any(not _HASH.fullmatch(item) for item in fingerprints)
            ):
                raise SchemaError(
                    "Attention contribution operation binding is invalid"
                )
            normalized.append((provider_id, operation_id) + fingerprints)
        identities = tuple(item[:2] for item in normalized)
        if len(set(identities)) != len(identities):
            raise SchemaError("Attention contribution operation identities duplicate")
        object.__setattr__(self, "contribution_id", str(self.contribution_id))
        object.__setattr__(
            self, "contribution_fingerprint", str(self.contribution_fingerprint)
        )
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(
            self,
            "operation_bindings",
            tuple(sorted(normalized, key=lambda item: item[:2])),
        )

    @property
    def identities(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(item[:2] for item in self.operation_bindings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contribution_id": self.contribution_id,
            "contribution_fingerprint": self.contribution_fingerprint,
            "provider_id": self.provider_id,
            "operation_bindings": [list(item) for item in self.operation_bindings],
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionOperatorProviderIntegrationContributionBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "Attention provider contribution binding fields are invalid"
            )
        if not isinstance(data.get("operation_bindings"), (list, tuple)):
            raise SchemaError(
                "Attention contribution operation bindings must be an array"
            )
        data["operation_bindings"] = tuple(data["operation_bindings"])
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                "Attention provider contribution binding fields are invalid"
            ) from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class _ContributionMaterialization:
    provider_id: str
    operation_catalog: AttentionOperatorOperationCatalog
    scoring_manifest: AttentionOperatorPlanScoringManifest
    runtime_specs: Tuple[AttentionOperatorPackageRuntimeSpec, ...]
    registrations: Tuple[AttentionDeclaredOperatorPackageRuntimeSpec, ...]
    routes: Tuple[AttentionOperatorPackageLoaderRoute, ...]
    operation_bindings: Tuple[Tuple[str, str, str, str, str, str], ...]


def _materialize_contribution(
    contribution_id: str,
    operations,
    runtime_specs,
    scoring_policies,
    package_loader_routes,
) -> _ContributionMaterialization:
    operation_values = tuple(operations)
    spec_values = tuple(runtime_specs)
    policy_values = tuple(scoring_policies)
    route_values = tuple(package_loader_routes)
    if not operation_values or any(
        not isinstance(item, AttentionOperatorOperationSpec)
        for item in operation_values
    ):
        raise TypeError("operations must contain AttentionOperatorOperationSpec")
    if not spec_values or any(
        not isinstance(item, AttentionOperatorPackageRuntimeSpec)
        for item in spec_values
    ):
        raise TypeError(
            "runtime_specs must contain AttentionOperatorPackageRuntimeSpec"
        )
    if not policy_values or any(
        not isinstance(item, AttentionOperatorPlanScoringPolicy)
        for item in policy_values
    ):
        raise TypeError(
            "scoring_policies must contain AttentionOperatorPlanScoringPolicy"
        )
    if not route_values or any(
        not isinstance(item, AttentionOperatorPackageLoaderRoute)
        for item in route_values
    ):
        raise TypeError(
            "package_loader_routes must contain package loader routes"
        )

    provider_ids = {item.provider_id for item in operation_values}
    if len(provider_ids) != 1:
        raise SchemaError(
            "Attention provider contribution must belong to exactly one provider"
        )
    provider_id = next(iter(provider_ids))
    catalog = AttentionOperatorOperationCatalog(
        name=_local_identity(contribution_id, "catalog"),
        operations=operation_values,
    )
    manifest = AttentionOperatorPlanScoringManifest(
        manifest_id=_local_identity(contribution_id, "scoring"),
        policies=policy_values,
    )
    bound_specs = bind_attention_operator_plan_scoring_manifest(
        spec_values,
        manifest,
    )
    registrations = tuple(
        AttentionDeclaredOperatorPackageRuntimeSpec(
            declaration=describe_attention_operator_package_runtime(
                spec,
                operation_catalog=catalog,
            ),
            runtime_spec=spec,
        )
        for spec in bound_specs
    )
    routed_loader = AttentionOperatorRoutedPackageLoader(
        operation_catalog=catalog,
        routes=route_values,
    )
    identities = tuple(
        (item.provider_id, item.operation_id) for item in catalog.operations
    )
    if any(item[0] != provider_id for item in identities):
        raise SchemaError(
            "Attention provider contribution contains another provider identity"
        )
    registrations_by_identity = {
        (item.declaration.provider_id, item.declaration.operation_id): item
        for item in registrations
    }
    policies_by_identity = {
        (item.provider_id, item.operation_id): item for item in manifest.policies
    }
    routes_by_identity = {
        (item.provider_id, item.operation_id): item for item in routed_loader.routes
    }
    operations_by_identity = {
        (item.provider_id, item.operation_id): item for item in catalog.operations
    }
    if not (
        set(identities)
        == set(registrations_by_identity)
        == set(policies_by_identity)
        == set(routes_by_identity)
    ):
        raise SchemaError("Attention provider contribution identity set differs")
    operation_bindings = tuple(
        (
            provider,
            operation,
            operations_by_identity[(provider, operation)].fingerprint,
            registrations_by_identity[(provider, operation)].fingerprint,
            policies_by_identity[(provider, operation)].fingerprint,
            routes_by_identity[(provider, operation)].binding.fingerprint,
        )
        for provider, operation in sorted(identities)
    )
    return _ContributionMaterialization(
        provider_id=provider_id,
        operation_catalog=catalog,
        scoring_manifest=manifest,
        runtime_specs=tuple(
            sorted(
                bound_specs,
                key=lambda item: (item.provider_id, item.operation_id),
            )
        ),
        registrations=tuple(
            sorted(
                registrations,
                key=lambda item: (
                    item.declaration.provider_id,
                    item.declaration.operation_id,
                ),
            )
        ),
        routes=routed_loader.routes,
        operation_bindings=operation_bindings,
    )


def _contribution_dict(
    contribution_id: str,
    materialized: _ContributionMaterialization,
) -> Dict[str, Any]:
    return {
        "schema_version": ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION,
        "kind": "attention_operator_provider_integration_contribution",
        "contribution_id": contribution_id,
        "provider_id": materialized.provider_id,
        "local_catalog_name": materialized.operation_catalog.name,
        "local_catalog_fingerprint": materialized.operation_catalog.fingerprint,
        "local_scoring_manifest_id": materialized.scoring_manifest.manifest_id,
        "local_scoring_manifest_fingerprint": (
            materialized.scoring_manifest.fingerprint
        ),
        "operation_bindings": [
            list(item) for item in materialized.operation_bindings
        ],
    }


@dataclass(frozen=True)
class AttentionOperatorProviderIntegrationContribution:
    """One provider's complete, independently reviewable integration inputs."""

    contribution_id: str
    operations: Tuple[AttentionOperatorOperationSpec, ...] = field(
        repr=False, compare=False
    )
    runtime_specs: Tuple[AttentionOperatorPackageRuntimeSpec, ...] = field(
        repr=False, compare=False
    )
    scoring_policies: Tuple[AttentionOperatorPlanScoringPolicy, ...] = field(
        repr=False, compare=False
    )
    package_loader_routes: Tuple[AttentionOperatorPackageLoaderRoute, ...] = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION
    _provider_id: str = field(init=False, repr=False, compare=False)
    _operation_bindings: Tuple[
        Tuple[str, str, str, str, str, str], ...
    ] = field(init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)
    _local_catalog_name: str = field(init=False, repr=False, compare=False)
    _local_catalog_fingerprint: str = field(init=False, repr=False, compare=False)
    _local_scoring_manifest_id: str = field(init=False, repr=False, compare=False)
    _local_scoring_manifest_fingerprint: str = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION:
            raise SchemaError("unsupported Attention provider contribution")
        contribution_id = str(self.contribution_id)
        if not _IDENTIFIER.fullmatch(contribution_id):
            raise SchemaError("invalid Attention provider contribution_id")
        materialized = _materialize_contribution(
            contribution_id,
            self.operations,
            self.runtime_specs,
            self.scoring_policies,
            self.package_loader_routes,
        )
        object.__setattr__(self, "contribution_id", contribution_id)
        object.__setattr__(
            self, "operations", materialized.operation_catalog.operations
        )
        object.__setattr__(self, "runtime_specs", materialized.runtime_specs)
        object.__setattr__(
            self, "scoring_policies", materialized.scoring_manifest.policies
        )
        object.__setattr__(self, "package_loader_routes", materialized.routes)
        object.__setattr__(self, "_provider_id", materialized.provider_id)
        object.__setattr__(
            self, "_operation_bindings", materialized.operation_bindings
        )
        object.__setattr__(
            self, "_local_catalog_name", materialized.operation_catalog.name
        )
        object.__setattr__(
            self,
            "_local_catalog_fingerprint",
            materialized.operation_catalog.fingerprint,
        )
        object.__setattr__(
            self,
            "_local_scoring_manifest_id",
            materialized.scoring_manifest.manifest_id,
        )
        object.__setattr__(
            self,
            "_local_scoring_manifest_fingerprint",
            materialized.scoring_manifest.fingerprint,
        )
        object.__setattr__(
            self,
            "_fingerprint",
            _canonical_hash(_contribution_dict(contribution_id, materialized)),
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def identities(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(item[:2] for item in self._operation_bindings)

    @property
    def operation_bindings(
        self,
    ) -> Tuple[Tuple[str, str, str, str, str, str], ...]:
        return self._operation_bindings

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_operator_provider_integration_contribution",
            "contribution_id": self.contribution_id,
            "provider_id": self.provider_id,
            "local_catalog_name": self._local_catalog_name,
            "local_catalog_fingerprint": self._local_catalog_fingerprint,
            "local_scoring_manifest_id": self._local_scoring_manifest_id,
            "local_scoring_manifest_fingerprint": (
                self._local_scoring_manifest_fingerprint
            ),
            "operation_bindings": [list(item) for item in self.operation_bindings],
        }

    @property
    def binding(self) -> AttentionOperatorProviderIntegrationContributionBinding:
        return AttentionOperatorProviderIntegrationContributionBinding(
            contribution_id=self.contribution_id,
            contribution_fingerprint=self.fingerprint,
            provider_id=self.provider_id,
            operation_bindings=self.operation_bindings,
        )

    def validate(self) -> None:
        """Recompute all identities and reject component or loader drift."""

        materialized = _materialize_contribution(
            self.contribution_id,
            self.operations,
            self.runtime_specs,
            self.scoring_policies,
            self.package_loader_routes,
        )
        current = _contribution_dict(self.contribution_id, materialized)
        if _canonical_hash(current) != self.fingerprint:
            raise SchemaError("Attention provider contribution identity changed")


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_CONTRIBUTION_VERSION",
    "AttentionOperatorProviderIntegrationContribution",
    "AttentionOperatorProviderIntegrationContributionBinding",
]
