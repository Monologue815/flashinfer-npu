"""Exact multi-package loader routing for one Attention integration bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import AttentionOperatorOperationCatalog
from .operator_package import AttentionOperatorPackageLoader


ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION = 1


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
        raise SchemaError("Attention package route loader type is not stable")
    return result


@dataclass(frozen=True)
class AttentionOperatorPackageLoaderRouteBinding:
    """Non-executable identity for one exact operation-to-loader route."""

    provider_id: str
    operation_id: str
    package_name: str
    callable_path: str
    delegate_loader_id: str
    delegate_loader_type: str
    schema_version: int = ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION:
            raise SchemaError("unsupported Attention package loader route binding")
        for name in (
            "provider_id",
            "operation_id",
            "package_name",
            "callable_path",
            "delegate_loader_id",
            "delegate_loader_type",
        ):
            value = str(getattr(self, name))
            if not value or any(item.isspace() for item in value):
                raise SchemaError("Attention package loader route identity is invalid")
            object.__setattr__(self, name, value)

    @property
    def identity(self) -> Tuple[str, str]:
        return (self.provider_id, self.operation_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "package_name": self.package_name,
            "callable_path": self.callable_path,
            "delegate_loader_id": self.delegate_loader_id,
            "delegate_loader_type": self.delegate_loader_type,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorPackageLoaderRoute:
    """One executable delegate bound to an exact catalog operation."""

    provider_id: str
    operation_id: str
    package_name: str
    callable_path: str
    package_loader: AttentionOperatorPackageLoader = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION
    _delegate_loader_id: str = field(init=False, repr=False, compare=False)
    _delegate_loader_type: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION:
            raise SchemaError("unsupported Attention package loader route")
        if not isinstance(self.package_loader, AttentionOperatorPackageLoader):
            raise TypeError(
                "package_loader must implement AttentionOperatorPackageLoader"
            )
        binding_values = (
            str(self.provider_id),
            str(self.operation_id),
            str(self.package_name),
            str(self.callable_path),
        )
        if any(
            not value or any(item.isspace() for item in value)
            for value in binding_values
        ):
            raise SchemaError("Attention package loader route identity is invalid")
        loader_id = str(self.package_loader.loader_id)
        if not loader_id or any(item.isspace() for item in loader_id):
            raise SchemaError("Attention package route loader_id is invalid")
        loader_type = _stable_type_name(self.package_loader)
        for name, value in zip(
            ("provider_id", "operation_id", "package_name", "callable_path"),
            binding_values,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_delegate_loader_id", loader_id)
        object.__setattr__(self, "_delegate_loader_type", loader_type)

    @classmethod
    def from_catalog_operation(cls, operation, package_loader):
        """Create a route from one already validated catalog operation."""

        from .operation_catalog import AttentionOperatorOperationSpec

        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        return cls(
            provider_id=operation.provider_id,
            operation_id=operation.operation_id,
            package_name=operation.package_name,
            callable_path=operation.callable_path,
            package_loader=package_loader,
        )

    @property
    def delegate_loader_id(self) -> str:
        return self._delegate_loader_id

    @property
    def delegate_loader_type(self) -> str:
        return self._delegate_loader_type

    def validate_loader_identity(self) -> None:
        if (
            str(self.package_loader.loader_id) != self.delegate_loader_id
            or _stable_type_name(self.package_loader) != self.delegate_loader_type
        ):
            raise SchemaError("Attention package route loader identity changed")

    @property
    def binding(self) -> AttentionOperatorPackageLoaderRouteBinding:
        return AttentionOperatorPackageLoaderRouteBinding(
            provider_id=self.provider_id,
            operation_id=self.operation_id,
            package_name=self.package_name,
            callable_path=self.callable_path,
            delegate_loader_id=self.delegate_loader_id,
            delegate_loader_type=self.delegate_loader_type,
        )


@dataclass(frozen=True)
class AttentionOperatorRoutedPackageLoader:
    """Route exact package and callable observations to reviewed delegates."""

    operation_catalog: AttentionOperatorOperationCatalog = field(
        repr=False, compare=False
    )
    routes: Tuple[AttentionOperatorPackageLoaderRoute, ...] = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION
    _loader_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION:
            raise SchemaError("unsupported routed Attention package loader")
        if not isinstance(self.operation_catalog, AttentionOperatorOperationCatalog):
            raise TypeError(
                "operation_catalog must be AttentionOperatorOperationCatalog"
            )
        routes = tuple(self.routes)
        if not routes or any(
            not isinstance(item, AttentionOperatorPackageLoaderRoute)
            for item in routes
        ):
            raise TypeError("routes must contain Attention package loader routes")
        identities = tuple(
            (item.provider_id, item.operation_id) for item in routes
        )
        if len(set(identities)) != len(identities):
            raise SchemaError("Attention package loader routes duplicate")
        catalog_identities = tuple(
            (item.provider_id, item.operation_id)
            for item in self.operation_catalog.operations
        )
        if set(identities) != set(catalog_identities):
            raise SchemaError(
                "Attention package loader route identity set differs from catalog"
            )
        for route in routes:
            operation = self.operation_catalog.get(route.operation_id)
            if (
                route.provider_id != operation.provider_id
                or route.package_name != operation.package_name
                or route.callable_path != operation.callable_path
            ):
                raise SchemaError(
                    "Attention package loader route differs from catalog operation"
                )
            route.validate_loader_identity()
        self._validate_unambiguous_delegates(routes)
        normalized = tuple(
            sorted(routes, key=lambda item: (item.provider_id, item.operation_id))
        )
        object.__setattr__(self, "routes", normalized)
        object.__setattr__(
            self,
            "_loader_id",
            "flashinfer_npu.attention.routed_package_loader.v1:%s"
            % self.routing_fingerprint,
        )

    @staticmethod
    def _validate_unambiguous_delegates(routes) -> None:
        for attribute in ("package_name", "callable_path"):
            values = {}
            for route in routes:
                key = getattr(route, attribute)
                existing = values.get(key)
                if existing is not None and existing.package_loader is not (
                    route.package_loader
                ):
                    raise SchemaError(
                        "Attention package loader routes are ambiguous for %s"
                        % key
                    )
                values[key] = route

    @property
    def loader_id(self) -> str:
        self._validate_all_loader_identities()
        return self._loader_id

    def _validate_all_loader_identities(self) -> None:
        for route in self.routes:
            route.validate_loader_identity()

    @property
    def route_bindings(self) -> Tuple[
        AttentionOperatorPackageLoaderRouteBinding, ...
    ]:
        return tuple(item.binding for item in self.routes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "attention_operator_routed_package_loader",
            "catalog_name": self.operation_catalog.name,
            "catalog_fingerprint": self.operation_catalog.fingerprint,
            "routes": [item.to_dict() for item in self.route_bindings],
        }

    @property
    def routing_fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def _route_for(self, attribute: str, value: str):
        matches = tuple(
            item for item in self.routes if getattr(item, attribute) == str(value)
        )
        if not matches:
            raise SchemaError(
                "no exact Attention package loader route for %s" % value
            )
        for route in matches:
            route.validate_loader_identity()
        return matches[0]

    def package_version(self, package_name: str) -> Optional[str]:
        self._validate_all_loader_identities()
        route = self._route_for("package_name", package_name)
        return route.package_loader.package_version(str(package_name))

    def resolve_callable(self, callable_path: str):
        self._validate_all_loader_identities()
        route = self._route_for("callable_path", callable_path)
        return route.package_loader.resolve_callable(str(callable_path))


__all__ = [
    "ATTENTION_OPERATOR_PACKAGE_LOADER_ROUTING_VERSION",
    "AttentionOperatorPackageLoaderRoute",
    "AttentionOperatorPackageLoaderRouteBinding",
    "AttentionOperatorRoutedPackageLoader",
]
