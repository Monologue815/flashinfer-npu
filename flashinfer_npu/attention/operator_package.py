"""Lazy external-package resolution for versioned Attention operations.

Package presence is not capability evidence.  This module only turns an
explicit adapter compatibility declaration into an exact callable binding.
It does not register any default package, initialize a device, plan an
operation, or invoke an Attention callable.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorOperationSpec,
)
from .operator_callable import (
    AttentionObservedOperatorCallable,
    AttentionOperatorCallableBinding,
    bind_attention_operator_callable,
    explain_attention_operator_callable,
    observe_python_callable_signature,
)
from .operator_execution import AttentionInjectedCallableExecutor
from .operator_provider import AttentionOperatorProviderProbe


ATTENTION_OPERATOR_PACKAGE_VERSION = 1

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class AttentionOperatorPackageLoader(Protocol):
    """Provider-owned loading boundary used only after metadata acceptance."""

    loader_id: str

    def package_version(self, package_name: str) -> Optional[str]:
        """Return distribution metadata without importing the package."""

    def resolve_callable(self, callable_path: str) -> Any:
        """Resolve one trusted catalog path without invoking the callable."""


class ImportlibAttentionOperatorPackageLoader:
    """Generic Python loader; never installed into the default NPU registry."""

    loader_id = "python.importlib.metadata.v1"

    def package_version(self, package_name: str) -> Optional[str]:
        try:
            return importlib.metadata.version(str(package_name))
        except importlib.metadata.PackageNotFoundError:
            return None

    def resolve_callable(self, callable_path: str) -> Any:
        module_name, separator, attribute_name = str(callable_path).rpartition(".")
        if not separator or not module_name or not attribute_name:
            raise SchemaError("operator callable path must include a module and name")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, attribute_name)
        except AttributeError as error:
            raise SchemaError(
                "operator callable %r is absent" % callable_path
            ) from error


@dataclass(frozen=True)
class AttentionOperatorPackageCompatibility:
    """Adapter-owned exact distribution versions for one catalog operation."""

    provider_id: str
    operation_id: str
    adapter_version: str
    supported_package_versions: Tuple[str, ...]
    schema_version: int = ATTENTION_OPERATOR_PACKAGE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PACKAGE_VERSION:
            raise SchemaError("unsupported Attention package compatibility version")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid Attention package compatibility provider_id")
        if not str(self.operation_id) or not str(self.adapter_version):
            raise SchemaError("package operation and adapter versions must be non-empty")
        versions = tuple(str(item) for item in self.supported_package_versions)
        if not versions or any(not item for item in versions):
            raise SchemaError("supported package versions must be non-empty")
        if len(set(versions)) != len(versions):
            raise SchemaError("supported package versions must be unique")
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(self, "adapter_version", str(self.adapter_version))
        object.__setattr__(
            self, "supported_package_versions", tuple(sorted(versions))
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "adapter_version": self.adapter_version,
            "supported_package_versions": list(self.supported_package_versions),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorPackageResolutionReport:
    """Inspectable metadata/callable result; no capability claim is implied."""

    compatibility_fingerprint: str
    loader_id: str
    provider_id: str
    operation_id: str
    package_name: str
    observed_package_version: Optional[str]
    stage: str
    callable_loaded: bool
    callable_observation_fingerprint: Optional[str]
    accepted: bool
    reasons: Tuple[str, ...]
    schema_version: int = ATTENTION_OPERATOR_PACKAGE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PACKAGE_VERSION:
            raise SchemaError("unsupported Attention package resolution version")
        if len(self.compatibility_fingerprint) != 64 or any(
            item not in "0123456789abcdef"
            for item in self.compatibility_fingerprint
        ):
            raise SchemaError("package compatibility fingerprint must be SHA-256")
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid package resolution provider_id")
        for name in ("loader_id", "operation_id", "package_name"):
            if not str(getattr(self, name)):
                raise SchemaError("package resolution %s must be non-empty" % name)
        if self.observed_package_version is not None and not str(
            self.observed_package_version
        ):
            raise SchemaError("observed package version cannot be empty")
        if self.stage not in {"metadata", "callable"}:
            raise SchemaError("unknown package resolution stage")
        observation = self.callable_observation_fingerprint
        if observation is not None and (
            len(observation) != 64
            or any(item not in "0123456789abcdef" for item in observation)
        ):
            raise SchemaError("callable observation fingerprint must be SHA-256")
        if self.stage == "metadata" and (
            self.callable_loaded or observation is not None
        ):
            raise SchemaError("metadata report cannot claim a loaded callable")
        if observation is not None and not self.callable_loaded:
            raise SchemaError("callable observation requires a loaded callable")
        reasons = tuple(str(item) for item in self.reasons)
        if any(not item for item in reasons) or len(set(reasons)) != len(reasons):
            raise SchemaError("package resolution reasons must be unique")
        if self.accepted == bool(reasons):
            raise SchemaError("accepted package resolution must have no reasons")
        object.__setattr__(self, "loader_id", str(self.loader_id))
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(self, "package_name", str(self.package_name))
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "loader_id": self.loader_id,
            "provider_id": self.provider_id,
            "operation_id": self.operation_id,
            "package_name": self.package_name,
            "observed_package_version": self.observed_package_version,
            "stage": self.stage,
            "callable_loaded": self.callable_loaded,
            "callable_observation_fingerprint": (
                self.callable_observation_fingerprint
            ),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class AttentionOperatorPackageResolutionError(RuntimeError):
    def __init__(
        self, message: str, report: AttentionOperatorPackageResolutionReport
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class AttentionResolvedOperatorPackage:
    """Unexecuted callable candidate ready for final plan-time authorization."""

    operation: AttentionOperatorOperationSpec
    provider_probe: AttentionOperatorProviderProbe
    observation: AttentionObservedOperatorCallable
    callable_binding: AttentionOperatorCallableBinding
    executor: AttentionInjectedCallableExecutor = field(repr=False, compare=False)
    report: AttentionOperatorPackageResolutionReport = field(
        repr=False, compare=False
    )
    schema_version: int = ATTENTION_OPERATOR_PACKAGE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PACKAGE_VERSION:
            raise SchemaError("unsupported resolved Attention package version")
        if not isinstance(self.operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if not isinstance(self.provider_probe, AttentionOperatorProviderProbe):
            raise TypeError("provider_probe must be AttentionOperatorProviderProbe")
        if not isinstance(self.observation, AttentionObservedOperatorCallable):
            raise TypeError("observation must be AttentionObservedOperatorCallable")
        if not isinstance(self.callable_binding, AttentionOperatorCallableBinding):
            raise TypeError("callable_binding must be AttentionOperatorCallableBinding")
        if not isinstance(self.executor, AttentionInjectedCallableExecutor):
            raise TypeError("executor must be AttentionInjectedCallableExecutor")
        if not isinstance(self.report, AttentionOperatorPackageResolutionReport):
            raise TypeError("report must be AttentionOperatorPackageResolutionReport")
        if not self.report.accepted or self.report.stage != "callable":
            raise SchemaError("resolved package requires an accepted callable report")
        if self.executor.is_runtime_bound:
            raise SchemaError("resolved package executor cannot pre-bind a runtime")
        if (
            self.operation.provider_id != self.provider_probe.provider_id
            or self.operation.provider_id != self.observation.provider_id
            or self.operation.provider_id != self.callable_binding.provider_id
            or self.operation.operation_id != self.callable_binding.operation_id
            or self.operation.operation_id != self.executor.operation_id
            or self.operation.fingerprint
            != self.callable_binding.operation_fingerprint
            or self.provider_probe.fingerprint
            != self.callable_binding.provider_probe_fingerprint
            or self.observation.fingerprint
            != self.callable_binding.observation_fingerprint
        ):
            raise SchemaError("resolved package authority chain is stale")


class AttentionOperatorPackageResolver:
    """Resolve one declared package/API without treating import as capability."""

    def __init__(
        self,
        operation_catalog: AttentionOperatorOperationCatalog,
        compatibility: AttentionOperatorPackageCompatibility,
        loader: AttentionOperatorPackageLoader,
    ) -> None:
        if not isinstance(operation_catalog, AttentionOperatorOperationCatalog):
            raise TypeError("operation_catalog must be AttentionOperatorOperationCatalog")
        if not isinstance(compatibility, AttentionOperatorPackageCompatibility):
            raise TypeError(
                "compatibility must be AttentionOperatorPackageCompatibility"
            )
        if not isinstance(loader, AttentionOperatorPackageLoader):
            raise TypeError("loader must implement AttentionOperatorPackageLoader")
        if not str(loader.loader_id):
            raise SchemaError("Attention package loader_id must be non-empty")
        operation = operation_catalog.get(compatibility.operation_id)
        if operation.provider_id != compatibility.provider_id:
            raise SchemaError("package compatibility does not match catalog provider")
        self._operation = operation
        self._compatibility = compatibility
        self._loader = loader

    @property
    def operation(self) -> AttentionOperatorOperationSpec:
        return self._operation

    @property
    def compatibility(self) -> AttentionOperatorPackageCompatibility:
        return self._compatibility

    def _report(
        self,
        *,
        observed_package_version,
        stage,
        callable_loaded,
        callable_observation_fingerprint,
        reasons,
    ) -> AttentionOperatorPackageResolutionReport:
        reasons = tuple(str(item) for item in reasons)
        return AttentionOperatorPackageResolutionReport(
            compatibility_fingerprint=self._compatibility.fingerprint,
            loader_id=self._loader.loader_id,
            provider_id=self._operation.provider_id,
            operation_id=self._operation.operation_id,
            package_name=self._operation.package_name,
            observed_package_version=observed_package_version,
            stage=stage,
            callable_loaded=callable_loaded,
            callable_observation_fingerprint=callable_observation_fingerprint,
            accepted=not reasons,
            reasons=reasons,
        )

    def explain(self) -> AttentionOperatorPackageResolutionReport:
        """Inspect distribution metadata only; never import a package."""

        observed_version = self._loader.package_version(
            self._operation.package_name
        )
        reasons = []
        if observed_version is None:
            reasons.append("package %s is not installed" % self._operation.package_name)
        elif str(observed_version) not in (
            self._compatibility.supported_package_versions
        ):
            reasons.append(
                "package %s version %s is not adapter-authorized"
                % (self._operation.package_name, observed_version)
            )
        return self._report(
            observed_package_version=(
                None if observed_version is None else str(observed_version)
            ),
            stage="metadata",
            callable_loaded=False,
            callable_observation_fingerprint=None,
            reasons=reasons,
        )

    def resolve(self) -> AttentionResolvedOperatorPackage:
        metadata_report = self.explain()
        if not metadata_report.accepted:
            raise AttentionOperatorPackageResolutionError(
                "Attention operator package is unavailable or incompatible: %s"
                % "; ".join(metadata_report.reasons),
                metadata_report,
            )
        callable_object = self._loader.resolve_callable(
            self._operation.callable_path
        )
        if not callable(callable_object):
            report = self._report(
                observed_package_version=metadata_report.observed_package_version,
                stage="callable",
                callable_loaded=True,
                callable_observation_fingerprint=None,
                reasons=("resolved package object is not callable",),
            )
            raise AttentionOperatorPackageResolutionError(
                "Attention package object is not callable", report
            )
        try:
            signature = observe_python_callable_signature(callable_object)
        except (TypeError, SchemaError) as error:
            report = self._report(
                observed_package_version=metadata_report.observed_package_version,
                stage="callable",
                callable_loaded=True,
                callable_observation_fingerprint=None,
                reasons=("resolved callable signature is not inspectable",),
            )
            raise AttentionOperatorPackageResolutionError(
                "Attention package callable signature is not inspectable", report
            ) from error
        observation = AttentionObservedOperatorCallable(
            provider_id=self._operation.provider_id,
            package_name=self._operation.package_name,
            package_version=metadata_report.observed_package_version,
            callable_path=self._operation.callable_path,
            api_version=self._operation.api_version,
            available=True,
            signature=signature,
        )
        probe = AttentionOperatorProviderProbe(
            provider_id=self._operation.provider_id,
            adapter_version=self._compatibility.adapter_version,
            available=True,
            package_versions=((self._operation.package_name, observation.package_version),),
        )
        callable_report = explain_attention_operator_callable(
            probe, self._operation, observation
        )
        if not callable_report.accepted:
            report = self._report(
                observed_package_version=metadata_report.observed_package_version,
                stage="callable",
                callable_loaded=True,
                callable_observation_fingerprint=observation.fingerprint,
                reasons=callable_report.reasons,
            )
            raise AttentionOperatorPackageResolutionError(
                "Attention package callable is incompatible: %s"
                % "; ".join(report.reasons),
                report,
            )
        binding = bind_attention_operator_callable(
            probe, self._operation, observation
        )
        executor = AttentionInjectedCallableExecutor(
            self._operation, binding, callable_object
        )
        report = self._report(
            observed_package_version=metadata_report.observed_package_version,
            stage="callable",
            callable_loaded=True,
            callable_observation_fingerprint=observation.fingerprint,
            reasons=(),
        )
        return AttentionResolvedOperatorPackage(
            operation=self._operation,
            provider_probe=probe,
            observation=observation,
            callable_binding=binding,
            executor=executor,
            report=report,
        )


__all__ = [
    "ATTENTION_OPERATOR_PACKAGE_VERSION",
    "AttentionOperatorPackageCompatibility",
    "AttentionOperatorPackageLoader",
    "AttentionOperatorPackageResolutionError",
    "AttentionOperatorPackageResolutionReport",
    "AttentionOperatorPackageResolver",
    "AttentionResolvedOperatorPackage",
    "ImportlibAttentionOperatorPackageLoader",
]
