"""Versioned external Attention operation signatures.

This catalog records documented Python call surfaces only.  It is not runtime
capability evidence and cannot by itself authorize dispatch or operator use.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from flashinfer_npu.runtime import SchemaError

from .operator_plan import AttentionOperatorActivePlan
from .schema import AttentionMode


ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION = 2

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OPERATION_ID = re.compile(r"^[A-Za-z0-9_.@-]+$")
_CALLABLE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


def _names(name: str, values) -> Tuple[str, ...]:
    try:
        result = tuple(str(item) for item in values)
    except TypeError as error:
        raise SchemaError("%s must be a sequence" % name) from error
    if any(not _ARGUMENT_NAME.fullmatch(item) for item in result):
        raise SchemaError("%s contains an invalid name" % name)
    if len(set(result)) != len(result):
        raise SchemaError("%s must contain unique names" % name)
    return result


@dataclass(frozen=True)
class AttentionOperatorOperationSpec:
    """Exact documented signature for one provider API variant."""

    operation_id: str
    provider_id: str
    package_name: str
    callable_path: str
    api_version: str
    candidate_modes: Tuple[AttentionMode, ...]
    positional_arguments: Tuple[str, ...]
    keyword_arguments: Tuple[str, ...]
    return_names: Tuple[str, ...]
    mutable_arguments: Tuple[str, ...] = ()
    host_sequence_arguments: Tuple[str, ...] = ()
    quant_arguments: Tuple[str, ...] = ()
    paged_table_argument: Optional[str] = None
    lse_control_argument: Optional[str] = None
    output_buffer_argument: Optional[str] = None
    lse_buffer_argument: Optional[str] = None
    source_url: str = ""
    schema_version: int = ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION:
            raise SchemaError("unsupported Attention operation catalog version")
        operation_id = str(self.operation_id)
        provider_id = str(self.provider_id)
        if not _OPERATION_ID.fullmatch(operation_id):
            raise SchemaError("invalid Attention operation_id")
        if not _PROVIDER_ID.fullmatch(provider_id):
            raise SchemaError("invalid Attention operation provider_id")
        if not str(self.package_name) or not str(self.api_version):
            raise SchemaError("operation package_name and api_version must be non-empty")
        if not _CALLABLE_PATH.fullmatch(str(self.callable_path)):
            raise SchemaError("invalid Attention operation callable_path")
        modes = tuple(AttentionMode(item) for item in self.candidate_modes)
        if not modes or len(set(modes)) != len(modes):
            raise SchemaError("operation candidate_modes must be non-empty and unique")
        positional = _names("positional_arguments", self.positional_arguments)
        keyword = _names("keyword_arguments", self.keyword_arguments)
        if not positional:
            raise SchemaError("operation must declare positional arguments")
        if set(positional).intersection(keyword):
            raise SchemaError("operation positional and keyword arguments overlap")
        all_arguments = set(positional).union(keyword)
        returns = _names("return_names", self.return_names)
        if not returns:
            raise SchemaError("operation must declare at least one return name")
        mutable = _names("mutable_arguments", self.mutable_arguments)
        host_sequence = _names(
            "host_sequence_arguments", self.host_sequence_arguments
        )
        quant = _names("quant_arguments", self.quant_arguments)
        for name, values in (
            ("mutable_arguments", mutable),
            ("host_sequence_arguments", host_sequence),
            ("quant_arguments", quant),
        ):
            if not set(values).issubset(all_arguments):
                raise SchemaError("%s must name declared arguments" % name)
        for name in (
            "paged_table_argument",
            "lse_control_argument",
            "output_buffer_argument",
            "lse_buffer_argument",
        ):
            value = getattr(self, name)
            if value is not None and value not in all_arguments:
                raise SchemaError("%s must name a declared argument" % name)
        buffer_arguments = tuple(
            value
            for value in (
                self.output_buffer_argument,
                self.lse_buffer_argument,
            )
            if value is not None
        )
        if len(set(buffer_arguments)) != len(buffer_arguments):
            raise SchemaError("output and LSE buffers must use different arguments")
        if not set(buffer_arguments).issubset(mutable):
            raise SchemaError("caller-owned buffer arguments must be mutable")
        if not set(buffer_arguments).issubset(keyword):
            raise SchemaError("caller-owned buffers must be keyword arguments")
        if self.output_buffer_argument is not None and "output" not in returns:
            raise SchemaError("output buffer operation must return output")
        if self.lse_buffer_argument is not None and "softmax_lse" not in returns:
            raise SchemaError("LSE buffer operation must return softmax_lse")
        if not str(self.source_url).startswith("https://"):
            raise SchemaError("operation source_url must be HTTPS")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "package_name", str(self.package_name))
        object.__setattr__(self, "callable_path", str(self.callable_path))
        object.__setattr__(self, "api_version", str(self.api_version))
        object.__setattr__(self, "candidate_modes", modes)
        object.__setattr__(self, "positional_arguments", positional)
        object.__setattr__(self, "keyword_arguments", keyword)
        object.__setattr__(self, "return_names", returns)
        object.__setattr__(self, "mutable_arguments", mutable)
        object.__setattr__(self, "host_sequence_arguments", host_sequence)
        object.__setattr__(self, "quant_arguments", quant)
        object.__setattr__(self, "source_url", str(self.source_url))

    @property
    def supports_lse(self) -> bool:
        return "softmax_lse" in self.return_names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "provider_id": self.provider_id,
            "package_name": self.package_name,
            "callable_path": self.callable_path,
            "api_version": self.api_version,
            "candidate_modes": [item.value for item in self.candidate_modes],
            "positional_arguments": list(self.positional_arguments),
            "keyword_arguments": list(self.keyword_arguments),
            "return_names": list(self.return_names),
            "mutable_arguments": list(self.mutable_arguments),
            "host_sequence_arguments": list(self.host_sequence_arguments),
            "quant_arguments": list(self.quant_arguments),
            "paged_table_argument": self.paged_table_argument,
            "lse_control_argument": self.lse_control_argument,
            "output_buffer_argument": self.output_buffer_argument,
            "lse_buffer_argument": self.lse_buffer_argument,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionOperatorOperationSpec":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention operation spec fields are invalid")
        try:
            for name in (
                "candidate_modes",
                "positional_arguments",
                "keyword_arguments",
                "return_names",
                "mutable_arguments",
                "host_sequence_arguments",
                "quant_arguments",
            ):
                data[name] = tuple(data[name])
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention operation spec fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorOperationCatalog:
    """Strict collection of external operation signatures."""

    name: str
    operations: Tuple[AttentionOperatorOperationSpec, ...]
    schema_version: int = ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION:
            raise SchemaError("unsupported Attention operation catalog version")
        if not str(self.name):
            raise SchemaError("Attention operation catalog name must be non-empty")
        operations = tuple(self.operations)
        if not operations or any(
            not isinstance(item, AttentionOperatorOperationSpec)
            for item in operations
        ):
            raise SchemaError("Attention operation catalog must contain operation specs")
        operation_ids = [item.operation_id for item in operations]
        if len(set(operation_ids)) != len(operation_ids):
            raise SchemaError("Attention operation catalog has duplicate operation_id")
        signatures = [
            (item.provider_id, item.callable_path, item.api_version)
            for item in operations
        ]
        if len(set(signatures)) != len(signatures):
            raise SchemaError("Attention operation catalog has duplicate API signature")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self, "operations", tuple(sorted(operations, key=lambda item: item.operation_id))
        )

    @property
    def operation_ids(self) -> Tuple[str, ...]:
        return tuple(item.operation_id for item in self.operations)

    def get(self, operation_id: str) -> AttentionOperatorOperationSpec:
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        raise SchemaError("unknown Attention operation_id %r" % operation_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "operations": [item.to_dict() for item in self.operations],
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AttentionOperatorOperationCatalog":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("Attention operation catalog fields are invalid")
        try:
            data["operations"] = tuple(
                AttentionOperatorOperationSpec.from_dict(item)
                for item in data["operations"]
            )
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("Attention operation catalog fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AttentionOperatorOperationBinding:
    """Identity link from an active provider plan to a catalog operation."""

    active_plan_fingerprint: str
    provider_id: str
    mode: AttentionMode
    operation_id: str
    operation_fingerprint: str
    api_version: str
    schema_version: int = ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION:
            raise SchemaError("unsupported Attention operation binding version")
        _require_hash("active_plan_fingerprint", self.active_plan_fingerprint)
        _require_hash("operation_fingerprint", self.operation_fingerprint)
        if not _PROVIDER_ID.fullmatch(str(self.provider_id)):
            raise SchemaError("invalid operation binding provider_id")
        object.__setattr__(self, "mode", AttentionMode(self.mode))
        if not _OPERATION_ID.fullmatch(str(self.operation_id)):
            raise SchemaError("invalid operation binding operation_id")
        if not str(self.api_version):
            raise SchemaError("operation binding api_version must be non-empty")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "active_plan_fingerprint": self.active_plan_fingerprint,
                "provider_id": self.provider_id,
                "mode": self.mode.value,
                "operation_id": self.operation_id,
                "operation_fingerprint": self.operation_fingerprint,
                "api_version": self.api_version,
            }
        )


def bind_attention_operator_operation(
    catalog: AttentionOperatorOperationCatalog,
    active_plan: AttentionOperatorActivePlan,
) -> AttentionOperatorOperationBinding:
    """Bind the plan-frozen implementation id to one exact catalog signature."""

    if not isinstance(catalog, AttentionOperatorOperationCatalog):
        raise TypeError("catalog must be AttentionOperatorOperationCatalog")
    if not isinstance(active_plan, AttentionOperatorActivePlan):
        raise TypeError("active_plan must be AttentionOperatorActivePlan")
    operation = catalog.get(active_plan.prepared_plan.implementation_id)
    provider_id = active_plan.provider_selection.provider_id
    if operation.provider_id != provider_id:
        raise SchemaError("catalog operation does not match the active provider")
    mode = active_plan.framework_plan.spec.mode
    if mode not in operation.candidate_modes:
        raise SchemaError("catalog operation is not a candidate for the planned mode")
    return AttentionOperatorOperationBinding(
        active_plan_fingerprint=active_plan.fingerprint,
        provider_id=provider_id,
        mode=mode,
        operation_id=operation.operation_id,
        operation_fingerprint=operation.fingerprint,
        api_version=operation.api_version,
    )


def packaged_attention_operator_catalog_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "attention_operator_operations.json"
    )


def load_attention_operator_operation_catalog(
    path: Union[str, Path],
) -> AttentionOperatorOperationCatalog:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise SchemaError("Attention operation catalog is unreadable") from error
    if not isinstance(value, Mapping):
        raise SchemaError("Attention operation catalog root must be an object")
    return AttentionOperatorOperationCatalog.from_dict(value)


def load_packaged_attention_operator_catalog() -> AttentionOperatorOperationCatalog:
    return load_attention_operator_operation_catalog(
        packaged_attention_operator_catalog_path()
    )


__all__ = [
    "ATTENTION_OPERATOR_OPERATION_CATALOG_VERSION",
    "AttentionOperatorOperationBinding",
    "AttentionOperatorOperationCatalog",
    "AttentionOperatorOperationSpec",
    "bind_attention_operator_operation",
    "load_attention_operator_operation_catalog",
    "load_packaged_attention_operator_catalog",
    "packaged_attention_operator_catalog_path",
]
