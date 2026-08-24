"""Injected JIT module loading and exact exported-symbol receipts.

This module defines the framework boundary corresponding to FlashInfer's
``JitSpec.load`` result.  It deliberately installs no dynamic-library loader
and never calls an exported symbol.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Tuple, runtime_checkable

from flashinfer_npu.runtime import SchemaError

from .artifacts import JitArtifactVerification
from .cache import JitCacheRecord


JIT_MODULE_LOAD_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("JIT loaded module %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class JitResolvedSymbol:
    """Opaque symbol identity; never a callable or process address."""

    name: str
    symbol_token: str
    schema_version: int = JIT_MODULE_LOAD_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_MODULE_LOAD_VERSION:
            raise SchemaError("unsupported JIT resolved symbol version")
        if not _SYMBOL.fullmatch(str(self.name)):
            raise SchemaError("invalid JIT resolved symbol name")
        if not str(self.symbol_token) or any(
            item in str(self.symbol_token) for item in ("\x00", "\n", "\r")
        ):
            raise SchemaError("invalid JIT resolved symbol token")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "symbol_token", str(self.symbol_token))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "symbol_token": self.symbol_token,
        }


@dataclass(frozen=True)
class JitModuleLoadReceipt:
    """Identity of one loaded artifact and its exact exported symbols."""

    spec_name: str
    spec_fingerprint: str
    cache_record_fingerprint: str
    artifact_verification_fingerprint: str
    artifact_fingerprint: str
    loader_id: str
    loader_version: str
    module_token: str
    load_generation: int
    symbols: Tuple[JitResolvedSymbol, ...]
    schema_version: int = JIT_MODULE_LOAD_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_MODULE_LOAD_VERSION:
            raise SchemaError("unsupported JIT module load receipt version")
        if not str(self.spec_name):
            raise SchemaError("JIT loaded module spec_name must be non-empty")
        if not _IDENTIFIER.fullmatch(str(self.loader_id)):
            raise SchemaError("invalid JIT module loader_id")
        for name in ("loader_version", "module_token"):
            value = str(getattr(self, name))
            if not value or any(item in value for item in ("\x00", "\n", "\r")):
                raise SchemaError("JIT module %s must be safe and non-empty" % name)
            object.__setattr__(self, name, value)
        for name in (
            "spec_fingerprint",
            "cache_record_fingerprint",
            "artifact_verification_fingerprint",
            "artifact_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if (
            not isinstance(self.load_generation, int)
            or isinstance(self.load_generation, bool)
            or self.load_generation < 1
        ):
            raise SchemaError("JIT module load generation must be positive")
        symbols = tuple(self.symbols)
        if not symbols or any(not isinstance(item, JitResolvedSymbol) for item in symbols):
            raise SchemaError("JIT loaded module must contain resolved symbols")
        names = tuple(item.name for item in symbols)
        if len(set(names)) != len(names):
            raise SchemaError("JIT loaded module symbols must be unique")
        object.__setattr__(self, "spec_name", str(self.spec_name))
        object.__setattr__(self, "loader_id", str(self.loader_id))
        object.__setattr__(self, "symbols", tuple(sorted(symbols, key=lambda item: item.name)))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "cache_record_fingerprint": self.cache_record_fingerprint,
            "artifact_verification_fingerprint": self.artifact_verification_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "loader_id": self.loader_id,
            "loader_version": self.loader_version,
            "module_token": self.module_token,
            "load_generation": self.load_generation,
            "symbols": [item.to_dict() for item in self.symbols],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate(
        self,
        record: JitCacheRecord,
        verification: JitArtifactVerification,
        required_entry_points: Tuple[str, ...],
    ) -> None:
        if not isinstance(record, JitCacheRecord):
            raise TypeError("record must be JitCacheRecord")
        if not isinstance(verification, JitArtifactVerification):
            raise TypeError("verification must be JitArtifactVerification")
        verification.validate_record(record)
        required = tuple(str(item) for item in required_entry_points)
        if not required or any(not _SYMBOL.fullmatch(item) for item in required):
            raise SchemaError("required JIT entry points are invalid")
        if len(set(required)) != len(required):
            raise SchemaError("required JIT entry points must be unique")
        if (
            self.spec_name != record.spec_name
            or self.spec_fingerprint != record.spec_fingerprint
            or self.cache_record_fingerprint != record.fingerprint
            or self.artifact_verification_fingerprint != verification.fingerprint
            or self.artifact_fingerprint != record.artifact.fingerprint
            or tuple(item.name for item in self.symbols) != tuple(sorted(required))
        ):
            raise SchemaError("JIT loaded module does not bind artifact and symbols")


@dataclass(frozen=True)
class JitLoadedModule:
    """Validated receipt plus an opaque loader-owned module object."""

    receipt: JitModuleLoadReceipt
    opaque_module: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, JitModuleLoadReceipt):
            raise TypeError("receipt must be JitModuleLoadReceipt")
        if self.opaque_module is None:
            raise SchemaError("JIT loaded module opaque object must be present")

    @property
    def fingerprint(self) -> str:
        return self.receipt.fingerprint


@runtime_checkable
class JitModuleLoader(Protocol):
    """Private module loader supplied by an authorized runtime integration."""

    loader_id: str
    loader_version: str

    def load(
        self,
        record: JitCacheRecord,
        verification: JitArtifactVerification,
        required_entry_points: Tuple[str, ...],
    ) -> JitLoadedModule:
        """Load and resolve exact symbols; never called by the model user."""


__all__ = [
    "JIT_MODULE_LOAD_VERSION",
    "JitLoadedModule",
    "JitModuleLoadReceipt",
    "JitModuleLoader",
    "JitResolvedSymbol",
]
