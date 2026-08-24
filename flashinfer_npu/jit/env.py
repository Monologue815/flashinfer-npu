"""Pure Host contracts for the Ascend JIT environment and policy.

The module deliberately performs no environment probing.  A trusted runtime
adapter must provide the exact toolchain snapshot before a cache entry or a
future compilation request can be authorized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

from flashinfer_npu.runtime import SchemaError


JIT_ENVIRONMENT_SCHEMA_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class JitCompilationPolicy(str, Enum):
    """Whether a missing cache artifact may become a build request."""

    DISABLED = "disabled"
    CACHE_ONLY = "cache_only"
    ENABLED = "enabled"


@dataclass(frozen=True)
class JitEnvironment:
    """Fully explicit identity of an Ascend JIT toolchain target."""

    target_soc: str
    soc_revision: str
    cann_version: str
    compiler_id: str
    compiler_version: str
    torch_version: str
    torch_npu_version: str
    python_abi: str
    build_type: str = "release"
    features: Tuple[str, ...] = ()
    schema_version: int = JIT_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_ENVIRONMENT_SCHEMA_VERSION:
            raise SchemaError("unsupported JIT environment schema version")
        for name in (
            "target_soc",
            "soc_revision",
            "cann_version",
            "compiler_id",
            "compiler_version",
            "torch_version",
            "torch_npu_version",
            "python_abi",
        ):
            value = str(getattr(self, name))
            if not value or "\x00" in value or "\n" in value or "\r" in value:
                raise SchemaError(
                    "JIT environment %s must be a safe non-empty string" % name
                )
            object.__setattr__(self, name, value)
        if self.build_type not in {"release", "debug"}:
            raise SchemaError("JIT build_type must be release or debug")
        features = tuple(sorted(str(item) for item in self.features))
        if any(not item for item in features) or len(set(features)) != len(features):
            raise SchemaError(
                "JIT environment features must be unique non-empty strings"
            )
        object.__setattr__(self, "features", features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_soc": self.target_soc,
            "soc_revision": self.soc_revision,
            "cann_version": self.cann_version,
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "torch_version": self.torch_version,
            "torch_npu_version": self.torch_npu_version,
            "python_abi": self.python_abi,
            "build_type": self.build_type,
            "features": list(self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JitEnvironment":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("JitEnvironment fields are invalid")
        features = data.get("features")
        if not isinstance(features, (list, tuple)):
            raise SchemaError("JIT environment features must be an array")
        data["features"] = tuple(features)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("JitEnvironment fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "JIT_ENVIRONMENT_SCHEMA_VERSION",
    "JitCompilationPolicy",
    "JitEnvironment",
]
