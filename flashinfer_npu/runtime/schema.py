"""Versioned, framework-independent schemas used by the runtime.

These classes intentionally contain no torch or torch_npu dependency. They are
the host-side representation of the stable runtime contracts described in the
architecture document.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a versioned runtime schema is invalid."""


class Backend(str, Enum):
    ASCENDC_AOT = "ascendc_aot"
    ASCENDC_JIT = "ascendc_jit"
    ACLNN = "aclnn"
    REFERENCE = "reference"


class ArtifactKind(str, Enum):
    FILE = "file"
    BUILTIN = "builtin"
    JIT_SOURCE = "jit_source"


class ArtifactFormat(str, Enum):
    ASCENDC_OBJECT = "ascendc_object"
    SHARED_LIBRARY = "shared_library"
    ACLNN_BUILTIN = "aclnn_builtin"
    ASCENDC_SOURCE = "ascendc_source"


class ArtifactVerificationError(RuntimeError):
    """Raised when artifact bytes do not match declared provenance."""


class CPrimitive(str, Enum):
    U8 = "u8"
    I8 = "i8"
    U16 = "u16"
    I16 = "i16"
    U32 = "u32"
    I32 = "i32"
    U64 = "u64"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"


_C_PRIMITIVE_LAYOUT = {
    CPrimitive.U8: (1, "B"),
    CPrimitive.I8: (1, "b"),
    CPrimitive.U16: (2, "H"),
    CPrimitive.I16: (2, "h"),
    CPrimitive.U32: (4, "I"),
    CPrimitive.I32: (4, "i"),
    CPrimitive.U64: (8, "Q"),
    CPrimitive.I64: (8, "q"),
    CPrimitive.F32: (4, "f"),
    CPrimitive.F64: (8, "d"),
}


class KernelArgumentPassing(str, Enum):
    POINTER = "pointer"
    VALUE = "value"
    OPAQUE_HANDLE = "opaque_handle"


class KernelArgumentDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


def _tuple_of_str(name: str, values: Iterable[str]) -> Tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise SchemaError("%s cannot contain empty values" % name)
    return result


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable identity and provenance for a kernel artifact/provider."""

    kind: ArtifactKind
    format: ArtifactFormat
    locator: str
    digest: str
    target_soc: str
    build_id: str
    size_bytes: Optional[int] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported ArtifactRef schema version")
        object.__setattr__(self, "kind", ArtifactKind(self.kind))
        object.__setattr__(self, "format", ArtifactFormat(self.format))
        for name in ("locator", "target_soc", "build_id"):
            if not str(getattr(self, name)):
                raise SchemaError("artifact %s must be non-empty" % name)
        _require_sha256("artifact digest", self.digest)
        format_kinds = {
            ArtifactFormat.ASCENDC_OBJECT: ArtifactKind.FILE,
            ArtifactFormat.SHARED_LIBRARY: ArtifactKind.FILE,
            ArtifactFormat.ACLNN_BUILTIN: ArtifactKind.BUILTIN,
            ArtifactFormat.ASCENDC_SOURCE: ArtifactKind.JIT_SOURCE,
        }
        if format_kinds[self.format] != self.kind:
            raise SchemaError("artifact kind does not match artifact format")
        if self.kind == ArtifactKind.BUILTIN:
            if self.size_bytes is not None:
                raise SchemaError("builtin artifact cannot declare byte size")
            if not self.locator.startswith("builtin:"):
                raise SchemaError("builtin artifact locator must start with builtin:")
        else:
            if (
                not isinstance(self.size_bytes, int)
                or isinstance(self.size_bytes, bool)
                or self.size_bytes < 0
            ):
                raise SchemaError("file/source artifact requires non-negative size")
            path = PurePosixPath(self.locator)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "." in path.parts
                or "\\" in self.locator
                or str(path) != self.locator
            ):
                raise SchemaError("artifact locator must be a normalized relative path")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "format": self.format.value,
            "locator": self.locator,
            "digest": self.digest,
            "target_soc": self.target_soc,
            "build_id": self.build_id,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("ArtifactRef fields do not match schema version 1")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("ArtifactRef fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def verify_bytes(self, payload: bytes) -> None:
        if self.kind == ArtifactKind.BUILTIN:
            raise ArtifactVerificationError(
                "builtin provider identity cannot be verified as file bytes"
            )
        if not isinstance(payload, bytes):
            raise TypeError("artifact payload must be bytes")
        if len(payload) != self.size_bytes:
            raise ArtifactVerificationError("artifact byte size mismatch")
        if hashlib.sha256(payload).hexdigest() != self.digest:
            raise ArtifactVerificationError("artifact digest mismatch")

    def verify_file(self, package_root: Path) -> Path:
        if self.kind != ArtifactKind.FILE:
            raise ArtifactVerificationError("artifact is not a file artifact")
        root = Path(package_root).resolve()
        candidate = (root / self.locator).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ArtifactVerificationError("artifact path escapes package root") from error
        try:
            payload = candidate.read_bytes()
        except OSError as error:
            raise ArtifactVerificationError("artifact file is unreadable") from error
        self.verify_bytes(payload)
        return candidate


@dataclass(frozen=True)
class KernelLaunchABI:
    """Framework-independent, versioned kernel entry-point contract."""

    abi_name: str
    entry_point: str
    argument_names: Tuple[str, ...]
    mutable_arguments: Tuple[str, ...]
    stream_argument: str
    pointer_width_bits: int = 64
    metadata_schema_version: int = SCHEMA_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported KernelLaunchABI schema version")
        for name in ("abi_name", "entry_point", "stream_argument"):
            if not str(getattr(self, name)) or any(
                character.isspace() for character in str(getattr(self, name))
            ):
                raise SchemaError("launch ABI %s must be non-empty without spaces" % name)
        arguments = _tuple_of_str("argument_names", self.argument_names)
        mutable = _tuple_of_str("mutable_arguments", self.mutable_arguments)
        if not arguments or len(set(arguments)) != len(arguments):
            raise SchemaError("launch ABI arguments must be non-empty and unique")
        if len(set(mutable)) != len(mutable) or not set(mutable) <= set(arguments):
            raise SchemaError("mutable arguments must be a unique argument subset")
        if self.stream_argument not in arguments:
            raise SchemaError("stream argument must occur in argument_names")
        if self.pointer_width_bits not in {32, 64}:
            raise SchemaError("pointer_width_bits must be 32 or 64")
        if self.metadata_schema_version <= 0:
            raise SchemaError("metadata_schema_version must be positive")
        object.__setattr__(self, "argument_names", arguments)
        object.__setattr__(self, "mutable_arguments", mutable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "abi_name": self.abi_name,
            "entry_point": self.entry_point,
            "argument_names": list(self.argument_names),
            "mutable_arguments": list(self.mutable_arguments),
            "stream_argument": self.stream_argument,
            "pointer_width_bits": self.pointer_width_bits,
            "metadata_schema_version": self.metadata_schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelLaunchABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("KernelLaunchABI fields do not match schema version 1")
        for name in ("argument_names", "mutable_arguments"):
            if not isinstance(data.get(name), (list, tuple)):
                raise SchemaError("KernelLaunchABI %s must be an array" % name)
            data[name] = tuple(data[name])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("KernelLaunchABI fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CFieldABI:
    name: str
    primitive: CPrimitive
    count: int = 1
    reserved: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported CFieldABI schema version")
        if not self.name or any(character.isspace() for character in self.name):
            raise SchemaError("C ABI field name must be non-empty without spaces")
        object.__setattr__(self, "primitive", CPrimitive(self.primitive))
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise SchemaError("C ABI field count must be positive")
        if not isinstance(self.reserved, bool):
            raise SchemaError("C ABI field reserved must be boolean")

    @property
    def element_size(self) -> int:
        return _C_PRIMITIVE_LAYOUT[self.primitive][0]

    @property
    def size_bytes(self) -> int:
        return self.element_size * self.count

    @property
    def alignment(self) -> int:
        return self.element_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "primitive": self.primitive.value,
            "count": self.count,
            "reserved": self.reserved,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CFieldABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("CFieldABI fields do not match schema version 1")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("CFieldABI fields are invalid") from error


@dataclass(frozen=True)
class CStructABI:
    name: str
    fields: Tuple[CFieldABI, ...]
    alignment: int = 8
    endianness: str = "little"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported CStructABI schema version")
        if not self.name or any(character.isspace() for character in self.name):
            raise SchemaError("C struct name must be non-empty without spaces")
        values = tuple(self.fields)
        if not values or len({item.name for item in values}) != len(values):
            raise SchemaError("C struct fields must be non-empty with unique names")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise SchemaError("C struct alignment must be a power of two")
        if self.alignment < max(item.alignment for item in values):
            raise SchemaError("C struct alignment is smaller than a field alignment")
        if self.endianness != "little":
            raise SchemaError("C ABI v1 supports little-endian layout only")
        object.__setattr__(self, "fields", values)

    @property
    def field_offsets(self) -> Tuple[Tuple[str, int], ...]:
        offset = 0
        result = []
        for field_value in self.fields:
            alignment = field_value.alignment
            offset = (offset + alignment - 1) & ~(alignment - 1)
            result.append((field_value.name, offset))
            offset += field_value.size_bytes
        return tuple(result)

    @property
    def size_bytes(self) -> int:
        offsets = dict(self.field_offsets)
        last = self.fields[-1]
        raw = offsets[last.name] + last.size_bytes
        return (raw + self.alignment - 1) & ~(self.alignment - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "fields": [item.to_dict() for item in self.fields],
            "alignment": self.alignment,
            "endianness": self.endianness,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CStructABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("CStructABI fields do not match schema version 1")
        fields = data.get("fields")
        if not isinstance(fields, (list, tuple)):
            raise SchemaError("CStructABI fields must be an array")
        data["fields"] = tuple(CFieldABI.from_dict(item) for item in fields)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("CStructABI fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def pack(self, values: Mapping[str, Any]) -> bytes:
        expected = {item.name for item in self.fields if not item.reserved}
        if set(values) != expected:
            raise SchemaError("C struct values do not match non-reserved fields")
        result = bytearray(self.size_bytes)
        offsets = dict(self.field_offsets)
        for field_value in self.fields:
            value = 0 if field_value.reserved else values[field_value.name]
            if field_value.reserved:
                items = (0,) * field_value.count
            elif field_value.count == 1:
                items = (value,)
            else:
                if not isinstance(value, (list, tuple)) or len(value) != field_value.count:
                    raise SchemaError(
                        "C struct field %s requires %d values"
                        % (field_value.name, field_value.count)
                    )
                items = tuple(value)
            fmt = "<%d%s" % (
                field_value.count,
                _C_PRIMITIVE_LAYOUT[field_value.primitive][1],
            )
            try:
                encoded = struct.pack(fmt, *items)
            except (struct.error, TypeError, ValueError) as error:
                raise SchemaError(
                    "C struct field %s value is out of range" % field_value.name
                ) from error
            start = offsets[field_value.name]
            result[start : start + len(encoded)] = encoded
        return bytes(result)

    def unpack(self, payload: bytes) -> Dict[str, Any]:
        if not isinstance(payload, bytes) or len(payload) != self.size_bytes:
            raise SchemaError("C struct payload has the wrong byte size")
        offsets = dict(self.field_offsets)
        result: Dict[str, Any] = {}
        for field_value in self.fields:
            fmt = "<%d%s" % (
                field_value.count,
                _C_PRIMITIVE_LAYOUT[field_value.primitive][1],
            )
            values = struct.unpack_from(fmt, payload, offsets[field_value.name])
            if field_value.reserved and any(value != 0 for value in values):
                raise SchemaError("C struct reserved field must be zero")
            result[field_value.name] = values[0] if field_value.count == 1 else values
        return result


@dataclass(frozen=True)
class KernelErrorCodeABI:
    name: str
    code: int
    retryable: bool = False
    asynchronous: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported KernelErrorCodeABI schema version")
        if not self.name or any(character.isspace() for character in self.name):
            raise SchemaError("kernel error name must be non-empty without spaces")
        if not isinstance(self.code, int) or isinstance(self.code, bool) or self.code < 0:
            raise SchemaError("kernel error code must be a non-negative integer")
        if not isinstance(self.retryable, bool) or not isinstance(self.asynchronous, bool):
            raise SchemaError("kernel error flags must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelErrorCodeABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("KernelErrorCodeABI fields are invalid")
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("KernelErrorCodeABI fields are invalid") from error


@dataclass(frozen=True)
class KernelErrorABI:
    name: str
    codes: Tuple[KernelErrorCodeABI, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported KernelErrorABI schema version")
        values = tuple(self.codes)
        if not self.name or not values:
            raise SchemaError("kernel error ABI name/codes must be non-empty")
        if len({item.name for item in values}) != len(values) or len(
            {item.code for item in values}
        ) != len(values):
            raise SchemaError("kernel error names and codes must be unique")
        success = tuple(item for item in values if item.code == 0)
        if len(success) != 1 or success[0].name != "success":
            raise SchemaError("kernel error ABI requires success=0")
        object.__setattr__(self, "codes", values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "codes": [item.to_dict() for item in self.codes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelErrorABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("KernelErrorABI fields are invalid")
        codes = data.get("codes")
        if not isinstance(codes, (list, tuple)):
            raise SchemaError("KernelErrorABI codes must be an array")
        data["codes"] = tuple(KernelErrorCodeABI.from_dict(item) for item in codes)
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("KernelErrorABI fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class KernelArgumentABI:
    name: str
    passing: KernelArgumentPassing
    direction: KernelArgumentDirection
    primitive: CPrimitive = CPrimitive.U64
    nullable: bool = False
    required_alignment: int = 1
    pointee_abi_name: Optional[str] = None
    pointee_abi_fingerprint: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported KernelArgumentABI schema version")
        if not self.name or any(character.isspace() for character in self.name):
            raise SchemaError("kernel argument name must be non-empty without spaces")
        object.__setattr__(self, "passing", KernelArgumentPassing(self.passing))
        object.__setattr__(self, "direction", KernelArgumentDirection(self.direction))
        object.__setattr__(self, "primitive", CPrimitive(self.primitive))
        if not isinstance(self.nullable, bool):
            raise SchemaError("kernel argument nullable must be boolean")
        if self.required_alignment <= 0 or self.required_alignment & (
            self.required_alignment - 1
        ):
            raise SchemaError("kernel argument alignment must be a power of two")
        pointee = (self.pointee_abi_name, self.pointee_abi_fingerprint)
        if self.passing == KernelArgumentPassing.POINTER:
            if (pointee[0] is None) != (pointee[1] is None):
                raise SchemaError("pointee ABI name/fingerprint become present together")
            if pointee[1] is not None:
                _require_sha256("pointee ABI fingerprint", pointee[1])
            if self.primitive != CPrimitive.U64:
                raise SchemaError("C ABI pointers must use u64 storage")
        elif any(item is not None for item in pointee):
            raise SchemaError("non-pointer argument cannot declare pointee ABI")
        if self.passing != KernelArgumentPassing.POINTER and self.nullable:
            raise SchemaError("only pointer arguments can be nullable")
        if self.passing == KernelArgumentPassing.VALUE and self.direction != (
            KernelArgumentDirection.INPUT
        ):
            raise SchemaError("value arguments must be input-only")
        if self.passing == KernelArgumentPassing.OPAQUE_HANDLE and (
            self.primitive != CPrimitive.U64 or self.nullable
        ):
            raise SchemaError("opaque handle must be a non-null u64")

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["passing"] = self.passing.value
        result["direction"] = self.direction.value
        result["primitive"] = self.primitive.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelArgumentABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("KernelArgumentABI fields are invalid")
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("KernelArgumentABI fields are invalid") from error


@dataclass(frozen=True)
class KernelBinaryABI:
    abi_name: str
    arguments: Tuple[KernelArgumentABI, ...]
    error_abi: KernelErrorABI
    return_primitive: CPrimitive = CPrimitive.I32
    calling_convention: str = "c"
    pointer_width_bits: int = 64
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported KernelBinaryABI schema version")
        values = tuple(self.arguments)
        if not self.abi_name or not values or len(
            {item.name for item in values}
        ) != len(values):
            raise SchemaError("kernel binary ABI arguments must be non-empty and unique")
        if not isinstance(self.error_abi, KernelErrorABI):
            raise TypeError("error_abi must be KernelErrorABI")
        object.__setattr__(self, "return_primitive", CPrimitive(self.return_primitive))
        if self.return_primitive != CPrimitive.I32:
            raise SchemaError("kernel binary ABI error return must be i32")
        if self.calling_convention != "c" or self.pointer_width_bits != 64:
            raise SchemaError("kernel binary ABI v1 requires C calling convention/u64 pointers")
        object.__setattr__(self, "arguments", values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "abi_name": self.abi_name,
            "arguments": [item.to_dict() for item in self.arguments],
            "error_abi": self.error_abi.to_dict(),
            "return_primitive": self.return_primitive.value,
            "calling_convention": self.calling_convention,
            "pointer_width_bits": self.pointer_width_bits,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelBinaryABI":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("KernelBinaryABI fields are invalid")
        arguments = data.get("arguments")
        if not isinstance(arguments, (list, tuple)) or not isinstance(
            data.get("error_abi"), Mapping
        ):
            raise SchemaError("KernelBinaryABI arguments/error_abi are invalid")
        data["arguments"] = tuple(
            KernelArgumentABI.from_dict(item) for item in arguments
        )
        data["error_abi"] = KernelErrorABI.from_dict(data["error_abi"])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("KernelBinaryABI fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate_logical(self, logical: KernelLaunchABI) -> None:
        names = tuple(item.name for item in self.arguments)
        if names != logical.argument_names:
            raise SchemaError("binary ABI arguments do not match logical launch order")
        mutable = {
            item.name
            for item in self.arguments
            if item.direction
            in {KernelArgumentDirection.OUTPUT, KernelArgumentDirection.INOUT}
        }
        if mutable != set(logical.mutable_arguments):
            raise SchemaError("binary ABI mutation set does not match logical launch ABI")
        stream = next(
            item for item in self.arguments if item.name == logical.stream_argument
        )
        if stream.passing != KernelArgumentPassing.OPAQUE_HANDLE:
            raise SchemaError("logical stream argument must be an opaque handle")


@dataclass(frozen=True)
class QuantSpec:
    """Logical and physical contract for a quantized tensor.

    Packed sub-byte formats must declare ``packing_order``. Scale and zero
    point tensors remain explicit runtime arguments; this object describes how
    their shapes and values are interpreted.
    """

    scheme: str
    storage_dtype: str
    compute_dtype: str
    accumulator_dtype: str
    scale_dtype: str = "float32"
    granularity: str = "tensor"
    group_size: Optional[Tuple[int, ...]] = None
    axis: Optional[Tuple[int, ...]] = None
    has_zero_point: bool = False
    physical_layout: str = "logical"
    packing_order: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        valid_schemes = {"symmetric", "asymmetric", "mx"}
        valid_granularities = {
            "tensor",
            "channel",
            "token",
            "group",
            "block",
            "page",
        }
        if self.group_size is not None:
            object.__setattr__(
                self, "group_size", tuple(int(size) for size in self.group_size)
            )
        if self.axis is not None:
            object.__setattr__(self, "axis", tuple(int(axis) for axis in self.axis))
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                "unsupported QuantSpec schema_version %s" % self.schema_version
            )
        if self.scheme not in valid_schemes:
            raise SchemaError("unsupported quantization scheme: %s" % self.scheme)
        if self.granularity not in valid_granularities:
            raise SchemaError(
                "unsupported quantization granularity: %s" % self.granularity
            )
        for name in (
            "storage_dtype",
            "compute_dtype",
            "accumulator_dtype",
            "scale_dtype",
            "physical_layout",
        ):
            if not getattr(self, name):
                raise SchemaError("%s must be non-empty" % name)
        if self.group_size is not None and any(size <= 0 for size in self.group_size):
            raise SchemaError("group_size values must be positive")
        if self.granularity in {"group", "block"} and not self.group_size:
            raise SchemaError(
                "%s quantization requires group_size" % self.granularity
            )
        if self.axis is not None and len(set(self.axis)) != len(self.axis):
            raise SchemaError("axis cannot contain duplicates")
        if self.scheme == "asymmetric" and not self.has_zero_point:
            raise SchemaError("asymmetric quantization requires a zero point")
        if self.storage_dtype in {"int4", "int4_packed", "uint4_packed"}:
            if not self.packing_order:
                raise SchemaError("packed 4-bit storage requires packing_order")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scheme": self.scheme,
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "scale_dtype": self.scale_dtype,
            "granularity": self.granularity,
            "group_size": list(self.group_size) if self.group_size else None,
            "axis": list(self.axis) if self.axis is not None else None,
            "has_zero_point": self.has_zero_point,
            "physical_layout": self.physical_layout,
            "packing_order": self.packing_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantSpec":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("QuantSpec fields do not match schema version 1")
        if data.get("group_size") is not None:
            data["group_size"] = tuple(int(item) for item in data["group_size"])
        if data.get("axis") is not None:
            data["axis"] = tuple(int(item) for item in data["axis"])
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("QuantSpec fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class WorkloadSpec:
    """Serializable kernel-selection inputs that do not contain device data."""

    op: str
    dtypes: Tuple[str, ...]
    layouts: Tuple[str, ...] = ()
    static_dims: Tuple[int, ...] = ()
    dynamic_bounds: Tuple[int, ...] = ()
    quant_specs: Tuple[QuantSpec, ...] = ()
    causal: Optional[bool] = None
    pos_encoding: Optional[str] = None
    deterministic: bool = False
    attributes: Tuple[Tuple[str, str], ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                "unsupported WorkloadSpec schema_version %s" % self.schema_version
            )
        if not self.op:
            raise SchemaError("op must be non-empty")
        object.__setattr__(self, "dtypes", _tuple_of_str("dtypes", self.dtypes))
        object.__setattr__(self, "layouts", _tuple_of_str("layouts", self.layouts))
        object.__setattr__(
            self, "static_dims", tuple(int(dim) for dim in self.static_dims)
        )
        object.__setattr__(
            self, "dynamic_bounds", tuple(int(bound) for bound in self.dynamic_bounds)
        )
        object.__setattr__(self, "quant_specs", tuple(self.quant_specs))
        object.__setattr__(
            self,
            "attributes",
            tuple(sorted((str(name), str(value)) for name, value in self.attributes)),
        )
        if any(dim < 0 for dim in self.static_dims):
            raise SchemaError("static_dims cannot contain negative values")
        if any(bound < 0 for bound in self.dynamic_bounds):
            raise SchemaError("dynamic_bounds cannot contain negative values")
        attribute_names = tuple(name for name, _ in self.attributes)
        if len(set(attribute_names)) != len(attribute_names):
            raise SchemaError("attributes cannot contain duplicate names")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "op": self.op,
            "dtypes": list(self.dtypes),
            "layouts": list(self.layouts),
            "static_dims": list(self.static_dims),
            "dynamic_bounds": list(self.dynamic_bounds),
            "quant_specs": [spec.to_dict() for spec in self.quant_specs],
            "causal": self.causal,
            "pos_encoding": self.pos_encoding,
            "deterministic": self.deterministic,
            "attributes": [list(item) for item in self.attributes],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class DeviceCapability:
    """Runtime-observed and build-manifest device capability."""

    soc_version: str
    soc_revision: str = "unknown"
    ai_core_count: int = 0
    supported_dtypes: Tuple[str, ...] = ()
    features: Tuple[str, ...] = ()
    cann_version: str = "unknown"
    torch_npu_version: str = "unavailable"
    compiler_version: str = "unavailable"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                "unsupported DeviceCapability schema_version %s"
                % self.schema_version
            )
        if not self.soc_version:
            raise SchemaError("soc_version must be non-empty")
        if self.ai_core_count < 0:
            raise SchemaError("ai_core_count cannot be negative")
        object.__setattr__(
            self,
            "supported_dtypes",
            _tuple_of_str("supported_dtypes", self.supported_dtypes),
        )
        object.__setattr__(self, "features", _tuple_of_str("features", self.features))

    def supports_dtype(self, dtype: str) -> bool:
        return dtype in self.supported_dtypes

    def has_feature(self, feature: str) -> bool:
        return feature in self.features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "soc_version": self.soc_version,
            "soc_revision": self.soc_revision,
            "ai_core_count": self.ai_core_count,
            "supported_dtypes": list(self.supported_dtypes),
            "features": list(self.features),
            "cann_version": self.cann_version,
            "torch_npu_version": self.torch_npu_version,
            "compiler_version": self.compiler_version,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class WorkspaceFormula:
    """A checked affine formula over WorkloadSpec.dynamic_bounds."""

    constant_bytes: int = 0
    dynamic_coefficients: Tuple[int, ...] = ()
    alignment: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dynamic_coefficients",
            tuple(int(value) for value in self.dynamic_coefficients),
        )
        if self.constant_bytes < 0:
            raise SchemaError("constant_bytes cannot be negative")
        if any(coefficient < 0 for coefficient in self.dynamic_coefficients):
            raise SchemaError("dynamic workspace coefficients cannot be negative")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise SchemaError("workspace alignment must be a positive power of two")

    def size_for(self, workload: WorkloadSpec) -> int:
        if len(self.dynamic_coefficients) > len(workload.dynamic_bounds):
            raise SchemaError(
                "workspace formula expects %d dynamic bounds, got %d"
                % (len(self.dynamic_coefficients), len(workload.dynamic_bounds))
            )
        raw = self.constant_bytes + sum(
            coefficient * bound
            for coefficient, bound in zip(
                self.dynamic_coefficients, workload.dynamic_bounds
            )
        )
        if raw == 0:
            return 0
        return (raw + self.alignment - 1) & ~(self.alignment - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constant_bytes": self.constant_bytes,
            "dynamic_coefficients": list(self.dynamic_coefficients),
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class KernelConstraints:
    """Serializable capability predicate for a kernel descriptor."""

    supported_socs: Tuple[str, ...] = ()
    dtype_signatures: Tuple[Tuple[str, ...], ...] = ()
    layout_signatures: Tuple[Tuple[str, ...], ...] = ()
    required_features: Tuple[str, ...] = ()
    quant_storage_dtypes: Tuple[str, ...] = ()
    deterministic: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_socs", tuple(self.supported_socs))
        object.__setattr__(
            self,
            "dtype_signatures",
            tuple(tuple(signature) for signature in self.dtype_signatures),
        )
        object.__setattr__(
            self,
            "layout_signatures",
            tuple(tuple(signature) for signature in self.layout_signatures),
        )
        object.__setattr__(self, "required_features", tuple(self.required_features))
        object.__setattr__(
            self, "quant_storage_dtypes", tuple(self.quant_storage_dtypes)
        )

    def unsupported_reasons(
        self, workload: WorkloadSpec, capability: DeviceCapability
    ) -> Tuple[str, ...]:
        reasons = []
        if self.supported_socs and capability.soc_version not in self.supported_socs:
            reasons.append("unsupported SoC %s" % capability.soc_version)
        if self.dtype_signatures and workload.dtypes not in self.dtype_signatures:
            reasons.append("unsupported dtype signature %r" % (workload.dtypes,))
        unsupported_device_dtypes = tuple(
            dtype
            for dtype in workload.dtypes
            if capability.supported_dtypes
            and not capability.supports_dtype(dtype)
        )
        if unsupported_device_dtypes:
            reasons.append(
                "device does not support dtypes %r" % (unsupported_device_dtypes,)
            )
        if self.layout_signatures and workload.layouts not in self.layout_signatures:
            reasons.append("unsupported layout signature %r" % (workload.layouts,))
        missing_features = tuple(
            feature
            for feature in self.required_features
            if not capability.has_feature(feature)
        )
        if missing_features:
            reasons.append("missing features %r" % (missing_features,))
        quant_dtypes = tuple(spec.storage_dtype for spec in workload.quant_specs)
        if self.quant_storage_dtypes and any(
            dtype not in self.quant_storage_dtypes for dtype in quant_dtypes
        ):
            reasons.append("unsupported quant storage dtype %r" % (quant_dtypes,))
        if workload.deterministic and self.deterministic is not True:
            reasons.append("kernel does not guarantee deterministic execution")
        return tuple(reasons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported_socs": list(self.supported_socs),
            "dtype_signatures": [list(item) for item in self.dtype_signatures],
            "layout_signatures": [list(item) for item in self.layout_signatures],
            "required_features": list(self.required_features),
            "quant_storage_dtypes": list(self.quant_storage_dtypes),
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class KernelCapabilityBinding:
    """Opaque link from a kernel descriptor to a domain capability rule."""

    domain: str
    profile_id: str
    rule_id: str
    profile_fingerprint: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported KernelCapabilityBinding schema version")
        for name in ("domain", "profile_id", "rule_id"):
            if not str(getattr(self, name)):
                raise SchemaError("%s must be non-empty" % name)
        fingerprint = str(self.profile_fingerprint)
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise SchemaError("profile_fingerprint must be lowercase SHA-256")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "profile_id": self.profile_id,
            "rule_id": self.rule_id,
            "profile_fingerprint": self.profile_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelCapabilityBinding":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError(
                "KernelCapabilityBinding fields do not match schema version 1"
            )
        try:
            return cls(**data)
        except TypeError as error:
            raise SchemaError("KernelCapabilityBinding fields are invalid") from error


@dataclass(frozen=True)
class KernelDescriptor:
    kernel_id: str
    op: str
    backend: Backend
    constraints: KernelConstraints = field(default_factory=KernelConstraints)
    workspace: WorkspaceFormula = field(default_factory=WorkspaceFormula)
    int_workspace: WorkspaceFormula = field(default_factory=WorkspaceFormula)
    artifact: Optional[ArtifactRef] = None
    launch_abi: Optional[KernelLaunchABI] = None
    binary_abi: Optional[KernelBinaryABI] = None
    priority: int = 0
    op_schema_version: int = SCHEMA_VERSION
    tiling_schema_version: int = SCHEMA_VERSION
    capability_binding: Optional[KernelCapabilityBinding] = None

    def __post_init__(self) -> None:
        if not self.kernel_id or not self.op:
            raise SchemaError("kernel_id and op must be non-empty")
        if self.op_schema_version <= 0 or self.tiling_schema_version <= 0:
            raise SchemaError("kernel schema versions must be positive")
        object.__setattr__(self, "backend", Backend(self.backend))
        if self.backend == Backend.REFERENCE:
            if any(
                item is not None
                for item in (self.artifact, self.launch_abi, self.binary_abi)
            ):
                raise SchemaError(
                    "reference kernels cannot claim artifact or launch ABI"
                )
        else:
            if not isinstance(self.artifact, ArtifactRef):
                raise SchemaError("non-reference kernels require ArtifactRef")
            if not isinstance(self.launch_abi, KernelLaunchABI):
                raise SchemaError("non-reference kernels require KernelLaunchABI")
            if not isinstance(self.binary_abi, KernelBinaryABI):
                raise SchemaError("non-reference kernels require KernelBinaryABI")
            self.binary_abi.validate_logical(self.launch_abi)
            expected = {
                Backend.ASCENDC_AOT: {
                    ArtifactFormat.ASCENDC_OBJECT,
                    ArtifactFormat.SHARED_LIBRARY,
                },
                Backend.ASCENDC_JIT: {ArtifactFormat.ASCENDC_SOURCE},
                Backend.ACLNN: {ArtifactFormat.ACLNN_BUILTIN},
            }[self.backend]
            if self.artifact.format not in expected:
                raise SchemaError("artifact format does not match kernel backend")
        if self.capability_binding is not None and not isinstance(
            self.capability_binding, KernelCapabilityBinding
        ):
            raise TypeError("capability_binding must be KernelCapabilityBinding")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "op": self.op,
            "backend": self.backend.value,
            "constraints": self.constraints.to_dict(),
            "workspace": self.workspace.to_dict(),
            "int_workspace": self.int_workspace.to_dict(),
            "artifact": (
                self.artifact.to_dict() if self.artifact is not None else None
            ),
            "launch_abi": (
                self.launch_abi.to_dict() if self.launch_abi is not None else None
            ),
            "binary_abi": (
                self.binary_abi.to_dict() if self.binary_abi is not None else None
            ),
            "priority": self.priority,
            "op_schema_version": self.op_schema_version,
            "tiling_schema_version": self.tiling_schema_version,
            "capability_binding": (
                self.capability_binding.to_dict()
                if self.capability_binding is not None
                else None
            ),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())
