"""Host-only compatibility contract for FlashInfer single-request JIT entry points.

This module models buffer allocation and argument forwarding only.  It does
not compile, load, authenticate, or execute an Ascend artifact by itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Callable, Protocol, Tuple

from flashinfer_npu.runtime import SchemaError

from .reference import ReferenceBuffer, ReferenceTensor
from .schema import KVLayout
from .protocol_trace import (
    AttentionProtocolRoute,
    AttentionProtocolState,
    attention_protocol_fingerprint,
    start_attention_protocol_recording,
)


ATTENTION_SINGLE_JIT_TEMP_BYTES = 32 * 1024 * 1024
ATTENTION_UPSTREAM_MASK_MODES = (0, 1, 2, 3)


class AttentionSingleJITModule(Protocol):
    """Structural protocol implemented by an injected module with ``run``."""

    def run(self, *args: Any) -> Any:
        ...


@dataclass(frozen=True)
class AttentionJITTemporaryBuffer:
    """Metadata-only representation of FlashInfer's 32 MiB scratch buffer.

    A scalar Host model must not materialize 32 MiB as Python float objects.
    The injected test module can inspect the exact capacity, dtype and device;
    a future tensor frontend will replace this record with real storage.
    """

    capacity_bytes: int
    device: str
    dtype: str = "uint8"

    def __post_init__(self) -> None:
        if self.capacity_bytes != ATTENTION_SINGLE_JIT_TEMP_BYTES:
            raise SchemaError("single JIT temporary buffer must be exactly 32 MiB")
        if not self.device:
            raise SchemaError("single JIT temporary buffer device must be non-empty")
        if self.dtype != "uint8":
            raise SchemaError("single JIT temporary buffer dtype must be uint8")

    @property
    def shape(self) -> Tuple[int]:
        return (self.capacity_bytes,)


def require_jit_run(jit_module: AttentionSingleJITModule) -> Callable[..., Any]:
    run = getattr(jit_module, "run", None)
    if not callable(run):
        raise TypeError("jit_module must provide a callable run method")
    return run


def upstream_kv_layout_code(layout: KVLayout) -> int:
    """Return FlashInfer's public JIT ABI layout code, not our C ABI code."""

    if layout == KVLayout.NHD:
        return 0
    if layout == KVLayout.HND:
        return 1
    raise SchemaError("JIT KV layout must be NHD or HND")


def validate_upstream_mask_mode(mask_mode: int) -> int:
    if (
        not isinstance(mask_mode, int)
        or isinstance(mask_mode, bool)
        or mask_mode not in ATTENTION_UPSTREAM_MASK_MODES
    ):
        raise SchemaError("mask_mode must be one of 0, 1, 2, 3")
    return mask_mode


def make_single_jit_buffers(
    q: ReferenceTensor,
    *,
    output_shape: Tuple[int, ...],
    lse_shape: Tuple[int, ...],
    return_lse: bool,
) -> Tuple[AttentionJITTemporaryBuffer, ReferenceBuffer, ReferenceBuffer | None]:
    tmp = AttentionJITTemporaryBuffer(ATTENTION_SINGLE_JIT_TEMP_BYTES, q.device)
    output = ReferenceBuffer.zeros(output_shape, dtype=q.dtype, device=q.device)
    lse = (
        ReferenceBuffer.zeros(lse_shape, dtype="float32", device=q.device)
        if return_lse
        else None
    )
    return tmp, output, lse


def freeze_jit_buffer(buffer: ReferenceBuffer, name: str) -> ReferenceTensor:
    try:
        return ReferenceTensor(buffer.shape, tuple(buffer.data), buffer.dtype, buffer.device)
    except (TypeError, ValueError) as error:
        raise SchemaError("JIT module produced invalid %s buffer" % name) from error


def _value_signature(value: Any):
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None and dtype is not None and device is not None:
        return {
            "kind": "tensor",
            "shape": [int(dim) for dim in shape],
            "dtype": str(dtype),
            "device": str(device),
        }
    if isinstance(value, float) and not math.isfinite(value):
        encoded = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        return {"kind": "scalar", "type": "float", "nonfinite": encoded}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "type": type(value).__name__, "value": value}
    return {
        "kind": "opaque",
        "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
    }


@contextmanager
def record_single_jit_protocol_call(
    *,
    mode: str,
    jit_module: AttentionSingleJITModule,
    q: ReferenceTensor,
    k: Any,
    v: Any,
    tmp: AttentionJITTemporaryBuffer,
    output: ReferenceBuffer,
    lse: ReferenceBuffer | None,
    layout_code: int,
    mask_code: int | None,
    window_left: int,
    return_lse: bool,
    extra_args: Tuple[Any, ...],
):
    """Record one opt-in injected-JIT call without changing its public signature."""

    module_type = "%s.%s" % (
        type(jit_module).__module__,
        type(jit_module).__qualname__,
    )
    subject = attention_protocol_fingerprint(
        {
            "route": "single_jit",
            "mode": str(mode),
            "module_type": module_type,
            "q": _value_signature(q),
            "k": _value_signature(k),
            "v": _value_signature(v),
            "layout_code": layout_code,
            "mask_code": mask_code,
            "window_left": window_left,
            "return_lse": bool(return_lse),
            "extra_args": [_value_signature(value) for value in extra_args],
        }
    )
    stream = attention_protocol_fingerprint(
        {
            "device": q.device,
            "stream_id": "host-synchronous",
            "ordered": True,
        }
    )
    owner = attention_protocol_fingerprint(
        {
            "subject_fingerprint": subject,
            "scratch": {
                "capacity_bytes": tmp.capacity_bytes,
                "dtype": tmp.dtype,
                "device": tmp.device,
            },
            "output": _value_signature(output),
            "lse": _value_signature(lse),
        }
    )
    recorder = start_attention_protocol_recording(
        AttentionProtocolRoute.SINGLE_JIT,
        subject,
        stream,
        owner,
    )
    if recorder is None:
        yield
        return
    recorder.record(AttentionProtocolState.PREPARED, subject)
    recorder.record(
        AttentionProtocolState.INVOKED,
        {"operation": "jit_module.run", "module_type": module_type},
    )
    try:
        yield
    except BaseException as error:
        recorder.record(
            AttentionProtocolState.FAILED_SYNC,
            {
                "operation": "jit_module.run_or_freeze",
                "exception_type": "%s.%s"
                % (type(error).__module__, type(error).__qualname__),
            },
        )
        recorder.record(
            AttentionProtocolState.RELEASED,
            {"outcome": "exception_unwind", "owner": owner},
        )
        raise
    recorder.record(
        AttentionProtocolState.COMPLETED,
        {
            "operation": "jit_module.run_or_freeze",
            "output": _value_signature(output),
            "lse": _value_signature(lse),
        },
    )
    recorder.record(
        AttentionProtocolState.RELEASED,
        {"outcome": "completed", "owner": owner},
    )


__all__ = [
    "ATTENTION_SINGLE_JIT_TEMP_BYTES",
    "ATTENTION_UPSTREAM_MASK_MODES",
    "AttentionJITTemporaryBuffer",
    "AttentionSingleJITModule",
    "freeze_jit_buffer",
    "make_single_jit_buffers",
    "record_single_jit_protocol_call",
    "require_jit_run",
    "upstream_kv_layout_code",
    "validate_upstream_mask_mode",
]
