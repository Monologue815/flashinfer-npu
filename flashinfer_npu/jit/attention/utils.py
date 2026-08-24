"""Naming helpers for Attention JIT recipes."""

from __future__ import annotations

from flashinfer_npu.attention.schema import AttentionMode


def attention_jit_module_name(
    mode: AttentionMode,
    specialization_fingerprint: str,
    kernel_fingerprint: str,
) -> str:
    """Return a stable registry name without exposing dynamic request lengths."""

    mode = AttentionMode(mode)
    for value in (specialization_fingerprint, kernel_fingerprint):
        if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
            raise ValueError("Attention JIT module name requires SHA-256 identities")
    return "attention.%s.%s.%s" % (
        mode.value,
        specialization_fingerprint[:16],
        kernel_fingerprint[:16],
    )


__all__ = ["attention_jit_module_name"]
