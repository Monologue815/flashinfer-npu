#!/usr/bin/env python3
"""Validate packaged parity and kernel manifests without extra dependencies."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_npu.cli import (
    packaged_attention_capability_manifest_path,
    packaged_kernel_manifest_path,
)
from flashinfer_npu.attention import (
    load_attention_capability_manifest,
    validate_attention_kernel_bindings,
)
from flashinfer_npu.parity import load_packaged_manifest
from flashinfer_npu.runtime import load_kernel_manifest


def main() -> int:
    parity = load_packaged_manifest("all")
    attention_parity = load_packaged_manifest("attention")
    kernels = load_kernel_manifest(packaged_kernel_manifest_path())
    attention_capabilities = load_attention_capability_manifest(
        packaged_attention_capability_manifest_path()
    )
    attention_bindings = validate_attention_kernel_bindings(
        attention_capabilities, kernels
    )
    print(parity.report())
    print(attention_parity.report())
    print("Validated kernel descriptors: %d" % len(kernels))
    print(
        "Validated Attention capability profiles: %d"
        % len(attention_capabilities)
    )
    print("Validated Attention kernel bindings: %d" % attention_bindings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
