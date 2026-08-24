# Attention capability matrix

This document describes the capabilities represented by the repository. It is
not a validation log or a claim that an Ascend kernel is available.

## Status vocabulary

| Status | Meaning |
|---|---|
| `reference` | Executable scalar Host semantics used as the correctness definition |
| `framework` | Schema, planning, dispatch or adapter contract exists, but no production NPU implementation is enabled |
| `integration-required` | A version-pinned external provider adapter and capability evidence must be installed |
| `planned` | The public or internal contract is not complete |

## Public Attention surface

| Capability | Status | Notes |
|---|---|---|
| Single prefill | `reference` | FlashInfer-style function surface with Host oracle semantics |
| Single decode | `reference` | FlashInfer-style function surface with Host oracle semantics |
| Batch paged prefill | `framework` | Plan/metadata/reference contracts exist; no default NPU provider |
| Batch ragged prefill | `framework` | Plan/metadata/reference contracts exist; no default NPU provider |
| Batch paged decode | `framework` | Plan/metadata/reference contracts exist; no default NPU provider |
| Mixed paged `BatchAttention` | `framework` | Wrapper-owned `plan()`/`run()` lifecycle and automatic resolver contract |
| NHD/HND KV layouts | `framework` | Logical tensor and metadata contracts are defined |
| Causal/window/custom masks | `framework` | Admission and reference semantics are defined |
| RoPE/ALiBi | `framework` | Plan and reference semantics are defined |
| Output and LSE | `framework` | Shape, allocation and normalization contracts are defined |

## Quantized KV cache

| Format | Logical/reference | Provider physical layout | Production NPU execution |
|---|---|---|---|
| INT8 | `reference` | `framework` | `integration-required` |
| UINT8 | `reference` | `framework` | `integration-required` |
| Packed INT4 | `reference` | `framework` | `integration-required` |
| Packed UINT4 | `reference` | `framework` | `integration-required` |
| FP8/NVFP4/MX | `planned` | `planned` | `planned` |

Scale granularity, independent K/V scale sources, asymmetric zero points,
runtime multipliers, physical blocking and padding participate in exact plan
and provider identities. A provider is not admitted merely because its Python
API contains quantization-related parameter names.

## Backend integration

| Backend | Repository state | Default availability |
|---|---|---|
| Host reference | Scalar correctness implementation | Explicit opt-in only |
| CANN/aclnn | Operation catalog, plan gate and adapter composition contracts | Disabled; no version-pinned runtime spec installed |
| flash-attention-npu | Operation catalog, plan gate and adapter composition contracts | Disabled; no version-pinned runtime spec installed |
| Ascend C AOT | Artifact, ABI, dispatch and launcher contracts | Disabled; no kernel artifact installed |
| Ascend C JIT | Spec, cache, artifact verification, module/symbol and active-plan contracts | Disabled; no compiler or concrete loader installed |

The packaged runtime registry is intentionally empty. An NPU route becomes
eligible only after an integration supplies exact package versions, runtime
environment, capability and numerical evidence, kernel/artifact/ABI
provenance, tensor materialization and an authorized executor.

## JIT lifecycle

| Stage | Status | Boundary |
|---|---|---|
| Specialization and `JitSpec` identity | `framework` | Static Attention choices and complete environment identity |
| Registry and cache decision | `framework` | Exact cache hit, build required or unavailable |
| Artifact byte verification | `framework` | Injected reader, exact size and SHA-256 receipt |
| Module and symbol resolution | `framework` | Injected loader, exact single/batch entry-point set |
| Compiler/build provider | `planned` | No source generator or compiler invocation is installed |
| Concrete filesystem cache | `planned` | No default cache root, lock or atomic writer is installed |
| Callable NPU executor | `planned` | No loaded symbol is invoked |

## Explicit non-claims

The repository does not currently claim production correctness, performance or
availability for any Ascend Attention operator. It does not install or call a
real CANN, flash-attention-npu or Ascend C kernel by default. Capability rows
describe contracts that a future integration must satisfy, not detected
hardware support.
