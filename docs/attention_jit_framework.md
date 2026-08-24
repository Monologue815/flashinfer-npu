# Attention JIT framework contract

> Status: checkpoint 030, Host framework only. No Ascend source generation,
> compiler invocation, artifact loading, NPU runtime initialization, or
> operator execution is implemented or claimed.

## 1. Why this package exists

FlashInfer keeps JIT specifications, registry, environment, cache/build policy,
and operator-specific module generation under `flashinfer/jit`.  Keeping only
the injected `jit_module.run()` compatibility ABI inside
`flashinfer_npu/attention` does not reproduce that architecture.  The
`flashinfer_npu/jit` package closes the source-layout and responsibility gap
while preserving the current non-executing development boundary.

The two layers have different jobs:

- `flashinfer_npu/attention/jit_protocol.py` validates the public injected
  single-request call ABI used by FlashInfer-compatible frontends;
- `flashinfer_npu/jit` describes how an internally selected JIT implementation
  is identified, registered, found in cache, or declared to require a future
  authorized builder.

Neither layer currently proves a compiled Ascend implementation.

## 2. Package boundary

```text
flashinfer_npu/jit/
├── __init__.py
├── core.py                 # JitSpec, JitSpecStatus, JitSpecRegistry
├── env.py                  # explicit toolchain target and compilation policy
├── cache.py                # verified cache identity and pure resolution
└── attention/
    ├── __init__.py
    ├── modules.py          # selected plan -> AttentionJitModuleSpec
    ├── variants.py         # static Attention specialization identity
    └── utils.py            # stable generated-module naming
```

There is intentionally no compiler, subprocess runner, file cache writer,
dynamic loader, or NPU launcher in checkpoint 030.  Consequently the package
does not yet expose an upstream-style `build_jit_specs()` implementation.

## 3. Wrapper-owned selection flow

The model-facing contract remains unchanged:

```text
user plan()
  -> framework Attention plan
  -> evidence-bearing automatic backend selection
  -> selected backend == ascendc_jit
  -> generate immutable JIT module spec
  -> exact cache lookup
     -> cache_hit: future loader may consume the verified artifact
     -> build_required: future authorized build provider is needed
     -> unavailable: policy forbids satisfying the cache miss
user run()
  -> consumes only the wrapper-owned active plan
```

Users do not pass `JitSpec`, cache records, provider handles, or a plan token to
`run()`.  CANN and flash-attention-npu adapters remain internal candidate
providers selected by the same wrapper-owned planning path.

## 4. `JitSpec` identity

A v1 `JitSpec` hashes all declared build inputs that may change generated code:

- domain, generator id/version, target SoC and exact JIT environment;
- Attention specialization fingerprint;
- selected kernel recipe, source artifact, logical launch ABI and binary ABI
  fingerprints;
- optional canonical Ascend C source artifact identities;
- ordered compile options and exported entry-point names.

The spec may be recipe-only before source generation.  `source_materialized`
distinguishes that state without treating absence of sources as a compiled or
cache-ready module.  A same-name/same-fingerprint registry publication is
idempotent; a same-name/different-fingerprint publication fails closed.

## 5. Environment and cache identity

`JitEnvironment` is provided explicitly by a trusted runtime adapter.  Importing
the package never probes the machine.  Its identity covers target SoC/revision,
CANN, compiler id/version, Torch/torch-npu ABI, Python ABI, build type and
feature set.

A `JitCacheRecord` must bind all of the following exactly:

- spec name and full spec fingerprint;
- JIT environment fingerprint;
- target SoC;
- compiled file artifact identity;
- producer and build-metadata fingerprints.

Builtin aclnn symbols, JIT source identities, mismatched environments and stale
specs cannot be reported as compiled cache hits.

Resolution is a pure decision:

| Cache | Policy | Result |
|---|---|---|
| exact verified record | any | `cache_hit` |
| miss | `enabled` | `build_required` |
| miss | `cache_only` or `disabled` | `unavailable` |

`build_required` is not execution.  It is evidence that a later checkpoint
must supply an authorized builder before the plan can become runnable.

## 6. Attention specialization

`AttentionJitVariant` derives static generation choices from
`AttentionPlanSpec`: mode, Q/KV/output dtype, head counts and dimensions,
NHD/HND layout, position encoding, mask kind, sliding-window fields,
multi-token decode width, soft-cap enablement, reduction/profiler switches and
the complete quantization fingerprint.

Dynamic sequence lengths do not change the reusable recipe.  The enclosing
`AttentionJitModuleSpec` separately binds the exact framework plan, workload
and dispatch receipt, so runtime metadata cannot be substituted after
planning.

Only a receipt whose selected backend is `ascendc_jit` may generate this spec.
An AOT, aclnn or reference selection is rejected rather than silently routed
through JIT.

## 7. Checkpoint evidence and non-claims

Checkpoint 030 tests canonical serialization, environment sensitivity, source
identity, conflict-safe registration, cache drift, all compilation policies,
plan/environment/backend binding, dynamic-shape recipe reuse, quantization
identity and isolated imports.  They also assert that no compiler or loader API
is present.

This evidence proves only the Host framework contract.  It does not prove:

- an Ascend C source template or generated source;
- CMake/Ninja or compiler integration;
- a real on-disk cache, file lock or atomic artifact publication;
- symbol loading or an NPU launch;
- numerical correctness or performance of any JIT kernel.

Those capabilities must be introduced and verified as separate checkpoints.
