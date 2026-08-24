# Attention JIT framework

This document defines the JIT boundary for Attention implementations. The design
follows FlashInfer's central ownership rule: JIT specialization and loading are
library internals; callers continue to use the normal Attention functions and
batch `plan()` / `run()` wrappers.

## 1. Goals

The JIT framework exists to support Ascend-specific implementations whose exact
executable depends on Attention semantics, quantization, runtime environment and
device characteristics. It must:

- create deterministic specialization identities;
- separate policy decisions from compilation and loading side effects;
- verify cache artifacts before they become executable;
- bind loaded modules and symbols to the active Attention plan;
- allow injected CANN/Ascend C implementations without changing public APIs;
- remain importable and explainable on a Host-only machine.

It is not a user-facing kernel compiler API.

## 2. Package structure

The `flashinfer_npu.jit` package is divided by responsibility:

| Area | Responsibility |
|---|---|
| `core` | Canonical JIT specification, identity and policy contracts |
| `env` | Runtime, compiler, ABI and device environment identity |
| `cache` | Cache records and side-effect-free cache decisions |
| `artifacts` | Artifact metadata and verified-byte receipts |
| `loading` | Module loader and exact symbol-resolution contracts |
| `attention` | Attention specialization, registry and wrapper integration |

Concrete compilers, filesystem caches and NPU loaders are dependencies injected
behind these contracts. They are not activated by importing the package.

## 3. Specialization identity

An Attention specialization includes every static choice that may alter code or
ABI, for example:

- single/batch and prefill/decode mode;
- paged/ragged storage mode;
- query, KV and output dtypes;
- logical and provider physical KV layout;
- head dimension, page size and grouping constraints;
- causal/window/custom-mask mode;
- RoPE, ALiBi and soft-cap mode;
- quantized storage, packing, scales and zero-point rules;
- requested entry-point family and ABI version.

Dynamic lengths and tensor addresses remain run-time values unless a provider
explicitly proves they are specialization parameters.

Canonical serialization produces a stable specialization fingerprint. Unknown
fields, ambiguous defaults and non-canonical encodings are rejected rather than
silently normalized.

## 4. Environment identity

The same specialization can produce incompatible binaries in different
environments. The JIT environment therefore contributes:

- provider/backend identity;
- package and compiler versions;
- target SoC/architecture;
- ABI and framework versions;
- compile flags that affect generated code;
- relevant runtime generation and device features.

The complete JIT identity is derived from both specialization and environment.
A cache entry created under a different environment cannot be reused by name
alone.

## 5. `JitSpec` and registry

`JitSpec` is an immutable internal build/load description. It contains canonical
identity, required sources or build inputs, required entry points and artifact
expectations. Attention implementations register a factory that derives one or
more specs from the selected framework plan.

Registry lookup is deterministic and side-effect free. Registration conflicts,
identity drift or multiple incompatible factories for the same key are errors.

The registry is not exposed through public Attention signatures. The wrapper
selects the factory after normal provider admission.

## 6. Policy and cache decisions

Policy evaluation separates “what should happen” from “perform the operation.”
The result is one of these conceptual states:

- `cache_hit`: an exact candidate record exists and may proceed to byte
  verification;
- `build_required`: policy permits construction but no valid artifact exists;
- `unavailable`: neither a usable artifact nor an authorized builder exists;
- `rejected`: policy or identity rules prohibit this route.

A cache hit is metadata only. It does not prove that artifact bytes still exist
or match their recorded identity. `build_required` likewise does not mean a
compiler has been invoked.

## 7. Artifact verification

An artifact record identifies expected path/key, size, digest, format and ABI.
Before module loading, an injected reader returns the artifact bytes and the
framework verifies:

- the record belongs to the active JIT identity;
- bytes are present and within configured bounds;
- actual size equals the recorded size;
- SHA-256 equals the recorded digest;
- artifact format and ABI match the requested operation.

Successful verification produces an immutable receipt. Loading APIs accept the
receipt, not an unverified cache path.

A future filesystem implementation must additionally provide bounded paths,
locking, temporary-file isolation and atomic publication.

## 8. Module and symbol loading

An injected loader converts a verified artifact into an opaque module object and
stable module token. The resolver then requires the exact entry-point set
declared by the operation:

- single-request operations require their single run symbol;
- reusable batch operations require the declared plan and run symbols;
- missing or unexpected symbols are rejected;
- loader, module and symbol identities are recorded in the receipt.

The framework treats loaded objects as opaque. Backend-specific handles must not
escape into user code.

## 9. Planner, callable and executor binding

Resolved symbols are not usable merely because they exist. For batch Attention,
a planner binder associates the module's `plan` entry point with the provider
plan factory. The resulting factory owns provider plan state while the public
wrapper continues to own the planning transaction.

An executor binder associates the same loaded module with an authorized
callable/executor contract that defines:

- accepted lowered argument schema;
- workspace, stream and device ownership;
- provider plan creation where required;
- output and LSE conventions;
- asynchronous completion and error behavior;
- lifetime and teardown rules.

Both binding identities participate in the active plan. `run()` accepts only
the planner state and executor bound during the successful planning transaction.

No default production Ascend binder or executor is installed by the repository.

## 10. Wrapper integration

For a JIT-backed batch implementation, `plan()` performs this transaction:

1. build the canonical Attention plan;
2. select and authorize a provider candidate;
3. derive the Attention specialization and environment;
4. resolve an exact `JitSpec`;
5. make the cache/build policy decision;
6. obtain and verify artifact bytes;
7. load the module and resolve exact symbols;
8. bind the module plan entry point to a provider plan factory;
9. bind an executor and create provider plan state;
10. publish one immutable active plan.

If any stage fails, the wrapper publishes nothing and preserves its previous
active plan.

`run()` validates input compatibility, lowers canonical tensors and metadata to
the bound schema, verifies active-plan/executor identity and invokes the bound
executor. It does not rebuild, reload or reselect an implementation implicitly.

## 11. Relationship to CANN and flash-attention-npu

Prebuilt package operators normally use the package-callable path rather than
the JIT artifact path. They still share the same provider admission, active-plan
and execution rules.

A package may expose its own JIT or compilation mechanism. An adapter may use it
only if it can produce the same framework receipts: complete specialization and
environment identity, verified artifact/module identity, exact symbols and an
authorized executor. Package-specific build objects remain internal.

## 12. Upstream FlashInfer alignment

NVIDIA FlashInfer keeps JIT construction and loading behind internal `JitSpec`
and cached loader helpers while public Attention wrappers retain their ordinary
call signatures. This design preserves that user experience while replacing
CUDA-specific compilation, module and stream details with explicit Ascend
provider contracts.

Alignment is behavioral rather than a copy of CUDA internals:

- public APIs express Attention semantics;
- the library selects and owns implementation state;
- batch wrappers own planning and reusable execution state;
- specialization and caching remain internal;
- backend-specific code is reached through implementation registries/loaders.

## 13. Required components for production use

Enabling a real JIT implementation requires all of the following:

1. an Attention specialization factory for a narrowly defined capability set;
2. a versioned Ascend environment probe;
3. an authorized source generator or external build provider;
4. a bounded, locked and atomic artifact cache;
5. an artifact reader and verifier;
6. an Ascend module loader and exact symbol resolver;
7. a module planner binder and callable/executor binder;
8. provider plan/run lowering;
9. stream, completion, error and teardown integration;
10. numerical and physical-layout evidence for the enabled capability.

Until these components are installed, the JIT package represents framework
contracts only and must fail closed for NPU execution.
