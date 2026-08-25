# Attention `plan()` / `run()` dispatch design

This document defines the long-lived architecture of the Attention framework.
It describes the public contract, ownership boundaries, backend selection and
extension rules. It is deliberately independent of any particular development
milestone or validation run.

## 1. Scope

The first library surface covers Attention only. The framework is designed for
Ascend inference, including quantized KV-cache execution, while retaining the
FlashInfer programming model:

- users call a stable wrapper and do not select a kernel directly;
- the wrapper owns a reusable `plan()` / `run()` lifecycle;
- planning resolves an exact implementation from request metadata and runtime
  evidence;
- execution uses the plan that the wrapper published atomically;
- CANN, flash-attention-npu and future Ascend C implementations are providers
  behind the same internal contract.

The repository does not provide a production NPU operator by default. A
provider is enabled only when its package, version, capability and callable
evidence are complete.

## 2. Public programming model

The intended batch surface is conceptually:

```python
wrapper = BatchAttention(workspace_buffer)

wrapper.plan(
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    **attention_options,
)

output = wrapper.run(query, paged_kv_cache, **runtime_options)
```

The exact Python signatures are defined by the public modules. The architectural
rules are:

1. The caller supplies tensors and semantic Attention options.
2. The caller does not supply a provider name, kernel name, JIT handle, module
   handle or executable object.
3. `plan()` may be reused by subsequent compatible `run()` calls.
4. A new incompatible request requires replanning.
5. Planning failure leaves the previous active plan unchanged.

Single-request prefill and decode functions follow the same policy but do not
need to expose a reusable batch plan object.

## 3. Layering and ownership

```text
Public Attention wrapper
        |
        v
Canonical framework plan
        |
        v
Runtime registry snapshot
        |
        v
Candidate explain / select
        |
        v
Package authority + provider plan gate
        |
        +---- package callable path
        |
        +---- AOT/JIT artifact path
        |
        v
Provider materialization + provider plan
        |
        v
Wrapper-owned active plan
        |
        v
Run validation + lowering
        |
        v
Authorized executor
```

Each boundary has one owner:

- the public wrapper owns user-visible lifecycle and the active plan;
- the framework owns canonical semantics and deterministic selection;
- a provider adapter owns translation to one external package/API;
- the executor owns the actual callable invocation;
- external packages own their kernels and device-specific behavior.

This prevents backend-specific objects from leaking into the public API.

Each batch wrapper owns a mode-bound internal runtime. The mode is frozen when
the wrapper is constructed, so a paged-prefill wrapper cannot accidentally
publish a decode or mixed-batch plan. Mode validation precedes package probing,
callable loading and provider planning. The holistic `BatchAttention` runtime
is the `BATCH_MIXED_PAGED` specialization of this shared runtime contract.

For paged/ragged prefill and paged decode, `backend="reference"` selects only
the explicit Host oracle. `backend="auto"` requires an NPU workspace and
snapshots the installed runtime registry together with its versioned operation
catalog when the wrapper is constructed. Public `plan()` publishes the exact
provider plan atomically; public `run()` reuses it and preserves FlashInfer's
output-versus-`(output, lse)` return convention. Provider options without an
authorized lowering are rejected before invocation.

Provider routing is an explicit per-wrapper capability, not a side effect of
the shared batch base class. A wrapper mode that has no complete public
`plan()`/`run()` lowering must reject `backend="auto"` before registry
resolution. This prevents a wrapper from publishing provider state that its
public execution path cannot consume.

## 4. Canonical framework plan

`plan()` first converts public arguments into an immutable, backend-neutral
plan. The plan records every value that can change semantics or implementation
eligibility, including:

- prefill/decode and paged/ragged mode;
- query and KV dtypes;
- logical KV layout (`NHD` or `HND`);
- head counts, head dimension, page size and sequence bounds;
- causal, window, custom-mask and soft-cap behavior;
- positional encoding choices such as RoPE or ALiBi;
- requested output and LSE behavior;
- quantized storage, scale and zero-point definitions;
- stream, device and workspace requirements where they affect execution.

Dynamic tensor contents are not embedded in the reusable plan. Shape or layout
facts that affect compatibility are represented explicitly and checked again at
`run()`.

## 5. Runtime registry snapshot

The wrapper reads one immutable registry snapshot at the start of `plan()`.
The snapshot contains registered runtime implementations, their priority,
operation catalogs, package declarations and generation identity.

Snapshot isolation provides three guarantees:

- registry changes do not alter a plan already being constructed;
- existing wrappers do not silently inherit a different backend generation;
- a stale provider generation can be rejected before execution.

The default packaged registry is empty for NPU execution. Importing the library
must not import CANN or flash-attention-npu, inspect a device or compile code.

## 6. Candidate explanation and selection

Every registered implementation is evaluated through a pure plan gate. The gate
returns structured acceptance or rejection reasons and must not import the
external package or invoke an operator.

Selection is deterministic. Ordering is based on declared priority and stable
implementation identity, never discovery order. A candidate is eligible only
when all of the following agree:

- the canonical plan;
- the provider operation catalog;
- provider-specific plan constraints;
- declared package/version range;
- runtime and device authority;
- numerical and physical-layout evidence;
- callable or artifact identity.

`explain()` may expose these reasons for diagnostics without performing package
loading or device work.

## 7. Operation catalog

The operation catalog is the provider's versioned declaration of callable
surface. It maps a framework operation to an exact external API contract:

- module and symbol path;
- single or batch lifecycle;
- required and optional parameter names;
- argument ownership and defaults;
- supported modes, layouts and dtypes;
- output and workspace conventions;
- quantization parameter bindings;
- provider-plan and run-time lowering rules.

The catalog and provider factory share one rule source. A plan accepted by the
catalog must not later be rejected by a duplicated, drifting eligibility rule.

## 8. Package runtime authority

Package metadata alone is insufficient authorization to execute. The package
resolver creates an immutable authority receipt containing at least:

- distribution and import identity;
- observed version and allowed version rule;
- implementation and catalog identity;
- runtime/device generation where relevant;
- evidence manifest identity;
- callable loader identity and resolved symbol identity.

Resolution is lazy and occurs only for the selected candidate. Failure to
resolve a package or prove authority fails closed and does not publish a partial
active plan.

### CANN and flash-attention-npu adapters

These packages are treated as independent providers. An adapter may use their
operators only when the selected operation exactly represents the requested
plan. The framework must not silently emulate unsupported semantics by dropping
arguments or changing layout/quantization assumptions.

Each adapter must provide:

1. a version-pinned package declaration;
2. an operation catalog derived from the package API being integrated;
3. pure capability/plan gates;
4. tensor and metadata materializers;
5. a callable resolver with exact signature checks;
6. an executor that returns framework-normalized outputs;
7. evidence for numerical behavior and non-logical physical layouts.

## 9. Quantized KV contract

Quantization is part of the plan identity, not an optional execution hint. The
canonical quantization specification distinguishes:

- storage format and packing order;
- signedness and logical value domain;
- scale granularity and axis mapping;
- independent K and V scale sources;
- symmetric or asymmetric zero points;
- runtime scale multipliers;
- group size, padding and physical blocking;
- provider API parameter names and tensor shapes.

Before publication, the provider binding must prove an exact mapping from this
specification to its API. At `run()`, lowering checks tensor metadata, device,
aliasing, physical descriptor and active-plan identity. A matching Python dtype
alone is never sufficient evidence.

## 10. Provider materialization

Once a candidate has authority, the framework may materialize provider-owned
objects such as:

- backend metadata tensors;
- converted page tables or auxiliary tables;
- physical-layout descriptors;
- workspace reservations;
- an external package's reusable plan object.

Materialization is part of the planning transaction. Objects are not visible to
`run()` until every required stage succeeds. Their lifetime is tied to the
active plan and provider generation.

## 11. JIT and artifact path

The JIT path preserves the same wrapper-owned model. A user never receives or
passes a `JitSpec`, cache record, artifact, module or symbol.

The internal sequence is:

```text
canonical plan
  -> attention specialization
  -> JIT spec + environment identity
  -> registry policy
  -> cache decision
  -> artifact-byte verification
  -> module loading
  -> exact entry-point resolution
  -> module plan-factory binding
  -> callable/executor binding
  -> active-plan publication
```

A cache metadata hit is not enough. Artifact bytes must match the recorded size
and digest before loading. The loaded module must expose exactly the entry-point
set required by the operation. Loader identity, module fingerprint and symbol
identity participate in the active-plan receipt.

The repository supplies these framework contracts but does not install a source
generator, compiler, filesystem cache or production NPU module loader by
default. See [Attention JIT framework](attention_jit_framework.md).

## 12. Active plan

The wrapper publishes one immutable active plan only after the full planning
transaction succeeds. It binds:

- canonical framework-plan fingerprint;
- registry snapshot and provider generation;
- implementation, catalog and operation identity;
- package/runtime authority;
- provider materialization and external plan identity;
- quantization and physical-layout bindings;
- callable or verified artifact/module/symbol identity;
- executor identity;
- compatibility rules for future `run()` inputs.

Publication is atomic. Any error keeps the previous active plan intact.

## 13. `run()` validation and lowering

`run()` performs no automatic backend reselection. It:

1. requires an active plan;
2. validates query, KV and auxiliary inputs against that plan;
3. validates device, stream, workspace and provider generation;
4. lowers canonical inputs to the selected operation's exact arguments;
5. verifies the final executable identity;
6. invokes the authorized executor once;
7. normalizes output/LSE and completion ownership to the public contract.

Provider-specific arguments are produced internally. Unknown arguments, missing
bindings, stale receipts or identity drift are hard errors.

The internal run request carries `return_lse` as a required boolean semantic,
even though it is not a provider handle and does not alter plan selection.
Paged/ragged wrappers set it from the public flag. A provider adapter must map
that intent to the selected operation's exact LSE-control argument and return
schema. The holistic `BatchAttention` contract always requests LSE because its
public return value is fixed to `(output, lse)`.

An immutable resource binding is derived from the selected operation before
active-plan publication. It distinguishes package-managed from caller-managed
workspace and returned tensors from mutable output arguments. The current
documented package APIs expose returned output/LSE values and no wrapper
workspace argument. Their caller workspace requirement is therefore zero while
package-internal scratch remains outside this contract. Public `out` or `lse`
buffers are rejected unless an exact mutable-argument binding exists.

## 14. Failure and replanning rules

The framework is fail-closed:

- no eligible implementation: planning fails with structured reasons;
- package missing or version rejected: planning fails before callable loading;
- incomplete evidence: the candidate is ineligible;
- artifact digest or symbol mismatch: the candidate is not executable;
- stale registry/runtime generation: execution is rejected;
- incompatible `run()` tensors: execution is rejected and the caller replans;
- failed replan: the previous active plan remains available.

Fallback is an explicit candidate-selection decision made during `plan()`, not a
silent retry after a provider has partially executed.

## 15. Adding a real provider

A production integration should be added in this order:

1. document the exact external package version and API;
2. add catalog entries for only the supported Attention operations;
3. implement pure plan gates and structured rejection reasons;
4. declare package/bootstrap metadata without import-time side effects;
5. implement lazy package and callable resolution;
6. implement tensor/metadata materialization and physical-layout evidence;
7. bind quantization parameters exactly;
8. implement provider planning and callable execution;
9. connect completion, stream, workspace and error ownership;
10. enable the provider in a versioned runtime registry declaration.

Unsupported combinations stay ineligible. They are not approximated merely to
increase the reported capability surface.

## 16. Architectural non-goals

At the current framework stage, the repository does not claim:

- an installed or callable Ascend Attention kernel;
- performance parity with NVIDIA FlashInfer;
- automatic support for arbitrary CANN or flash-attention-npu versions;
- correctness of a provider without its required evidence;
- a public API for choosing kernels or managing JIT objects;
- coverage of non-Attention FlashInfer components.

These boundaries keep the public API stable while allowing real operators to be
introduced incrementally behind auditable provider contracts.
