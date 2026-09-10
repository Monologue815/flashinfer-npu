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

The mixed-batch public surface is:

```python
from flashinfer_npu.attention import BatchAttention

wrapper = BatchAttention(kv_layout="NHD", device="npu:0")

wrapper.plan(
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_len_arr,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    page_size,
    causal=True,
    q_data_type="bfloat16",
    kv_data_type="bfloat16",
)

output, lse = wrapper.run(query, paged_kv_cache)
```

`BatchAttention` owns its workspace. `kv_len_arr` holds each request's total KV
length; it is not the classic paged wrapper's last-page-length array. The return
value always contains both output and LSE. The NPU example requires an installed
provider integration. A runnable dense/INT8 CPU example is available in
[`examples/attention_plan_run.py`](../examples/attention_plan_run.py); run it from
the repository root with `python3 -m examples.attention_plan_run`.

The exact Python signatures are defined by the public modules. The architectural
rules are:

1. The caller supplies tensors and semantic Attention options.
2. On the normal high-level path, the caller does not supply a provider name,
   kernel name, JIT handle, module handle or executable object.
3. `plan()` may be reused by subsequent compatible `run()` calls.
4. A new incompatible request requires replanning.
5. Planning failure leaves the previous active plan unchanged.

Single-request prefill and decode functions follow the same policy but do not
need to expose a reusable batch plan object.

FlashInfer also publishes separately named `*_with_jit_module` functions for
advanced callers that intentionally inject a compiled module. FlashInfer-NPU
keeps those low-level compatibility entries outside this normal automatic
dispatch contract; their existence does not make module handles part of a
batch wrapper's public `plan()` / `run()` lifecycle.

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
Adapter restrictions + capability/evidence admission
        |
        v
Candidate priority / score / select
        |
        v
Selected package authority revalidation
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

The declarative package bootstrap composes each adapter's plan gate with the
same evidence-bearing dispatch checks used by final authorization. A candidate
must support the canonical mode, dtype, exact QuantSpec, dimensions and metadata,
and have valid bound evidence, before it participates in priority or score
selection. Logical and provider-specific physical KV layouts use their respective
evidence paths. Rejected candidates retain diagnostic reasons and are not scored;
their callables are not resolved. The selected candidate is checked again before
callable binding. This is plan-time filtering, not execution-time fallback.

Admission is short-circuiting: when adapter, quantization or evidence checks
reject a candidate, the framework does not inspect that candidate's package
version. Its report contains the admission reasons, not speculative package
availability. An incompatible optional dependency therefore cannot interrupt
another valid candidate's plan. Candidates that pass admission still undergo
the exact package-version checks; unexpected errors from a relevant package
loader propagate, and do not silently authorize another implementation.

A wrapper may therefore plan dense KV, replan INT8 KV, and later replan dense KV
again using the same public methods. Each successful plan owns its exact operation,
adapter and scale bindings; `run()` neither selects again nor carries quantization
arguments over from an earlier plan. Inputs must match the current plan. If no
candidate supports a newly requested QuantSpec, replanning fails and leaves the
previous active plan usable. Actual choices depend on the installed, reviewed
provider capabilities, not on a built-in assumption that one named package owns
dense or quantized Attention.

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

The functional `single_prefill_with_kv_cache` and
`single_decode_with_kv_cache` APIs have no reusable public wrapper in
FlashInfer. For NPU tensor-like inputs they therefore create one ephemeral
mode-bound runtime per call, snapshot the registry/catalog, build a canonical
single-request plan, execute it once and discard the private runtime. Their
function signatures expose no provider, module or plan handle. Single prefill
uses explicit `backend="reference"` for the Host oracle; single decode uses a
`ReferenceTensor` query as the explicit Host opt-in because its upstream
signature has no backend parameter.

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

### 4.1 Custom-mask provider binding design

Status: proposed framework extension, not an enabled provider capability.
The Host oracle and `CustomMaskSpec` already describe masks, but public provider
single/paged/ragged prefill currently rejects `custom_mask` and
`packed_custom_mask`. A capability profile alone cannot close this gap: it says
which semantics an operation supports, not where the actual mask data comes from.

The pinned [upstream prefill implementation](https://github.com/flashinfer-ai/flashinfer/blob/919a24e5b1d971d50c97a3cd38862f801527eab5/flashinfer/prefill.py)
prepares masks during planning, packs an unpacked mask only when a packed one is
absent, uses little-endian segment packing, and retains mask buffers in the batch
wrapper. Graph-mode buffers have a separate lifetime/capacity contract. Ascend
must preserve the public lifecycle, without assuming CUDA buffer operations or a
particular provider's mask representation.

The extension should divide responsibilities as follows:

| Boundary | Required responsibility |
|---|---|
| Public frontend | Preserve packed-mask precedence; derive per-request QO/KV segment lengths; normalize rank, dtype and device facts without reading device values |
| Canonical plan | Record mask kind, bit order, segment offsets and logical sizes; preserve existing effective-causal semantics |
| Private plan resources | Carry an opaque mask payload separately from serializable plan metadata and keep it alive for every run using that plan |
| Provider admission | Require both semantic capability and an exact mask binding/representation; reject unsupported formats before package resolution |
| Materializer | Perform only declared copy/packing/format transformations through an injected implementation; account for buffers and workspace |
| Prepared provider plan | Bind materialized mask resources to the exact active-plan generation and operation |
| Run adapter | Inject the retained mask and any segment metadata into the catalog-authorized arguments, without adding a public run mask/plan handle |
| Completion/lifetime | Retain resources until outstanding execution completes; replacement planning must not release resources still in use |

Logical mask length for request `i` is `qo_len[i] * kv_len[i]`. Each request's
packed storage uses `ceil(logical_length / 8)` bytes; padding bits must not join
the next request. Element offsets and packed-byte offsets are different units
and must never share an untyped field. Mapping this representation into an
external dense/additive mask is an explicit provider transformation, not a
silent reinterpretation of bytes.

Mask ownership is part of the binding. A borrowed immutable tensor needs a
documented no-mutation-until-completion rule and a retained owner/lease; a copied
snapshot needs a declared copy operation and completion ordering before use.
The framework must not promise a snapshot while merely retaining a Python
reference. It also must not place device addresses or tensor payloads into the
public plan fingerprint. Private resource identity and prepared-plan binding
track the concrete payload separately from reusable semantic/cache identity.

Publication remains transactional: validate metadata and admission, prepare all
mask resources, then publish the new plan and resource set together. Failure
keeps the previous plan and mask resources usable. Replanning without a mask
must remove the previous mask binding. No automatic Host fallback, implicit
device-to-host read or framework-written packing kernel is permitted.

Implementation can proceed in independently verifiable increments: private
resource/binding schemas; injected metadata/materialization contracts; adapter
lowering with synthetic callables; public prefill activation after end-to-end
lifetime and failed-replan checks. Graph capture needs its own stable-address
and capacity proof and must remain unavailable until that proof exists. The
current rejection guards must remain in place until the public payload path is
complete. Real provider packing/copy/operator implementations require a separate
integration decision; none is introduced by this design.

The first internal building block is `attention.operator_mask`.
`AttentionMaskPlanMetadata.from_plan()` derives separately named
`logical_element_indptr` and `packed_byte_indptr` from the existing mask semantics,
and binds them to the plan fingerprint, admission fingerprint and generation.
Reusing equivalent semantics in a new generation still requires a new resource
binding. `CustomMaskSpec.numel` is strictly integer-valued and `packed` is a boolean.

`AttentionMaskPlanResource` retains an opaque payload and its owner by reference.
It never inspects payload contents, performs packing/copying, or serializes the
payload as part of its metadata. Two resources are not equal merely because they
share semantic metadata. Borrowing requires the owner to keep the payload
immutable; retaining the owner is not a device lease or a copied snapshot.
Only metadata has a diagnostic dictionary/fingerprint. These types are private
building blocks, not new model-facing parameters or provider execution authority.
`inspect_attention_mask_plan_resource()` supplies the next metadata-only boundary.
It validates the plan binding before invoking the injected tensor inspector once,
requests a read-only view, and requires a flat contiguous view with the exact
planned element count, dtype, expected device and requested power-of-two alignment.
Storage bounds are validated by the existing `TensorView` contract. Bool masks
remain bool and packed masks remain uint8; there is no implicit reshape, flatten,
packing, device transfer or contiguous copy. A higher-rank public source therefore
needs an explicit frontend normalization step before reaching this private boundary.

The returned `AttentionInspectedMaskPlanResource` retains both the original resource
and the metadata snapshot. It does not establish content immutability, inspect
padding-bit values or create a device lease. The borrowed-owner contract still
applies. `revalidate_attention_mask_plan_resource()` rechecks the exact plan binding
before inspecting the borrowed source again. All view fields must still match the
original snapshot, including storage identity, offset, capacity and alignment;
being compatible in shape and dtype alone is insufficient. A failed check never
refreshes the retained snapshot or authorizes the changed source. A successful
check returns the original retained resource, not a replacement binding.

This private check is intended for materializer/run adapters immediately before
use. It does not detect content-only mutation or allocation reuse hidden by an
inspector's storage identity, and cannot prevent concurrent changes after the
check. The integrating adapter must supply trustworthy storage identity and
separately enforce ownership, allocation lifetime and execution ordering through
completion. Public provider execution is not yet connected to this boundary.
`attention.operator_mask_binding` provides a private no-conversion argument
fragment. An `AttentionOperatorMaskArgumentSpec` explicitly maps the payload and
typed offsets to keyword arguments of one exact operation fingerprint. It accepts
only canonical row-major allow-mask segments: bool values or little-endian packed
bits, where true/set means visible. Packed-byte offsets are mandatory for the
packed representation and distinct from logical-element offsets. Offset arguments
must be declared host sequences; payload/offset arguments cannot overlap mutable,
quantization, page-table, output or LSE control roles. This mapping is an integration
declaration, not proof that a real package implements those semantics.

`lower_attention_mask_arguments()` rejects an incompatible declaration, plan,
mode or encoding before touching the tensor inspector, then revalidates the
borrowed source. Its result retains the inspected resource and owner alongside
the argument fragment. The run adapter must retain that result through completion;
keeping only the extracted argument tuple does not retain a separate owner. The
fragment alone is neither an active-provider binding nor execution authority.
No packaged CANN or flash-attention-npu operation is given a mask mapping by
default. Inverted, additive, dense provider masks and device-resident offset
tables require explicit transformation/materialization support, not reinterpretation.

`AttentionOperatorMaskRunAdapter` composes this fragment with an existing run
adapter, outside the tensor-validation adapters. It is constructed for one exact
active plan and catalog operation. A different prepared state or generation
requires a new adapter. It validates the request, base call identity and signature
before inspecting the mask, and rejects existing mask/offset arguments (including
`None` placeholders) or a preexisting `custom_mask` input view instead of overwriting
them. After source revalidation it appends the mask view for downstream completion
checks and rejects writable operation arguments that alias the borrowed mask.
Completion validation likewise rejects returned output/LSE views that alias
`custom_mask`, even when a profile permits general output/input aliasing. This
post-call check detects an invalid result; it cannot undo a provider's writes.

Internal run contract version 10 adds `AttentionLoweredOperatorCall.retained_resources`:
a process-local immutable tuple of owners, excluded from representation and value
comparison. The mask adapter preserves existing retained resources and appends its
mask fragment. Ordinary call decoration through `dataclasses.replace` preserves
these owners. The execution integration must retain the call through completion;
Python return, an output metadata receipt or a new plan is not proof that queued
device reads have finished. This field is not a lease, event tracker or public
model-facing run parameter.

The private [call retention registry](attention_operator_call_retention.md) can
retain complete lowered calls before invocation and release each only after its
own injected event reports completion. It preserves pending calls on invocation,
recording or query failures. Internal runtimes can receive an event recorder at
construction and keep a stable registry across plan publication and executor
replacement. Result-validation failures do not release pending calls. Without an
event recorder, calls carrying explicit retained owners fail before execution.
This is an opt-in internal runtime path, not an automatic device-event integration
or a change to public `run()` results.

With a recorder configured, later `plan()`/`run()` entries make one non-waiting
collection pass over prior calls. Only completed invocations are released;
unfinished or unrecorded calls remain retained. Query failures stop new work with
a metadata-only collection report, while independent completed calls can still
be reclaimed. This does not supply idle-time polling. The internal runtime's
non-waiting `close()` enters a closing state and rejects new work, then can be
retried until all tracked calls complete. It drops active-plan and execution
references only after the retention registry becomes empty. Public-wrapper
teardown and independently queued preparation work still need integration.

`AttentionMaskPlanRunAdapterBinder` prepares a borrowed canonical source during
the private plan transaction. It derives resource metadata from the actual
candidate generation, validates the selected operation mapping and encoding,
inspects the source, and constructs the plan-bound mask adapter. Internal
`AttentionOperatorRuntime.plan()` accepts a `run_adapter_plan_binder`; the wrapper
session invokes it before publishing its candidate adapter. Later workspace or
executor-binding failures still leave the runtime's old plan and adapter intact.
Successful planning without a binder installs the ordinary adapter, removing any
old mask binding while pending calls continue to retain their own resources.

The binder protocol declares whether call retention is required. A mask binder
requires a configured event recorder, and a custom-mask runtime plan requires a
resource adapter; both missing-input checks precede package resolution. Binder
results must match the selected provider and operation. These are private
integration hooks, not additional model-facing `plan()` parameters. No CANN or
flash-attention-npu mask binder is registered automatically.

Mask binders also implement metadata-only candidate admission. One binder can
carry several `AttentionOperatorMaskArgumentSpec` mappings, keyed by exact
operation fingerprint and encoding. Duplicate operation/encoding pairs are
rejected. Admission checks the mapping, encoding, argument roles, mode and planned
device without inspecting the source tensor. The implementation registry applies
this check before a candidate's own package/evidence probe and before priority or
score selection. Incompatible candidates remain in the resolution report with
reasons, receive no score and cannot shadow compatible lower-priority candidates.
Candidates that pass still undergo all existing capability/evidence checks;
declaring a mask mapping does not authorize an operation.

These checks are per-plan and do not mutate the frozen registry. An unmasked plan
without a resource binder uses ordinary selection again. Concrete source inspection
and adapter binding remain after selection and revalidate the chosen mapping
before publication. A late tensor-metadata failure rolls back the plan; it does
not trigger an implicit retry with another provider. Custom-mask binders must
provide pre-probe admission. Custom resolver implementations must implement
`resolve_with_admission()` to participate; unsupported resolvers are rejected
before their ordinary resolution method is called.

Materialization, device-specific event recording/polling, wrapper teardown
integration and public activation remain to be implemented before the rejection
guards can be removed. The bundled provider catalogs still have no automatically
installed real mask representation mappings.

### 4.2 Borrowed batch-mask frontend metadata

The private `adapt_framework_batch_custom_mask()` frontend adapter accepts flat
bool or per-segment packed uint8 tensor-like inputs and returns a `CustomMaskSpec`
with the original selected payload. It reads only shape, dtype and device facts;
it does not inspect tensor values, pack bits, copy data or invoke a tensor library.
When both inputs are present, the packed input takes precedence and the bool
input is not inspected. Invalid packed metadata is an error, not a reason to
fall back to the bool input.

Segment sizes must be nonnegative integers. Bool length is their sum; packed
length is the sum of each segment's individually rounded byte count. Empty
segments contribute no elements or bytes. The selected payload must be rank one
with exactly that length, the corresponding dtype and the workspace device.
This helper does not flatten single-request rank-two masks.

Frontend acceptance establishes plan facts only. Integration must still pass the
borrowed payload and its owner to the plan-bound mask binder, whose inspector
checks storage, contiguity and alignment and whose admission checks the selected
operation. Public provider wrappers do not yet call this helper: their rejection
guards remain until resource preparation and completion tracking are integrated.

### 4.3 Quantized KV and custom-mask composition

Quantization and mask bindings describe independent inputs to the same selected
operation. The quantization adapter validates the active `QuantSpec`, logical
storage and scale metadata, then maps K/V storage and their scales to authorized
arguments. The outer plan-bound mask adapter preserves those arguments and adds
the borrowed mask and segment offsets. Neither adapter dequantizes KV, converts
mask encodings or performs Attention arithmetic.

Bool masks carry logical segment offsets; packed masks additionally carry
per-segment byte offsets. These offsets describe QO/KV sequence lengths, not
quantization groups. A provider must independently satisfy both bindings and all
existing capability and authorization checks; support for either feature alone
does not establish support for their combination.

With the default access policy, caller-owned output/LSE buffers must not overlap
quantized storage or scales. These checks precede invocation. Borrowed masks
remain protected even when general output/input aliasing is explicitly allowed.
Completion validation includes quantized input views and the mask view; invalid
returned aliases do not produce a successful run receipt. Post-invocation checks
cannot undo writes already performed by an external callable.

An input validation failure leaves the active plan reusable and submits no call.
Replanning between quantized and dense KV replaces the adapter chain, so dense
runs do not inherit scale arguments from an earlier generation. Already submitted
calls retain their own lowered arguments and mask resources until their matching
completion events report completion, including when result validation fails.

This composition is an internal framework contract. It does not enable public
provider mask paths, prove numerical accuracy or establish a real package's
quantized-mask capability. Model-facing interfaces remain unchanged.

## 5. Runtime registry snapshot

The provider wrapper captures one immutable registry snapshot when constructed;
each subsequent `plan()` uses that captured generation.
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

Selection is deterministic and has two explicit levels. Declared priority is
an administrative deployment rank. After all plan gates have run, only
accepted candidates at the highest priority are passed to their optional
plan-specific scorer. Stable implementation identity determines report order,
never selection or discovery order. A candidate is eligible only when all of
the following agree:

- the canonical plan;
- the provider operation catalog;
- provider-specific plan constraints;
- declared package/version range;
- runtime and device authority;
- numerical and physical-layout evidence;
- callable or artifact identity.

The scorer returns a bounded integer preference plus a non-empty `source` and
`reason`. A candidate without a scorer receives score zero. Higher scores win;
an equal top score is ambiguous and fails closed. Rejected and lower-priority
candidates are not scored, so an irrelevant integration cannot block the
selected priority tier.

A scorer is integration-owned, identity-bound to one provider operation and
side-effect-free. It may inspect the immutable canonical plan and injected,
prevalidated policy or tuning records. It must not import the external package,
probe a device, read tensor contents, run an operator or perform online tuning.
`explain()` exposes gate reasons and score evidence without package loading or
device work, and its fingerprint therefore binds the decision to the plan.
`run()` never rescales, rescores, retries or switches the selected provider.
The versioned rule schema and ambiguity rules are defined in
[Attention plan scoring policy](attention_plan_scoring_policy.md).

## 7. Operation catalog

The operation catalog is the provider's versioned declaration of callable
surface. It maps a framework operation to an exact external API contract:

- module and symbol path;
- single or batch lifecycle;
- required and optional parameter names;
- argument ownership and defaults;
- supported modes, layouts and dtypes;
- output and workspace conventions;
- exact optional caller-owned `out`/`lse` argument names and mutability;
- quantization parameter bindings;
- provider-plan and run-time lowering rules.

The catalog and provider factory share one rule source. A plan accepted by the
catalog must not later be rejected by a duplicated, drifting eligibility rule.

When one production bundle contains operations that require different loading
boundaries, an `AttentionOperatorRoutedPackageLoader` maps every catalog
operation to an exact delegate. Package names and callable paths use exact
lookup, and the complete canonical route fingerprint enters the bundle loader
identity. It performs no prefix guessing, import fallback or provider
selection. See
[Attention package loader routing](attention_package_loader_routing.md).

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
- complete runtime-resolution report fingerprint, including provider scores;
- implementation, catalog and operation identity;
- package/runtime authority;
- provider materialization and external plan identity;
- quantization and physical-layout bindings;
- callable or verified artifact/module/symbol identity;
- executor identity;
- compatibility rules for future `run()` inputs.

Publication is atomic. Any error keeps the previous active plan intact.

After publication, batch wrappers expose `plan_selection` as a read-only
diagnostic value. It contains only the Attention mode, route, backend,
provider/operation identifiers, registry generation and plan fingerprints. A
scored selection also includes the selected integer score, source, reason and
complete resolution-report fingerprint. For a declaration-bound scoring
manifest it additionally includes manifest id/fingerprint and the selected
policy id/fingerprint, after the runtime has matched those structured score
fields against the frozen manifest binding. For a production bundle install it
also includes the bundle id/fingerprint after the runtime has matched the
selected declaration to that bundle's exact registration set. It contains no callable, module,
executor, opaque provider state or mutable plan handle. The property is not an
input to `run()` and does not transfer plan ownership to the caller.

Reference plans report `route="reference"` and contain no provider identity.
Provider plans report the exact registry generation captured by the wrapper.
If replanning fails, both the old active plan and its selection summary remain
unchanged.

For production provider installations, the captured registry snapshot carries
both the non-executable scoring-manifest binding and provider-bundle binding for
that generation. The latter closes the catalog, loader type/id, registration
declarations and manifest into one fingerprint. These are integration audit
surfaces, not `plan()` or `run()` arguments. Legacy and synthetic registry
installs carry no bundle binding.

A successful completion-validated provider run copies the same bundle,
manifest and selected-policy identity into its atomic run receipt. The receipt also binds the
active-plan fingerprint, which transitively contains the structured policy
identity through the complete resolution report. Execution or completion
failure publishes no receipt.

`workspace_size()` uses the same frozen registry and plan gates through an
unpublished runtime fork. It may resolve and prepare the selected provider in
order to derive its exact resource binding, but it never executes Attention or
changes the wrapper's active plan. Returned sizes describe caller-owned wrapper
workspace only; package-managed internal scratch is not reported as zero memory
usage.

## 13. `run()` validation and lowering

`run()` performs no automatic backend reselection. It:

1. requires an active plan;
2. validates query, KV and auxiliary inputs against that plan;
3. validates device, stream, workspace and provider generation;
4. lowers canonical inputs to the selected operation's exact arguments;
5. verifies the final executable identity;
6. invokes the authorized executor once;
7. normalizes output/LSE and completion ownership to the public contract.

Every package-backed provider installs a common run-tensor validation adapter
before its operation-specific lowering. Bootstrap therefore requires a
metadata-only tensor inspector and an explicit `AttentionTensorAccessPolicy`,
even when the operation is not quantized. The adapter checks the exact
mode-dependent query shape, planned Q dtype and provider device on every run.
For an unquantized plan it also normalizes FlashInfer's packed paged KV tensor
or separate `(K, V)` pair into a metadata-only `KVCacheView`. NHD/HND shape,
page capacity, planned KV dtype, device, alignment, provider KV contiguity
policy and separate K/V overlap are closed before provider-specific lowering.
Quantized KV remains on the exact QuantSpec/physical-layout validation path;
its physical storage, scale and optional zero-point views also obey provider
alignment and participate in the same output/input alias gate. Virtual implicit
unit scales are logical omissions and therefore have no device-address
alignment requirement.
For optional caller-owned `out`/`lse`, it checks planned output/LSE shape,
output dtype/FP32 LSE dtype, device, writable storage, alignment, provider
contiguity policy and forbidden aliases against Q and every validated KV
component. The catalog must explicitly name each buffer argument and mark it
mutable; a separate generic adapter then injects
only the provided buffers under those exact names. Operations without those
declarations keep rejecting caller-owned buffers. Original tensor objects are
forwarded unchanged after validation; this layer neither reads device data nor
performs a hidden copy/cast. A provider that needs a conversion must declare a
separate materialization path rather than weakening the run contract.
After the provider callable completes, the executor validates the public return
arity again. A single-output run must return one non-container value; an LSE run
must return exactly `(output, softmax_lse)`, and neither public value may be
missing. When the caller supplied `out` or `lse`, the corresponding returned
object must be that exact buffer object. The completion receipt records which
public results retained caller ownership. This follows the FlashInfer wrapper
contract and prevents an external package from silently replacing a validated
buffer with a newly allocated tensor or an unrelated view.

Provider-allocated results additionally have a plan-bound metadata completion
contract. It uses the same injected tensor inspector and access policy as run
lowering, without importing torch or reading device data. `output` must match
the planned output shape, output dtype and provider device; `softmax_lse` must
match the planned LSE shape, FP32 and the same device. Both results must be
writable, meet provider alignment/contiguity rules and occupy non-overlapping
storage. A completion receipt binds their metadata fingerprints to the active
plan, exact operation and access policy. This validator is a distinct boundary
so provider invocation authority and returned-tensor acceptance cannot be
conflated.
Package bootstrap enables this result validation by default. The validator
factory is carried by the resolved runtime, then bound only after the complete
provider active plan exists and before that generation is atomically published.
`run()` clears the previous completion receipt, invokes the already authorized
provider exactly once, validates the returned tensors and only then exposes the
result to the public wrapper. Validation failure therefore cannot publish a
result or a success receipt and does not trigger a second provider invocation.
Replanning constructs a new validator for the new active-plan fingerprint.
JIT and non-JIT runtimes share this completion boundary; it wraps neither the
callable nor the JIT executor, so their existing identity bindings remain
unchanged. A bootstrap spec can disable result validation only explicitly,
which is intended for metadata-free synthetic integration fixtures rather than
production provider registrations.
The outer run-tensor validator also freezes the exact metadata views it has
accepted into the lowered call. For dense paths these are Q/K/V; for quantized
paths they include Q, physical key/value storage, scales, optional zero-points
and every tensor-valued runtime/head scale. Provider-specific inner adapters
cannot inject this evidence: the quantization validator must create its portion
first and the outer validator then prepends the query view (and dense KV views).
The completion validator requires that query evidence and, unless the one
shared access policy explicitly permits output/input aliasing, rejects output
or LSE overlap with every frozen input view. The completion receipt includes
both input and result view fingerprints, closing the metadata chain from run
admission through public result publication.

After execution and completion both succeed, the runtime publishes one atomic
run receipt containing their exact receipt fingerprints. Active plan, provider,
operation and ordered return names must agree. Callable failure, completion
failure and replanning all clear the previous atomic receipt, so “the selected
callable ran” and “the returned tensors were accepted” form one auditable run
fact rather than two unrelated observations.

The active-plan fingerprint includes the runtime-resolution fingerprint. Every
execution and completion receipt already binds the active-plan fingerprint, so
a successful atomic run receipt transitively proves the exact scored provider
decision used for execution. `plan_selection` publishes both fingerprints and
rejects a resolution fingerprint that differs from the active plan.

For JIT, the module/callable executor binding is necessarily created before the
final active-plan runtime binding. `bind_runtime()` may then produce a different
executor object. A separate JIT runtime-executor binding joins the original JIT
executor-binding fingerprint, final operator runtime-binding fingerprint,
active-plan fingerprint and exact post-bind executor object. Every JIT run
validates this final binding before invocation. A strict JIT provider's atomic
run receipt also embeds the JIT runtime-executor binding fingerprint.

Provider-specific arguments are produced internally. Unknown arguments, missing
bindings, stale receipts or identity drift are hard errors.

The internal run request carries `return_lse` as a required boolean semantic and
keeps `q_scale`, `k_scale`, `v_scale` and ragged-prefill `o_scale` as four
independent optional sources; none is a provider handle or alters plan
selection.
When one of these sources or a dedicated Q/K/V head scale is a tensor, the
quantization adapter retains its validated metadata view, enforces provider
alignment, and includes it in the output/input alias gate. Finite scalar forms
remain value-only inputs and therefore have no storage identity.
Paged/ragged wrappers set it from the public flag. A provider adapter must map
that intent to the selected operation's exact LSE-control argument and return
schema. The holistic `BatchAttention` contract always requests LSE because its
public return value is fixed to `(output, lse)`.

For a quantized plan, the quantization binding independently decides whether
`run.q_scale`, `run.k_scale`, `run.v_scale` and `run.o_scale` map to exact catalog
arguments. The default policy rejects each source. An `o_scale` argument binding
also declares its eligible plan output dtypes, so output quantization cannot leak
onto a float-output route that merely shares the same KV `QuantSpec`. The base
provider adapter sees none of the authorized scale values; the quantization
adapter injects them only after the selected operation, complete `QuantSpec`,
output dtype and argument names are closed.

An immutable resource binding is derived from the selected operation before
active-plan publication. It distinguishes package-managed from caller-managed
workspace and returned tensors from mutable output arguments. The current
documented package APIs expose returned output/LSE values and no wrapper
workspace argument. Their caller workspace requirement is therefore zero while
package-internal scratch remains outside this contract. Public `out` or `lse`
buffers are rejected unless an exact mutable-argument binding exists.
Package-managed workspace replacement may retain the active plan because the
buffers are not submitted to the operation; wrapper and runtime publish the new
binding generation together. Caller-managed replacement requires an explicit
completion/lease binding and otherwise fails closed.

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

A production integration follows the version-pinned procedure in
[`attention_provider_onboarding.md`](attention_provider_onboarding.md). At a high
level it is added in this order:

1. document the exact external package version and API;
2. add catalog entries for only the supported Attention operations;
3. implement pure plan gates and structured rejection reasons;
4. declare package/bootstrap metadata without import-time side effects;
5. publish a data-only runtime declaration and reject spec/catalog drift;
6. implement lazy package and callable resolution;
7. implement tensor/metadata materialization and physical-layout evidence;
8. bind quantization parameters exactly;
9. implement provider planning and callable execution;
10. connect completion, stream, workspace and error ownership;
11. enable the provider in a versioned runtime registry declaration.

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
