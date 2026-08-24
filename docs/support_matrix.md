# Support Matrix

This file records only configurations that have been exercised by automated
checks. Absence from the table means unverified, not implicitly supported.

## Host-side architecture checks

| Date | Platform | Python | Result | Scope |
| --- | --- | --- | --- | --- |
| 2026-08-05 | macOS arm64 | 3.9.6 | Pass | 129 Host tests: Attention schema/plan/reference, quantized KV, trace JSON/replay CLI, manifests, wheel install |
| 2026-08-11 | macOS arm64 | 3.9.6 | Pass | 145 Host tests: Attention workspace query/lifecycle, 6-case corpus, 36/36 coverage cells, replay CLI |
| 2026-08-13 | macOS arm64 | 3.9.6 | Pass | 156 Host tests: TensorView stride/bounds/alignment, quantized views, alias/writable/device/stream contract |
| 2026-08-13 | macOS arm64 | 3.9.6 | Pass | 167 Host tests: lazy Torch metadata adapter using protocol fakes; dense/packed/quantized KV, stream and failure boundaries |
| 2026-08-13 | macOS arm64 | 3.9.6 | Pass | 184 Host tests: 10-case/42-cell corpus, numerics policy, 9-dimension metadata admission, 32 dense-vs-INT8 and 24 NHD-vs-HND property seeds |
| 2026-08-13 | macOS arm64 | 3.9.6 | Pass | 197 Host tests: 12-case/45-cell corpus, paged INT4, groupwise decode, manifest-revalidated capability evidence; registered runtime profiles remain 0 |
| 2026-08-19 | macOS arm64 | 3.9.6 | Pass | 214 Host tests: strict bounded JSON envelope and bidirectional capability-profile/kernel-descriptor binding; runtime profiles/kernels remain 0 |
| 2026-08-19 | macOS arm64 | 3.9.6 | Pass | 224 Host tests: versioned Attention execution identity, persistent graph-resource fingerprint and stale Host structural-capture rejection; no device graph/runtime claimed |
| 2026-08-19 | macOS arm64 | 3.9.6 | Pass | 234 Host tests: evidence-bearing Attention dispatch receipt, independent float/int workspace formulas, receipt-to-kernel-identity revalidation; packaged profiles/kernels remain 0 |
| 2026-08-19 | macOS arm64 | 3.9.6 | Pass | 243 Host tests: kernel manifest v2, structured artifact provenance/content checks and versioned launch ABI; packaged kernel manifest remains empty |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 255 Host tests: kernel manifest v3, frozen Attention C POD/error ABI, logical/binary ABI binding, address/allocation-generation/stream/event lease lifecycle; no NPU runtime used |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 262 Host tests: six-mode canonical Attention metadata wire, auxiliary/run-options ABI, INT4 logical-vs-physical dtype separation, strict bounds/enum/offset/padding/stale-plan rejection; local Host only |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 270 Host tests: auxiliary role/shape/access contract, 64-byte run-options, TensorView/KV/aux POD materialization, component-table/address lease binding and capture scalar stale rejection; local Host only |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 277 Host tests: Host/device lease-domain separation and ownership-complete 13-argument Attention launch packet; metadata/options/component/workspace/stream/generation mutation rejection; local Host only |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 286 Host tests: fake-provider artifact bytes/builtin verification, resolved-symbol lifetime, sync/retry/async error ownership, submit-unknown device/Host lease retention, Host descriptor concurrency and completion-event gates; no artifact/runtime loaded |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 294 Host tests: four-state submit-unknown recovery, exact runtime-generation teardown/quiescence, shared device/Host/symbol registries and 8-way concurrent prepare gate; fake provider only, no artifact/runtime loaded |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 306 Host tests: Attention core 0 missing (32 reference/2 framework), injected single-JIT call ABI, three deprecated forward aliases, shared-page/segment-packed-mask property, execution identity v3 stream claim, multi-stream writable lease gate, partial-event and poll/release/teardown fault interleavings; no NPU runtime used |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 314 Host tests: mixed BatchAttention empty-Q/KV, shared/repeated pages, causal NHD/HND GQA single-oracle equivalence and exact INT8/dense equivalence; protocol trace v1 route/state/stream/owner/recovery/quiescence gates; no NPU runtime used |
| 2026-08-20 | macOS arm64 | 3.9.6 | Pass | 322 Host tests: context-local automatic JIT/provider protocol recorder, success/failure/recovery/quiescence evidence, nested/8-way thread isolation, incomplete-publication gate, protocol corpus numerical binding and validation CLI; no NPU runtime used |
| 2026-08-21 | macOS arm64 | 3.9.6 | Pass | 326 Host tests: mixed packed INT4 and asymmetric UINT8 channel/per-head runtime scale, shared/repeated pages, window/soft-cap explicit-dequant properties; 14-case corpus v4 and 51/51 coverage; isolated wheel pass; no NPU runtime used |
| 2026-08-21 | macOS arm64 | 3.9.6 | Pass | 334 Host tests: versioned quantization/backend accuracy budgets and metrics, strict paired-trace identity, exact/lossy/nonfinite/backend-injection checks, 4-case accuracy corpus including expected scale-overflow rejection, CLI and isolated wheel pass; no NPU runtime used |
| 2026-08-21 | macOS arm64 | 3.9.6 | Pass | 338 Host tests: post-dispatch accuracy binding revalidates paired result, dispatch receipt, profile/rule/environment, capability evidence, kernel/artifact/ABIs; cross-workload, expected-rejection and tamper gates; isolated wheel pass; synthetic Host fixtures only |
| 2026-08-21 | macOS arm64 | 3.9.6 | Pass | 342 Host tests: accuracy execution binding joins candidate to launch packet, execution/lease/stream/runtime identity, successful provider submit/completion/event and released protocol trace; async failure and drift gates; runner origin remains explicitly unattested |
| 2026-08-21 | macOS arm64 | 3.9.6 | Pass | 353 Host tests: reversible quantized storage/scale/zero-point blocked/permuted physical descriptors, packed-INT4-before-blocking, canonical padding, catalog and explicit logical/physical conversion plans; KV POD v1 rejects under-described non-logical launch; synthetic layouts only, no converter/kernel/NPU runtime used |
| 2026-08-21 | macOS arm64 | 3.9.6 | Pass | 359 Host tests: frozen 192-byte KV POD v2 and independent binary ABI v2, exact descriptor/catalog/profile/rule/environment/kernel/evidence/receipt native-layout binding, tamper and v1 compatibility gates; v2 packet/provider not connected, synthetic Host fixtures only, no converter/kernel/NPU runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 418 Host tests through checkpoint 017: public BatchAttention provider runtime, deterministic implementation selection, plan-time fake tensor materialization, final-plan-bound injected callable execution and output/LSE normalization; no external operator package or NPU runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 426 Host tests through checkpoint 018: lazy distribution metadata gate, exact adapter-authorized package versions, catalog callable resolution/signature binding and unbound executor publication; fake loaders only except Python stdlib path resolution, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 433 Host tests through checkpoint 019: package metadata and capability authority before callable import, mandatory plan tensor materialization, final runtime binding, repeat run reuse and failed-replan atomicity; injected fakes only, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 438 Host tests through checkpoint 020: CANN v2 and flash-attention-npu v3 pure plan gates share exact admission rules with prepare factories, deterministic multi-reason rejection and gate/factory matrix consistency; no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 445 Host tests through checkpoint 021: evidence-bearing package authority composes exact environment, conformance evidence, kernel/artifact/launch+binary ABI provenance, provider ownership and strict NPU device identity; synthetic fixtures only, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 453 Host tests through checkpoint 022: declarative package runtime bootstrap composes exact catalog/package policy, evidence authority, deterministic NPU resolver, provider lowering and materializer while preserving staged metadata/import/materialization effects; synthetic fixtures only, default integration set empty, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 461 Host tests through checkpoint 023: atomic process-bootstrap registry snapshots, generation compare-and-swap, future-wrapper-only integration swaps and unchanged provider-free public plan/run signatures; injected fake resolver only, default registry empty, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 469 Host tests through checkpoint 024: exact capability QuantSpec-to-catalog argument closure, independent K/V scale and asymmetric zero-point sources, explicit runtime scale policy and composed plan gate; synthetic bindings only, real provider quant gates remain closed, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 477 Host tests through checkpoint 025: generic quantized provider KV input, exact active-plan binding, storage unwrap, independent scale/zero-point and explicit runtime multiplier injection, collision protection and non-executing lowering; synthetic objects only, real provider quant gates remain closed, no NPU package/runtime used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 485 Host tests through checkpoint 026: injected metadata-only tensor inspection before provider lowering, exact dtype/shape/contiguous-stride/device/page-capacity/zero-point/alias validation through TensorView, QuantizedTensorView and KVCacheView; synthetic views only, no NPU package/runtime/operator used |
| 2026-08-24 | macOS arm64 | 3.9.6 | Pass | 493 Host tests through checkpoint 027: exact provider physical-layout catalog closure, descriptor-driven physical shape/alignment validation, unblocked page-axis requirement, KV POD v2 identity gate and intentional fail-closed runtime authority without real non-logical conformance evidence; synthetic metadata only, no NPU package/runtime/operator used |

## Remote framework checks

| Date | Platform | Python | Result | Scope |
| --- | --- | --- | --- | --- |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 017 isolated wheel, 5/5 guarded callable framework tests; no CANN environment load, NPU query, external operator import or Attention operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 018 isolated wheel, 8/8 lazy package metadata/callable resolution tests with fake package loader and Python stdlib path only; no CANN environment load, NPU query, external Attention package import or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 019 isolated wheel, 6/6 package-to-authority-to-materialization-to-bound-run transaction tests with injected fakes; no CANN environment load, NPU query, external Attention package import or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 020 isolated wheel, 5/5 CANN v2 and flash-attention-npu v3 plan-gate/factory shared-rule tests; no CANN environment load, NPU query, external Attention package import or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 021 isolated wheel, 7/7 evidence authority, exact device/backend policy, environment drift, provider ownership and tuned-kernel admission tests with synthetic fixtures; no CANN environment load, NPU query, external Attention package import or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 022 isolated wheel, 8/8 declarative bootstrap, empty default, staged metadata/import/materialization, NPU routing and identity rejection tests with synthetic fixtures; no CANN environment load, NPU query, external Attention package import or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 023 isolated wheel, 8/8 atomic registry snapshot, generation compare-and-swap, future-wrapper isolation and provider-free public signature tests with a fake resolver; no package probe, CANN environment load, NPU query, external Attention package import or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 024 isolated wheel, 8/8 exact capability QuantSpec-to-catalog argument closure, independent scale/zero-point source, runtime scale policy and composed plan-gate tests with synthetic bindings; real provider quant gates remain closed, no package import, NPU query or operator call |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 025 isolated wheel, 8/8 generic quantized KV input, storage unwrap, independent bound scale/zero-point/runtime multiplier injection, collision protection and plan-materialization reuse tests with opaque Python objects; callable not executed, no NPU package/runtime used |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 026 isolated wheel, matching SHA-256 inputs and 8/8 metadata-only quantized provider KV validation tests for dtype/shape/contiguous stride/exact device/page capacity/zero-point/alias gates; forbidden module set empty, no torch_npu/flash_attn import, package callable or Attention operator execution |
| 2026-08-24 | Ubuntu 24.04 x86_64 | 3.10.8 | Pass | Checkpoint 027 isolated wheel, matching SHA-256 inputs and 8/8 exact physical-layout catalog/descriptor, shape/alignment/page-axis, KV POD v2 and fail-closed evidence-authority tests; forbidden module set empty, no torch_npu/flash_attn import, package callable or Attention operator execution |

Verified Host quantized KV storage contracts are `int8`, `uint8`,
`int4_packed`, and `uint4_packed` with `physical_layout="logical"`. This is
framework/reference evidence only and does not imply Ascend runtime support.

The Torch adapter row is protocol-level evidence only. PyTorch and torch_npu are
not installed in this configuration, so no real `torch.Tensor`, dispatcher,
allocator, device, or stream behavior has been verified.

## Ascend runtime configurations

No Ascend runtime tuple has been verified yet. Before the first kernel is
registered, this table must pin all of the following:

- SoC model and revision
- Driver and firmware
- CANN toolkit and runtime
- Python, PyTorch, and torch_npu
- Ascend C compiler
- Supported dtype and feature probes

The runtime must reject an artifact whose manifest fingerprint does not match
the observed tuple. Broad version claims such as `CANN >= x` are not allowed
until covered by CI evidence.
