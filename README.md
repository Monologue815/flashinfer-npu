# FlashInfer-NPU

FlashInfer-NPU is a work-in-progress inference kernel library for the Ascend
software stack. It follows FlashInfer's API domains and plan/run execution
model while implementing kernels for Ascend AI Core rather than translating
CUDA kernels.

The active development track is currently limited to the framework layer of
FlashInfer-compatible Attention: single/batch prefill and decode, paged/ragged
KV metadata, mixed `BatchAttention`, plan/run semantics, and a zero-dependency
scalar reference oracle for small conformance cases. No NPU kernel code is
being developed in this phase.

The intended public lifecycle matches FlashInfer: users call `plan()` once and
then call `run()` without passing a plan or provider handle. Provider selection,
page-table materialization, operation-version binding and callable authorization
remain internal to the wrapper:

```python
from flashinfer_npu.attention import BatchAttention

attention = BatchAttention(kv_layout="HND", device="npu:0")
attention.plan(
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
)
output, lse = attention.run(q, (k_cache, v_cache))
```

This example documents the target interface. Internally, a declarative
bootstrap composes exact package versions, capability evidence, kernel/ABI
provenance, provider adapters and tensor materializers into the automatic NPU
resolver. The packaged declaration set is intentionally empty today: no real
CANN or flash-attention-npu package is imported or called until a separately
verified integration is installed.

Provider integrations are installed atomically at process bootstrap. Each NPU
`BatchAttention` instance captures one immutable resolver generation when it is
constructed, so a later integration update cannot change an already-created
wrapper's plan/run authority. This bootstrap control remains separate from the
model-facing API.

Quantized provider candidates have an additional admission boundary: every
capability `QuantSpec` must map key/value scale and, when applicable,
independent zero-point sources to exact quantization arguments in the selected
catalog operation. Merely exposing quantization-named package parameters is
not treated as runtime support.

The unchanged `kv_cache` run slot can carry an
`AttentionOperatorQuantizedKVInput` for a future provider integration. The
framework unwraps K/V storage and injects only the scale, zero-point and
optional runtime multiplier arguments authorized by the selected exact
binding; this lowering remains non-executing until the callable authority is
completed.

The authoritative design is in
[`docs/architecture.md`](docs/architecture.md). The current repository is a
Phase 0 host-side architecture skeleton; it does not yet contain runnable
Ascend C kernels.

The Attention-specific framework contract is documented in
[`docs/attention_framework.md`](docs/attention_framework.md).
The FlashInfer-aligned wrapper-owned plan/run lifecycle and the boundary for
future CANN/flash-attention-npu provider selection are documented in
[`docs/attention_plan_run_dispatch_design.md`](docs/attention_plan_run_dispatch_design.md).
The executable Host contract for INT8/UINT8/packed-INT4 KV Cache is documented
in [`docs/attention_quantization.md`](docs/attention_quantization.md).
Versioned correctness trace and replay semantics are documented in
[`docs/attention_trace.md`](docs/attention_trace.md).
The separate lifecycle/stream/resource-ownership trace schema for injected JIT
and provider routes is documented in
[`docs/attention_protocol_trace.md`](docs/attention_protocol_trace.md).
Workspace capacity/query/lifecycle semantics are documented in
[`docs/attention_workspace.md`](docs/attention_workspace.md).
The joint plan/workspace/tensor/capability identity used to guard execution
reuse is documented in
[`docs/attention_execution_identity.md`](docs/attention_execution_identity.md).
The framework-independent tensor/stride/alias/stream adapter target is in
[`docs/attention_tensor_contract.md`](docs/attention_tensor_contract.md).
The optional metadata-only Torch adapter and its explicit runtime verification
boundary are documented in
[`docs/attention_torch_adapter.md`](docs/attention_torch_adapter.md).
The versioned exceptional-value and metadata admission policies are in
[`docs/attention_numerics.md`](docs/attention_numerics.md) and
[`docs/attention_resource_limits.md`](docs/attention_resource_limits.md).
The evidence-bearing backend support declaration and dispatch gate are defined
in [`docs/attention_capability_profile.md`](docs/attention_capability_profile.md).
The revalidatable plan-to-profile-to-kernel selection record is specified in
[`docs/attention_dispatch_receipt.md`](docs/attention_dispatch_receipt.md).
Structured kernel provenance and launcher ABI requirements are documented in
[`docs/attention_kernel_artifact.md`](docs/attention_kernel_artifact.md).
The frozen Attention C/POD/error ABI and address-generation/event lease state
machine are specified in
[`docs/attention_launcher_abi.md`](docs/attention_launcher_abi.md).
The canonical six-mode CSR/page-table plan metadata wire format is documented
in [`docs/attention_plan_metadata_wire.md`](docs/attention_plan_metadata_wire.md).
The validated auxiliary/run-options contract and TensorView/KV/aux POD
materialization path are documented in
[`docs/attention_launch_binding.md`](docs/attention_launch_binding.md).
The ownership-complete Host/device lease model and canonical 13-argument
submission record are documented in
[`docs/attention_launch_packet.md`](docs/attention_launch_packet.md).
The Host-only artifact verification, symbol resolution, provider submission,
completion-event, and unload lifecycle are documented in
[`docs/attention_launcher_provider.md`](docs/attention_launcher_provider.md).
The two single-request injected-JIT compatibility entries model upstream scratch,
output/LSE allocation and argument ordering only; they do not compile or load an
Ascend artifact and remain `framework` parity.
Untrusted trace/corpus decoding limits are specified in
[`docs/attention_json_envelope.md`](docs/attention_json_envelope.md).
Independent quantization-drift and future backend-error budgets are specified
in [`docs/attention_accuracy.md`](docs/attention_accuracy.md), with a replayable
paired accuracy corpus.
Explicit quantized storage/scale/zero-point blocking, padding, layout catalogs,
and non-executing converter plans are specified in
[`docs/attention_quant_physical_layout.md`](docs/attention_quant_physical_layout.md).
The two single-request APIs and three paged/ragged batch wrappers now have
explicit Host-reference facades; the remaining FlashInfer-compatible public
surface and current gaps are documented in
[`docs/attention_frontend_contract.md`](docs/attention_frontend_contract.md).

## Development checks

The host-side tests have no third-party dependencies:

```bash
python3 -m unittest discover -s tests -v
python3 -m flashinfer_npu show-config
python3 -m flashinfer_npu parity-report --scope attention
python3 -m flashinfer_npu list-kernels
python3 -m flashinfer_npu attention-capabilities
python3 -m flashinfer_npu attention-replay path/to/case.json
python3 -m flashinfer_npu attention-coverage --replay --require-complete
python3 -m flashinfer_npu attention-corpus --pretty
python3 -m flashinfer_npu attention-accuracy
python3 -m flashinfer_npu attention-accuracy-corpus --pretty
python3 -m flashinfer_npu attention-protocol-validate path/to/protocol.json
```

The current Host-only suite contains 509 tests. It validates framework
contracts and injected fake callables; passing it is not evidence of NPU
operator correctness or performance.
