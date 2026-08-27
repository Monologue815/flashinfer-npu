# FlashInfer API Parity

The active machine-readable source of truth is
`flashinfer_npu/data/attention_api_parity.json`. It is complete for the scoped
core Attention framework surface defined in that manifest. The broader
`flashinfer_npu/data/api_parity.json` remains a bootstrap and is intentionally
not part of the current development phase.

Parity schema v3 separates three views:

- `entries` tracks each upstream symbol's semantic and executable level;
- `attention_surfaces` tracks the six user-facing Attention modes, their
  one-shot or `plan()`/`run()` lifecycle, Host oracle, private provider routing
  and production-NPU boundary;
- `attention_interface_contract` freezes the model-facing `private_auto`
  dispatch policy, forbids plan handles and provider selection in normal calls,
  and inventories the separately named advanced injected-module APIs.

This prevents an executable Host `reference` from hiding the presence of a
framework-only provider resolver, while also preventing that resolver from
being reported as a callable NPU implementation. It also prevents the low-level
`*_with_jit_module` compatibility surface from being mistaken for the normal
model-facing Attention interface.

The packaged manifest is also bound to live Python objects by a read-only
interface audit. It resolves every one-shot function and every wrapper
constructor/`plan()`/`run()` method, freezes ordered parameter names, kinds and
scalar defaults, rejects provider/plan/executable handles on model-facing
callables, and verifies that advanced injected-module functions take
`jit_module` as their first argument. Upstream-compatible `jit_args` and
`jit_kwargs` configuration slots are not executable module handles and remain
valid where the classic wrapper signature defines them. The resulting report
has a deterministic fingerprint and does not instantiate a wrapper, inspect a
device, import an operator package or run an Attention implementation.

```bash
python3 -m flashinfer_npu attention-interface
```

Render the manifest as a parity report with:

```bash
python3 -m flashinfer_npu parity-report
```

Inspect the deferred broad inventory with:

```bash
python3 -m flashinfer_npu parity-report --scope all
```

Release CI will eventually use:

```bash
python3 -m flashinfer_npu parity-report --scope attention --require-complete
```

The release gate succeeds only when every scoped stable Attention symbol has
assessed semantics plus a functional or optimized implementation.
Framework-only coverage is reported separately and does not count as a
runnable NPU implementation. The manifest uses the following durable meaning:

- `reference`: executable Host oracle semantics;
- `framework`: public facade, planning, dispatch, adapter or JIT contract;
- `functional`: a callable NPU implementation with the required capability
  and numerical evidence;
- `optimized`: a functional NPU implementation that also satisfies the
  repository's performance policy;
- `missing`: a scoped upstream symbol whose contract has not been represented.

Counts are generated from the machine-readable manifest and CI output. They
are deliberately not copied into this design document, where they would become
stale test or progress records.
