# FlashInfer API Parity

The active machine-readable source of truth is
`flashinfer_npu/data/attention_api_parity.json`. It is complete for the scoped
core Attention framework surface defined in that manifest. The broader
`flashinfer_npu/data/api_parity.json` remains a bootstrap and is intentionally
not part of the current development phase.

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
