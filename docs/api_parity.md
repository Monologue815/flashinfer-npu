# FlashInfer API Parity

The active machine-readable source of truth is
`flashinfer_npu/data/attention_api_parity.json`. It is complete for the scoped
core Attention framework surface defined in that manifest. The broader
`flashinfer_npu/data/api_parity.json` remains a bootstrap and is intentionally
not part of the current development phase.

Generate the current summary with:

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

That command must remain failing until every scoped stable Attention symbol
has assessed semantics plus a functional or optimized implementation.
Framework-only validation is reported separately and does not count as a
runnable implementation.

The current Attention inventory has no `missing` symbol: 32 entries are Host
`reference` and the two injected single-request JIT entry points are
`framework`. This is API/framework coverage only; with zero `functional` or
`optimized` entries, release compatibility intentionally remains false.
