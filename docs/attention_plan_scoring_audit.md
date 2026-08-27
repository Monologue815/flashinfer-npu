# Attention plan scoring audit

## Purpose

Provider selection is automatic and private to the Attention wrapper. A
deployment nevertheless needs to prove, after planning or execution, that the
selected provider came from the reviewed scoring manifest and that the
published diagnostics were not mixed across registry generations.

`verify_attention_plan_scoring_chain()` is the pure Host verifier for that
purpose. It replays declarative policies and joins existing immutable evidence;
it does not resolve a package, inspect a device, read tensors or execute an
operator. Verification failure raises `AttentionPlanScoringAuditError` and
produces no success report.

## Inputs

The verifier requires:

- the canonical `AttentionFrameworkPlan` used for resolution;
- the complete `AttentionOperatorRuntimeResolutionReport`;
- the captured `AttentionOperatorRuntimeRegistrySnapshot`;
- the reviewed `AttentionOperatorPlanScoringManifest`;
- the public read-only `AttentionPlanSelection`;
- optionally, the successful `AttentionOperatorRunReceipt`.

These are in-memory, versioned framework objects. Loading JSON, checking a file
signature and selecting the trusted manifest remain deployment responsibilities
outside the verifier.

## Verification algorithm

The verifier fails closed unless all of the following are true:

1. the plan fingerprint, generation and mode match the selection and resolution
   report;
2. the resolution-report fingerprint matches the one frozen by the selection;
3. the report has exactly one scored winner and its provider-operation identity
   matches the selection;
4. the selection registry generation matches the captured snapshot;
5. the snapshot manifest binding is exactly the binding derived from the
   supplied manifest;
6. the resolution candidate identity set equals the manifest binding identity
   set, preventing omitted or injected provider operations;
7. the selected runtime declaration, manifest and policy identities match the
   snapshot and selection;
8. when the snapshot is provider-bundle-bound, the selected operation belongs
   to that bundle and the selection carries its exact bundle id/fingerprint;
9. every candidate that the resolver actually scored is replayed with its exact
   declarative policy, canonical plan and recorded device; value, source,
   reason, policy id and policy fingerprint must all match;
10. the report's unique winner and the public score diagnostics agree;
11. when a run receipt is supplied, its active plan, provider, operation,
    declaration, provider-bundle, manifest and policy identities exactly match
    the selection.

Rejected candidates and accepted candidates below the highest static priority
do not have plan scores by design and are not evaluated during replay. Their
identities must still be present in the exact candidate set. The verifier never
changes provider selection and never performs fallback.

## Audit report

Success returns an immutable `AttentionPlanScoringAuditReport` containing:

- plan, resolution, selection and active-plan fingerprints;
- registry generation and device;
- selected provider, operation and declaration fingerprint;
- optional provider integration bundle id/fingerprint from the captured
  registry generation;
- manifest and selected-policy identities;
- selected score;
- a canonical list of every replayed top-tier score;
- the run-receipt fingerprint when execution evidence was supplied.

The report has a canonical `to_dict()` representation and SHA-256 fingerprint.
It contains no policy rule objects, package handles, callables, tensors, device
addresses or opaque provider plans.

## Usage

```python
from flashinfer_npu.attention import verify_attention_plan_scoring_chain

audit = verify_attention_plan_scoring_chain(
    plan=canonical_plan,
    resolution_report=resolution_report,
    registry_snapshot=wrapper_registry_snapshot,
    scoring_manifest=reviewed_scoring_manifest,
    plan_selection=attention.plan_selection,
    run_receipt=attention.last_run_receipt,  # optional for plan-only audit
)
audit_fingerprint = audit.fingerprint
```

The verifier is an integration and operations audit surface. Model code does
not call it in the inference hot path, and it does not alter the FlashInfer-like
`plan()` / `run()` interface.
