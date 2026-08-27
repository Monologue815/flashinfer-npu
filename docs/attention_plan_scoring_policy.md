# Attention plan scoring policy

## Purpose

The public Attention API owns provider selection. Model code supplies ordinary
`plan()` arguments and later calls `run()`; it never chooses CANN,
flash-attention-npu, a kernel identifier or an executable plan handle.

An integration may use `AttentionOperatorPlanScoringPolicy` to declare which of
its already-compatible operations it prefers for a canonical plan. The policy
is framework metadata, not executable operator code. It exists so reviewed
offline tuning records and deterministic heuristics can participate in provider
selection without embedding custom selector functions in each integration.

Capability admission and preference scoring are separate:

1. provider and capability gates reject plans the operation cannot implement;
2. static integration priority selects the highest accepted deployment tier;
3. only candidates in that tier evaluate their plan scoring policy;
4. the unique highest provider score is bound into the active plan;
5. `run()` consumes that frozen choice without rescoring or fallback.

A score can never turn an unsupported plan into a supported plan.

## Policy identity

One policy is bound to exactly one `(provider_id, operation_id)` pair. It has a
stable `policy_id`, schema version, ordered semantic content and SHA-256
fingerprint. The policy fingerprint is included in the selected score source,
so changing an unmatched rule still changes the decision authority identity.
The same fingerprint is recorded as the `plan_scorer` component of runtime
declaration v2; reviewed declarations therefore reject policy drift before any
provider package probe.

Policies and rules support strict `to_dict()` / `from_dict()` round trips.
Unknown, missing or version-mismatched fields fail closed. Sequence-valued
predicates are canonicalized so declaration order does not change the policy
fingerprint.

## Bounded manifest ingestion

Deployments may group reviewed policies in an
`AttentionOperatorPlanScoringManifest`. A manifest has its own stable
`manifest_id` and fingerprint, contains at most one policy for each exact
`(provider_id, operation_id)` identity, and is canonicalized independently of
input policy order. `manifest.get(provider_id, operation_id)` performs exact
identity lookup; it never selects by provider name alone or substitutes another
operation version.

`load_attention_operator_plan_scoring_manifest()` is the only JSON ingestion
boundary. It first applies the shared strict JSON envelope, which rejects
duplicate object keys, non-finite numbers and inputs exceeding configured byte,
depth, node, string or container limits. Before constructing policy and rule
objects, it then enforces Attention-specific limits for:

- policies per manifest;
- rules per policy and total rules;
- values per sequence predicate and total predicate values.

All limits are explicit through `AttentionJsonEnvelopeLimits` and
`AttentionOperatorPlanScoringManifestLimits`. Unknown fields, malformed array
shapes, duplicate policy ids and duplicate provider-operation identities fail
closed. The loader returns both the immutable manifest and measured envelope
usage so the bootstrap owner can audit the accepted input size.

The loader accepts JSON text, not a path, URL or package name. Reading files,
verifying signatures and choosing which reviewed manifest to trust belong to
the deployment/bootstrap layer. Loading and looking up a manifest therefore
performs no filesystem access, provider import, device probe or operator call.

## Rule model

Each `AttentionOperatorPlanScoreRule` contains:

- `rule_id`: stable identity within the policy;
- `precedence`: rule selection level inside one provider policy;
- `score`: bounded signed 32-bit provider preference;
- `reason`: human-readable explanation published with the selected plan;
- zero or more canonical-plan predicates.

Supported predicates are intentionally finite and serializable:

| Predicate | Meaning |
| --- | --- |
| `modes` | Attention mode such as mixed paged, prefill or decode |
| `kv_layouts` | Canonical HND/NHD KV layout |
| `dtype_signatures` | Exact `(q, kv, output)` dtype triples |
| `quantization` | Any, dense-only or quantized-only plan |
| `quant_spec_fingerprints` | Exact reviewed `QuantSpec` identities |
| `page_sizes` | Exact page-size buckets; zero represents non-paged workloads |
| `head_dim_qk_values` / `head_dim_vo_values` | Exact head dimensions |
| `gqa_group_sizes` | Exact query-head to KV-head ratios |
| `causal_values` | Effective causal semantics after custom-mask handling |
| batch/QO/KV token bounds | Inclusive workload bucket ranges |
| `workload_fingerprints` | Exact canonical workload/tuning records |

Every rule must contain at least one predicate. A policy-level default score and
reason handle plans that match no rule.

Exact offline tuning records should use `workload_fingerprints` with a higher
precedence than broader deterministic heuristic buckets. A QuantSpec-specific
rule must declare `quantization="quantized"`; dense and quantized identities
cannot be conflated.

## Determinism and ambiguity

All matching rules are collected. Only rules at the highest matching
`precedence` are finalists. Exactly one finalist is required. Two overlapping
rules at the same highest precedence are a policy error even if their numeric
scores happen to agree; registration or file order is never a tie breaker.

The selected rule returns an `AttentionOperatorRuntimePlanScore` containing:

- the rule's integer score;
- a source string binding policy id, policy fingerprint and rule id;
- the rule's declared reason.
- structured `policy_id` and `policy_fingerprint` fields for framework
  validation; custom/non-manifest scorers leave both fields absent.

If no rule matches, the same structure binds the policy fingerprint and the
explicit default. Provider-level equal top scores remain ambiguous in the
runtime resolver and fail before a plan is published.

The complete resolution-report fingerprint is part of the immutable active-plan
fingerprint. Execution and completion receipts bind that active-plan identity,
so a stored run receipt and its `plan_selection` record form one auditable chain
back to the exact scoring policy and selected rule.

## Side-effect boundary

Policy evaluation reads only `AttentionFrameworkPlan`, `AttentionPlanSpec` and
`WorkloadSpec` values already constructed by the framework. It does not:

- import CANN, torch-npu or flash-attention-npu;
- initialize or query an NPU device;
- inspect tensor contents or addresses;
- compile, load or execute an operator;
- benchmark or tune online;
- mutate the registry or active plan.

The `device` argument is checked only for a non-empty plan context and is not a
source of observed capability. Hardware/software compatibility remains the
responsibility of versioned capability and runtime authority records.

## Bootstrap integration

The declarative policy implements the same identity-bound scorer protocol as an
advanced injected scorer and can be assigned directly to
`AttentionOperatorPackageRuntimeSpec.plan_scorer`. Bootstrap verifies that its
provider and operation identity match the package runtime before registration.
No provider package is loaded while the policy is installed or evaluated.

```python
from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorPlanScoreRule,
    AttentionOperatorPlanScoringPolicy,
)

scorer = AttentionOperatorPlanScoringPolicy(
    policy_id="cann.mixed_paged.preference.v1",
    provider_id="cann",
    operation_id="cann.operation@version",
    rules=(
        AttentionOperatorPlanScoreRule(
            rule_id="int8_page_128_v1",
            precedence=20,
            score=90,
            reason="reviewed INT8 page-128 preference",
            modes=(AttentionMode.BATCH_MIXED_PAGED,),
            quantization="quantized",
            page_sizes=(128,),
        ),
    ),
    default_score=0,
    default_reason="no reviewed preference for this plan",
)
```

The policy is internal bootstrap data. It is not added to the model-facing
`plan()` signature. After successful planning, callers may inspect only the
selected score, source, reason and resolution fingerprint through
`attention.plan_selection`.

When policies are supplied as JSON, bootstrap first loads the bounded manifest,
looks up the exact package runtime identity, and assigns that returned policy as
`plan_scorer`. Runtime declaration v2 records the selected policy fingerprint,
not the manifest location. Packaging, signature and rollout metadata may change
without changing runtime identity; any policy-content change still produces a
new fingerprint and requires a matching reviewed declaration.

For a complete provider bootstrap set, use
`bind_attention_operator_plan_scoring_manifest(specs, manifest)`. The set of
`(provider_id, operation_id)` identities in the manifest must exactly equal the
runtime-spec identity set: missing policies and orphan policies both fail. A
duplicate runtime identity also fails. The binding is immutable and validates
the whole set before returning replacement specs, so no caller can observe a
partly bound set.

Binding is idempotent when a spec already contains the same declarative policy
fingerprint. It never overwrites a different declarative policy or an injected
custom scorer. `build_attention_operator_runtime_resolvers()` accepts the same
manifest as `plan_scoring_manifest` for framework composition; production
integrations bind first, then generate and review runtime declarations from the
bound specs. This ordering ensures the declaration contains each exact policy
fingerprint before the declaration-bound registry is installed.

The production installer receives the same manifest through
`plan_scoring_manifest`. It does not bind or rewrite an unreviewed spec at
install time. Instead it requires every registration to have been bound before
its declaration was created, rechecks the complete identity set and every
policy fingerprint, and then validates declaration drift. All of these checks
finish before package metadata is observed.

On commit, resolver, operation catalog, declaration bindings and an
`AttentionOperatorPlanScoringManifestBinding` are published under one registry
generation. The binding contains only manifest id/fingerprint and exact
provider-operation-policy fingerprints; it contains no rules, scorer callable
or provider object. `attention_operator_runtime_registry_snapshot()` exposes
this immutable audit identity. A stale expected generation cannot publish a
new manifest, and a legacy/synthetic installation explicitly clears the
binding.

At plan time, a manifest-bound runtime looks up the selected
`(provider_id, operation_id)` in the frozen binding and requires the structured
policy identity on the resolved score to match exactly. A source string is
diagnostic text and is never parsed as authority. Mismatch fails before the
candidate framework plan is committed, preserving the previous plan.

After a successful plan, `plan_selection` exposes the manifest id/fingerprint
and selected policy id/fingerprint alongside the score and complete resolution
fingerprint. A successful validated provider call copies the same four fields
into `AttentionOperatorRunReceipt`, whose active-plan fingerprint already binds
the resolution report. Completion failure publishes no receipt. Reference,
legacy and custom-scored paths keep the optional manifest fields absent.
