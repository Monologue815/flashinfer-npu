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
