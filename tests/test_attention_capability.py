import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from flashinfer_npu.attention import (
    AttentionBackendCapabilityProfile,
    AttentionCapabilityError,
    AttentionCapabilityEvidence,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionMetadataLimits,
    AttentionNumericsPolicy,
    AttentionRuntimeEnvironment,
    AttentionDispatchError,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionTraceCorpus,
    ATTENTION_LAUNCH_ARGUMENT_NAMES,
    attention_kernel_binary_abi,
    build_framework_attention_corpus,
    framework_attention_coverage_policy,
    load_attention_capability_manifest,
    explain_attention_dispatch,
    select_attention_dispatch,
    validate_attention_kernel_bindings,
)
from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    Backend,
    KernelCapabilityBinding,
    KernelConstraints,
    KernelDescriptor,
    KernelLaunchABI,
    QuantSpec,
    SchemaError,
    WorkspaceFormula,
    load_kernel_manifest,
)


def pinned_environment():
    return AttentionRuntimeEnvironment(
        soc_version="Ascend910B",
        soc_revision="rev1",
        driver_version="25.0.test",
        firmware_version="7.0.test",
        cann_version="8.0.test",
        torch_version="2.6.test",
        torch_npu_version="2.6.test",
        compiler_version="8.0.test",
        python_abi="cp39",
        ai_core_count=20,
        features=("cube", "int8-group-dequant"),
    )


def group_case():
    corpus = build_framework_attention_corpus()
    case = next(
        item
        for item in corpus.cases
        if item.case_id == "paged_decode_int8_group_gqa_distinct_dims"
    )
    return corpus, case


def functional_profile():
    corpus, case = group_case()
    policy = framework_attention_coverage_policy()
    subset = AttentionTraceCorpus(
        "synthetic-evidence-subset", (case,), "one exact capability rule"
    )
    coverage = policy.evaluate(subset)
    spec = case.trace.spec
    rule = AttentionCapabilityRule(
        rule_id="paged_decode_int8_group_v1",
        modes=(spec.mode,),
        kv_layouts=(spec.kv_layout,),
        dtype_signatures=((spec.q_dtype, spec.kv_dtype, spec.o_dtype),),
        supports_dense_kv=False,
        quant_specs=(spec.kv_quant_spec,),
        pos_encoding_modes=(spec.pos_encoding_mode,),
        mask_kinds=("none",),
        causal_values=(spec.effective_causal,),
        max_head_dim_qk=3,
        max_head_dim_vo=2,
        max_gqa_group_size=2,
        metadata_limits=AttentionMetadataLimits(
            max_batch_size=2,
            max_total_qo_tokens=2,
            max_total_kv_tokens=3,
            max_total_pages=2,
            max_page_size=2,
        ),
        required_features=("int8-group-dequant",),
    )
    evidence = AttentionCapabilityEvidence(
        evidence_id="synthetic-functional-run-v1",
        level=AttentionCapabilityStatus.FUNCTIONAL,
        runner="synthetic-npu-harness",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
        result_digest=hashlib.sha256(b"synthetic-result-record").hexdigest(),
    )
    return AttentionBackendCapabilityProfile(
        profile_id="ascend910b.synthetic.attention.v1",
        backend=Backend.ASCENDC_AOT,
        environment=pinned_environment(),
        status=AttentionCapabilityStatus.FUNCTIONAL,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(rule,),
        evidence=(evidence,),
    )


def attention_artifact(
    locator="artifacts/ascend910b/paged_decode_int8_group.o",
):
    payload = ("synthetic:" + locator).encode("utf-8")
    return ArtifactRef(
        kind=ArtifactKind.FILE,
        format=ArtifactFormat.ASCENDC_OBJECT,
        locator=locator,
        digest=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        target_soc="Ascend910B",
        build_id="synthetic-attention-test",
    )


def attention_launch_abi(entry_point="attention_paged_decode_entry"):
    return KernelLaunchABI(
        abi_name="flashinfer_npu.attention.v1",
        entry_point=entry_point,
        argument_names=ATTENTION_LAUNCH_ARGUMENT_NAMES,
        mutable_arguments=(
            "aux",
            "out",
            "lse",
            "float_workspace",
            "int_workspace",
        ),
        stream_argument="stream",
    )


def aclnn_attention_artifact():
    locator = "builtin:aclnn:attention"
    return ArtifactRef(
        kind=ArtifactKind.BUILTIN,
        format=ArtifactFormat.ACLNN_BUILTIN,
        locator=locator,
        digest=hashlib.sha256(locator.encode("utf-8")).hexdigest(),
        target_soc="Ascend910B",
        build_id="synthetic-aclnn-provider",
    )


def bound_kernel(profile=None, **overrides):
    profile = functional_profile() if profile is None else profile
    rule = profile.rules[0]
    values = {
        "kernel_id": "paged_decode_int8_group_910b_aot_v1",
        "op": "attention.%s" % rule.modes[0].value,
        "backend": profile.backend,
        "constraints": KernelConstraints(
            supported_socs=(profile.environment.soc_version,),
            dtype_signatures=(rule.dtype_signatures[0],),
            layout_signatures=((rule.kv_layouts[0].value,),),
            required_features=rule.required_features,
            quant_storage_dtypes=(rule.quant_specs[0].storage_dtype,),
        ),
        "artifact": attention_artifact(),
        "launch_abi": attention_launch_abi(),
        "binary_abi": attention_kernel_binary_abi(),
        "capability_binding": KernelCapabilityBinding(
            domain="attention",
            profile_id=profile.profile_id,
            rule_id=rule.rule_id,
            profile_fingerprint=profile.fingerprint,
        ),
    }
    values.update(overrides)
    return KernelDescriptor(**values)


def group_plan():
    _, case = group_case()
    return AttentionFrameworkSession(case.trace.spec.mode).plan(
        case.trace.spec, case.trace.metadata
    )


class AttentionCapabilitySchemaTests(unittest.TestCase):
    def test_profile_round_trip_preserves_fingerprint(self):
        profile = functional_profile()
        restored = AttentionBackendCapabilityProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)
        self.assertEqual(restored.fingerprint, profile.fingerprint)
        self.assertTrue(restored.environment.is_fully_pinned)

    def test_runnable_profile_requires_pinned_environment_and_matching_evidence(self):
        profile = functional_profile()
        unresolved = replace(
            profile.environment, driver_version="unknown", ai_core_count=0
        )
        with self.assertRaisesRegex(SchemaError, "fully pinned"):
            replace(profile, environment=unresolved)
        with self.assertRaisesRegex(SchemaError, "requires evidence"):
            replace(profile, evidence=())

    def test_draft_profile_is_not_dispatchable(self):
        functional = functional_profile()
        draft = replace(
            functional,
            status=AttentionCapabilityStatus.DRAFT,
            evidence=(),
        )
        _, case = group_case()
        report = draft.explain(
            case.trace.spec, case.trace.metadata, draft.environment
        )
        self.assertFalse(report.accepted)
        self.assertIn("not runnable", report.global_reasons[0])
        protocol_report = draft.explain(
            case.trace.spec,
            case.trace.metadata,
            draft.environment,
            require_functional=False,
        )
        self.assertTrue(protocol_report.accepted)


class AttentionCapabilityMatchingTests(unittest.TestCase):
    def test_exact_rule_selects_supported_groupwise_decode(self):
        profile = functional_profile()
        _, case = group_case()
        selected = profile.select_rule(
            case.trace.spec, case.trace.metadata, profile.environment
        )
        self.assertEqual(selected.rule_id, "paged_decode_int8_group_v1")

    def test_quant_spec_is_exact_not_storage_dtype_only(self):
        profile = functional_profile()
        _, case = group_case()
        original = case.trace.spec.kv_quant_spec
        tensor_quant = QuantSpec(
            scheme=original.scheme,
            storage_dtype=original.storage_dtype,
            compute_dtype=original.compute_dtype,
            accumulator_dtype=original.accumulator_dtype,
        )
        changed = replace(case.trace.spec, kv_quant_spec=tensor_quant)
        report = profile.explain(changed, case.trace.metadata, profile.environment)
        self.assertFalse(report.accepted)
        self.assertIn("unsupported exact QuantSpec", report.rules[0].reasons[0])

    def test_environment_and_resource_mismatches_are_explainable(self):
        profile = functional_profile()
        _, case = group_case()
        changed_environment = replace(
            profile.environment, firmware_version="different"
        )
        report = profile.explain(
            case.trace.spec, case.trace.metadata, changed_environment
        )
        self.assertFalse(report.accepted)
        self.assertIn("fingerprint mismatch", report.global_reasons[0])

        too_small = replace(
            profile.rules[0],
            metadata_limits=AttentionMetadataLimits(max_total_kv_tokens=2),
        )
        limited = replace(profile, rules=(too_small,))
        report = limited.explain(
            case.trace.spec, case.trace.metadata, limited.environment
        )
        self.assertFalse(report.accepted)
        self.assertTrue(
            any("exceeds limit" in reason for reason in report.rules[0].reasons)
        )

    def test_select_rule_reports_all_rejection_reasons(self):
        profile = functional_profile()
        _, case = group_case()
        wrong = replace(case.trace.spec, head_dim_qk=4)
        with self.assertRaisesRegex(
            AttentionCapabilityError, "head_dim_qk exceeds"
        ):
            profile.select_rule(wrong, case.trace.metadata, profile.environment)


class AttentionCapabilityEvidenceTests(unittest.TestCase):
    def test_evidence_is_bound_to_corpus_policy_cases_and_rules(self):
        profile = functional_profile()
        corpus, _ = group_case()
        evidence = profile.validate_evidence(
            corpus, framework_attention_coverage_policy(), replay=True
        )
        self.assertEqual(evidence.evidence_id, "synthetic-functional-run-v1")

    def test_unknown_evidence_case_is_rejected(self):
        profile = functional_profile()
        corpus, _ = group_case()
        changed_evidence = replace(
            profile.evidence[0], passed_case_ids=("not_in_corpus",)
        )
        changed = replace(profile, evidence=(changed_evidence,))
        with self.assertRaisesRegex(AttentionCapabilityError, "does not match"):
            changed.validate_evidence(corpus, framework_attention_coverage_policy())

    def test_manifest_loader_rejects_duplicate_profile_ids(self):
        profile = functional_profile().to_dict()
        payload = {
            "schema_version": 1,
            "generated_at": "2026-08-13",
            "profiles": [profile, profile],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capabilities.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "duplicate profile_id"):
                load_attention_capability_manifest(path)


class AttentionKernelBindingTests(unittest.TestCase):
    def test_valid_binding_closes_profile_rule_mode_and_kernel_loop(self):
        profile = functional_profile()
        self.assertEqual(
            validate_attention_kernel_bindings(
                (profile,), (bound_kernel(profile),)
            ),
            1,
        )
        self.assertEqual(validate_attention_kernel_bindings((), ()), 0)

    def test_non_reference_attention_kernel_requires_binding(self):
        kernel = bound_kernel()
        kernel = replace(kernel, capability_binding=None)
        with self.assertRaisesRegex(SchemaError, "requires a capability binding"):
            validate_attention_kernel_bindings((), (kernel,))

    def test_runnable_profile_rule_mode_requires_a_kernel(self):
        with self.assertRaisesRegex(SchemaError, "has no kernel descriptor"):
            validate_attention_kernel_bindings((functional_profile(),), ())

    def test_binding_identity_backend_and_op_are_exact(self):
        profile = functional_profile()
        base = bound_kernel(profile)
        cases = (
            (
                replace(
                    base,
                    capability_binding=replace(
                        base.capability_binding, profile_id="unknown.profile"
                    ),
                ),
                "unknown profile",
            ),
            (
                replace(
                    base,
                    capability_binding=replace(
                        base.capability_binding, profile_fingerprint="0" * 64
                    ),
                ),
                "fingerprint mismatch",
            ),
            (
                replace(
                    base,
                    capability_binding=replace(
                        base.capability_binding, rule_id="unknown_rule"
                    ),
                ),
                "unknown rule",
            ),
            (
                replace(
                    base,
                    backend=Backend.ACLNN,
                    artifact=aclnn_attention_artifact(),
                ),
                "backend does not match",
            ),
            (replace(base, op="attention.single_decode"), "outside its capability rule"),
        )
        for kernel, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    validate_attention_kernel_bindings((profile,), (kernel,))

    def test_attention_binding_cannot_be_attached_to_other_domains(self):
        profile = functional_profile()
        attention = bound_kernel(profile)
        wrong_domain = replace(
            attention,
            capability_binding=replace(
                attention.capability_binding, domain="sampling"
            ),
        )
        with self.assertRaisesRegex(SchemaError, "domain='attention'"):
            validate_attention_kernel_bindings((profile,), (wrong_domain,))
        non_attention = replace(attention, op="rmsnorm")
        with self.assertRaisesRegex(SchemaError, "non-Attention kernel"):
            validate_attention_kernel_bindings((profile,), (non_attention,))

    def test_draft_profile_cannot_authorize_a_kernel(self):
        functional = functional_profile()
        draft = replace(
            functional,
            status=AttentionCapabilityStatus.DRAFT,
            evidence=(),
        )
        kernel = bound_kernel(draft)
        with self.assertRaisesRegex(SchemaError, "non-runnable profile"):
            validate_attention_kernel_bindings((draft,), (kernel,))

    def test_descriptor_constraints_must_be_narrower_than_rule(self):
        profile = functional_profile()
        base = bound_kernel(profile)
        cases = (
            replace(
                base,
                constraints=replace(
                    base.constraints,
                    supported_socs=(profile.environment.soc_version, "OtherSoC"),
                ),
            ),
            replace(
                base,
                constraints=replace(
                    base.constraints,
                    dtype_signatures=(("float16", "int8", "float16"),),
                ),
            ),
            replace(
                base,
                constraints=replace(
                    base.constraints, layout_signatures=(("HND",),)
                ),
            ),
            replace(
                base,
                constraints=replace(
                    base.constraints, quant_storage_dtypes=("int4_packed",)
                ),
            ),
            replace(
                base,
                constraints=replace(base.constraints, required_features=()),
            ),
        )
        messages = (
            "exact profile SoC",
            "dtype constraints",
            "layout constraints",
            "quant dtype constraints",
            "omits rule required features",
        )
        for kernel, message in zip(cases, messages):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    validate_attention_kernel_bindings((profile,), (kernel,))

    def test_artifact_target_and_attention_launch_abi_are_exact(self):
        profile = functional_profile()
        base = bound_kernel(profile)
        cases = (
            (
                replace(
                    base,
                    artifact=replace(base.artifact, target_soc="OtherSoC"),
                ),
                "artifact target",
            ),
            (
                replace(
                    base,
                    launch_abi=replace(
                        base.launch_abi,
                        abi_name="flashinfer_npu.other.v1",
                    ),
                ),
                "launch ABI",
            ),
            (
                replace(
                    base,
                    launch_abi=replace(
                        base.launch_abi,
                        argument_names=tuple(
                            item
                            for item in base.launch_abi.argument_names
                            if item != "plan_metadata"
                        ),
                    ),
                    binary_abi=replace(
                        base.binary_abi,
                        arguments=tuple(
                            item
                            for item in base.binary_abi.arguments
                            if item.name != "plan_metadata"
                        ),
                    ),
                ),
                "launch ABI",
            ),
        )
        for kernel, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    validate_attention_kernel_bindings((profile,), (kernel,))

    def test_serialized_manifests_round_trip_then_cross_validate(self):
        profile = functional_profile()
        kernel = bound_kernel(profile)
        capability_payload = {
            "schema_version": 1,
            "generated_at": "2026-08-19",
            "profiles": [profile.to_dict()],
        }
        kernel_payload = {
            "schema_version": 3,
            "generated_at": "2026-08-19",
            "kernels": [kernel.to_dict()],
        }
        with tempfile.TemporaryDirectory() as directory:
            capability_path = Path(directory) / "capabilities.json"
            kernel_path = Path(directory) / "kernels.json"
            capability_path.write_text(
                __import__("json").dumps(capability_payload), encoding="utf-8"
            )
            kernel_path.write_text(
                __import__("json").dumps(kernel_payload), encoding="utf-8"
            )
            profiles = load_attention_capability_manifest(capability_path)
            kernels = load_kernel_manifest(kernel_path)
        self.assertEqual(kernels[0], kernel)
        self.assertEqual(validate_attention_kernel_bindings(profiles, kernels), 1)

    def test_manifest_loader_revalidates_runnable_profile_evidence(self):
        profile = functional_profile()
        invalid = replace(
            profile,
            evidence=(
                replace(
                    profile.evidence[0],
                    covered_cells=profile.evidence[0].covered_cells + 1,
                ),
            ),
        )
        payload = {
            "schema_version": 1,
            "generated_at": "2026-08-13",
            "profiles": [invalid.to_dict()],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capabilities.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "invalid packaged evidence"):
                load_attention_capability_manifest(path)


class AttentionDispatchReceiptTests(unittest.TestCase):
    def test_selection_closes_plan_profile_evidence_and_kernel_chain(self):
        profile = functional_profile()
        kernel = bound_kernel(profile)
        plan = group_plan()
        receipt = select_attention_dispatch(
            plan,
            (profile,),
            (kernel,),
            profile.environment,
        )
        self.assertEqual(receipt.mode, plan.spec.mode)
        self.assertEqual(receipt.plan_fingerprint, plan.fingerprint)
        self.assertEqual(receipt.admission_fingerprint, plan.admission_fingerprint)
        self.assertEqual(receipt.profile_fingerprint, profile.fingerprint)
        self.assertEqual(receipt.evidence_id, profile.evidence[0].evidence_id)
        self.assertEqual(receipt.kernel_fingerprint, kernel.fingerprint)
        self.assertEqual(receipt.selection_source, "priority")
        restored = AttentionDispatchReceipt.from_dict(receipt.to_dict())
        self.assertEqual(restored, receipt)
        self.assertEqual(restored.fingerprint, receipt.fingerprint)
        restored.validate(plan, profile, kernel, profile.environment)

    def test_explain_preserves_environment_and_backend_rejections(self):
        profile = functional_profile()
        kernel = bound_kernel(profile)
        changed_environment = replace(
            profile.environment, firmware_version="other"
        )
        report = explain_attention_dispatch(
            group_plan(),
            (profile,),
            (kernel,),
            changed_environment,
            backend=Backend.ACLNN,
        )
        self.assertEqual(len(report.candidates), 1)
        candidate = report.candidates[0]
        self.assertFalse(candidate.accepted)
        self.assertTrue(
            any("environment fingerprint mismatch" in item for item in candidate.reasons)
        )
        self.assertTrue(
            any("backend policy" in item for item in candidate.reasons)
        )
        with self.assertRaisesRegex(
            AttentionDispatchError, "environment fingerprint mismatch"
        ):
            select_attention_dispatch(
                group_plan(),
                (profile,),
                (kernel,),
                changed_environment,
            )

    def test_tuning_can_only_select_an_accepted_bound_kernel(self):
        profile = functional_profile()
        preferred = bound_kernel(
            profile,
            kernel_id="paged_decode_int8_group_910b_tuned_v1",
            artifact=attention_artifact(
                "artifacts/ascend910b/paged_decode_int8_group_tuned.o"
            ),
            priority=1,
        )
        priority = bound_kernel(profile, priority=20)
        receipt = select_attention_dispatch(
            group_plan(),
            (profile,),
            (priority, preferred),
            profile.environment,
            tuned_kernel_ids=("unknown", preferred.kernel_id),
        )
        self.assertEqual(receipt.kernel_id, preferred.kernel_id)
        self.assertEqual(receipt.selection_source, "tuning")

    def test_invalid_evidence_is_a_dispatch_rejection(self):
        profile = functional_profile()
        invalid = replace(
            profile,
            evidence=(
                replace(
                    profile.evidence[0],
                    covered_cells=profile.evidence[0].covered_cells + 1,
                ),
            ),
        )
        kernel = bound_kernel(invalid)
        report = explain_attention_dispatch(
            group_plan(), (invalid,), (kernel,), invalid.environment
        )
        self.assertFalse(report.candidates[0].accepted)
        self.assertTrue(
            any("evidence invalid" in item for item in report.candidates[0].reasons)
        )

    def test_receipt_revalidation_detects_each_authority_change(self):
        profile = functional_profile()
        kernel = bound_kernel(profile)
        plan = group_plan()
        receipt = select_attention_dispatch(
            plan, (profile,), (kernel,), profile.environment
        )
        cases = (
            (replace(receipt, plan_fingerprint="0" * 64), "plan"),
            (replace(receipt, profile_fingerprint="1" * 64), "profile"),
            (replace(receipt, evidence_result_digest="2" * 64), "evidence"),
            (replace(receipt, kernel_fingerprint="3" * 64), "kernel"),
            (replace(receipt, artifact_fingerprint="4" * 64), "artifact"),
            (replace(receipt, launch_abi_fingerprint="5" * 64), "ABI"),
            (replace(receipt, binary_abi_fingerprint="6" * 64), "ABI"),
            (replace(receipt, float_workspace_bytes=1), "workspace"),
        )
        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(AttentionDispatchError, message):
                    changed.validate(plan, profile, kernel, profile.environment)

    def test_workspace_formula_is_frozen_in_receipt(self):
        profile = functional_profile()
        kernel = bound_kernel(
            profile,
            workspace=WorkspaceFormula(
                constant_bytes=17,
                dynamic_coefficients=(1, 2, 3),
                alignment=64,
            ),
            int_workspace=WorkspaceFormula(
                constant_bytes=9,
                dynamic_coefficients=(0, 1),
                alignment=16,
            ),
        )
        plan = group_plan()
        receipt = select_attention_dispatch(
            plan, (profile,), (kernel,), profile.environment
        )
        self.assertEqual(
            receipt.workspace_bytes,
            (
                kernel.workspace.size_for(plan.workload),
                kernel.int_workspace.size_for(plan.workload),
            ),
        )
        self.assertEqual(receipt.workspace_alignments, (64, 16))


if __name__ == "__main__":
    unittest.main()
