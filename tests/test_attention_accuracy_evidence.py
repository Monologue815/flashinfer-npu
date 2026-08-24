import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionAccuracyBudget,
    AttentionAccuracyCase,
    AttentionAccuracyCorpus,
    AttentionAccuracyBindingError,
    AttentionAccuracyDispatchBinding,
    AttentionTrace,
    ReferenceAttentionResult,
    ReferenceKVData,
    ReferenceTensor,
    bind_attention_accuracy_dispatch,
    build_attention_accuracy_corpus,
    evaluate_attention_accuracy,
    select_attention_dispatch,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import (
    bound_kernel,
    functional_profile,
    group_case,
    group_plan,
)


def accuracy_dispatch_fixture():
    conformance_corpus, quantized_case = group_case()
    quantized_trace = quantized_case.trace
    quantized_kv = quantized_trace.kv_data
    dense_spec = replace(
        quantized_trace.spec, kv_dtype="float32", kv_quant_spec=None
    )
    dense_cache = replace(
        quantized_kv.spec, dtype="float32", quant_spec=None
    )
    dense_trace = AttentionTrace.capture(
        spec=dense_spec,
        metadata=quantized_trace.metadata,
        q=quantized_trace.q,
        kv_data=ReferenceKVData(
            dense_cache,
            (
                quantized_kv.key_data.dequantize(),
                quantized_kv.value_data.dequantize(),
            ),
        ),
        custom_mask_data=quantized_trace.custom_mask_data,
        alibi_slopes=quantized_trace.alibi_slopes,
        q_scale=quantized_trace.q_scale,
        k_scale=quantized_trace.k_scale,
        v_scale=quantized_trace.v_scale,
        logits_soft_cap=quantized_trace.logits_soft_cap,
        return_lse=quantized_trace.return_lse,
    )
    accuracy_case = AttentionAccuracyCase(
        "paged_decode_int8_group_exact_dequant",
        dense_trace,
        quantized_trace,
        AttentionAccuracyBudget(),
        True,
        "Synthetic Host binding fixture; not measured NPU evidence.",
    )
    accuracy_corpus = AttentionAccuracyCorpus(
        "synthetic-accuracy-dispatch-v1", (accuracy_case,)
    )
    candidate = quantized_trace.replay()
    report = evaluate_attention_accuracy(
        dense_trace,
        quantized_trace,
        budget=accuracy_case.budget,
        candidate=candidate,
    )
    profile = functional_profile()
    descriptor = bound_kernel(profile)
    receipt = select_attention_dispatch(
        group_plan(),
        (profile,),
        (descriptor,),
        profile.environment,
    )
    return (
        conformance_corpus,
        accuracy_case,
        accuracy_corpus,
        candidate,
        report,
        profile,
        descriptor,
        receipt,
    )


class AttentionAccuracyDispatchBindingTests(unittest.TestCase):
    def test_binding_round_trips_and_revalidates_the_full_authority_chain(self):
        (
            conformance_corpus,
            case,
            corpus,
            candidate,
            report,
            profile,
            descriptor,
            receipt,
        ) = accuracy_dispatch_fixture()
        binding = bind_attention_accuracy_dispatch(
            binding_id="synthetic-host-binding-v1",
            runner="host-contract-test",
            accuracy_case=case,
            accuracy_corpus=corpus,
            accuracy_report=report,
            candidate=candidate,
            dispatch_receipt=receipt,
            profile=profile,
            descriptor=descriptor,
            environment=profile.environment,
            conformance_corpus=conformance_corpus,
        )
        restored = AttentionAccuracyDispatchBinding.from_dict(binding.to_dict())
        self.assertEqual(restored, binding)
        self.assertEqual(restored.fingerprint, binding.fingerprint)
        restored.validate(
            case,
            corpus,
            report,
            candidate,
            receipt,
            profile,
            descriptor,
            profile.environment,
            conformance_corpus=conformance_corpus,
        )
        self.assertEqual(binding.kernel_id, descriptor.kernel_id)
        self.assertEqual(binding.artifact_fingerprint, receipt.artifact_fingerprint)
        self.assertEqual(binding.accuracy_report_fingerprint, report.fingerprint)
        with self.assertRaisesRegex(SchemaError, "backend is invalid"):
            replace(binding, backend="cuda-guess")

    def test_report_candidate_and_binding_tampering_are_rejected(self):
        (
            conformance_corpus,
            case,
            corpus,
            candidate,
            report,
            profile,
            descriptor,
            receipt,
        ) = accuracy_dispatch_fixture()
        changed_candidate = ReferenceAttentionResult(
            ReferenceTensor(
                candidate.output.shape,
                tuple(value + 0.25 for value in candidate.output.data),
                candidate.output.dtype,
                candidate.output.device,
            ),
            candidate.lse,
        )
        with self.assertRaisesRegex(
            AttentionAccuracyBindingError, "does not match replayed"
        ):
            bind_attention_accuracy_dispatch(
                binding_id="synthetic-host-binding-v1",
                runner="host-contract-test",
                accuracy_case=case,
                accuracy_corpus=corpus,
                accuracy_report=report,
                candidate=changed_candidate,
                dispatch_receipt=receipt,
                profile=profile,
                descriptor=descriptor,
                environment=profile.environment,
                conformance_corpus=conformance_corpus,
            )

        binding = bind_attention_accuracy_dispatch(
            binding_id="synthetic-host-binding-v1",
            runner="host-contract-test",
            accuracy_case=case,
            accuracy_corpus=corpus,
            accuracy_report=report,
            candidate=candidate,
            dispatch_receipt=receipt,
            profile=profile,
            descriptor=descriptor,
            environment=profile.environment,
            conformance_corpus=conformance_corpus,
        )
        stale = replace(binding, artifact_fingerprint="0" * 64)
        with self.assertRaisesRegex(
            AttentionAccuracyBindingError, "stale or inconsistent"
        ):
            stale.validate(
                case,
                corpus,
                report,
                candidate,
                receipt,
                profile,
                descriptor,
                profile.environment,
                conformance_corpus=conformance_corpus,
            )

    def test_cross_workload_and_expected_rejection_cases_cannot_bind(self):
        (
            conformance_corpus,
            _case,
            _corpus,
            _candidate,
            _report,
            profile,
            descriptor,
            receipt,
        ) = accuracy_dispatch_fixture()
        builtin = build_attention_accuracy_corpus()
        for case_id, reason in (
            ("exact_int8", "dispatch receipt revalidation failed"),
            ("int8_scale_overflow_rejected", "cannot bind runnable dispatch"),
        ):
            case = next(item for item in builtin.cases if item.case_id == case_id)
            candidate = case.quantized_trace.replay()
            report = evaluate_attention_accuracy(
                case.dense_trace,
                case.quantized_trace,
                budget=case.budget,
                candidate=candidate,
            )
            with self.subTest(case_id=case_id):
                with self.assertRaisesRegex(AttentionAccuracyBindingError, reason):
                    bind_attention_accuracy_dispatch(
                        binding_id="invalid-cross-workload",
                        runner="host-contract-test",
                        accuracy_case=case,
                        accuracy_corpus=builtin,
                        accuracy_report=report,
                        candidate=candidate,
                        dispatch_receipt=receipt,
                        profile=profile,
                        descriptor=descriptor,
                        environment=profile.environment,
                        conformance_corpus=conformance_corpus,
                    )

    def test_binding_schema_rejects_malformed_hashes(self):
        values = {name: "x" for name in AttentionAccuracyDispatchBinding.__dataclass_fields__}
        values["schema_version"] = 1
        with self.assertRaisesRegex(SchemaError, "SHA-256"):
            AttentionAccuracyDispatchBinding(**values)


if __name__ == "__main__":
    unittest.main()
