import unittest

from flashinfer_npu.attention import (
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorCallableBindingError,
    AttentionOperatorCallableInspector,
    AttentionOperatorProviderProbe,
    bind_attention_operator_callable,
    explain_attention_operator_callable,
    load_packaged_attention_operator_catalog,
    observe_python_callable_signature,
)
from flashinfer_npu.runtime import SchemaError


OPERATION_ID = "flash_attention_npu.flash_attn_with_kvcache@v2"


def fake_flash_attn_with_kvcache_v2(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
):
    raise AssertionError("signature fixture must never execute")


def available_probe(version="test-build-v2"):
    return AttentionOperatorProviderProbe(
        provider_id="flash_attention_npu",
        adapter_version="test-adapter",
        available=True,
        package_versions=(("flash-attention-npu", version),),
    )


def observation(signature=None, **changes):
    values = {
        "provider_id": "flash_attention_npu",
        "package_name": "flash-attention-npu",
        "package_version": "test-build-v2",
        "callable_path": "flash_attn.flash_attn_with_kvcache",
        "api_version": "v2",
        "available": True,
        "signature": (
            observe_python_callable_signature(fake_flash_attn_with_kvcache_v2)
            if signature is None
            else signature
        ),
    }
    values.update(changes)
    return AttentionObservedOperatorCallable(**values)


class FakeInspector:
    provider_id = "flash_attention_npu"

    def inspect(self, operation):
        return observation()


class AttentionCallableObservationCheckpoint(unittest.TestCase):
    """Checkpoint 011: fake callable inspection, never real package import/call."""

    def test_exact_fake_signature_binds_package_api_and_catalog_identity(self):
        operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)
        observed = observation()
        report = explain_attention_operator_callable(
            available_probe(), operation, observed
        )

        self.assertTrue(report.accepted)
        self.assertEqual(report.reasons, ())
        self.assertEqual(
            observed.signature,
            AttentionObservedCallableSignature.from_dict(
                observed.signature.to_dict()
            ),
        )
        binding = bind_attention_operator_callable(
            available_probe(), operation, observed
        )
        self.assertEqual(binding.operation_id, OPERATION_ID)
        self.assertEqual(binding.api_version, "v2")
        self.assertEqual(binding.package_version, "test-build-v2")
        self.assertEqual(len(binding.fingerprint), 64)
        self.assertIsInstance(FakeInspector(), AttentionOperatorCallableInspector)

    def test_renamed_extra_and_variadic_parameters_cannot_fake_exact_match(self):
        operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)
        exact = observation().signature

        renamed = AttentionObservedCallableSignature(
            exact.positional_arguments,
            tuple(
                "page_table" if item == "block_table" else item
                for item in exact.keyword_arguments
            ),
            observation_kind="test-drift",
        )
        report = explain_attention_operator_callable(
            available_probe(), operation, observation(renamed)
        )
        self.assertIn("observed keyword signature differs from catalog", report.reasons)

        extra = AttentionObservedCallableSignature(
            exact.positional_arguments,
            exact.keyword_arguments + ("debug_option",),
            observation_kind="test-drift",
        )
        with self.assertRaisesRegex(
            AttentionOperatorCallableBindingError, "keyword signature"
        ):
            bind_attention_operator_callable(
                available_probe(), operation, observation(extra)
            )

        variadic = AttentionObservedCallableSignature(
            exact.positional_arguments,
            exact.keyword_arguments,
            has_var_keyword=True,
            observation_kind="test-drift",
        )
        with self.assertRaisesRegex(
            AttentionOperatorCallableBindingError, "variadic callable"
        ):
            bind_attention_operator_callable(
                available_probe(), operation, observation(variadic)
            )

    def test_unavailable_package_version_path_and_api_variant_are_explainable(self):
        operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)

        unavailable_probe = AttentionOperatorProviderProbe(
            provider_id="flash_attention_npu",
            adapter_version="test-adapter",
            available=False,
            unavailable_reasons=("package metadata is missing",),
        )
        unavailable_observation = AttentionObservedOperatorCallable(
            provider_id="flash_attention_npu",
            package_name="flash-attention-npu",
            package_version="unknown",
            callable_path="flash_attn.flash_attn_with_kvcache",
            api_version="v2",
            available=False,
            unavailable_reasons=("callable was not resolved",),
        )
        report = explain_attention_operator_callable(
            unavailable_probe, operation, unavailable_observation
        )
        self.assertIn(
            "provider unavailable: package metadata is missing", report.reasons
        )
        self.assertIn(
            "callable unavailable: callable was not resolved", report.reasons
        )

        version_report = explain_attention_operator_callable(
            available_probe("different-build"), operation, observation()
        )
        self.assertIn(
            "observed package version differs from provider probe",
            version_report.reasons,
        )
        api_report = explain_attention_operator_callable(
            available_probe(),
            operation,
            observation(
                api_version="v3",
                callable_path="flash_attn.v3.flash_attn_with_kvcache",
            ),
        )
        self.assertIn(
            "observed API version does not match catalog operation",
            api_report.reasons,
        )
        self.assertIn(
            "observed callable path does not match catalog operation",
            api_report.reasons,
        )

        with self.assertRaisesRegex(SchemaError, "requires reasons"):
            AttentionObservedOperatorCallable(
                provider_id="flash_attention_npu",
                package_name="flash-attention-npu",
                package_version="unknown",
                callable_path="flash_attn.flash_attn_with_kvcache",
                api_version="v2",
                available=False,
            )


if __name__ == "__main__":
    unittest.main()
