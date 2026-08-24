import unittest

from flashinfer_npu.attention import (
    ATTENTION_OPERATOR_PROVIDER_VERSION,
    CANN_ATTENTION_PROVIDER_ID,
    FLASH_ATTENTION_NPU_PROVIDER_ID,
    AttentionBackendCapabilityProfile,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionNumericsPolicy,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderRegistry,
    AttentionRuntimeEnvironment,
    AttentionMode,
    KVLayout,
)
from flashinfer_npu.runtime import Backend, SchemaError


def draft_profile(profile_id, backend):
    return AttentionBackendCapabilityProfile(
        profile_id=profile_id,
        backend=backend,
        environment=AttentionRuntimeEnvironment(
            soc_version="unknown",
            soc_revision="unknown",
            driver_version="unknown",
            firmware_version="unknown",
            cann_version="unknown",
            torch_version="unknown",
            torch_npu_version="unknown",
            compiler_version="unknown",
            python_abi="unknown",
        ),
        status=AttentionCapabilityStatus.DRAFT,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(
            AttentionCapabilityRule(
                rule_id="paged_prefill_dense_v1",
                modes=(AttentionMode.BATCH_PREFILL_PAGED,),
                kv_layouts=(KVLayout.NHD,),
                dtype_signatures=(("float16", "float16", "float16"),),
            ),
        ),
    )


class FakeProvider:
    def __init__(self, provider_id, probe, profiles):
        self.provider_id = provider_id
        self._probe = probe
        self._profiles = tuple(profiles)
        self.probe_calls = 0

    def probe(self):
        self.probe_calls += 1
        return self._probe

    def capability_profiles(self):
        return self._profiles


class AttentionOperatorProviderSPICheckpoint(unittest.TestCase):
    """Checkpoint 004: package discovery and capability ownership only."""

    def test_registration_is_lazy_and_discovery_is_identity_bound(self):
        probe = AttentionOperatorProviderProbe(
            provider_id=CANN_ATTENTION_PROVIDER_ID,
            adapter_version="0.1",
            available=True,
            package_versions=(("torch_npu", "test-only"), ("cann", "test-only")),
        )
        provider = FakeProvider(
            CANN_ATTENTION_PROVIDER_ID,
            probe,
            (draft_profile("cann.test.attention.v1", Backend.ACLNN),),
        )
        registry = AttentionOperatorProviderRegistry((provider,))

        self.assertEqual(provider.probe_calls, 0)
        records = registry.discover()
        self.assertEqual(provider.probe_calls, 1)
        self.assertEqual(records[0].provider_id, CANN_ATTENTION_PROVIDER_ID)
        self.assertEqual(records[0].probe, probe)
        self.assertEqual(len(records[0].fingerprint), 64)
        self.assertEqual(
            AttentionOperatorProviderProbe.from_dict(probe.to_dict()).fingerprint,
            probe.fingerprint,
        )

    def test_unavailable_provider_keeps_explicit_reason_and_declared_profiles(self):
        probe = AttentionOperatorProviderProbe(
            provider_id=FLASH_ATTENTION_NPU_PROVIDER_ID,
            adapter_version="0.1",
            available=False,
            unavailable_reasons=("package is not installed",),
        )
        registry = AttentionOperatorProviderRegistry(
            (
                FakeProvider(
                    FLASH_ATTENTION_NPU_PROVIDER_ID,
                    probe,
                    (
                        draft_profile(
                            "flash_attention_npu.test.attention.v1",
                            Backend.ASCENDC_AOT,
                        ),
                    ),
                ),
            )
        )
        record = registry.discover()[0]

        self.assertFalse(record.probe.available)
        self.assertEqual(
            record.probe.unavailable_reasons, ("package is not installed",)
        )
        self.assertEqual(
            record.profiles[0].profile_id,
            "flash_attention_npu.test.attention.v1",
        )

    def test_duplicate_provider_and_profile_ownership_are_rejected(self):
        self.assertEqual(ATTENTION_OPERATOR_PROVIDER_VERSION, 1)
        cann_probe = AttentionOperatorProviderProbe(
            CANN_ATTENTION_PROVIDER_ID, "0.1", True
        )
        cann = FakeProvider(
            CANN_ATTENTION_PROVIDER_ID,
            cann_probe,
            (draft_profile("shared.attention.v1", Backend.ACLNN),),
        )
        with self.assertRaisesRegex(SchemaError, "duplicate.*provider_id"):
            AttentionOperatorProviderRegistry((cann, cann))

        flash_probe = AttentionOperatorProviderProbe(
            FLASH_ATTENTION_NPU_PROVIDER_ID, "0.1", True
        )
        flash = FakeProvider(
            FLASH_ATTENTION_NPU_PROVIDER_ID,
            flash_probe,
            (draft_profile("shared.attention.v1", Backend.ASCENDC_AOT),),
        )
        with self.assertRaisesRegex(SchemaError, "multiple provider owners"):
            AttentionOperatorProviderRegistry((cann, flash)).discover()


if __name__ == "__main__":
    unittest.main()
