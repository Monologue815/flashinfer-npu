import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorNvfp4PackedKVBinding,
    AttentionOperatorNvfp4ScaleFactorBinding,
    validate_attention_operator_nvfp4_packed_kv_bindings,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_101_nvfp4_runtime_registration import runtime_values


def binding_with_structures(values, *, combined, separate):
    original = values["packed_binding"]
    scale_binding = AttentionOperatorNvfp4ScaleFactorBinding(
        provider_id=original.provider_id,
        operation_id=original.operation_id,
        quant_spec=original.quant_spec,
        combined_argument="kv_cache_sf" if combined else None,
        key_argument="k_sf" if separate else None,
        value_argument="v_sf" if separate else None,
    )
    return AttentionOperatorNvfp4PackedKVBinding(
        scale_binding, original.layout_descriptor
    )


def profiles_for_mode(values, mode):
    profile = values["profile"]
    rule = replace(profile.rules[0], modes=(mode,))
    return (replace(profile, rules=(rule,)),)


class Nvfp4PublicStructureAdmissionCheckpoint(unittest.TestCase):
    """Plan-time candidates cover every run-time KV structure of the facade."""

    def test_paged_capability_requires_combined_and_separate_routes(self):
        values = runtime_values()
        profiles = profiles_for_mode(values, AttentionMode.BATCH_DECODE_PAGED)

        accepted = validate_attention_operator_nvfp4_packed_kv_bindings(
            values["operation"], profiles, (values["packed_binding"],)
        )

        self.assertEqual(accepted, (values["packed_binding"],))
        for combined, separate in ((True, False), (False, True)):
            with self.subTest(combined=combined, separate=separate):
                with self.assertRaisesRegex(
                    SchemaError, "combined and separate public KV"
                ):
                    validate_attention_operator_nvfp4_packed_kv_bindings(
                        values["operation"],
                        profiles,
                        (
                            binding_with_structures(
                                values,
                                combined=combined,
                                separate=separate,
                            ),
                        ),
                    )

    def test_single_capability_requires_separate_kv(self):
        values = runtime_values()
        profiles = profiles_for_mode(values, AttentionMode.SINGLE_PREFILL)

        with self.assertRaisesRegex(SchemaError, "separate public K/V"):
            validate_attention_operator_nvfp4_packed_kv_bindings(
                values["operation"],
                profiles,
                (
                    binding_with_structures(
                        values, combined=True, separate=False
                    ),
                ),
            )

    def test_ragged_nvfp4_is_rejected_at_registration(self):
        values = runtime_values()
        profiles = profiles_for_mode(values, AttentionMode.BATCH_PREFILL_RAGGED)

        with self.assertRaisesRegex(SchemaError, "cannot authorize ragged"):
            validate_attention_operator_nvfp4_packed_kv_bindings(
                values["operation"], profiles, (values["packed_binding"],)
            )

    def test_rejection_is_metadata_only(self):
        values = runtime_values()
        profiles = profiles_for_mode(values, AttentionMode.BATCH_PREFILL_PAGED)

        with self.assertRaisesRegex(SchemaError, "combined and separate"):
            validate_attention_operator_nvfp4_packed_kv_bindings(
                values["operation"],
                profiles,
                (
                    binding_with_structures(
                        values, combined=False, separate=True
                    ),
                ),
            )

        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])


if __name__ == "__main__":
    unittest.main()
