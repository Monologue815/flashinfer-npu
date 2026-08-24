import struct
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    ATTENTION_METADATA_SECTION_C_ABI,
    ATTENTION_PLAN_CONFIG_C_ABI,
    ATTENTION_PLAN_METADATA_HEADER_C_ABI,
    AttentionFrameworkSession,
    AttentionMetadataSectionKind,
    AttentionMode,
    AttentionPlanMetadataDecodeLimits,
    AttentionPlanMetadataWire,
    AttentionPlanFlags,
    PagedKVMetadata,
    build_framework_attention_corpus,
    materialize_attention_plan_metadata,
)
from flashinfer_npu.runtime import SchemaError


DISPATCH_FINGERPRINT = "d" * 64


def plans_by_mode():
    result = {}
    for case in build_framework_attention_corpus().cases:
        trace = case.trace
        if trace.spec.mode not in result:
            result[trace.spec.mode] = AttentionFrameworkSession(
                trace.spec.mode
            ).plan(trace.spec, trace.metadata)
    return result


class AttentionMetadataWireTests(unittest.TestCase):
    def test_config_directory_layout_and_canonical_blob_are_frozen(self):
        self.assertEqual(ATTENTION_PLAN_CONFIG_C_ABI.size_bytes, 128)
        self.assertEqual(ATTENTION_METADATA_SECTION_C_ABI.size_bytes, 32)
        self.assertEqual(
            ATTENTION_PLAN_CONFIG_C_ABI.fingerprint,
            "904cb34545a527c447ae575f037ec7c9764fc3938dc62bddd7be8c8db4c023cb",
        )
        self.assertEqual(
            ATTENTION_METADATA_SECTION_C_ABI.fingerprint,
            "e8af6b5d791411cb58ace8c446d526007c41cda0a9b56fbeac851c5913a90e9d",
        )
        wire = materialize_attention_plan_metadata(
            plans_by_mode()[AttentionMode.BATCH_PREFILL_PAGED],
            dispatch_fingerprint=DISPATCH_FINGERPRINT,
        )
        self.assertEqual(len(wire.to_bytes()), 488)
        self.assertEqual(
            wire.fingerprint,
            "f04f7b3f0f1a8ce0550e265c4d563a6da0a53260d1baab8eacbfb56af6068352",
        )

    def test_all_six_modes_round_trip_with_exact_section_sets(self):
        expected = {
            AttentionMode.SINGLE_PREFILL: (),
            AttentionMode.SINGLE_DECODE: (),
            AttentionMode.BATCH_PREFILL_PAGED: (
                AttentionMetadataSectionKind.QO_INDPTR,
                AttentionMetadataSectionKind.KV_INDPTR,
                AttentionMetadataSectionKind.KV_INDICES,
                AttentionMetadataSectionKind.LAST_PAGE_LEN,
                AttentionMetadataSectionKind.MASK_INDPTR,
            ),
            AttentionMode.BATCH_PREFILL_RAGGED: (
                AttentionMetadataSectionKind.QO_INDPTR,
                AttentionMetadataSectionKind.KV_INDPTR,
                AttentionMetadataSectionKind.MASK_INDPTR,
            ),
            AttentionMode.BATCH_DECODE_PAGED: (
                AttentionMetadataSectionKind.KV_INDPTR,
                AttentionMetadataSectionKind.KV_INDICES,
                AttentionMetadataSectionKind.LAST_PAGE_LEN,
            ),
            AttentionMode.BATCH_MIXED_PAGED: (
                AttentionMetadataSectionKind.QO_INDPTR,
                AttentionMetadataSectionKind.KV_INDPTR,
                AttentionMetadataSectionKind.KV_INDICES,
                AttentionMetadataSectionKind.KV_LEN,
            ),
        }
        plans = plans_by_mode()
        self.assertEqual(set(plans), set(AttentionMode))
        for mode, plan in plans.items():
            with self.subTest(mode=mode):
                wire = materialize_attention_plan_metadata(
                    plan, dispatch_fingerprint=DISPATCH_FINGERPRINT
                )
                payload = wire.to_bytes()
                restored = AttentionPlanMetadataWire.from_bytes(payload)
                self.assertEqual(restored, wire)
                self.assertEqual(restored.to_bytes(), payload)
                self.assertEqual(
                    tuple(section.kind for section in restored.sections),
                    expected[mode],
                )
                restored.validate_plan(plan)

    def test_custom_mask_offsets_flags_and_quantization_are_derived(self):
        plans = plans_by_mode()
        paged = materialize_attention_plan_metadata(
            plans[AttentionMode.BATCH_PREFILL_PAGED],
            dispatch_fingerprint=DISPATCH_FINGERPRINT,
        )
        self.assertTrue(paged.flags & AttentionPlanFlags.CUSTOM_MASK)
        self.assertTrue(paged.flags & AttentionPlanFlags.QUANTIZED_KV)
        self.assertEqual(
            paged.section_map[AttentionMetadataSectionKind.MASK_INDPTR],
            (0, 2),
        )
        ragged = materialize_attention_plan_metadata(
            plans[AttentionMode.BATCH_PREFILL_RAGGED],
            dispatch_fingerprint=DISPATCH_FINGERPRINT,
        )
        self.assertTrue(ragged.flags & AttentionPlanFlags.CUSTOM_MASK_PACKED)
        self.assertEqual(
            ragged.section_map[AttentionMetadataSectionKind.MASK_INDPTR],
            (0, 1),
        )

    def test_noncanonical_lengths_offsets_padding_and_reserved_are_rejected(self):
        plan = plans_by_mode()[AttentionMode.BATCH_PREFILL_PAGED]
        wire = materialize_attention_plan_metadata(
            plan, dispatch_fingerprint=DISPATCH_FINGERPRINT
        )
        payload = wire.to_bytes()
        with self.assertRaisesRegex(SchemaError, "byte length"):
            AttentionPlanMetadataWire.from_bytes(payload[:-1])

        reserved = bytearray(payload)
        reserved[dict(ATTENTION_PLAN_METADATA_HEADER_C_ABI.field_offsets)["reserved"]] = 1
        with self.assertRaisesRegex(SchemaError, "reserved"):
            AttentionPlanMetadataWire.from_bytes(bytes(reserved))

        offset = bytearray(payload)
        first_entry = (
            ATTENTION_PLAN_METADATA_HEADER_C_ABI.size_bytes
            + ATTENTION_PLAN_CONFIG_C_ABI.size_bytes
        )
        struct.pack_into("<Q", offset, first_entry + 8, 0)
        with self.assertRaisesRegex(SchemaError, "offset"):
            AttentionPlanMetadataWire.from_bytes(bytes(offset))

        header_size = ATTENTION_PLAN_METADATA_HEADER_C_ABI.size_bytes
        config = ATTENTION_PLAN_CONFIG_C_ABI.unpack(
            payload[header_size : header_size + ATTENTION_PLAN_CONFIG_C_ABI.size_bytes]
        )
        padding = bytearray(payload)
        for index in range(config["section_count"]):
            start = header_size + ATTENTION_PLAN_CONFIG_C_ABI.size_bytes + index * ATTENTION_METADATA_SECTION_C_ABI.size_bytes
            entry = ATTENTION_METADATA_SECTION_C_ABI.unpack(
                payload[start : start + ATTENTION_METADATA_SECTION_C_ABI.size_bytes]
            )
            if entry["nbytes"] % 8:
                padding[header_size + entry["offset_bytes"] + entry["nbytes"]] = 1
                break
        else:  # pragma: no cover - fixture deliberately contains odd int32 counts
            self.fail("fixture has no padded section")
        with self.assertRaisesRegex(SchemaError, "padding"):
            AttentionPlanMetadataWire.from_bytes(bytes(padding))

    def test_decode_limits_and_int32_index_boundary_are_enforced(self):
        plan = plans_by_mode()[AttentionMode.BATCH_DECODE_PAGED]
        payload = materialize_attention_plan_metadata(
            plan, dispatch_fingerprint=DISPATCH_FINGERPRINT
        ).to_bytes()
        with self.assertRaisesRegex(SchemaError, "byte limit"):
            AttentionPlanMetadataWire.from_bytes(
                payload,
                limits=AttentionPlanMetadataDecodeLimits(
                    max_total_nbytes=len(payload) - 1,
                    max_section_count=16,
                    max_section_elements=1024,
                ),
            )
        with self.assertRaisesRegex(SchemaError, "element limit"):
            AttentionPlanMetadataWire.from_bytes(
                payload,
                limits=AttentionPlanMetadataDecodeLimits(
                    max_total_nbytes=len(payload),
                    max_section_count=16,
                    max_section_elements=1,
                ),
            )

        metadata = PagedKVMetadata((0, 1), (1 << 31,), (1,), 1)
        overflow = AttentionFrameworkSession(AttentionMode.BATCH_DECODE_PAGED).plan(
            replace(plan.spec, q_len_per_req=1), metadata
        )
        with self.assertRaisesRegex(SchemaError, "outside int32"):
            materialize_attention_plan_metadata(
                overflow, dispatch_fingerprint=DISPATCH_FINGERPRINT
            )

    def test_unknown_mode_flags_config_and_section_enums_are_rejected(self):
        plan = plans_by_mode()[AttentionMode.BATCH_PREFILL_PAGED]
        payload = materialize_attention_plan_metadata(
            plan, dispatch_fingerprint=DISPATCH_FINGERPRINT
        ).to_bytes()
        header_size = ATTENTION_PLAN_METADATA_HEADER_C_ABI.size_bytes

        mode = bytearray(payload)
        struct.pack_into("<H", mode, 4, 0xFFFF)
        with self.assertRaisesRegex(SchemaError, "mode code"):
            AttentionPlanMetadataWire.from_bytes(bytes(mode))

        flags = bytearray(payload)
        struct.pack_into("<H", flags, 6, 1 << 15)
        with self.assertRaisesRegex(SchemaError, "unknown bits"):
            AttentionPlanMetadataWire.from_bytes(bytes(flags))

        config = bytearray(payload)
        struct.pack_into("<H", config, header_size + 36, 0xFFFF)
        with self.assertRaisesRegex(SchemaError, "wire enum"):
            AttentionPlanMetadataWire.from_bytes(bytes(config))

        section = bytearray(payload)
        struct.pack_into(
            "<H",
            section,
            header_size + ATTENTION_PLAN_CONFIG_C_ABI.size_bytes,
            0xFFFF,
        )
        with self.assertRaisesRegex(SchemaError, "unknown enum"):
            AttentionPlanMetadataWire.from_bytes(bytes(section))

    def test_stale_plan_and_binary_fingerprint_are_rejected(self):
        plans = plans_by_mode()
        wire = materialize_attention_plan_metadata(
            plans[AttentionMode.SINGLE_DECODE],
            dispatch_fingerprint=DISPATCH_FINGERPRINT,
        )
        with self.assertRaisesRegex(SchemaError, "stale"):
            wire.validate_plan(plans[AttentionMode.SINGLE_PREFILL])
        with self.assertRaisesRegex(SchemaError, "canonical binary ABI"):
            materialize_attention_plan_metadata(
                plans[AttentionMode.SINGLE_DECODE],
                dispatch_fingerprint=DISPATCH_FINGERPRINT,
                binary_abi_fingerprint="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
