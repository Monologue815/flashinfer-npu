import unittest

from flashinfer_npu.attention.schema import (
    MixedPagedKVMetadata,
    PagedKVMetadata,
    PagedPrefillMetadata,
    RaggedKVMetadata,
    TensorSpec,
    attention_metadata_from_dict,
)
from flashinfer_npu.runtime import SchemaError


class IndexValue:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

    def __int__(self):
        raise AssertionError("metadata must use the integer index protocol")


class IntOnly:
    def __int__(self):
        return 1


class MetadataIntegerContractCheckpoint(unittest.TestCase):
    def factories(self):
        paged = PagedKVMetadata((0, 1), (0,), (1,), 8)
        return (
            lambda x: TensorSpec((x,), "float16"),
            lambda x: PagedKVMetadata((0, x), (0,), (1,), 8),
            lambda x: PagedKVMetadata((0, 1), (x,), (1,), 8),
            lambda x: PagedKVMetadata((0, 1), (0,), (x,), 8),
            lambda x: PagedPrefillMetadata((0, x), paged),
            lambda x: RaggedKVMetadata((0, x), (0, 1)),
            lambda x: RaggedKVMetadata((0, 1), (0, x)),
            lambda x: MixedPagedKVMetadata((0, x), (0, 1), (0,), (1,), 8),
            lambda x: MixedPagedKVMetadata((0, 1), (0, x), (0,), (1,), 8),
            lambda x: MixedPagedKVMetadata((0, 1), (0, 1), (x,), (1,), 8),
            lambda x: MixedPagedKVMetadata((0, 1), (0, 1), (0,), (x,), 8),
        )

    def test_all_integer_arrays_reject_lossy_values(self):
        for field, factory in enumerate(self.factories()):
            for value in (True, False, 1.0, 1.5, -0.5, "1", float("inf"),
                          float("nan"), IntOnly()):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SchemaError):
                        factory(value)

    def test_index_protocol_preserves_canonical_values(self):
        for field, factory in enumerate(self.factories()):
            with self.subTest(field=field):
                self.assertEqual(factory(IndexValue(1)), factory(1))
        metadata = PagedKVMetadata([IndexValue(0), IndexValue(1)],
                                   [IndexValue(2)], [IndexValue(3)], 8)
        expected = PagedKVMetadata((0, 1), (2,), (3,), 8)
        self.assertEqual(metadata.fingerprint, expected.fingerprint)
        self.assertIs(type(metadata.indices[0]), int)

    def test_serialized_metadata_uses_same_integer_gate(self):
        fixtures = (
            PagedKVMetadata((0, 1), (0,), (1,), 8),
            RaggedKVMetadata((0, 1), (0, 1)),
            MixedPagedKVMetadata((0, 1), (0, 1), (0,), (1,), 8),
        )
        for metadata in fixtures:
            self.assertEqual(attention_metadata_from_dict(metadata.to_dict()), metadata)
            for field, values in metadata.to_dict().items():
                if not isinstance(values, list):
                    continue
                for invalid in (True, 1.5, "1"):
                    with self.subTest(kind=type(metadata).__name__, field=field,
                                      invalid=invalid):
                        payload = metadata.to_dict()
                        payload[field][-1] = invalid
                        with self.assertRaises(SchemaError):
                            attention_metadata_from_dict(payload)
