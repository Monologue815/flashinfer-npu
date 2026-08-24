import random
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionMode,
    AttentionPlanSpec,
    CustomMaskSpec,
    KVLayout,
    MixedPagedKVMetadata,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    ReferenceAttentionExecutor,
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    RaggedKVCacheSpec,
    SingleAttentionMetadata,
)
from flashinfer_npu.runtime import QuantSpec


def flat(shape, data, dtype="float32"):
    return ReferenceTensor(tuple(shape), tuple(data), dtype=dtype, device="cpu")


def assert_tensor_close(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    for left, right in zip(actual.data, expected.data):
        test.assertAlmostEqual(left, right, places=12)


class DeterministicAttentionPropertyTests(unittest.TestCase):
    def test_dense_and_exact_int8_dequant_are_equivalent_for_random_small_shapes(self):
        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="float32",
            accumulator_dtype="float32",
        )
        executor = ReferenceAttentionExecutor()
        for seed in range(32):
            rng = random.Random(seed)
            qo_len = rng.randint(1, 3)
            kv_len = rng.randint(1, 4)
            num_kv_heads = rng.choice((1, 2))
            group_size = rng.choice((1, 2))
            num_qo_heads = num_kv_heads * group_size
            qk_dim = rng.randint(1, 3)
            vo_dim = rng.randint(1, 3)
            q_shape = (qo_len, num_qo_heads, qk_dim)
            key_shape = (kv_len, num_kv_heads, qk_dim)
            value_shape = (kv_len, num_kv_heads, vo_dim)
            q = flat(
                q_shape,
                (rng.randint(-8, 8) / 4.0 for _ in range(
                    qo_len * num_qo_heads * qk_dim
                )),
            )
            key_values = tuple(
                rng.randint(-4, 4)
                for _ in range(kv_len * num_kv_heads * qk_dim)
            )
            value_values = tuple(
                rng.randint(-4, 4)
                for _ in range(kv_len * num_kv_heads * vo_dim)
            )
            mask = tuple(
                rng.choice((False, True)) for _ in range(qo_len * kv_len)
            )
            mask_spec = CustomMaskSpec(len(mask)) if seed % 2 else None
            metadata = SingleAttentionMetadata(qo_len, kv_len)
            common = dict(
                mode=AttentionMode.SINGLE_PREFILL,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=qk_dim,
                head_dim_vo=vo_dim,
                q_dtype="float32",
                o_dtype="float32",
                sm_scale=0.75,
                custom_mask=mask_spec,
            )
            dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
            quant_spec = AttentionPlanSpec(
                kv_dtype="int8", kv_quant_spec=quant, **common
            )
            dense_cache = RaggedKVCacheSpec(
                kv_len,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "float32",
                device="cpu",
            )
            int8_cache = RaggedKVCacheSpec(
                kv_len,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "int8",
                device="cpu",
                quant_spec=quant,
            )
            dense_kv = ReferenceKVData(
                dense_cache,
                (flat(key_shape, key_values), flat(value_shape, value_values)),
            )
            scale = flat((), (1.0,))
            quant_kv = ReferenceQuantizedKVData(
                int8_cache,
                ReferenceQuantizedTensor(
                    key_shape, flat(key_shape, key_values, "int8"), scale, quant
                ),
                ReferenceQuantizedTensor(
                    value_shape,
                    flat(value_shape, value_values, "int8"),
                    scale,
                    quant,
                ),
            )
            mask_data = mask if mask_spec is not None else None
            dense_result = executor.execute(
                AttentionFrameworkSession(dense_spec.mode).plan(
                    dense_spec, metadata
                ),
                q,
                dense_kv,
                return_lse=True,
                custom_mask_data=mask_data,
            )
            quant_result = executor.execute(
                AttentionFrameworkSession(quant_spec.mode).plan(
                    quant_spec, metadata
                ),
                q,
                quant_kv,
                return_lse=True,
                custom_mask_data=mask_data,
            )
            with self.subTest(seed=seed):
                assert_tensor_close(self, dense_result.output, quant_result.output)
                assert_tensor_close(self, dense_result.lse, quant_result.lse)

    def test_nhd_and_hnd_paged_layouts_are_equivalent_for_random_page_tables(self):
        executor = ReferenceAttentionExecutor()
        for seed in range(24):
            rng = random.Random(1000 + seed)
            num_pages = rng.randint(1, 3)
            page_size = rng.randint(1, 3)
            num_heads = rng.choice((1, 2))
            qk_dim = rng.randint(1, 3)
            vo_dim = rng.randint(1, 3)
            page_order = list(range(num_pages))
            rng.shuffle(page_order)
            page_count = rng.randint(1, num_pages)
            page_order = tuple(page_order[:page_count])
            last_len = rng.randint(1, page_size)
            metadata = PagedKVMetadata(
                (0, page_count), page_order, (last_len,), page_size
            )
            q = flat(
                (1, num_heads, qk_dim),
                (
                    rng.randint(-8, 8) / 4.0
                    for _ in range(num_heads * qk_dim)
                ),
            )
            keys = [
                [
                    [
                        [rng.randint(-4, 4) for _ in range(qk_dim)]
                        for _ in range(num_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            values = [
                [
                    [
                        [rng.randint(-4, 4) for _ in range(vo_dim)]
                        for _ in range(num_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]

            def flatten_nhd(data):
                return tuple(
                    component
                    for page in data
                    for token in page
                    for head in token
                    for component in head
                )

            def flatten_hnd(data):
                return tuple(
                    data[page][token][head][component]
                    for page in range(num_pages)
                    for head in range(num_heads)
                    for token in range(page_size)
                    for component in range(len(data[page][token][head]))
                )

            results = []
            for layout, key_data, value_data in (
                (KVLayout.NHD, flatten_nhd(keys), flatten_nhd(values)),
                (KVLayout.HND, flatten_hnd(keys), flatten_hnd(values)),
            ):
                spec = AttentionPlanSpec(
                    mode=AttentionMode.BATCH_DECODE_PAGED,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim_qk=qk_dim,
                    head_dim_vo=vo_dim,
                    kv_layout=layout,
                    q_dtype="float32",
                    kv_dtype="float32",
                    sm_scale=0.5,
                )
                cache = PagedKVCacheSpec(
                    num_pages,
                    page_size,
                    num_heads,
                    qk_dim,
                    vo_dim,
                    "float32",
                    layout=layout,
                    structure="separate",
                    device="cpu",
                )
                key_shape, value_shape = cache.expected_shapes
                result = executor.execute(
                    AttentionFrameworkSession(spec.mode).plan(spec, metadata),
                    q,
                    ReferenceKVData(
                        cache,
                        (
                            flat(key_shape, key_data),
                            flat(value_shape, value_data),
                        ),
                    ),
                    return_lse=True,
                )
                results.append(result)
            with self.subTest(seed=seed):
                assert_tensor_close(self, results[0].output, results[1].output)
                assert_tensor_close(self, results[0].lse, results[1].lse)

    def test_shared_pages_and_segment_packed_masks_match_independent_requests(self):
        executor = ReferenceAttentionExecutor()

        def flatten_nested(value):
            if isinstance(value, (list, tuple)):
                return tuple(item for child in value for item in flatten_nested(child))
            return (value,)

        def pack_little(bits):
            values = []
            for offset in range(0, len(bits), 8):
                byte = 0
                for bit, enabled in enumerate(bits[offset : offset + 8]):
                    if enabled:
                        byte |= 1 << bit
                values.append(byte)
            return tuple(values)

        for seed in range(24):
            rng = random.Random(2000 + seed)
            page_size = rng.randint(2, 3)
            num_pages = 3
            num_kv_heads = rng.choice((1, 2))
            group_size = rng.choice((1, 2))
            num_qo_heads = num_kv_heads * group_size
            qk_dim = rng.randint(1, 3)
            vo_dim = rng.randint(1, 3)
            layout = rng.choice((KVLayout.NHD, KVLayout.HND))
            page_lists = ((0,), (0, 1), (2, 0))
            last_page_len = tuple(rng.randint(1, page_size) for _ in page_lists)
            kv_lengths = tuple(
                (len(pages) - 1) * page_size + last
                for pages, last in zip(page_lists, last_page_len)
            )
            qo_lengths = tuple(rng.randint(1, min(2, length)) for length in kv_lengths)
            qo_indptr = [0]
            page_indptr = [0]
            page_indices = []
            for qo_len, pages in zip(qo_lengths, page_lists):
                qo_indptr.append(qo_indptr[-1] + qo_len)
                page_indices.extend(pages)
                page_indptr.append(page_indptr[-1] + len(pages))

            keys = [
                [
                    [
                        [rng.randint(-4, 4) for _ in range(qk_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            values = [
                [
                    [
                        [rng.randint(-4, 4) for _ in range(vo_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            queries = [
                [
                    [rng.randint(-8, 8) / 4.0 for _ in range(qk_dim)]
                    for _ in range(num_qo_heads)
                ]
                for _ in range(qo_indptr[-1])
            ]

            mask_bits = []
            mask_segments = []
            mask_bytes = []
            for qo_len, kv_len in zip(qo_lengths, kv_lengths):
                request_bits = []
                for _ in range(qo_len):
                    row = [rng.choice((False, True)) for _ in range(kv_len)]
                    row[rng.randrange(kv_len)] = True
                    request_bits.extend(row)
                request_bytes = pack_little(request_bits)
                mask_bits.append(tuple(request_bits))
                mask_segments.append(request_bytes)
                mask_bytes.extend(request_bytes)

            paged_metadata = PagedKVMetadata(
                tuple(page_indptr),
                tuple(page_indices),
                last_page_len,
                page_size,
            )
            batch_spec = AttentionPlanSpec(
                mode=AttentionMode.BATCH_PREFILL_PAGED,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=qk_dim,
                head_dim_vo=vo_dim,
                kv_layout=layout,
                q_dtype="float32",
                kv_dtype="float32",
                sm_scale=0.75,
                custom_mask=CustomMaskSpec(len(mask_bytes), packed=True),
            )

            def flatten_layout(data):
                if layout == KVLayout.NHD:
                    return flatten_nested(data)
                return tuple(
                    data[page][token][head][component]
                    for page in range(num_pages)
                    for head in range(num_kv_heads)
                    for token in range(page_size)
                    for component in range(len(data[page][token][head]))
                )

            cache = PagedKVCacheSpec(
                num_pages,
                page_size,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "float32",
                layout=layout,
                structure="separate",
                device="cpu",
            )
            key_shape, value_shape = cache.expected_shapes
            batch_result = executor.execute(
                AttentionFrameworkSession(batch_spec.mode).plan(
                    batch_spec,
                    PagedPrefillMetadata(tuple(qo_indptr), paged_metadata),
                ),
                flat(
                    (qo_indptr[-1], num_qo_heads, qk_dim),
                    flatten_nested(queries),
                ),
                ReferenceKVData(
                    cache,
                    (
                        flat(key_shape, flatten_layout(keys)),
                        flat(value_shape, flatten_layout(values)),
                    ),
                ),
                return_lse=True,
                custom_mask_data=tuple(mask_bytes),
            )

            expected_output = []
            expected_lse = []
            for request, (q_start, q_end, pages, kv_len) in enumerate(
                zip(qo_indptr, qo_indptr[1:], page_lists, kv_lengths)
            ):
                logical_keys = []
                logical_values = []
                for token in range(kv_len):
                    page = pages[token // page_size]
                    offset = token % page_size
                    logical_keys.append(keys[page][offset])
                    logical_values.append(values[page][offset])
                single_spec = replace(
                    batch_spec,
                    mode=AttentionMode.SINGLE_PREFILL,
                    custom_mask=CustomMaskSpec(
                        len(mask_segments[request]), packed=True
                    ),
                )
                ragged_cache = RaggedKVCacheSpec(
                    kv_len,
                    num_kv_heads,
                    qk_dim,
                    vo_dim,
                    "float32",
                    layout=KVLayout.NHD,
                    device="cpu",
                )
                single_result = executor.execute(
                    AttentionFrameworkSession(single_spec.mode).plan(
                        replace(single_spec, kv_layout=KVLayout.NHD),
                        SingleAttentionMetadata(q_end - q_start, kv_len),
                    ),
                    flat(
                        (q_end - q_start, num_qo_heads, qk_dim),
                        flatten_nested(queries[q_start:q_end]),
                    ),
                    ReferenceKVData(
                        ragged_cache,
                        (
                            flat(
                                (kv_len, num_kv_heads, qk_dim),
                                flatten_nested(logical_keys),
                            ),
                            flat(
                                (kv_len, num_kv_heads, vo_dim),
                                flatten_nested(logical_values),
                            ),
                        ),
                    ),
                    return_lse=True,
                    custom_mask_data=mask_segments[request],
                )
                expected_output.extend(single_result.output.data)
                expected_lse.extend(single_result.lse.data)

            with self.subTest(seed=seed, layout=layout.value):
                assert_tensor_close(
                    self,
                    batch_result.output,
                    flat(batch_result.output.shape, expected_output),
                )
                assert_tensor_close(
                    self,
                    batch_result.lse,
                    flat(batch_result.lse.shape, expected_lse),
                )

    def test_mixed_batch_matches_independent_single_requests_with_empty_and_repeated_pages(self):
        executor = ReferenceAttentionExecutor()

        def flatten_nested(value):
            if isinstance(value, (list, tuple)):
                return tuple(item for child in value for item in flatten_nested(child))
            return (value,)

        for seed in range(24):
            rng = random.Random(3000 + seed)
            page_size = rng.randint(2, 3)
            num_pages = 3
            num_kv_heads = rng.choice((1, 2))
            num_qo_heads = num_kv_heads * rng.choice((1, 2))
            qk_dim = rng.randint(1, 3)
            vo_dim = rng.randint(1, 3)
            layout = rng.choice((KVLayout.NHD, KVLayout.HND))
            causal = bool(seed % 2)

            # Request 0 has no Q or KV, request 1 has KV but no Q, request 2 is
            # decode-like, and requests 3/4 exercise repeated and shared pages.
            page_lists = ((), (0,), (0,), (1, 1), (2, 0))
            kv_lengths = (
                0,
                rng.randint(1, page_size),
                rng.randint(1, page_size),
                page_size + rng.randint(1, page_size),
                page_size + rng.randint(1, page_size),
            )
            qo_lengths = (
                0,
                0,
                1,
                rng.randint(1, min(3, kv_lengths[3])),
                rng.randint(1, min(3, kv_lengths[4])),
            )
            qo_indptr = [0]
            kv_indptr = [0]
            kv_indices = []
            for qo_len, pages in zip(qo_lengths, page_lists):
                qo_indptr.append(qo_indptr[-1] + qo_len)
                kv_indices.extend(pages)
                kv_indptr.append(kv_indptr[-1] + len(pages))

            keys = [
                [
                    [
                        [rng.randint(-4, 4) for _ in range(qk_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            values = [
                [
                    [
                        [rng.randint(-4, 4) for _ in range(vo_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            queries = [
                [
                    [rng.randint(-8, 8) / 4.0 for _ in range(qk_dim)]
                    for _ in range(num_qo_heads)
                ]
                for _ in range(qo_indptr[-1])
            ]

            def flatten_layout(data):
                if layout == KVLayout.NHD:
                    return flatten_nested(data)
                return tuple(
                    data[page][token][head][component]
                    for page in range(num_pages)
                    for head in range(num_kv_heads)
                    for token in range(page_size)
                    for component in range(len(data[page][token][head]))
                )

            cache = PagedKVCacheSpec(
                num_pages,
                page_size,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "float32",
                layout=layout,
                structure="separate",
                device="cpu",
            )
            key_shape, value_shape = cache.expected_shapes
            kv_data = ReferenceKVData(
                cache,
                (
                    flat(key_shape, flatten_layout(keys)),
                    flat(value_shape, flatten_layout(values)),
                ),
            )
            batch_spec = AttentionPlanSpec(
                mode=AttentionMode.BATCH_MIXED_PAGED,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=qk_dim,
                head_dim_vo=vo_dim,
                kv_layout=layout,
                causal=causal,
                q_dtype="float32",
                kv_dtype="float32",
                o_dtype="float32",
                sm_scale=0.75,
            )
            metadata = MixedPagedKVMetadata(
                tuple(qo_indptr),
                tuple(kv_indptr),
                tuple(kv_indices),
                kv_lengths,
                page_size,
            )
            batch_result = executor.execute(
                AttentionFrameworkSession(batch_spec.mode).plan(batch_spec, metadata),
                flat(
                    (qo_indptr[-1], num_qo_heads, qk_dim),
                    flatten_nested(queries),
                ),
                kv_data,
                return_lse=True,
            )

            expected_output = []
            expected_lse = []
            for q_start, q_end, pages, kv_len in zip(
                qo_indptr, qo_indptr[1:], page_lists, kv_lengths
            ):
                qo_len = q_end - q_start
                if qo_len == 0:
                    continue
                logical_keys = []
                logical_values = []
                for token in range(kv_len):
                    page = pages[token // page_size]
                    offset = token % page_size
                    logical_keys.append(keys[page][offset])
                    logical_values.append(values[page][offset])
                single_spec = replace(
                    batch_spec,
                    mode=AttentionMode.SINGLE_PREFILL,
                    kv_layout=KVLayout.NHD,
                )
                ragged_cache = RaggedKVCacheSpec(
                    kv_len,
                    num_kv_heads,
                    qk_dim,
                    vo_dim,
                    "float32",
                    layout=KVLayout.NHD,
                    device="cpu",
                )
                single_result = executor.execute(
                    AttentionFrameworkSession(single_spec.mode).plan(
                        single_spec, SingleAttentionMetadata(qo_len, kv_len)
                    ),
                    flat(
                        (qo_len, num_qo_heads, qk_dim),
                        flatten_nested(queries[q_start:q_end]),
                    ),
                    ReferenceKVData(
                        ragged_cache,
                        (
                            flat(
                                (kv_len, num_kv_heads, qk_dim),
                                flatten_nested(logical_keys),
                            ),
                            flat(
                                (kv_len, num_kv_heads, vo_dim),
                                flatten_nested(logical_values),
                            ),
                        ),
                    ),
                    return_lse=True,
                )
                expected_output.extend(single_result.output.data)
                expected_lse.extend(single_result.lse.data)

            with self.subTest(seed=seed, layout=layout.value, causal=causal):
                assert_tensor_close(
                    self,
                    batch_result.output,
                    flat(batch_result.output.shape, expected_output),
                )
                assert_tensor_close(
                    self,
                    batch_result.lse,
                    flat(batch_result.lse.shape, expected_lse),
                )

    def test_mixed_shared_and_repeated_pages_match_exact_int8_dequantization(self):
        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="float32",
            accumulator_dtype="float32",
        )
        executor = ReferenceAttentionExecutor()
        metadata = MixedPagedKVMetadata(
            qo_indptr=(0, 0, 1, 3),
            kv_indptr=(0, 1, 2, 4),
            kv_indices=(0, 0, 1, 1),
            kv_len_arr=(2, 1, 4),
            page_size=2,
        )
        common = dict(
            mode=AttentionMode.BATCH_MIXED_PAGED,
            num_qo_heads=2,
            num_kv_heads=1,
            head_dim_qk=2,
            head_dim_vo=3,
            causal=True,
            q_dtype="float32",
            o_dtype="float32",
            sm_scale=0.5,
        )
        dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
        quant_spec = AttentionPlanSpec(
            kv_dtype="int8", kv_quant_spec=quant, **common
        )
        dense_cache = PagedKVCacheSpec(
            2, 2, 1, 2, 3, "float32", structure="separate", device="cpu"
        )
        int8_cache = PagedKVCacheSpec(
            2,
            2,
            1,
            2,
            3,
            "int8",
            structure="separate",
            device="cpu",
            quant_spec=quant,
        )
        key_values = (1, -2, 3, 0, -1, 4, 2, 1)
        value_values = (2, -1, 3, 4, 0, -2, 1, 5, -3, 2, 2, 0)
        key_shape, value_shape = dense_cache.expected_shapes
        dense_kv = ReferenceKVData(
            dense_cache,
            (
                flat(key_shape, key_values),
                flat(value_shape, value_values),
            ),
        )
        scale = flat((), (1.0,))
        quant_kv = ReferenceQuantizedKVData(
            int8_cache,
            ReferenceQuantizedTensor(
                key_shape, flat(key_shape, key_values, "int8"), scale, quant
            ),
            ReferenceQuantizedTensor(
                value_shape,
                flat(value_shape, value_values, "int8"),
                scale,
                quant,
            ),
        )
        q = flat(
            (3, 2, 2),
            (0.5, -1.0, 1.0, 0.25, -0.5, 2.0, 1.5, -0.25, 0.0, 1.0, 2.0, -1.0),
        )
        dense_result = executor.execute(
            AttentionFrameworkSession(dense_spec.mode).plan(dense_spec, metadata),
            q,
            dense_kv,
            return_lse=True,
        )
        quant_result = executor.execute(
            AttentionFrameworkSession(quant_spec.mode).plan(quant_spec, metadata),
            q,
            quant_kv,
            return_lse=True,
        )
        assert_tensor_close(self, dense_result.output, quant_result.output)
        assert_tensor_close(self, dense_result.lse, quant_result.lse)

    def test_mixed_packed_int4_matches_explicit_dequant_with_window_and_soft_cap(self):
        executor = ReferenceAttentionExecutor()

        def rows(data, layout, num_pages, page_size, num_heads):
            if layout == KVLayout.NHD:
                return [
                    data[page][token][head]
                    for page in range(num_pages)
                    for token in range(page_size)
                    for head in range(num_heads)
                ]
            return [
                data[page][token][head]
                for page in range(num_pages)
                for head in range(num_heads)
                for token in range(page_size)
            ]

        def pack_rows(values, order):
            packed = []
            for row in values:
                for offset in range(0, len(row), 2):
                    first = row[offset] & 0xF
                    second = row[offset + 1] & 0xF if offset + 1 < len(row) else 0
                    packed.append(
                        first | (second << 4)
                        if order == "low_nibble_first"
                        else (first << 4) | second
                    )
            return tuple(packed)

        for seed in range(16):
            rng = random.Random(4000 + seed)
            layout = rng.choice((KVLayout.NHD, KVLayout.HND))
            order = rng.choice(("low_nibble_first", "high_nibble_first"))
            page_size = 2
            num_pages = 3
            num_kv_heads = rng.choice((1, 2))
            num_qo_heads = num_kv_heads * rng.choice((1, 2))
            qk_dim = rng.choice((1, 3, 5))
            vo_dim = rng.choice((1, 3, 5))
            metadata = MixedPagedKVMetadata(
                qo_indptr=(0, 1, 3, 4),
                kv_indptr=(0, 1, 3, 5),
                kv_indices=(0, 0, 1, 2, 2),
                kv_len_arr=(2, 3, 4),
                page_size=page_size,
            )
            keys = [
                [
                    [
                        [rng.randint(-8, 7) for _ in range(qk_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            values = [
                [
                    [
                        [rng.randint(-8, 7) for _ in range(vo_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            q = flat(
                (4, num_qo_heads, qk_dim),
                (
                    rng.randint(-8, 8) / 4.0
                    for _ in range(4 * num_qo_heads * qk_dim)
                ),
            )
            quant = QuantSpec(
                scheme="symmetric",
                storage_dtype="int4_packed",
                compute_dtype="float32",
                accumulator_dtype="float32",
                packing_order=order,
            )
            common = dict(
                mode=AttentionMode.BATCH_MIXED_PAGED,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=qk_dim,
                head_dim_vo=vo_dim,
                kv_layout=layout,
                causal=bool(seed % 2),
                q_dtype="float32",
                o_dtype="float32",
                sm_scale=0.625,
                logits_soft_cap=2.0,
                window_left=(-1 if seed % 4 == 0 else seed % 3),
            )
            dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
            quant_spec = AttentionPlanSpec(
                kv_dtype="int4_packed", kv_quant_spec=quant, **common
            )
            dense_cache = PagedKVCacheSpec(
                num_pages,
                page_size,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "float32",
                layout=layout,
                structure="separate",
                device="cpu",
            )
            quant_cache = PagedKVCacheSpec(
                num_pages,
                page_size,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "int4_packed",
                layout=layout,
                structure="separate",
                device="cpu",
                quant_spec=quant,
            )
            key_shape, value_shape = dense_cache.expected_shapes
            key_rows = rows(keys, layout, num_pages, page_size, num_kv_heads)
            value_rows = rows(values, layout, num_pages, page_size, num_kv_heads)
            key_storage_shape = key_shape[:-1] + ((qk_dim + 1) // 2,)
            value_storage_shape = value_shape[:-1] + ((vo_dim + 1) // 2,)
            key_scale = 0.5
            value_scale = 1.25
            dense_kv = ReferenceKVData(
                dense_cache,
                (
                    flat(
                        key_shape,
                        (item * key_scale for row in key_rows for item in row),
                    ),
                    flat(
                        value_shape,
                        (item * value_scale for row in value_rows for item in row),
                    ),
                ),
            )
            quant_kv = ReferenceQuantizedKVData(
                quant_cache,
                ReferenceQuantizedTensor(
                    key_shape,
                    flat(key_storage_shape, pack_rows(key_rows, order), "uint8"),
                    flat((), (key_scale,)),
                    quant,
                ),
                ReferenceQuantizedTensor(
                    value_shape,
                    flat(value_storage_shape, pack_rows(value_rows, order), "uint8"),
                    flat((), (value_scale,)),
                    quant,
                ),
            )
            runtime_k_scale = tuple(
                0.75 + 0.25 * head for head in range(num_kv_heads)
            )
            runtime_v_scale = tuple(
                1.5 - 0.25 * head for head in range(num_kv_heads)
            )
            dense_result = executor.execute(
                AttentionFrameworkSession(dense_spec.mode).plan(dense_spec, metadata),
                q,
                dense_kv,
                return_lse=True,
                k_scale=runtime_k_scale,
                v_scale=runtime_v_scale,
                logits_soft_cap=1.25,
            )
            quant_result = executor.execute(
                AttentionFrameworkSession(quant_spec.mode).plan(quant_spec, metadata),
                q,
                quant_kv,
                return_lse=True,
                k_scale=runtime_k_scale,
                v_scale=runtime_v_scale,
                logits_soft_cap=1.25,
            )
            with self.subTest(seed=seed, layout=layout.value, order=order):
                assert_tensor_close(self, dense_result.output, quant_result.output)
                assert_tensor_close(self, dense_result.lse, quant_result.lse)

    def test_mixed_asymmetric_uint8_per_head_matches_explicit_dequantization(self):
        executor = ReferenceAttentionExecutor()

        for seed in range(16):
            rng = random.Random(5000 + seed)
            layout = rng.choice((KVLayout.NHD, KVLayout.HND))
            page_size = 2
            num_pages = 3
            num_kv_heads = 2
            num_qo_heads = 4
            qk_dim = rng.randint(1, 4)
            vo_dim = rng.randint(1, 4)
            metadata = MixedPagedKVMetadata(
                qo_indptr=(0, 1, 3, 4),
                kv_indptr=(0, 1, 3, 5),
                kv_indices=(0, 0, 1, 2, 2),
                kv_len_arr=(2, 3, 4),
                page_size=page_size,
            )
            head_axis = 2 if layout == KVLayout.NHD else 1
            quant = QuantSpec(
                scheme="asymmetric",
                storage_dtype="uint8",
                compute_dtype="float32",
                accumulator_dtype="float32",
                granularity="channel",
                axis=(head_axis,),
                has_zero_point=True,
            )
            key_scales = (0.25, 0.75)
            value_scales = (0.5, 1.25)
            key_zero = (3, 9)
            value_zero = (7, 2)
            keys = [
                [
                    [
                        [rng.randint(0, 15) for _ in range(qk_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]
            values = [
                [
                    [
                        [rng.randint(0, 15) for _ in range(vo_dim)]
                        for _ in range(num_kv_heads)
                    ]
                    for _ in range(page_size)
                ]
                for _ in range(num_pages)
            ]

            def flatten_layout(data, *, scales=None, zero_points=None):
                result = []
                order = (
                    (
                        (page, token, head)
                        for page in range(num_pages)
                        for token in range(page_size)
                        for head in range(num_kv_heads)
                    )
                    if layout == KVLayout.NHD
                    else (
                        (page, token, head)
                        for page in range(num_pages)
                        for head in range(num_kv_heads)
                        for token in range(page_size)
                    )
                )
                for page, token, head in order:
                    for item in data[page][token][head]:
                        result.append(
                            item
                            if scales is None
                            else (item - zero_points[head]) * scales[head]
                        )
                return tuple(result)

            common = dict(
                mode=AttentionMode.BATCH_MIXED_PAGED,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=qk_dim,
                head_dim_vo=vo_dim,
                kv_layout=layout,
                causal=True,
                q_dtype="float32",
                o_dtype="float32",
                sm_scale=0.5,
                logits_soft_cap=1.75,
            )
            dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
            quant_spec = AttentionPlanSpec(
                kv_dtype="uint8", kv_quant_spec=quant, **common
            )
            dense_cache = PagedKVCacheSpec(
                num_pages,
                page_size,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "float32",
                layout=layout,
                structure="separate",
                device="cpu",
            )
            quant_cache = PagedKVCacheSpec(
                num_pages,
                page_size,
                num_kv_heads,
                qk_dim,
                vo_dim,
                "uint8",
                layout=layout,
                structure="separate",
                device="cpu",
                quant_spec=quant,
            )
            key_shape, value_shape = dense_cache.expected_shapes
            dense_kv = ReferenceKVData(
                dense_cache,
                (
                    flat(
                        key_shape,
                        flatten_layout(
                            keys, scales=key_scales, zero_points=key_zero
                        ),
                    ),
                    flat(
                        value_shape,
                        flatten_layout(
                            values, scales=value_scales, zero_points=value_zero
                        ),
                    ),
                ),
            )
            quant_kv = ReferenceQuantizedKVData(
                quant_cache,
                ReferenceQuantizedTensor(
                    key_shape,
                    flat(key_shape, flatten_layout(keys), "uint8"),
                    flat((num_kv_heads,), key_scales),
                    quant,
                    flat((num_kv_heads,), key_zero, "int32"),
                ),
                ReferenceQuantizedTensor(
                    value_shape,
                    flat(value_shape, flatten_layout(values), "uint8"),
                    flat((num_kv_heads,), value_scales),
                    quant,
                    flat((num_kv_heads,), value_zero, "int32"),
                ),
            )
            q = flat(
                (4, num_qo_heads, qk_dim),
                (
                    rng.randint(-8, 8) / 4.0
                    for _ in range(4 * num_qo_heads * qk_dim)
                ),
            )
            runtime_k_scale = (0.75, 1.25)
            runtime_v_scale = (1.5, 0.5)
            dense_result = executor.execute(
                AttentionFrameworkSession(dense_spec.mode).plan(dense_spec, metadata),
                q,
                dense_kv,
                return_lse=True,
                k_scale=runtime_k_scale,
                v_scale=runtime_v_scale,
                logits_soft_cap=1.0,
            )
            quant_result = executor.execute(
                AttentionFrameworkSession(quant_spec.mode).plan(quant_spec, metadata),
                q,
                quant_kv,
                return_lse=True,
                k_scale=runtime_k_scale,
                v_scale=runtime_v_scale,
                logits_soft_cap=1.0,
            )
            with self.subTest(seed=seed, layout=layout.value):
                assert_tensor_close(self, dense_result.output, quant_result.output)
                assert_tensor_close(self, dense_result.lse, quant_result.lse)


if __name__ == "__main__":
    unittest.main()
