import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionFrameworkSession,
    AttentionOperatorOperationCatalog,
    AttentionTraceCorpus,
    AttentionTensorAccessPolicy,
    PagedKVCacheSpec,
    build_framework_attention_corpus,
    framework_attention_coverage_policy,
    inspect_attention_operator_dense_kv_input,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import fake_operation
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    metadata_tensor,
    query_input,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import (
    BufferPackageLoader,
    buffer_tensor,
)


def dense_plan():
    corpus = build_framework_attention_corpus()
    case = next(
        item for item in corpus.cases if item.case_id == "mixed_dense_distinct_dims"
    )
    return AttentionFrameworkSession(case.trace.spec.mode).plan(
        case.trace.spec, case.trace.metadata
    )


def dense_kv(plan, *, key=None, value=None):
    metadata = plan.metadata
    num_pages = metadata.max_page_index + 1
    cache_spec = PagedKVCacheSpec(
        num_pages=num_pages,
        page_size=metadata.page_size,
        num_kv_heads=plan.spec.num_kv_heads,
        head_dim_qk=plan.spec.head_dim_qk,
        head_dim_vo=plan.spec.head_dim_vo,
        dtype=plan.spec.kv_dtype,
        layout=plan.spec.kv_layout,
        structure="separate",
        device="npu:0",
    )
    key_shape, value_shape = cache_spec.expected_shapes
    return (
        metadata_tensor("dense-key", key_shape, plan.spec.kv_dtype)
        if key is None
        else key,
        metadata_tensor("dense-value", value_shape, plan.spec.kv_dtype)
        if value is None
        else value,
    )


def dense_runtime(*, access_policy=None, caller_buffers=False):
    values = bootstrap_components()
    plan = dense_plan()
    corpus = build_framework_attention_corpus()
    case = next(
        item for item in corpus.cases if item.case_id == "mixed_dense_distinct_dims"
    )
    policy = framework_attention_coverage_policy()
    subset = AttentionTraceCorpus(
        "checkpoint-057-dense-subset",
        (case,),
        "one exact unquantized provider KV contract case",
    )
    coverage = policy.evaluate(subset)
    profile = values["spec"].profiles[0]
    rule = replace(
        profile.rules[0],
        rule_id="mixed_dense_v1",
        modes=(plan.spec.mode,),
        kv_layouts=(plan.spec.kv_layout,),
        dtype_signatures=((plan.spec.q_dtype, plan.spec.kv_dtype, plan.spec.o_dtype),),
        supports_dense_kv=True,
        quant_specs=(),
        causal_values=(plan.spec.effective_causal,),
        required_features=(),
    )
    evidence = replace(
        profile.evidence[0],
        evidence_id="synthetic-dense-provider-run-v1",
        corpus_fingerprint=corpus.fingerprint,
        coverage_policy_name=policy.name,
        covered_cells=coverage.covered_cells,
        total_cells=len(coverage.requirements),
        passed_case_ids=(case.case_id,),
    )
    profile = replace(profile, rules=(rule,), evidence=(evidence,))
    descriptor = values["spec"].descriptors[0]
    descriptor = replace(
        descriptor,
        op="attention.%s" % plan.spec.mode.value,
        constraints=replace(
            descriptor.constraints,
            dtype_signatures=rule.dtype_signatures,
            layout_signatures=((plan.spec.kv_layout.value,),),
            required_features=(),
            quant_storage_dtypes=(),
        ),
        capability_binding=replace(
            descriptor.capability_binding,
            rule_id=rule.rule_id,
            profile_fingerprint=profile.fingerprint,
        ),
    )
    operation = fake_operation()
    loader = values["loader"]
    if caller_buffers:
        operation = replace(
            operation,
            callable_path="checkpoint_056_package.attention",
            keyword_arguments=(
                "table",
                "scale",
                "return_softmax_lse",
                "key_scale",
                "value_scale",
                "out",
                "lse",
            ),
            mutable_arguments=("out", "lse"),
            quant_arguments=("key_scale", "value_scale"),
            output_buffer_argument="out",
            lse_buffer_argument="lse",
        )
        loader = BufferPackageLoader(values["events"])
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-057-dense-kv-catalog", operations=(operation,)
    )
    runtime_spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(descriptor,),
        quantization_bindings=(),
        tensor_access_policy=(
            AttentionTensorAccessPolicy()
            if access_policy is None
            else access_policy
        ),
    )
    values.update(
        operation=operation,
        catalog=catalog,
        loader=loader,
        spec=runtime_spec,
    )
    resolved_plan, session = active_session(
        values,
        spec=runtime_spec,
        catalog=catalog,
        plan=plan,
    )
    return values, resolved_plan, session


class ProviderDenseKVContractTests(unittest.TestCase):
    """Unquantized provider K/V metadata is plan-bound before lowering."""

    def test_valid_separate_kv_is_inspected_and_forwarded_without_copy(self):
        values, plan, session = dense_runtime()
        key, value = dense_kv(plan)

        lowered = session.run(query_input(plan), (key, value))

        positional = dict(lowered.positional_arguments)
        self.assertIs(positional["key"], key)
        self.assertIs(positional["value"], value)
        self.assertEqual(
            tuple(name for _, name, _ in values["tensor_metadata_inspector"].calls),
            ("query", "kv.key", "kv.value"),
        )

    def test_kv_shape_dtype_and_device_are_fail_closed(self):
        _, plan, session = dense_runtime()
        key, value = dense_kv(plan)
        cases = (
            (
                metadata_tensor(
                    "bad-key-shape",
                    key.tensor_view.shape[:-1] + (plan.spec.head_dim_qk + 1,),
                    plan.spec.kv_dtype,
                ),
                value,
                "K/V view shapes",
            ),
            (
                key,
                metadata_tensor("bad-value-dtype", value.tensor_view.shape, "float16"),
                "K/V view dtype",
            ),
            (
                metadata_tensor(
                    "bad-key-device",
                    key.tensor_view.shape,
                    plan.spec.kv_dtype,
                    device="npu:1",
                ),
                value,
                "share device",
            ),
        )
        for bad_key, bad_value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    session.run(query_input(plan), (bad_key, bad_value))

    def test_kv_alignment_contiguity_and_separate_alias_are_enforced(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_kv=True,
            required_alignment=16,
        )
        _, plan, session = dense_runtime(access_policy=policy)
        key, value = dense_kv(plan)
        key = type(key)(replace(key.tensor_view, data_ptr_alignment=16))
        value = type(value)(replace(value.tensor_view, data_ptr_alignment=16))
        non_contiguous = type(key)(
            replace(
                key.tensor_view,
                strides=(32, 16, 8, 2),
                storage_nbytes=256,
            )
        )
        under_aligned = type(value)(
            replace(value.tensor_view, data_ptr_alignment=8)
        )
        aliased_value = type(value)(
            replace(value.tensor_view, storage_id=key.tensor_view.storage_id)
        )
        query = query_input(plan)
        query = type(query)(
            replace(query.tensor_view, data_ptr_alignment=16)
        )

        with self.assertRaisesRegex(SchemaError, "kv.key_storage must be contiguous"):
            session.run(query, (non_contiguous, value))
        with self.assertRaisesRegex(SchemaError, "kv.value_storage must be 16-byte aligned"):
            session.run(query, (key, under_aligned))
        with self.assertRaisesRegex(SchemaError, "K and V cannot alias"):
            session.run(query, (key, aliased_value))

    def test_output_cannot_alias_dense_kv_by_default(self):
        _, plan, session = dense_runtime(caller_buffers=True)
        key, value = dense_kv(plan)
        out = buffer_tensor(
            "aliased-out",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=key.tensor_view.storage_id,
        )

        with self.assertRaisesRegex(SchemaError, "out cannot alias kv.key_storage"):
            session.run(query_input(plan), (key, value), out=out)

    def test_packed_paged_kv_metadata_is_supported_before_provider_policy(self):
        plan = dense_plan()
        equal_dims = replace(plan.spec, head_dim_vo=plan.spec.head_dim_qk)
        plan = AttentionFrameworkSession(equal_dims.mode).plan(
            equal_dims, plan.metadata
        )
        num_pages = plan.metadata.max_page_index + 1
        cache_spec = PagedKVCacheSpec(
            num_pages=num_pages,
            page_size=plan.metadata.page_size,
            num_kv_heads=plan.spec.num_kv_heads,
            head_dim_qk=plan.spec.head_dim_qk,
            head_dim_vo=plan.spec.head_dim_vo,
            dtype=plan.spec.kv_dtype,
            layout=plan.spec.kv_layout,
            structure="packed",
            device="npu:0",
        )
        packed = metadata_tensor(
            "packed-kv", cache_spec.expected_shapes[0], plan.spec.kv_dtype
        )
        values = bootstrap_components()

        view = inspect_attention_operator_dense_kv_input(
            plan,
            packed,
            values["tensor_metadata_inspector"],
            "npu:0",
        )

        self.assertTrue(view.packed)
        self.assertIs(view.key, view.value)


if __name__ == "__main__":
    unittest.main()
