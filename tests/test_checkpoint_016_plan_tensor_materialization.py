import unittest

from flashinfer_npu.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionMaterializedOperatorPlanState,
    AttentionMaterializingPlanFactory,
    AttentionMaterializingRunAdapter,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorTensorPlan,
    AttentionOperatorWrapperSession,
    AttentionPlanSpec,
    AttentionMode,
    CannV2PagedPlanFactory,
    CannV2PagedRunAdapter,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedRunAdapter,
    KVLayout,
    MixedPagedKVMetadata,
    bind_attention_operator_callable,
    load_packaged_attention_operator_catalog,
)
from flashinfer_npu.runtime import Backend, SchemaError


def hash_value(character):
    return character * 64


def framework_plan(provider):
    layout = KVLayout.HND if provider == "cann" else KVLayout.NHD
    page_size = 128 if provider == "cann" else 16
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=layout,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        o_dtype="bfloat16",
    )
    metadata = MixedPagedKVMetadata(
        qo_indptr=(0, 2, 3),
        kv_indptr=(0, 2, 3),
        kv_indices=(7, 3, 9),
        kv_len_arr=(page_size + page_size // 2, page_size),
        page_size=page_size,
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def authority(plan, provider, operation_id, backend):
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.materialization.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_materialization_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-materialization-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-materialization-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=backend,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )
    operation = load_packaged_attention_operator_catalog().get(operation_id)
    probe = AttentionOperatorProviderProbe(
        provider_id=provider,
        adapter_version="checkpoint-016-test",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    selection = AttentionOperatorProviderSelection(
        provider_id=provider,
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=hash_value("9"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    observed = AttentionObservedOperatorCallable(
        provider_id=provider,
        package_name=operation.package_name,
        package_version="test-package",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=AttentionObservedCallableSignature(
            operation.positional_arguments,
            operation.keyword_arguments,
            observation_kind="checkpoint-016-test",
        ),
    )
    binding = bind_attention_operator_callable(probe, operation, observed)
    return receipt, selection, binding


class FakeTensor:
    def __init__(self, role, shape, dtype, device, materialization):
        self.role = role
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.materialization = materialization


class FakeMaterializer:
    def __init__(self, provider_id, fail_role=None, return_recipe=False):
        self.provider_id = provider_id
        self.materializer_id = "fake-%s-tensor-materializer-v1" % provider_id
        self.fail_role = fail_role
        self.return_recipe = return_recipe
        self.calls = []

    def materialize(self, tensor_plan, device):
        self.calls.append((tensor_plan, device))
        if tensor_plan.role == self.fail_role:
            raise RuntimeError("fake tensor materialization failure")
        if self.return_recipe:
            return tensor_plan
        return FakeTensor(
            tensor_plan.role,
            tensor_plan.shape,
            tensor_plan.dtype,
            device,
            tensor_plan.materialization,
        )


def components(provider, materializer):
    plan = framework_plan(provider)
    if provider == "cann":
        logical_factory = CannV2PagedPlanFactory()
        logical_adapter = CannV2PagedRunAdapter()
        operation_id = CANN_V2_OPERATION_ID
        backend = Backend.ACLNN
    else:
        logical_factory = FlashAttentionNpuV3PagedPlanFactory()
        logical_adapter = FlashAttentionNpuV3PagedRunAdapter()
        operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID
        backend = Backend.ASCENDC_AOT
    receipt, selection, binding = authority(
        plan, provider, operation_id, backend
    )
    factory = AttentionMaterializingPlanFactory(
        logical_factory, materializer, "npu:0"
    )
    adapter = AttentionMaterializingRunAdapter(logical_adapter)
    return plan, receipt, selection, binding, factory, adapter


def planned_wrapper(provider, materializer):
    plan, receipt, selection, binding, factory, adapter = components(
        provider, materializer
    )
    wrapper = AttentionOperatorWrapperSession(
        load_packaged_attention_operator_catalog()
    )
    wrapper.plan(factory, adapter, plan, receipt, selection, binding)
    return wrapper


def contains_recipe(value):
    if isinstance(value, AttentionOperatorTensorPlan):
        return True
    if isinstance(value, (tuple, list)):
        return any(contains_recipe(item) for item in value)
    return False


class PlanTensorMaterializationCheckpoint(unittest.TestCase):
    """Checkpoint 016: plan recipes become provider tensors once, before run."""

    def test_cann_page_table_and_causal_mask_materialize_once_and_reuse(self):
        materializer = FakeMaterializer("cann")
        wrapper = planned_wrapper("cann", materializer)
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertIsInstance(state, AttentionMaterializedOperatorPlanState)
        self.assertEqual(
            tuple(item.role for item in state.tensors),
            ("atten_mask", "block_table"),
        )
        self.assertEqual(len(materializer.calls), 2)

        first = wrapper.run("query-1", ("key-1", "value-1"))
        second = wrapper.run("query-2", ("key-2", "value-2"))
        self.assertEqual(len(materializer.calls), 2)
        first_keywords = dict(first.keyword_arguments)
        second_keywords = dict(second.keyword_arguments)
        self.assertIsInstance(first_keywords["atten_mask"], FakeTensor)
        self.assertIsInstance(first_keywords["block_table"], FakeTensor)
        self.assertIs(
            first_keywords["block_table"], second_keywords["block_table"]
        )
        self.assertFalse(contains_recipe(first.keyword_arguments))

    def test_flash_query_offsets_cache_lengths_and_page_table_materialize(self):
        materializer = FakeMaterializer("flash_attention_npu")
        wrapper = planned_wrapper("flash_attention_npu", materializer)
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertEqual(
            tuple(item.role for item in state.tensors),
            ("cache_seqlens", "cu_seqlens_q", "page_table"),
        )
        lowered = wrapper.run("query", ("key", "value"))
        keywords = dict(lowered.keyword_arguments)
        for name in ("cache_seqlens", "cu_seqlens_q", "page_table"):
            self.assertIsInstance(keywords[name], FakeTensor)
            self.assertEqual(keywords[name].device, "npu:0")
        self.assertFalse(contains_recipe(lowered.keyword_arguments))

    def test_failed_materializing_replan_preserves_previous_runtime(self):
        good = FakeMaterializer("cann")
        wrapper = planned_wrapper("cann", good)
        active = wrapper.active_plan
        runtime = wrapper.runtime_binding

        failing = FakeMaterializer("cann", fail_role="block_table")
        plan, receipt, selection, binding, factory, adapter = components(
            "cann", failing
        )
        with self.assertRaisesRegex(RuntimeError, "materialization failure"):
            wrapper.plan(factory, adapter, plan, receipt, selection, binding)
        self.assertIs(wrapper.active_plan, active)
        self.assertIs(wrapper.runtime_binding, runtime)
        lowered = wrapper.run("query", ("key", "value"))
        self.assertIsInstance(dict(lowered.keyword_arguments)["block_table"], FakeTensor)

    def test_foreign_or_fake_recipe_materializer_output_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "providers differ"):
            AttentionMaterializingPlanFactory(
                CannV2PagedPlanFactory(),
                FakeMaterializer("flash_attention_npu"),
                "npu:0",
            )

        materializer = FakeMaterializer("cann", return_recipe=True)
        plan, receipt, selection, _, factory, _ = components("cann", materializer)
        with self.assertRaisesRegex(SchemaError, "opaque provider tensor"):
            factory.prepare(plan, receipt, selection)


if __name__ == "__main__":
    unittest.main()
