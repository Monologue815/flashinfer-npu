import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionDispatchReceipt,
    AttentionLoweredOperatorCall,
    AttentionOperatorActivePlan,
    AttentionOperatorNvfp4ScaleFactorBinding,
    AttentionOperatorNvfp4ScaleFactorRunAdapter,
    AttentionOperatorOperationSpec,
    AttentionOperatorProviderSelection,
    AttentionOperatorRunRequest,
    AttentionPreparedOperatorPlan,
    AttentionTensorAccessPolicy,
    AttentionMode,
    attention_nvfp4_kv_quant_spec,
)
from flashinfer_npu.runtime import Backend, SchemaError
from tests.test_checkpoint_096_nvfp4_scale_factor_contract import (
    FakeInspector,
    FakeTensor,
    paged_plan,
)


OPERATION_ID = "test.nvfp4.attention@v1"
NVFP4_QUANT_SPEC = attention_nvfp4_kv_quant_spec(
    physical_layout="test_nvfp4_e2m1_packed",
    packing_order="test_low_nibble_first",
)


def hash_value(character):
    return character * 64


def operation(*, quant_arguments=("kv_cache_sf", "k_sf", "v_sf")):
    return AttentionOperatorOperationSpec(
        operation_id=OPERATION_ID,
        provider_id="cann",
        package_name="test-nvfp4-provider",
        callable_path="test_nvfp4.attention",
        api_version="1.0.0",
        candidate_modes=(AttentionMode.BATCH_PREFILL_PAGED,),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=("kv_cache_sf", "k_sf", "v_sf"),
        return_names=("output",),
        quant_arguments=quant_arguments,
        source_url="https://example.invalid/test-nvfp4-provider",
    )


def active_plan(quant_spec=NVFP4_QUANT_SPEC):
    plan = paged_plan(quant_spec=quant_spec)
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="test.nvfp4.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="test_nvfp4_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="test-nvfp4-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="test-nvfp4-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=Backend.ACLNN,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )
    selection = AttentionOperatorProviderSelection(
        provider_id="cann",
        provider_probe_fingerprint=hash_value("9"),
        provider_record_fingerprint=hash_value("a"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    prepared = AttentionPreparedOperatorPlan(
        provider_id="cann",
        provider_selection_fingerprint=selection.fingerprint,
        framework_plan_fingerprint=plan.fingerprint,
        framework_plan_generation=plan.generation,
        implementation_id=OPERATION_ID,
        opaque_plan_token="test-nvfp4-plan-v1",
        opaque_state=None,
    )
    return AttentionOperatorActivePlan(plan, receipt, selection, prepared)


class BaseAdapter:
    provider_id = "cann"

    def __init__(self, *, collision=None):
        self.calls = 0
        self.collision = collision

    def lower(self, active, request):
        self.calls += 1
        if request.kv_cache_sf is not None:
            raise AssertionError("outer NVFP4 adapter must consume kv_cache_sf")
        keyword = () if self.collision is None else ((self.collision, "existing"),)
        return AttentionLoweredOperatorCall(
            provider_id="cann",
            operation_id=OPERATION_ID,
            active_plan_fingerprint=active.fingerprint,
            positional_arguments=(("query", request.query), ("key", request.kv_cache[0]), ("value", request.kv_cache[1])),
            keyword_arguments=keyword,
            consumed_request_fields=request.consumed_fields,
        )


def request(active, scale_factors, *, out=None):
    return AttentionOperatorRunRequest.from_active_plan(
        active,
        "query",
        ("key", "value"),
        kv_cache_sf=scale_factors,
        out=out,
    )


def adapter(binding, *, base=None, policy=None):
    return AttentionOperatorNvfp4ScaleFactorRunAdapter(
        base or BaseAdapter(),
        operation(),
        binding,
        FakeInspector(),
        policy or AttentionTensorAccessPolicy(required_alignment=16),
        "npu:0",
    )


class Nvfp4OperationBindingCheckpoint(unittest.TestCase):
    """NVFP4 metadata enters lowering only through exact catalog arguments."""

    def test_binding_requires_exact_keyword_quant_arguments(self):
        binding = AttentionOperatorNvfp4ScaleFactorBinding(
            provider_id="cann", operation_id=OPERATION_ID,
            quant_spec=NVFP4_QUANT_SPEC,
            combined_argument="kv_cache_sf",
        )
        binding.validate_operation(operation())
        self.assertEqual(binding.accepted_structures, ("combined",))
        self.assertEqual(len(binding.fingerprint), 64)

        with self.assertRaisesRegex(SchemaError, "non-quant"):
            binding.validate_operation(operation(quant_arguments=("k_sf", "v_sf")))
        with self.assertRaisesRegex(SchemaError, "requires K and V"):
            AttentionOperatorNvfp4ScaleFactorBinding(
                provider_id="cann", operation_id=OPERATION_ID,
                quant_spec=NVFP4_QUANT_SPEC,
                key_argument="k_sf",
            )

    def test_combined_scale_factor_is_injected_and_carried_as_input_evidence(self):
        active = active_plan()
        scale = FakeTensor("combined", (6, 2, 16, 2, 8))
        lowered = adapter(AttentionOperatorNvfp4ScaleFactorBinding(
            provider_id="cann", operation_id=OPERATION_ID,
            quant_spec=NVFP4_QUANT_SPEC,
            combined_argument="kv_cache_sf",
        )).lower(active, request(active, scale))

        self.assertIs(dict(lowered.keyword_arguments)["kv_cache_sf"], scale)
        self.assertEqual(
            tuple(name for name, _ in lowered.validated_input_views),
            ("kv_cache_sf",),
        )
        self.assertIn("kv_cache_sf", lowered.consumed_request_fields)

    def test_separate_scale_factors_map_to_distinct_provider_arguments(self):
        active = active_plan()
        scales = (
            FakeTensor("key-sf", (6, 16, 2, 8)),
            FakeTensor("value-sf", (6, 16, 2, 8)),
        )
        lowered = adapter(AttentionOperatorNvfp4ScaleFactorBinding(
            provider_id="cann", operation_id=OPERATION_ID,
            quant_spec=NVFP4_QUANT_SPEC,
            key_argument="k_sf", value_argument="v_sf",
        )).lower(active, request(active, scales))

        arguments = dict(lowered.keyword_arguments)
        self.assertIs(arguments["k_sf"], scales[0])
        self.assertIs(arguments["v_sf"], scales[1])
        self.assertEqual(
            tuple(name for name, _ in lowered.validated_input_views),
            ("kv_cache_sf.key", "kv_cache_sf.value"),
        )

    def test_unbound_structure_fails_before_base_lowering(self):
        active = active_plan()
        base = BaseAdapter()
        wrapped = adapter(
            AttentionOperatorNvfp4ScaleFactorBinding(
                provider_id="cann", operation_id=OPERATION_ID,
                quant_spec=NVFP4_QUANT_SPEC,
                combined_argument="kv_cache_sf",
            ),
            base=base,
        )
        scales = (
            FakeTensor("key-sf", (6, 16, 2, 8)),
            FakeTensor("value-sf", (6, 16, 2, 8)),
        )

        with self.assertRaisesRegex(SchemaError, "not bound"):
            wrapped.lower(active, request(active, scales))
        self.assertEqual(base.calls, 0)

    def test_alignment_alias_and_argument_collision_fail_closed(self):
        active = active_plan()
        binding = AttentionOperatorNvfp4ScaleFactorBinding(
            provider_id="cann", operation_id=OPERATION_ID,
            quant_spec=NVFP4_QUANT_SPEC,
            combined_argument="kv_cache_sf",
        )
        misaligned = FakeTensor("misaligned", (6, 2, 16, 2, 8))
        misaligned.tensor_view = replace(misaligned.tensor_view, data_ptr_alignment=8)
        with self.assertRaisesRegex(SchemaError, "16-byte aligned"):
            adapter(binding).lower(active, request(active, misaligned))

        scale = FakeTensor("shared", (6, 2, 16, 2, 8))
        out = FakeTensor("output", (3, 8, 128))
        out.tensor_view = replace(out.tensor_view, storage_id="shared", writable=True)
        with self.assertRaisesRegex(SchemaError, "out cannot alias"):
            adapter(binding).lower(active, request(active, scale, out=out))

        with self.assertRaisesRegex(SchemaError, "collides"):
            adapter(binding, base=BaseAdapter(collision="kv_cache_sf")).lower(
                active, request(active, scale)
            )


if __name__ == "__main__":
    unittest.main()
