import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionLoweredOperatorCall,
    AttentionNvfp4PackedLayoutDescriptor,
    AttentionOperatorNvfp4PackedKVBinding,
    AttentionOperatorNvfp4PackedKVRunAdapter,
    AttentionOperatorNvfp4PackedKVRunAdapterFactory,
    AttentionOperatorNvfp4ScaleFactorBinding,
    AttentionOperatorOperationSpec,
    AttentionOperatorRunRequest,
    AttentionTensorAccessPolicy,
    attention_nvfp4_kv_quant_spec,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_096_nvfp4_scale_factor_contract import FakeInspector
from tests.test_checkpoint_097_nvfp4_operation_binding import (
    NVFP4_QUANT_SPEC,
    OPERATION_ID,
    active_plan,
)
from tests.test_checkpoint_099_nvfp4_packed_kv_view import scale, storage


def operation():
    return AttentionOperatorOperationSpec(
        operation_id=OPERATION_ID,
        provider_id="cann",
        package_name="test-nvfp4-provider",
        callable_path="test_nvfp4.attention",
        api_version="1.0.0",
        candidate_modes=(active_plan().framework_plan.spec.mode,),
        positional_arguments=("query", "kv_cache"),
        keyword_arguments=("kv_cache_sf", "k_sf", "v_sf"),
        return_names=("output",),
        quant_arguments=("kv_cache_sf", "k_sf", "v_sf"),
        source_url="https://example.invalid/test-nvfp4-provider",
    )


def packed_binding(*, combined=True, storage_alignment=16, scale_alignment=16):
    scale_binding = AttentionOperatorNvfp4ScaleFactorBinding(
        provider_id="cann",
        operation_id=OPERATION_ID,
        quant_spec=NVFP4_QUANT_SPEC,
        combined_argument="kv_cache_sf" if combined else None,
        key_argument=None if combined else "k_sf",
        value_argument=None if combined else "v_sf",
    )
    layout = AttentionNvfp4PackedLayoutDescriptor(
        physical_layout=NVFP4_QUANT_SPEC.physical_layout,
        packing_order=NVFP4_QUANT_SPEC.packing_order,
        storage_required_alignment=storage_alignment,
        scale_required_alignment=scale_alignment,
    )
    return AttentionOperatorNvfp4PackedKVBinding(scale_binding, layout)


class BaseAdapter:
    provider_id = "cann"

    def __init__(self, *, collision=None, input_view=None):
        self.calls = []
        self.collision = collision
        self.input_view = input_view

    def lower(self, active, request):
        if request.kv_cache_sf is not None:
            raise AssertionError("joint adapter must consume kv_cache_sf")
        self.calls.append(request)
        keyword = () if self.collision is None else ((self.collision, "existing"),)
        views = () if self.input_view is None else (self.input_view,)
        return AttentionLoweredOperatorCall(
            provider_id="cann",
            operation_id=OPERATION_ID,
            active_plan_fingerprint=active.fingerprint,
            positional_arguments=(
                ("query", request.query),
                ("kv_cache", request.kv_cache),
            ),
            keyword_arguments=keyword,
            validated_input_views=views,
            consumed_request_fields=request.consumed_fields,
        )


def run_request(active, kv_cache, scale_factors=None, *, out=None):
    return AttentionOperatorRunRequest.from_active_plan(
        active,
        "query",
        kv_cache,
        kv_cache_sf=scale_factors,
        out=out,
    )


def adapter(binding=None, *, base=None, inspector=None, policy=None):
    return AttentionOperatorNvfp4PackedKVRunAdapter(
        base or BaseAdapter(),
        operation(),
        binding or packed_binding(),
        inspector or FakeInspector(),
        policy or AttentionTensorAccessPolicy(required_alignment=16),
        "npu:0",
    )


class Nvfp4JointLoweringCheckpoint(unittest.TestCase):
    """Only one exact adapter may jointly consume packed KV and its scales."""

    def test_combined_input_is_validated_once_and_lowered_without_execution(self):
        active = active_plan()
        kv_cache = storage("combined-kv", (6, 2, 16, 2, 64))
        scales = scale("combined-sf", (6, 2, 16, 2, 8))
        base = BaseAdapter()

        lowered = adapter(base=base).lower(
            active, run_request(active, kv_cache, scales)
        )

        self.assertEqual(len(base.calls), 1)
        self.assertIs(base.calls[0].kv_cache, kv_cache)
        self.assertIsNone(base.calls[0].kv_cache_sf)
        self.assertIs(dict(lowered.keyword_arguments)["kv_cache_sf"], scales)
        self.assertEqual(
            tuple(name for name, _ in lowered.validated_input_views),
            ("kv.packed_storage", "kv_cache_sf"),
        )
        self.assertIn("kv_cache_sf", lowered.consumed_request_fields)

    def test_bound_plan_requires_scale_factor_before_base_lowering(self):
        active = active_plan()
        base = BaseAdapter()
        with self.assertRaisesRegex(SchemaError, "requires kv_cache_sf"):
            adapter(base=base).lower(
                active,
                run_request(
                    active, storage("combined-kv", (6, 2, 16, 2, 64))
                ),
            )
        self.assertEqual(base.calls, [])

    def test_unmatched_plan_delegates_only_when_scale_factor_is_absent(self):
        other_spec = attention_nvfp4_kv_quant_spec(
            physical_layout="other_nvfp4_layout_v1",
            packing_order="other_pair_order_v1",
        )
        active = active_plan(other_spec)
        base = BaseAdapter()
        request = run_request(active, "other-kv")

        lowered = adapter(base=base).lower(active, request)

        self.assertEqual(len(base.calls), 1)
        self.assertEqual(lowered.positional_arguments[-1], ("kv_cache", "other-kv"))
        inspector = FakeInspector()
        with self.assertRaisesRegex(SchemaError, "no exact NVFP4"):
            adapter(base=BaseAdapter(), inspector=inspector).lower(
                active,
                run_request(
                    active,
                    "other-kv",
                    scale("unbound-sf", (6, 2, 16, 2, 8)),
                ),
            )
        self.assertEqual(inspector.calls, [])

    def test_storage_structure_must_be_authorized_by_scale_binding(self):
        active = active_plan()
        with self.assertRaisesRegex(SchemaError, "not bound by operation"):
            adapter(packed_binding(combined=False)).lower(
                active,
                run_request(
                    active,
                    storage("combined-kv", (6, 2, 16, 2, 64)),
                    scale("combined-sf", (6, 2, 16, 2, 8)),
                ),
            )

    def test_runtime_alignment_alias_and_collisions_fail_closed(self):
        active = active_plan()
        kv_cache = storage(
            "combined-kv", (6, 2, 16, 2, 64), alignment=16
        )
        scales = scale("combined-sf", (6, 2, 16, 2, 8))
        with self.assertRaisesRegex(SchemaError, "32-byte aligned"):
            adapter(
                policy=AttentionTensorAccessPolicy(required_alignment=32)
            ).lower(active, run_request(active, kv_cache, scales))

        output = storage("combined-kv", (3, 8, 128))
        output.tensor_view = replace(output.tensor_view, writable=True)
        with self.assertRaisesRegex(SchemaError, "out cannot alias"):
            adapter().lower(
                active, run_request(active, kv_cache, scales, out=output)
            )

        with self.assertRaisesRegex(SchemaError, "argument collides"):
            adapter(base=BaseAdapter(collision="kv_cache_sf")).lower(
                active, run_request(active, kv_cache, scales)
            )
        with self.assertRaisesRegex(SchemaError, "view name collides"):
            adapter(
                base=BaseAdapter(
                    input_view=("kv_cache_sf", scales.tensor_view)
                )
            ).lower(active, run_request(active, kv_cache, scales))

    def test_binding_identity_and_factory_are_auditable(self):
        binding = packed_binding()
        changed = packed_binding(storage_alignment=32)
        self.assertNotEqual(binding.fingerprint, changed.fingerprint)
        self.assertEqual(binding.provider_id, "cann")
        self.assertEqual(binding.operation_id, OPERATION_ID)

        factory = AttentionOperatorNvfp4PackedKVRunAdapterFactory(
            operation(),
            binding,
            FakeInspector(),
            AttentionTensorAccessPolicy(required_alignment=16),
        )
        built = factory.build(BaseAdapter(), "npu:0")
        self.assertIsInstance(built, AttentionOperatorNvfp4PackedKVRunAdapter)

        wrong_layout = AttentionNvfp4PackedLayoutDescriptor(
            physical_layout="different-layout",
            packing_order=NVFP4_QUANT_SPEC.packing_order,
        )
        with self.assertRaisesRegex(SchemaError, "differs from QuantSpec"):
            AttentionOperatorNvfp4PackedKVBinding(
                binding.scale_factor_binding, wrong_layout
            )


if __name__ == "__main__":
    unittest.main()
