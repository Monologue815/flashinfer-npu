import unittest
from dataclasses import replace

from flashinfer_npu.attention import lower_attention_operator_run
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_097_nvfp4_operation_binding import (
    BaseAdapter as ScaleBase,
    adapter as scale_adapter,
    request as scale_request,
)
from tests.test_checkpoint_100_nvfp4_joint_lowering import (
    BaseAdapter as PackedBase,
    active_plan,
    adapter as packed_adapter,
    packed_binding,
    run_request,
    scale,
    storage,
)


class DelegatingBase:
    provider_id = "cann"

    def __init__(self, base, omit=None):
        self.base = base
        self.omit = omit
        self.requests = []

    def lower(self, active, request):
        self.requests.append(request)
        lowered = self.base.lower(active, request)
        return replace(
            lowered,
            keyword_arguments=lowered.keyword_arguments
            + (("global_k", request.k_scale), ("global_v", request.v_scale)),
            consumed_request_fields=tuple(
                name for name in lowered.consumed_request_fields
                if name != self.omit
            ),
        )


class Nvfp4DelegatedConsumptionCheckpoint(unittest.TestCase):
    def cases(self, omit=None):
        active = active_plan()
        scales = scale("combined-sf", (6, 2, 16, 2, 8))
        binding = packed_binding()
        for kind in ("packed", "scale"):
            base = DelegatingBase(
                PackedBase() if kind == "packed" else ScaleBase(), omit
            )
            if kind == "packed":
                wrapped = packed_adapter(base=base)
                request = run_request(
                    active, storage("combined-kv", (6, 2, 16, 2, 64)), scales
                )
            else:
                wrapped = scale_adapter(binding.scale_factor_binding, base=base)
                request = scale_request(active, scales)
            yield kind, active, wrapped, base, replace(
                request, k_scale=2.0, v_scale=3.0
            ), scales

    def test_outer_adapter_rejects_missing_delegated_field_receipts(self):
        for missing in ("query", "kv_cache", "return_lse", "logits_soft_cap",
                        "k_scale", "v_scale"):
            for kind, active, wrapped, base, request, _ in self.cases(missing):
                with self.subTest(kind=kind, missing=missing):
                    with self.assertRaisesRegex(SchemaError, "every run field"):
                        lower_attention_operator_run(wrapped, active, request)
                    self.assertEqual(len(base.requests), 1)

    def test_global_scales_remain_separate_from_block_scales(self):
        for kind, active, wrapped, base, request, scales in self.cases():
            with self.subTest(kind=kind):
                lowered = lower_attention_operator_run(wrapped, active, request)
                arguments = dict(lowered.keyword_arguments)
                self.assertEqual(arguments["global_k"], 2.0)
                self.assertEqual(arguments["global_v"], 3.0)
                self.assertIs(arguments["kv_cache_sf"], scales)
                self.assertIsNone(base.requests[0].kv_cache_sf)
                self.assertEqual(lowered.consumed_request_fields,
                                 request.consumed_fields)


if __name__ == "__main__":
    unittest.main()
