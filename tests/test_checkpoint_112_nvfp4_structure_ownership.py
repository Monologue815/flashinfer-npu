import unittest
from dataclasses import replace

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


class Nvfp4StructureOwnershipCheckpoint(unittest.TestCase):
    def cases(self, structure, collision=None):
        active = active_plan()
        original = packed_binding()
        binding = replace(
            original,
            scale_factor_binding=replace(
                original.scale_factor_binding,
                key_argument="k_sf",
                value_argument="v_sf",
            ),
        )
        if structure == "combined":
            cache = storage("combined", (6, 2, 16, 2, 64))
            scales = scale("combined-sf", (6, 2, 16, 2, 8))
        else:
            cache = tuple(storage(name, (6, 16, 2, 64))
                          for name in ("key", "value"))
            scales = tuple(scale(name, (6, 16, 2, 8))
                           for name in ("key-sf", "value-sf"))
        yield (
            "packed",
            packed_adapter(binding, base=PackedBase(collision=collision)),
            active,
            run_request(active, cache, scales),
        )
        yield (
            "scale",
            scale_adapter(binding.scale_factor_binding,
                          base=ScaleBase(collision=collision)),
            active,
            scale_request(active, scales),
        )

    def test_base_cannot_supply_the_other_public_structure(self):
        for structure, argument in (("combined", "k_sf"),
                                    ("combined", "v_sf"),
                                    ("separate", "kv_cache_sf")):
            for kind, adapter, active, request in self.cases(structure, argument):
                with self.subTest(kind=kind, structure=structure, argument=argument):
                    with self.assertRaisesRegex(SchemaError, "collides"):
                        adapter.lower(active, request)

    def test_dual_structure_binding_injects_only_the_selected_structure(self):
        for structure in ("combined", "separate"):
            for kind, adapter, active, request in self.cases(structure):
                with self.subTest(kind=kind, structure=structure):
                    lowered = adapter.lower(active, request)
                    arguments = dict(lowered.keyword_arguments)
                    if structure == "combined":
                        self.assertEqual(set(arguments), {"kv_cache_sf"})
                        self.assertIs(arguments["kv_cache_sf"], request.kv_cache_sf)
                    else:
                        self.assertEqual(set(arguments), {"k_sf", "v_sf"})
                        self.assertIs(arguments["k_sf"], request.kv_cache_sf[0])
                        self.assertIs(arguments["v_sf"], request.kv_cache_sf[1])


if __name__ == "__main__":
    unittest.main()
