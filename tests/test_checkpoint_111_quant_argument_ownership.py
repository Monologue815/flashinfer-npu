import unittest
from dataclasses import replace
from unittest.mock import patch

from flashinfer_npu.attention import (
    attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_public_quantized_paged_decode import (
    FakeNpuWorkspace,
    metadata_tensor,
    package_attention,
    plan_public_wrapper,
    public_paged_decode_runtime,
    query_tensor,
)


class QuantArgumentOwnershipCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        self.values, self.case, registry = public_paged_decode_runtime()
        install_attention_operator_runtime_resolvers(
            registry, operation_catalog=self.values["catalog"]
        )
        package_attention.calls[:] = []
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout=self.case.trace.spec.kv_layout.value
        )
        plan_public_wrapper(self.wrapper, self.case)
        shape = self.case.trace.kv_data.key_data.logical_shape
        self.cache = tuple(
            metadata_tensor(name, shape, "float8_e4m3fn")
            for name in ("key", "value")
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def test_base_cannot_override_omitted_quant_arguments(self):
        base = self.values["spec"].logical_run_adapter
        original_lower = base.lower
        active = self.wrapper.plan_state
        for argument in ("key_scale", "value_scale", "runtime_key_scale",
                         "runtime_value_scale", "runtime_query_scale"):
            with self.subTest(argument=argument):
                def conflicting_lower(plan, request):
                    lowered = original_lower(plan, request)
                    return replace(
                        lowered,
                        keyword_arguments=lowered.keyword_arguments
                        + ((argument, 2.0),),
                    )

                with patch.object(base, "lower", conflicting_lower):
                    with self.assertRaisesRegex(SchemaError, "collides"):
                        self.wrapper.run(query_tensor(self.case), self.cache)
                self.assertEqual(package_attention.calls, [])
                self.assertIs(self.wrapper.plan_state, active)

        self.assertEqual(
            self.wrapper.run(query_tensor(self.case, "reused"), self.cache),
            "package-output:reused",
        )
        self.assertEqual(package_attention.calls[0][6:11], (None,) * 5)


if __name__ == "__main__":
    unittest.main()
