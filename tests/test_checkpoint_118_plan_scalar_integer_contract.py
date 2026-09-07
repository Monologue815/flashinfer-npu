import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionMode, AttentionPlanSpec, MixedPagedKVMetadata, PagedKVMetadata,
    SingleAttentionMetadata, attention_operator_runtime_registry_snapshot,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.schema import attention_metadata_from_dict
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import build_components
from tests.test_checkpoint_043_provider_workspace_reset import (
    FakeNpuWorkspace, plan_wrapper, runtime_registry,
)
from tests.test_checkpoint_116_metadata_integer_contract import IndexValue, IntOnly


class PlanScalarIntegerContractCheckpoint(unittest.TestCase):
    def fixtures(self):
        return (
            (AttentionPlanSpec(AttentionMode.BATCH_DECODE_PAGED, 8, 2, 128),
             ("num_qo_heads", "num_kv_heads", "head_dim_qk", "head_dim_vo",
              "window_left", "window_right", "q_len_per_req")),
            (PagedKVMetadata((0, 1), (0,), (1,), 8), ("page_size",)),
            (MixedPagedKVMetadata((0, 1), (0, 1), (0,), (1,), 8), ("page_size",)),
            (SingleAttentionMetadata(1, 8), ("qo_len", "kv_len")),
        )

    def test_plan_scalars_reject_non_integer_types(self):
        for fixture, fields in self.fixtures():
            for field in fields:
                for value in (True, float(getattr(fixture, field)), 1.5, "1",
                              float("inf"), float("nan"), IntOnly()):
                    with self.subTest(schema=type(fixture).__name__, field=field, value=value):
                        with self.assertRaisesRegex(SchemaError, field):
                            replace(fixture, **{field: value})

    def test_index_objects_preserve_fingerprint_and_defaults(self):
        for fixture, fields in self.fixtures():
            candidate = replace(fixture, **{
                name: IndexValue(getattr(fixture, name)) for name in fields
            })
            self.assertEqual(candidate, fixture)
            self.assertEqual(candidate.fingerprint, fixture.fingerprint)
            for name in fields:
                self.assertIs(type(getattr(candidate, name)), int)
        spec = AttentionPlanSpec(AttentionMode.SINGLE_PREFILL, 8, 2, IndexValue(128))
        self.assertEqual(spec.head_dim_vo, 128)
        self.assertAlmostEqual(spec.sm_scale, 128 ** -0.5)

    def test_serialization_cannot_bypass_scalar_types(self):
        for fixture, fields in self.fixtures():
            decode = (AttentionPlanSpec.from_dict if isinstance(fixture, AttentionPlanSpec)
                      else attention_metadata_from_dict)
            self.assertEqual(decode(fixture.to_dict()), fixture)
            for field in fields:
                with self.subTest(schema=type(fixture).__name__, field=field):
                    payload = fixture.to_dict()
                    payload[field] = float(payload[field])
                    with self.assertRaisesRegex(SchemaError, field):
                        decode(payload)

    def test_invalid_public_replan_does_not_reach_provider(self):
        original = attention_operator_runtime_registry_snapshot()
        components = build_components()
        try:
            install_attention_operator_runtime_resolvers(
                runtime_registry(components), operation_catalog=components["catalog"]
            )
            wrapper = BatchDecodeWithPagedKVCacheWrapper(FakeNpuWorkspace(), kv_layout="HND")
            plan_wrapper(wrapper)
            plan = wrapper.plan_state
            workspace = wrapper.workspace_contract
            events = list(components["events"])
            valid = dict(indptr=[0, 1], indices=[7], last_page_len=[64],
                         num_qo_heads=8, num_kv_heads=2, head_dim=128, page_size=128,
                         q_len_per_req=1, window_left=-1, window_right=0,
                         q_data_type="bfloat16", kv_data_type="bfloat16")
            for field in ("num_qo_heads", "num_kv_heads", "head_dim", "page_size",
                          "q_len_per_req", "window_left", "window_right"):
                for value in (float(valid[field]), "1", True):
                    with self.subTest(field=field, value=value):
                        with self.assertRaises(SchemaError):
                            wrapper.plan(**dict(valid, **{field: value}))
                        self.assertIs(wrapper.plan_state, plan)
                        self.assertIs(wrapper.workspace_contract, workspace)
                        self.assertEqual(components["events"], events)
            self.assertEqual(wrapper.run("q", ("k", "v")), "package-output:q")
            integer_args = dict(valid)
            for name in ("num_qo_heads", "num_kv_heads", "head_dim", "page_size",
                         "q_len_per_req", "window_left", "window_right"):
                integer_args[name] = IndexValue(integer_args[name])
            wrapper.plan(**integer_args)
            self.assertEqual(wrapper.plan_state.spec.fingerprint, plan.spec.fingerprint)
            self.assertEqual(wrapper.run("q", ("k", "v")), "package-output:q")
        finally:
            install_attention_operator_runtime_resolvers(
                original.registry, operation_catalog=original.operation_catalog
            )
