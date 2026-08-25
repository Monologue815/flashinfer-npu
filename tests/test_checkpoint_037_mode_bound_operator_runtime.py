import unittest

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionOperatorRuntime,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionPlanSpec,
    KVLayout,
    PagedKVMetadata,
    PagedPrefillMetadata,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    build_components,
    framework_inputs,
    package_attention,
)


def prefill_inputs():
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=KVLayout.HND,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        o_dtype="bfloat16",
    )
    metadata = PagedPrefillMetadata(
        qo_indptr=(0, 2),
        paged_kv=PagedKVMetadata(
            indptr=(0, 1),
            indices=(7,),
            last_page_len=(64,),
            page_size=128,
        ),
    )
    return spec, metadata


def prefill_runtime(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry(
        (components["implementation"],)
    )
    resolvers = AttentionOperatorRuntimeResolverRegistry(
        (("npu", implementations),)
    )
    return AttentionOperatorRuntime(
        "npu:0",
        resolvers,
        components["catalog"],
        mode=AttentionMode.BATCH_PREFILL_PAGED,
    )


class ModeBoundAttentionOperatorRuntimeCheckpointTests(unittest.TestCase):
    """One provider runtime can serve any exact public Attention mode."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_paged_prefill_plan_selects_and_runs_through_the_same_runtime(self):
        components = build_components()
        runtime = prefill_runtime(components)
        spec, metadata = prefill_inputs()

        self.assertEqual(runtime.mode, AttentionMode.BATCH_PREFILL_PAGED)
        self.assertIsNone(runtime.plan(spec, metadata))
        result = runtime.run("q", ("k-cache", "v-cache"))

        self.assertEqual(result, ("package-output:q", "package-lse:0.25"))
        self.assertEqual(runtime.plan_state.spec.mode, runtime.mode)
        self.assertEqual(components["loader"].resolve_calls, 1)
        self.assertEqual(components["authority"].calls, 1)
        self.assertEqual(len(package_attention.calls), 1)

    def test_mode_mismatch_fails_before_provider_resolution(self):
        components = build_components()
        runtime = prefill_runtime(components)
        mixed_spec, mixed_metadata = framework_inputs()

        with self.assertRaisesRegex(SchemaError, "mode"):
            runtime.plan(mixed_spec, mixed_metadata)

        self.assertFalse(runtime.is_planned)
        self.assertEqual(components["loader"].version_calls, 0)
        self.assertEqual(components["authority"].calls, 0)

    def test_mode_is_required_and_must_be_an_attention_mode(self):
        components = build_components()
        implementations = AttentionOperatorRuntimeImplementationRegistry(
            (components["implementation"],)
        )
        resolvers = AttentionOperatorRuntimeResolverRegistry(
            (("npu", implementations),)
        )
        with self.assertRaisesRegex(TypeError, "mode"):
            AttentionOperatorRuntime(
                "npu:0",
                resolvers,
                components["catalog"],
                mode="batch_prefill_paged",
            )


if __name__ == "__main__":
    unittest.main()
