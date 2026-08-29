import inspect
import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionMode,
    AttentionNvfp4PackedLayoutDescriptor,
    AttentionOperatorNvfp4PackedKVBinding,
    AttentionOperatorNvfp4PackedKVRunAdapterFactory,
    AttentionOperatorNvfp4ScaleFactorBinding,
    AttentionOperatorOperationCatalog,
    AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageResolver,
    AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionTensorAccessPolicy,
    TensorView,
    attention_operator_runtime_registry_snapshot,
    flashinfer_nvfp4_kv_quant_spec,
    install_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.frontend import (
    canonicalize_flashinfer_paged_kv_dtype,
)
from flashinfer_npu.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    single_prefill_with_kv_cache,
)
from flashinfer_npu.runtime import QuantSpec, SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalRunAdapter,
    build_components,
)


class FakeNpuTensor:
    def __init__(self, name, shape, *, dtype, device="npu:0"):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        strides = []
        stride = 1
        for dim in reversed(self.shape):
            strides.append(stride)
            stride *= dim
        self.tensor_view = TensorView(
            shape=self.shape,
            strides=tuple(reversed(strides)),
            dtype=dtype,
            device=device,
            storage_id=name,
            storage_nbytes=max(1, stride * (2 if dtype == "bfloat16" else 1)),
            data_ptr_alignment=64,
        )

    def __str__(self):
        return self.name


class FakeNpuWorkspace:
    shape = (4096,)
    dtype = "uint8"
    device = "npu:0"


def explicit_uint8_spec():
    return QuantSpec(
        scheme="asymmetric",
        storage_dtype="uint8",
        compute_dtype="float32",
        accumulator_dtype="float32",
        scale_dtype="float32",
        granularity="tensor",
        has_zero_point=True,
    )


def nvfp4_runtime():
    values = build_components()
    quant_spec = flashinfer_nvfp4_kv_quant_spec()
    operation = replace(
        values["operation"],
        keyword_arguments=values["operation"].keyword_arguments
        + ("k_sf", "v_sf"),
        quant_arguments=values["operation"].quant_arguments + ("k_sf", "v_sf"),
    )
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-102-public-nvfp4", operations=(operation,)
    )
    scale_binding = AttentionOperatorNvfp4ScaleFactorBinding(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        quant_spec=quant_spec,
        key_argument="k_sf",
        value_argument="v_sf",
    )
    packed_binding = AttentionOperatorNvfp4PackedKVBinding(
        scale_binding,
        AttentionNvfp4PackedLayoutDescriptor(
            physical_layout=quant_spec.physical_layout,
            packing_order=quant_spec.packing_order,
        ),
    )
    calls = []

    def package_attention(
        query,
        key,
        value,
        *,
        table=None,
        scale=1.0,
        return_softmax_lse=False,
        key_scale=None,
        value_scale=None,
        runtime_key_scale=None,
        runtime_value_scale=None,
        runtime_query_scale=None,
        runtime_output_scale=None,
        runtime_query_head_scale=None,
        runtime_key_head_scale=None,
        runtime_value_head_scale=None,
        k_sf=None,
        v_sf=None,
    ):
        kwargs = {
            "table": table,
            "scale": scale,
            "return_softmax_lse": return_softmax_lse,
            "key_scale": key_scale,
            "value_scale": value_scale,
            "runtime_key_scale": runtime_key_scale,
            "runtime_value_scale": runtime_value_scale,
            "runtime_query_scale": runtime_query_scale,
            "runtime_output_scale": runtime_output_scale,
            "runtime_query_head_scale": runtime_query_head_scale,
            "runtime_key_head_scale": runtime_key_head_scale,
            "runtime_value_head_scale": runtime_value_head_scale,
            "k_sf": k_sf,
            "v_sf": v_sf,
        }
        calls.append((query, key, value, kwargs))
        output = "nvfp4-output:%s" % query
        if return_softmax_lse:
            return output, "nvfp4-lse"
        return output

    def resolve_callable(callable_path):
        values["loader"].resolve_calls += 1
        values["events"].append("resolve_callable")
        return package_attention

    values["loader"].resolve_callable = resolve_callable
    package_resolver = AttentionOperatorPackageResolver(
        catalog,
        AttentionOperatorPackageCompatibility(
            provider_id=operation.provider_id,
            operation_id=operation.operation_id,
            adapter_version="checkpoint-102-framework-only-v1",
            supported_package_versions=("1.0.0",),
        ),
        values["loader"],
    )

    class TensorInspector:
        def to_view(self, tensor, *, name, writable=False):
            view = getattr(tensor, "tensor_view", None)
            if view is None:
                raise SchemaError("synthetic tensor has no TensorView metadata")
            return view

    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=100,
        package_resolver=package_resolver,
        plan_gate=values["gate"],
        authority_resolver=values["authority"],
        logical_factory=values["factory"],
        logical_run_adapter=FakeLogicalRunAdapter(),
        tensor_materializer=values["materializer"],
        run_adapter_factory=AttentionOperatorNvfp4PackedKVRunAdapterFactory(
            operation,
            packed_binding,
            TensorInspector(),
            AttentionTensorAccessPolicy(),
        ),
    )
    registry = AttentionOperatorRuntimeResolverRegistry(
        (
            (
                "npu",
                AttentionOperatorRuntimeImplementationRegistry((implementation,)),
            ),
        )
    )
    return values, quant_spec, catalog, registry, calls


class PublicNvfp4CanonicalizationCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original = attention_operator_runtime_registry_snapshot()
        (
            self.values,
            self.quant_spec,
            self.catalog,
            self.registry,
            self.calls,
        ) = nvfp4_runtime()
        install_attention_operator_runtime_resolvers(
            self.registry, operation_catalog=self.catalog
        )

    def tearDown(self):
        install_attention_operator_runtime_resolvers(
            self.original.registry,
            operation_catalog=self.original.operation_catalog,
        )

    def test_bare_uint8_paged_plan_is_exact_public_nvfp4(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="NHD", backend="auto"
        )
        wrapper.plan(
            [0, 1],
            [0],
            [16],
            8,
            2,
            128,
            16,
            q_data_type="bfloat16",
            kv_data_type="uint8",
            o_data_type="bfloat16",
        )

        self.assertEqual(wrapper.plan_state.spec.kv_dtype, "uint8")
        self.assertEqual(wrapper.plan_state.spec.kv_quant_spec, self.quant_spec)
        key = FakeNpuTensor("key", (1, 16, 2, 64), dtype="uint8")
        value = FakeNpuTensor("value", (1, 16, 2, 64), dtype="uint8")
        k_sf = FakeNpuTensor(
            "k-sf", (1, 16, 2, 8), dtype="float8_e4m3fn"
        )
        v_sf = FakeNpuTensor(
            "v-sf", (1, 16, 2, 8), dtype="float8_e4m3fn"
        )

        self.assertEqual(
            wrapper.run("q", (key, value), kv_cache_sf=(k_sf, v_sf)),
            "nvfp4-output:q",
        )
        self.assertIs(self.calls[0][3]["k_sf"], k_sf)
        self.assertIs(self.calls[0][3]["v_sf"], v_sf)

    def test_paged_nvfp4_requires_scales_and_does_not_retry_dense(self):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="NHD", backend="auto"
        )
        wrapper.plan(
            [0, 1], [0], [16], 8, 2, 128, 16,
            q_data_type="bfloat16", kv_data_type="uint8",
        )
        key = FakeNpuTensor("key", (1, 16, 2, 64), dtype="uint8")
        value = FakeNpuTensor("value", (1, 16, 2, 64), dtype="uint8")

        with self.assertRaisesRegex(SchemaError, "requires kv_cache_sf"):
            wrapper.run("q", (key, value))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.values["loader"].resolve_calls, 1)

    def test_bare_uint8_paged_prefill_plan_uses_the_same_contract(self):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

        wrapper.plan(
            [0, 2],
            [0, 1],
            [0],
            [16],
            8,
            2,
            128,
            16,
            q_data_type="bfloat16",
            kv_data_type="uint8",
            o_data_type="bfloat16",
        )

        self.assertEqual(wrapper.plan_state.spec.kv_dtype, "uint8")
        self.assertEqual(wrapper.plan_state.spec.kv_quant_spec, self.quant_spec)
        self.assertEqual(wrapper.plan_state.spec.mode, AttentionMode.BATCH_PREFILL_PAGED)

    def test_single_prefill_uses_scale_presence_without_new_public_parameter(self):
        q = FakeNpuTensor("q", (3, 8, 128), dtype="bfloat16")
        key = FakeNpuTensor("key", (7, 2, 64), dtype="uint8")
        value = FakeNpuTensor("value", (7, 2, 64), dtype="uint8")
        k_sf = FakeNpuTensor("k-sf", (7, 2, 8), dtype="float8_e4m3fn")
        v_sf = FakeNpuTensor("v-sf", (7, 2, 8), dtype="float8_e4m3fn")

        output = single_prefill_with_kv_cache(
            q, key, value, kv_cache_sf=(k_sf, v_sf)
        )

        self.assertEqual(output, "nvfp4-output:q")
        self.assertIs(self.calls[0][3]["k_sf"], k_sf)
        self.assertIs(self.calls[0][3]["v_sf"], v_sf)
        parameters = inspect.signature(single_prefill_with_kv_cache).parameters
        self.assertIn("kv_cache_sf", parameters)
        self.assertNotIn("kv_quant_spec", parameters)
        self.assertNotIn("provider", parameters)
        self.assertNotIn("plan_handle", parameters)

    def test_explicit_uint8_quantspec_is_not_reclassified_as_nvfp4(self):
        explicit = explicit_uint8_spec()

        dtype, quant_spec = canonicalize_flashinfer_paged_kv_dtype(
            explicit, "bfloat16"
        )

        self.assertEqual(dtype, "uint8")
        self.assertIs(quant_spec, explicit)
        self.assertNotEqual(quant_spec.fingerprint, self.quant_spec.fingerprint)


if __name__ == "__main__":
    unittest.main()
