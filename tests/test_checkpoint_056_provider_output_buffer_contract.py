import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog,
    AttentionTensorAccessPolicy,
    TensorView,
    contiguous_strides,
    dtype_itemsize,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    FakePackageLoader,
    fake_operation,
    package_attention,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    MetadataTensor,
    active_session,
    quantized_input,
    query_input,
)


def package_attention_with_buffers(
    query,
    key,
    value,
    *,
    table=None,
    scale=1.0,
    return_softmax_lse=False,
    key_scale=None,
    value_scale=None,
    out=None,
    lse=None,
):
    return (out, lse) if return_softmax_lse else out


class BufferPackageLoader(FakePackageLoader):
    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append("resolve_callable")
        return package_attention_with_buffers


def buffer_tensor(
    name,
    shape,
    dtype,
    *,
    device="npu:0",
    writable=True,
    strides=None,
    storage_id=None,
    alignment=16,
):
    shape = tuple(shape)
    strides = contiguous_strides(shape) if strides is None else tuple(strides)
    storage_elements = (
        0
        if any(dim == 0 for dim in shape)
        else 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
    )
    return MetadataTensor(
        TensorView(
            shape=shape,
            strides=strides,
            dtype=dtype,
            device=device,
            storage_id=("checkpoint-056:" + name if storage_id is None else storage_id),
            storage_nbytes=storage_elements * dtype_itemsize(dtype),
            data_ptr_alignment=alignment,
            writable=writable,
        )
    )


def buffer_runtime(*, access_policy=None):
    values = bootstrap_components()
    operation = replace(
        fake_operation(),
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
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-056-buffer-catalog", operations=(operation,)
    )
    loader = BufferPackageLoader(values["events"])
    values = dict(values, loader=loader, operation=operation, catalog=catalog)
    spec = replace(
        values["spec"],
        tensor_access_policy=(
            AttentionTensorAccessPolicy(require_contiguous_q=True)
            if access_policy is None
            else access_policy
        ),
    )
    plan, session = active_session(values, spec=spec, catalog=catalog)
    return values, plan, session


class ProviderOutputBufferContractTests(unittest.TestCase):
    """Caller-owned out/LSE remain optional and catalog-bound."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_catalog_requires_explicit_mutable_buffer_arguments(self):
        operation = fake_operation()
        with self.assertRaisesRegex(SchemaError, "declared argument"):
            replace(operation, output_buffer_argument="out")
        with self.assertRaisesRegex(SchemaError, "must be mutable"):
            replace(
                operation,
                keyword_arguments=operation.keyword_arguments + ("out",),
                output_buffer_argument="out",
            )

    def test_valid_buffers_are_validated_and_injected_by_exact_name(self):
        values, plan, session = buffer_runtime()
        query = query_input(plan)
        out = buffer_tensor(
            "out", plan.expected_output_shape, plan.spec.o_dtype
        )
        lse = buffer_tensor("lse", plan.expected_lse_shape, "float32")

        lowered = session.run(
            query,
            quantized_input(plan.spec.kv_quant_spec),
            out=out,
            lse=lse,
            return_lse=True,
        )

        keywords = dict(lowered.keyword_arguments)
        self.assertIs(keywords["out"], out)
        self.assertIs(keywords["lse"], lse)
        self.assertEqual(lowered.mutable_argument_names, ("out", "lse"))
        self.assertEqual(
            session.resource_binding.output_binding, "caller_optional"
        )
        self.assertEqual(session.resource_binding.lse_binding, "caller_optional")
        self.assertEqual(
            tuple(name for _, name, _ in values["tensor_metadata_inspector"].calls[:3]),
            ("query", "out", "lse"),
        )
        self.assertEqual(package_attention.calls, [])

    def test_buffer_shape_dtype_device_and_writable_are_fail_closed(self):
        _, plan, session = buffer_runtime()
        query = query_input(plan)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        cases = (
            (
                "out",
                buffer_tensor(
                    "bad-out-shape",
                    plan.expected_output_shape[:-1] + (plan.spec.head_dim_vo + 1,),
                    plan.spec.o_dtype,
                ),
                "out shape",
            ),
            (
                "out",
                buffer_tensor(
                    "bad-out-dtype", plan.expected_output_shape, "float16"
                ),
                "out dtype",
            ),
            (
                "out",
                buffer_tensor(
                    "bad-out-device",
                    plan.expected_output_shape,
                    plan.spec.o_dtype,
                    device="npu:1",
                ),
                "out device",
            ),
            (
                "out",
                buffer_tensor(
                    "readonly-out",
                    plan.expected_output_shape,
                    plan.spec.o_dtype,
                    writable=False,
                ),
                "out must be writable",
            ),
            (
                "lse",
                buffer_tensor(
                    "bad-lse-shape",
                    plan.expected_lse_shape + (1,),
                    "float32",
                ),
                "lse shape",
            ),
            (
                "lse",
                buffer_tensor(
                    "bad-lse-dtype", plan.expected_lse_shape, "float16"
                ),
                "lse dtype",
            ),
            (
                "lse",
                buffer_tensor(
                    "bad-lse-device",
                    plan.expected_lse_shape,
                    "float32",
                    device="npu:1",
                ),
                "lse device",
            ),
            (
                "lse",
                buffer_tensor(
                    "readonly-lse",
                    plan.expected_lse_shape,
                    "float32",
                    writable=False,
                ),
                "lse must be writable",
            ),
        )
        for field, value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    session.run(
                        query,
                        kv_input,
                        return_lse=True,
                        **{field: value},
                    )
        self.assertEqual(package_attention.calls, [])

    def test_output_policy_alignment_contiguity_and_alias_are_enforced(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_q=True,
            require_contiguous_output=True,
            required_alignment=16,
        )
        _, plan, session = buffer_runtime(access_policy=policy)
        query = query_input(plan)
        query = type(query)(
            replace(query.tensor_view, data_ptr_alignment=16)
        )
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        non_contiguous = buffer_tensor(
            "strided-out",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            strides=(8, 4, 1),
        )
        under_aligned = buffer_tensor(
            "under-aligned-out",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            alignment=8,
        )
        aliased = buffer_tensor(
            "aliased-out",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=query.tensor_view.storage_id,
        )

        with self.assertRaisesRegex(SchemaError, "out must be contiguous"):
            session.run(query, kv_input, out=non_contiguous)
        with self.assertRaisesRegex(SchemaError, "out must be 16-byte aligned"):
            session.run(query, kv_input, out=under_aligned)
        with self.assertRaisesRegex(SchemaError, "out cannot alias query"):
            session.run(query, kv_input, out=aliased)
        self.assertEqual(package_attention.calls, [])

    def test_out_and_lse_must_not_overlap_each_other(self):
        _, plan, session = buffer_runtime()
        shared_storage = "checkpoint-056:shared-output-storage"
        out = buffer_tensor(
            "out",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=shared_storage,
        )
        lse = buffer_tensor(
            "lse",
            plan.expected_lse_shape,
            "float32",
            storage_id=shared_storage,
        )

        with self.assertRaisesRegex(SchemaError, "out and lse cannot alias"):
            session.run(
                query_input(plan),
                quantized_input(plan.spec.kv_quant_spec),
                out=out,
                lse=lse,
                return_lse=True,
            )
        self.assertEqual(package_attention.calls, [])


if __name__ == "__main__":
    unittest.main()
