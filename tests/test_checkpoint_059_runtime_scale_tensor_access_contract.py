import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorRunRequest,
    AttentionTensorAccessPolicy,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_019_package_runtime_integration import (
    FakePackageLoader,
    fake_operation,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    active_session,
    metadata_tensor,
    query_input,
    quantized_input,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor
from tests.test_checkpoint_058_quantized_kv_output_alias_contract import (
    aligned_quantized_input,
    aligned_tensor,
)


def scale_buffer_attention(
    query,
    key,
    value,
    *,
    table=None,
    scale=1.0,
    return_softmax_lse=False,
    key_scale=None,
    value_scale=None,
    runtime_query_head_scale=None,
    out=None,
    lse=None,
):
    return (out, lse) if return_softmax_lse else out


class ScaleBufferPackageLoader(FakePackageLoader):
    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append("resolve_callable")
        return scale_buffer_attention


def scale_buffer_runtime(*, access_policy=None):
    values = bootstrap_components()
    operation = replace(
        fake_operation(),
        callable_path="checkpoint_059_package.attention",
        keyword_arguments=(
            "table",
            "scale",
            "return_softmax_lse",
            "key_scale",
            "value_scale",
            "runtime_query_head_scale",
            "out",
            "lse",
        ),
        mutable_arguments=("out", "lse"),
        quant_arguments=(
            "key_scale",
            "value_scale",
            "runtime_query_head_scale",
        ),
        output_buffer_argument="out",
        lse_buffer_argument="lse",
    )
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-059-runtime-scale-catalog", operations=(operation,)
    )
    original = values["spec"].quantization_bindings[0]
    binding = replace(
        original,
        argument_bindings=original.argument_bindings
        + (
            AttentionOperatorQuantArgumentBinding(
                "run.q_head_scale", "runtime_query_head_scale"
            ),
        ),
        runtime_q_head_scale_policy="argument",
    )
    runtime_spec = replace(
        values["spec"],
        quantization_bindings=(binding,),
        tensor_access_policy=(
            AttentionTensorAccessPolicy(require_contiguous_q=True)
            if access_policy is None
            else access_policy
        ),
    )
    loader = ScaleBufferPackageLoader(values["events"])
    values.update(
        operation=operation,
        catalog=catalog,
        loader=loader,
        spec=runtime_spec,
    )
    plan, session = active_session(
        values, spec=runtime_spec, catalog=catalog
    )
    return values, plan, session


def lower_with_query_head_scale(
    session,
    plan,
    kv_input,
    query_head_scale,
    *,
    query=None,
    out=None,
):
    request = AttentionOperatorRunRequest.from_active_plan(
        session.active_plan,
        query_input(plan) if query is None else query,
        kv_input,
        q_head_scale=query_head_scale,
        out=out,
    )
    return session._lower_request(request)


class RuntimeScaleTensorAccessContractTests(unittest.TestCase):
    """Dynamic per-head scale tensors obey the provider tensor policy."""

    def test_runtime_head_scale_alignment_is_fail_closed(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_q=True,
            require_contiguous_kv=True,
            required_alignment=16,
        )
        _, plan, session = scale_buffer_runtime(access_policy=policy)
        query = aligned_tensor(query_input(plan), 16)
        kv_input = aligned_quantized_input(plan)
        query_head_scale = metadata_tensor(
            "under-aligned-query-head-scale",
            (plan.spec.num_qo_heads,),
            plan.spec.kv_quant_spec.scale_dtype,
        )
        query_head_scale = aligned_tensor(query_head_scale, 8)

        with self.assertRaisesRegex(
            SchemaError, "run.q_head_scale must be 16-byte aligned"
        ):
            lower_with_query_head_scale(
                session,
                plan,
                kv_input,
                query_head_scale,
                query=query,
            )

    def test_output_cannot_alias_runtime_head_scale_by_default(self):
        _, plan, session = scale_buffer_runtime()
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        query_head_scale = metadata_tensor(
            "query-head-scale",
            (plan.spec.num_qo_heads,),
            plan.spec.kv_quant_spec.scale_dtype,
        )
        out = buffer_tensor(
            "out-alias-query-head-scale",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=query_head_scale.tensor_view.storage_id,
        )

        with self.assertRaisesRegex(
            SchemaError, "out cannot alias run.q_head_scale"
        ):
            lower_with_query_head_scale(
                session, plan, kv_input, query_head_scale, out=out
            )

    def test_valid_runtime_head_scale_is_injected_without_copy(self):
        _, plan, session = scale_buffer_runtime()
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        query_head_scale = metadata_tensor(
            "query-head-scale",
            (plan.spec.num_qo_heads,),
            plan.spec.kv_quant_spec.scale_dtype,
        )

        lowered = lower_with_query_head_scale(
            session, plan, kv_input, query_head_scale
        )

        self.assertIs(
            dict(lowered.keyword_arguments)["runtime_query_head_scale"],
            query_head_scale,
        )


if __name__ == "__main__":
    unittest.main()
