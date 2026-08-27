import unittest
from dataclasses import replace

from flashinfer_npu.attention import AttentionTensorAccessPolicy
from flashinfer_npu.runtime import SchemaError
from tests.test_checkpoint_025_quantized_provider_run_lowering import (
    query_input,
    quantized_input,
)
from tests.test_checkpoint_056_provider_output_buffer_contract import (
    buffer_runtime,
    buffer_tensor,
)


def aligned_tensor(value, alignment):
    return type(value)(
        replace(value.tensor_view, data_ptr_alignment=alignment)
    )


def aligned_quantized_input(plan, alignment=16):
    value = quantized_input(plan.spec.kv_quant_spec)
    return replace(
        value,
        key_storage=aligned_tensor(value.key_storage, alignment),
        value_storage=aligned_tensor(value.value_storage, alignment),
        key_scale=aligned_tensor(value.key_scale, alignment),
        value_scale=aligned_tensor(value.value_scale, alignment),
    )


class QuantizedKVOutputAliasContractTests(unittest.TestCase):
    """Quantized KV components participate in the provider run alias gate."""

    def test_output_cannot_alias_quantized_storage_by_default(self):
        _, plan, session = buffer_runtime()
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        out = buffer_tensor(
            "out-alias-key-storage",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=kv_input.key_storage.tensor_view.storage_id,
        )

        with self.assertRaisesRegex(
            SchemaError, "out cannot alias kv.key_storage"
        ):
            session.run(query_input(plan), kv_input, out=out)

    def test_lse_cannot_alias_quantized_scale_by_default(self):
        _, plan, session = buffer_runtime()
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        lse = buffer_tensor(
            "lse-alias-value-scale",
            plan.expected_lse_shape,
            "float32",
            storage_id=kv_input.value_scale.tensor_view.storage_id,
        )

        with self.assertRaisesRegex(
            SchemaError, "lse cannot alias kv.value_scale"
        ):
            session.run(
                query_input(plan),
                kv_input,
                lse=lse,
                return_lse=True,
            )

    def test_quantized_component_alignment_uses_provider_policy(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_q=True,
            require_contiguous_kv=True,
            required_alignment=16,
        )
        _, plan, session = buffer_runtime(access_policy=policy)
        query = aligned_tensor(query_input(plan), 16)
        kv_input = aligned_quantized_input(plan)
        kv_input = replace(
            kv_input,
            key_scale=aligned_tensor(kv_input.key_scale, 8),
        )

        with self.assertRaisesRegex(
            SchemaError, "kv.key_scale must be 16-byte aligned"
        ):
            session.run(query, kv_input)

    def test_explicit_alias_permission_applies_to_quantized_kv(self):
        policy = AttentionTensorAccessPolicy(
            require_contiguous_q=True,
            permit_output_input_alias=True,
        )
        _, plan, session = buffer_runtime(access_policy=policy)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        out = buffer_tensor(
            "permitted-out-alias",
            plan.expected_output_shape,
            plan.spec.o_dtype,
            storage_id=kv_input.key_storage.tensor_view.storage_id,
        )

        lowered = session.run(query_input(plan), kv_input, out=out)

        self.assertIs(dict(lowered.keyword_arguments)["out"], out)


if __name__ == "__main__":
    unittest.main()
