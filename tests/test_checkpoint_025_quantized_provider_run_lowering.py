import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizedKVInput,
    AttentionOperatorWrapperSession,
    TensorView,
    build_attention_operator_package_runtime,
    contiguous_strides,
    dtype_itemsize,
    infer_quant_scale_shape,
    infer_quant_storage_shape,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import package_attention
from tests.test_checkpoint_022_operator_runtime_bootstrap import (
    bootstrap_components,
)


def active_session(values, *, spec=None, catalog=None, plan=None):
    spec = values["spec"] if spec is None else spec
    catalog = values["catalog"] if catalog is None else catalog
    implementation = build_attention_operator_package_runtime(
        spec,
        operation_catalog=catalog,
        package_loader=values["loader"],
    )
    plan = group_plan() if plan is None else plan
    resolved = implementation.resolve(plan, "npu:0")
    session = AttentionOperatorWrapperSession(catalog)
    session.plan(
        resolved.factory,
        resolved.run_adapter,
        plan,
        resolved.receipt,
        resolved.selection,
        resolved.callable_binding,
    )
    return plan, session


class MetadataTensor:
    def __init__(self, tensor_view):
        self.tensor_view = tensor_view


def metadata_tensor(name, shape, dtype, *, device="npu:0", strides=None):
    shape = tuple(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    return MetadataTensor(
        TensorView(
            shape=shape,
            strides=contiguous_strides(shape) if strides is None else tuple(strides),
            dtype=dtype,
            device=device,
            storage_id="checkpoint-025:" + name,
            storage_nbytes=numel * dtype_itemsize(dtype),
        )
    )


def quantized_input(quant_spec, **overrides):
    key_shape = (2, 2, 1, 3)
    value_shape = (2, 2, 1, 2)
    storage_dtype = (
        "uint8"
        if quant_spec.storage_dtype in {"int4_packed", "uint4_packed"}
        else quant_spec.storage_dtype
    )
    values = {
        "quant_spec": quant_spec,
        "key_storage": metadata_tensor(
            "key-storage",
            infer_quant_storage_shape(key_shape, quant_spec),
            storage_dtype,
        ),
        "value_storage": metadata_tensor(
            "value-storage",
            infer_quant_storage_shape(value_shape, quant_spec),
            storage_dtype,
        ),
        "key_scale": metadata_tensor(
            "key-scale",
            infer_quant_scale_shape(key_shape, quant_spec),
            quant_spec.scale_dtype,
        ),
        "value_scale": metadata_tensor(
            "value-scale",
            infer_quant_scale_shape(value_shape, quant_spec),
            quant_spec.scale_dtype,
        ),
    }
    values.update(overrides)
    return AttentionOperatorQuantizedKVInput(**values)


def query_input(plan, name="query"):
    return metadata_tensor(
        name,
        plan.expected_query_shape,
        plan.spec.q_dtype,
    )


class QuantizedProviderRunLoweringCheckpoint(unittest.TestCase):
    """Checkpoint 025: bound quant inputs lower without executing a package."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_symmetric_input_requires_independent_storage_and_scale_objects(self):
        quant_spec = group_plan().spec.kv_quant_spec
        value = quantized_input(quant_spec)

        self.assertIsNot(value.key_storage, value.value_storage)
        self.assertIsNot(value.key_scale, value.value_scale)
        for field_name in (
            "key_storage",
            "value_storage",
            "key_scale",
            "value_scale",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(SchemaError, "requires %s" % field_name):
                    quantized_input(quant_spec, **{field_name: None})

    def test_zero_point_presence_must_match_exact_quant_spec(self):
        symmetric = group_plan().spec.kv_quant_spec
        with self.assertRaisesRegex(SchemaError, "cannot carry zero points"):
            quantized_input(symmetric, key_zero_point=object())

        asymmetric = replace(symmetric, scheme="asymmetric", has_zero_point=True)
        with self.assertRaisesRegex(SchemaError, "requires independent zero points"):
            quantized_input(asymmetric)
        value = quantized_input(
            asymmetric,
            key_zero_point=object(),
            value_zero_point=object(),
        )
        self.assertIsNotNone(value.key_zero_point)
        self.assertIsNotNone(value.value_zero_point)

    def test_bootstrap_lowering_unwraps_storage_and_injects_bound_scales(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        lowered = session.run(query_input(plan), kv_input)

        positional = dict(lowered.positional_arguments)
        keywords = dict(lowered.keyword_arguments)
        self.assertIs(positional["key"], kv_input.key_storage)
        self.assertIs(positional["value"], kv_input.value_storage)
        self.assertIs(keywords["key_scale"], kv_input.key_scale)
        self.assertIs(keywords["value_scale"], kv_input.value_scale)
        self.assertEqual(
            lowered.consumed_request_fields,
            ("query", "kv_cache", "return_lse", "logits_soft_cap"),
        )
        self.assertEqual(package_attention.calls, [])

    def test_plain_or_different_quantspec_input_is_rejected_before_base_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        with self.assertRaisesRegex(SchemaError, "QuantizedKVInput"):
            session.run(query_input(plan), ("key", "value"))

        different = replace(
            plan.spec.kv_quant_spec,
            granularity="tensor",
            group_size=None,
            axis=None,
        )
        with self.assertRaisesRegex(SchemaError, "does not match"):
            session.run(query_input(plan), quantized_input(different))
        self.assertEqual(len(values["materializer"].calls), 1)
        self.assertEqual(package_attention.calls, [])

    def test_default_binding_rejects_public_runtime_scale_multipliers(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        for name in ("q_scale", "k_scale", "v_scale"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(SchemaError, "rejects run-time"):
                    session.run(query_input(plan), kv_input, **{name: object()})
        self.assertEqual(package_attention.calls, [])

    def test_explicit_runtime_scale_policy_injects_separate_arguments(self):
        values = bootstrap_components()
        original = values["spec"].quantization_bindings[0]
        extended = replace(
            original,
            argument_bindings=original.argument_bindings
            + (
                AttentionOperatorQuantArgumentBinding(
                    "run.q_scale", "runtime_query_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.k_scale", "runtime_key_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.v_scale", "runtime_value_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.o_scale", "runtime_output_scale"
                ),
            ),
            runtime_q_scale_policy="argument",
            runtime_k_scale_policy="argument",
            runtime_v_scale_policy="argument",
            runtime_o_scale_policy="argument",
            runtime_q_scale_input_kinds=("scalar",),
            runtime_k_scale_input_kinds=("scalar",),
            runtime_v_scale_input_kinds=("scalar",),
            runtime_o_scale_input_kinds=("scalar",),
            runtime_o_scale_output_dtypes=("float32",),
        )
        spec = replace(values["spec"], quantization_bindings=(extended,))
        plan, session = active_session(values, spec=spec)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        runtime_q_scale = 2.0
        runtime_k_scale = 1.5
        runtime_v_scale = 0.5
        runtime_o_scale = 0.25

        lowered = session.run(
            query_input(plan),
            kv_input,
            q_scale=runtime_q_scale,
            k_scale=runtime_k_scale,
            v_scale=runtime_v_scale,
            o_scale=runtime_o_scale,
        )

        keywords = dict(lowered.keyword_arguments)
        self.assertIs(keywords["runtime_query_scale"], runtime_q_scale)
        self.assertIs(keywords["runtime_key_scale"], runtime_k_scale)
        self.assertIs(keywords["runtime_value_scale"], runtime_v_scale)
        self.assertIs(keywords["runtime_output_scale"], runtime_o_scale)
        self.assertEqual(
            lowered.consumed_request_fields,
            (
                "query",
                "kv_cache",
                "return_lse",
                "q_scale",
                "k_scale",
                "v_scale",
                "o_scale",
                "logits_soft_cap",
            ),
        )
        self.assertEqual(package_attention.calls, [])

    def test_injected_argument_cannot_overwrite_base_provider_lowering(self):
        values = bootstrap_components()
        operation = replace(
            values["operation"],
            quant_arguments=values["operation"].quant_arguments + ("scale",),
        )
        catalog = AttentionOperatorOperationCatalog(
            name="checkpoint-025-collision", operations=(operation,)
        )
        original = values["spec"].quantization_bindings[0]
        collision = replace(
            original,
            argument_bindings=(
                AttentionOperatorQuantArgumentBinding("kv.key.scale", "scale"),
                AttentionOperatorQuantArgumentBinding(
                    "kv.value.scale", "value_scale"
                ),
            ),
        )
        spec = replace(values["spec"], quantization_bindings=(collision,))
        plan, session = active_session(values, spec=spec, catalog=catalog)

        with self.assertRaisesRegex(SchemaError, "collides.*scale"):
            session.run(
                query_input(plan), quantized_input(plan.spec.kv_quant_spec)
            )
        self.assertEqual(package_attention.calls, [])

    def test_lowering_reuses_one_plan_materialization_without_callable_execution(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        first = session.run(query_input(plan, "query-1"), kv_input)
        second = session.run(query_input(plan, "query-2"), kv_input)

        self.assertEqual(len(values["materializer"].calls), 1)
        self.assertIs(
            dict(first.keyword_arguments)["table"],
            dict(second.keyword_arguments)["table"],
        )
        self.assertEqual(package_attention.calls, [])


if __name__ == "__main__":
    unittest.main()
