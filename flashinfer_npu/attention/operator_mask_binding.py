"""Private no-conversion mask argument lowering, not provider launch authority.

An integration must explicitly declare canonical allow-mask semantics for its
exact operation. No packaged operation receives that declaration automatically.
"""

import re
from dataclasses import dataclass, field, replace
from typing import Optional

from flashinfer_npu.runtime import SchemaError

from .operation_catalog import (
    AttentionOperatorOperationCatalog, AttentionOperatorOperationSpec,
    bind_attention_operator_operation,
)
from .operator_mask import (
    AttentionInspectedMaskPlanResource,
    revalidate_attention_mask_plan_resource,
)
from .schema import _canonical_hash
from .operator_plan import AttentionOperatorActivePlan
from .operator_run import (
    AttentionOperatorRunAdapter, AttentionOperatorTensorMetadataInspector,
    lower_attention_operator_run, validate_attention_lowered_operator_call,
)
from .tensor_contract import TensorView


@dataclass(frozen=True)
class AttentionOperatorMaskArgumentSpec:
    """Exact signature mapping for borrowed, row-major allow-mask segments.

    True (or a set bit) means visible. Packed segments use little bit order
    and separate byte padding per request. Offsets are host integer sequences.
    Other semantics or device offset tensors require a separate transformation.
    """

    operation_fingerprint: str
    encoding: str
    mask_argument: str
    logical_element_indptr_argument: str
    packed_byte_indptr_argument: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.operation_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.operation_fingerprint
        ):
            raise SchemaError("mask operation_fingerprint must be lowercase SHA-256")
        if self.encoding not in ("bool_allow_flat", "packed_allow_little_segments"):
            raise SchemaError("unsupported direct mask encoding; explicit transformation required")
        if (self.encoding == "packed_allow_little_segments") != (
            self.packed_byte_indptr_argument is not None
        ):
            raise SchemaError("packed mask binding requires separately named byte offsets")
        names = (self.mask_argument, self.logical_element_indptr_argument)
        if self.packed_byte_indptr_argument is not None:
            names += (self.packed_byte_indptr_argument,)
        if any(not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ) for name in names):
            raise SchemaError("mask argument names must be valid identifiers")
        if len(set(names)) != len(names):
            raise SchemaError("mask payload and offset arguments must be distinct")

    @property
    def offset_arguments(self):
        names = (self.logical_element_indptr_argument,)
        if self.packed_byte_indptr_argument is not None:
            names += (self.packed_byte_indptr_argument,)
        return names

    def validate_operation(self, operation):
        if not isinstance(operation, AttentionOperatorOperationSpec):
            raise TypeError("operation must be AttentionOperatorOperationSpec")
        if operation.fingerprint != self.operation_fingerprint:
            raise SchemaError("mask argument mapping does not match the exact operation")
        names = {self.mask_argument, *self.offset_arguments}
        if not names.issubset(operation.keyword_arguments):
            raise SchemaError("direct mask bindings must name declared keyword arguments")
        reserved = set(operation.mutable_arguments) | set(operation.quant_arguments) | {
            operation.paged_table_argument, operation.lse_control_argument,
            operation.output_buffer_argument, operation.lse_buffer_argument,
        }
        if names.intersection(reserved):
            raise SchemaError("mask arguments conflict with another operation role")
        if self.mask_argument in operation.host_sequence_arguments:
            raise SchemaError("mask payload cannot be a host sequence argument")
        if not set(self.offset_arguments).issubset(operation.host_sequence_arguments):
            raise SchemaError("mask offset arguments must explicitly accept host sequences")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def fingerprint(self):
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True, eq=False)
class AttentionLoweredMaskArguments:
    """A non-executing argument fragment retaining the borrowed source owner.

    Keep this object, not only its argument tuple, alive through completion.
    It does not establish dispatch authority, device leases or a copied snapshot.
    """

    inspected: AttentionInspectedMaskPlanResource = field(repr=False)
    spec: AttentionOperatorMaskArgumentSpec

    def __post_init__(self):
        if not isinstance(self.inspected, AttentionInspectedMaskPlanResource):
            raise TypeError("inspected must be AttentionInspectedMaskPlanResource")
        if not isinstance(self.spec, AttentionOperatorMaskArgumentSpec):
            raise TypeError("spec must be AttentionOperatorMaskArgumentSpec")
        expected = ("packed_allow_little_segments"
                    if self.inspected.resource.metadata.mask_spec.packed else "bool_allow_flat")
        if self.spec.encoding != expected:
            raise SchemaError("mask encoding differs from the source; explicit transformation required")

    @property
    def keyword_arguments(self):
        resource = self.inspected.resource
        pairs = (
            (self.spec.mask_argument, resource.payload),
            (self.spec.logical_element_indptr_argument, resource.metadata.logical_element_indptr),
        )
        if self.spec.packed_byte_indptr_argument is not None:
            pairs += ((self.spec.packed_byte_indptr_argument,
                       resource.metadata.packed_byte_indptr),)
        return pairs


def lower_attention_mask_arguments(plan, inspected, operation, spec, inspector):
    """Validate an explicit mapping and recheck the source before forming arguments.

    Declaration and plan failures precede tensor inspection. No package lookup,
    allocation, packing, mutation, provider call or public-path activation occurs.
    The future run adapter must also bind this fragment to its selected active
    operation, reject argument collisions and retain it until execution completes.
    """
    if not isinstance(spec, AttentionOperatorMaskArgumentSpec):
        raise TypeError("spec must be AttentionOperatorMaskArgumentSpec")
    spec.validate_operation(operation)
    fragment = AttentionLoweredMaskArguments(inspected, spec)
    inspected.validate_plan(plan)
    if plan.spec.mode not in operation.candidate_modes:
        raise SchemaError("mask operation is not a candidate for the planned mode")
    revalidate_attention_mask_plan_resource(plan, inspected, inspector)
    return fragment


class AttentionOperatorMaskRunAdapter:
    """Private active-plan-bound decorator; install outside tensor validators.

    This adapter describes a call and retains owners, but neither launches it nor
    tracks asynchronous completion. No public wrapper installs it automatically.
    """

    def __init__(self, base_adapter, active_plan, operation, inspected, spec, inspector):
        if not isinstance(base_adapter, AttentionOperatorRunAdapter):
            raise TypeError("base_adapter must implement AttentionOperatorRunAdapter")
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        if not isinstance(spec, AttentionOperatorMaskArgumentSpec):
            raise TypeError("spec must be AttentionOperatorMaskArgumentSpec")
        spec.validate_operation(operation)
        fragment = AttentionLoweredMaskArguments(inspected, spec)
        inspected.validate_plan(active_plan.framework_plan)
        binding = bind_attention_operator_operation(
            AttentionOperatorOperationCatalog("private_mask_adapter", (operation,)), active_plan)
        if base_adapter.provider_id != binding.provider_id:
            raise SchemaError("mask adapter base provider differs from the active operation")
        if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
            raise TypeError("inspector must implement AttentionOperatorTensorMetadataInspector")
        self.provider_id = binding.provider_id
        self.operation_id = binding.operation_id
        self._binding = binding
        self._operation = operation
        self._base_adapter = base_adapter
        self._fragment = fragment
        self._inspector = inspector

    def lower(self, active_plan, request):
        if not isinstance(active_plan, AttentionOperatorActivePlan):
            raise TypeError("active_plan must be AttentionOperatorActivePlan")
        if active_plan.fingerprint != self._binding.active_plan_fingerprint:
            raise SchemaError("mask adapter does not bind this active plan")
        self._fragment.inspected.validate_plan(active_plan.framework_plan)
        lowered = lower_attention_operator_run(self._base_adapter, active_plan, request)
        validate_attention_lowered_operator_call(self._operation, self._binding, lowered)
        spec = self._fragment.spec
        names = {spec.mask_argument, *spec.offset_arguments}
        arguments = dict(lowered.positional_arguments + lowered.keyword_arguments)
        if names.intersection(arguments):
            raise SchemaError("mask argument collides with provider lowering")
        if "custom_mask" in dict(lowered.validated_input_views):
            raise SchemaError("custom_mask input view collides with provider lowering")
        fragment = lower_attention_mask_arguments(
            active_plan.framework_plan, self._fragment.inspected,
            self._operation, spec, self._inspector)
        mask_view = fragment.inspected.view
        # Borrowed mask contents must remain unchanged, even if an operation's
        # general access policy permits output/query aliasing.
        for name in lowered.mutable_argument_names:
            view = self._inspector.to_view(arguments[name], name=name, writable=True)
            if not isinstance(view, TensorView):
                raise TypeError("mutable argument inspector must return TensorView")
            if view.device != mask_view.device:
                raise SchemaError("mutable argument and mask must share the planned device")
            if view.overlaps(mask_view):
                raise SchemaError("mutable argument cannot alias the borrowed custom mask")
        result = replace(
            lowered,
            keyword_arguments=lowered.keyword_arguments + fragment.keyword_arguments,
            validated_input_views=lowered.validated_input_views + (("custom_mask", mask_view),),
            retained_resources=lowered.retained_resources + (fragment,),
        )
        return validate_attention_lowered_operator_call(self._operation, self._binding, result)
