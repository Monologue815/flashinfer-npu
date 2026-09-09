"""Private plan-bound mask metadata; no packing, copying or provider activation.

Payload descriptors retain borrowed owners only. They are not launch authority,
device leases, immutable snapshots, or evidence of provider mask support.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

from flashinfer_npu.runtime import SchemaError

from .operator_run import AttentionOperatorTensorMetadataInspector
from .planner import AttentionFrameworkPlan
from .schema import CustomMaskSpec, _as_integer, _as_int_tuple, _canonical_hash
from .tensor_contract import TensorView


@dataclass(frozen=True)
class AttentionMaskPlanMetadata:
    framework_plan_fingerprint: str
    admission_fingerprint: str
    plan_generation: int
    mask_spec: CustomMaskSpec
    logical_element_indptr: Tuple[int, ...]
    packed_byte_indptr: Tuple[int, ...]

    def __post_init__(self):
        for name in ("framework_plan_fingerprint", "admission_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError("%s must be lowercase SHA-256" % name)
        generation = _as_integer("plan_generation", self.plan_generation)
        if generation < 1:
            raise SchemaError("mask plan_generation must be positive")
        object.__setattr__(self, "plan_generation", generation)
        if not isinstance(self.mask_spec, CustomMaskSpec):
            raise TypeError("mask_spec must be CustomMaskSpec")
        for name in ("logical_element_indptr", "packed_byte_indptr"):
            values = _as_int_tuple(name, getattr(self, name))
            if len(values) < 2 or values[0] != 0 or any(
                a > b for a, b in zip(values, values[1:])
            ):
                raise SchemaError("%s must be monotone offsets starting at zero" % name)
            object.__setattr__(self, name, values)
        logical = self.logical_element_indptr
        packed = self.packed_byte_indptr
        if len(logical) != len(packed) or any(
            packed[i + 1] - packed[i] != (logical[i + 1] - logical[i] + 7) // 8
            for i in range(len(logical) - 1)
        ):
            raise SchemaError("packed byte offsets must preserve per-request padding")
        expected = packed[-1] if self.mask_spec.packed else logical[-1]
        if self.mask_spec.numel != expected:
            raise SchemaError("mask payload element count does not match segment offsets")

    @classmethod
    def from_plan(cls, plan: AttentionFrameworkPlan):
        if not isinstance(plan, AttentionFrameworkPlan):
            raise TypeError("plan must be AttentionFrameworkPlan")
        plan.spec.validate_metadata(plan.metadata)
        if plan.spec.custom_mask is None:
            raise SchemaError("mask resources require a custom-mask plan")
        logical, packed = [0], [0]
        for length in plan.spec._mask_segment_sizes(plan.metadata):
            logical.append(logical[-1] + length)
            packed.append(packed[-1] + (length + 7) // 8)
        return cls(plan.fingerprint, plan.admission_fingerprint, plan.generation,
                   plan.spec.custom_mask, tuple(logical), tuple(packed))

    def validate_plan(self, plan: AttentionFrameworkPlan):
        if self != type(self).from_plan(plan):
            raise SchemaError("mask resource metadata does not match the active plan")

    def to_dict(self):
        return {
            "schema_version": 1,
            "framework_plan_fingerprint": self.framework_plan_fingerprint,
            "admission_fingerprint": self.admission_fingerprint,
            "plan_generation": self.plan_generation,
            "mask_spec": self.mask_spec.to_dict(),
            "logical_element_indptr": list(self.logical_element_indptr),
            "packed_byte_indptr": list(self.packed_byte_indptr),
        }

    @property
    def fingerprint(self):
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True, eq=False)
class AttentionMaskPlanResource:
    """Retain a borrowed payload and owner without inspecting either object.

    The owner must keep the payload immutable and alive through all uses. This
    object alone does not enforce device lifetime or validate tensor metadata.
    Resource equality is identity-based, never equality of arbitrary payloads.
    """

    metadata: AttentionMaskPlanMetadata
    payload: Any = field(repr=False)
    owner: Any = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.metadata, AttentionMaskPlanMetadata):
            raise TypeError("metadata must be AttentionMaskPlanMetadata")
        if self.payload is None or self.owner is None:
            raise SchemaError("mask resource requires a payload and retained owner")

    def validate_plan(self, plan: AttentionFrameworkPlan):
        self.metadata.validate_plan(plan)


def _inspection_requirements(expected_device, required_alignment):
    if not isinstance(expected_device, str) or not expected_device:
        raise SchemaError("mask expected_device must be a non-empty string")
    alignment = _as_integer("mask required_alignment", required_alignment)
    if alignment < 1 or alignment & (alignment - 1):
        raise SchemaError("mask required_alignment must be a positive power of two")
    return expected_device, alignment


@dataclass(frozen=True, eq=False)
class AttentionInspectedMaskPlanResource:
    """A retained source and its validated metadata snapshot, not launch authority."""

    resource: AttentionMaskPlanResource = field(repr=False)
    view: TensorView
    expected_device: str
    required_alignment: int = 1

    def __post_init__(self):
        if not isinstance(self.resource, AttentionMaskPlanResource):
            raise TypeError("resource must be AttentionMaskPlanResource")
        if not isinstance(self.view, TensorView):
            raise TypeError("mask tensor metadata inspector must return TensorView")
        device, alignment = _inspection_requirements(self.expected_device, self.required_alignment)
        object.__setattr__(self, "required_alignment", alignment)
        spec = self.resource.metadata.mask_spec
        if self.view.shape != (spec.numel,):
            raise SchemaError("mask view must be a rank-1 view with the planned element count")
        if self.view.dtype != spec.dtype:
            raise SchemaError("mask view dtype does not match the planned mask encoding")
        if self.view.device != device:
            raise SchemaError("mask view device does not match the expected device")
        if not self.view.is_contiguous:
            raise SchemaError("mask view must be contiguous; implicit materialization is forbidden")
        self.view.require_alignment(alignment, "mask view")

    def validate_plan(self, plan: AttentionFrameworkPlan):
        self.resource.validate_plan(plan)


def inspect_attention_mask_plan_resource(
    plan: AttentionFrameworkPlan,
    resource: AttentionMaskPlanResource,
    inspector: AttentionOperatorTensorMetadataInspector,
    expected_device: str,
    *,
    required_alignment: int = 1,
) -> AttentionInspectedMaskPlanResource:
    """Inspect one already-flattened source without reading or transforming data.

    Unpacked sources remain bool arrays; packed sources remain uint8 arrays.
    A successful check proves metadata consistency, not mask values, immutable
    storage, completion ordering, or a provider's supported representation.
    """
    if not isinstance(resource, AttentionMaskPlanResource):
        raise TypeError("resource must be AttentionMaskPlanResource")
    resource.validate_plan(plan)
    device, alignment = _inspection_requirements(expected_device, required_alignment)
    if not isinstance(inspector, AttentionOperatorTensorMetadataInspector):
        raise TypeError("inspector must implement AttentionOperatorTensorMetadataInspector")
    view = inspector.to_view(resource.payload, name="custom_mask", writable=False)
    return AttentionInspectedMaskPlanResource(resource, view, device, alignment)
