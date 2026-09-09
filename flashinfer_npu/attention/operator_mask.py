"""Private plan-bound mask metadata; no packing, copying or provider activation.

Payload descriptors retain borrowed owners only. They are not launch authority,
device leases, immutable snapshots, or evidence of provider mask support.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

from flashinfer_npu.runtime import SchemaError

from .planner import AttentionFrameworkPlan
from .schema import CustomMaskSpec, _as_integer, _as_int_tuple, _canonical_hash


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
