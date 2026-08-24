"""Immutable host-side execution plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

from .registry import KernelRegistry
from .schema import Backend, DeviceCapability, KernelDescriptor, WorkloadSpec


@dataclass(frozen=True)
class Plan:
    descriptor: KernelDescriptor
    workspace_bytes: int
    int_workspace_bytes: int
    workload_fingerprint: str
    capability_fingerprint: str
    plan_schema_version: int = 1

    @property
    def kernel_id(self) -> str:
        return self.descriptor.kernel_id

    @property
    def backend(self) -> Backend:
        return self.descriptor.backend

    def validate_runtime(
        self, workload: WorkloadSpec, capability: DeviceCapability
    ) -> None:
        if workload.fingerprint != self.workload_fingerprint:
            raise ValueError("runtime workload does not match the planned workload")
        if capability.fingerprint != self.capability_fingerprint:
            raise ValueError("runtime capability does not match the planned device")


def build_plan(
    registry: KernelRegistry,
    workload: WorkloadSpec,
    capability: DeviceCapability,
    backend: Union[str, Backend] = "auto",
    allow_reference: bool = False,
    tuned_kernel_ids: Sequence[str] = (),
) -> Plan:
    descriptor = registry.select(
        workload,
        capability,
        backend=backend,
        allow_reference=allow_reference,
        tuned_kernel_ids=tuned_kernel_ids,
    )
    return Plan(
        descriptor=descriptor,
        workspace_bytes=descriptor.workspace.size_for(workload),
        int_workspace_bytes=descriptor.int_workspace.size_for(workload),
        workload_fingerprint=workload.fingerprint,
        capability_fingerprint=capability.fingerprint,
    )
