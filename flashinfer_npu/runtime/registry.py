"""Deterministic kernel registry and explainable dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .schema import Backend, DeviceCapability, KernelDescriptor, WorkloadSpec


class DispatchError(RuntimeError):
    """Raised when no kernel satisfies a workload and dispatch policy."""


@dataclass(frozen=True)
class CandidateReport:
    kernel_id: str
    backend: Backend
    accepted: bool
    reasons: Tuple[str, ...]


class KernelRegistry:
    """The single source of truth for runtime-visible kernels."""

    def __init__(self, descriptors: Iterable[KernelDescriptor] = ()) -> None:
        self._descriptors: Dict[str, KernelDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: KernelDescriptor) -> None:
        if descriptor.kernel_id in self._descriptors:
            raise ValueError("duplicate kernel_id: %s" % descriptor.kernel_id)
        self._descriptors[descriptor.kernel_id] = descriptor

    def descriptors(self, op: Optional[str] = None) -> Tuple[KernelDescriptor, ...]:
        values = self._descriptors.values()
        if op is not None:
            values = (descriptor for descriptor in values if descriptor.op == op)
        return tuple(sorted(values, key=lambda descriptor: descriptor.kernel_id))

    def explain(
        self,
        workload: WorkloadSpec,
        capability: DeviceCapability,
        backend: Union[str, Backend] = "auto",
        allow_reference: bool = False,
    ) -> Tuple[CandidateReport, ...]:
        requested_backend = None if backend == "auto" else Backend(backend)
        reports: List[CandidateReport] = []
        for descriptor in self.descriptors(op=workload.op):
            reasons = list(
                descriptor.constraints.unsupported_reasons(workload, capability)
            )
            if requested_backend is not None and descriptor.backend != requested_backend:
                reasons.append("excluded by backend policy %s" % requested_backend.value)
            if descriptor.backend == Backend.REFERENCE and not allow_reference:
                reasons.append("reference backend requires explicit opt-in")
            reports.append(
                CandidateReport(
                    kernel_id=descriptor.kernel_id,
                    backend=descriptor.backend,
                    accepted=not reasons,
                    reasons=tuple(reasons),
                )
            )
        return tuple(reports)

    def select(
        self,
        workload: WorkloadSpec,
        capability: DeviceCapability,
        backend: Union[str, Backend] = "auto",
        allow_reference: bool = False,
        tuned_kernel_ids: Sequence[str] = (),
    ) -> KernelDescriptor:
        reports = self.explain(
            workload,
            capability,
            backend=backend,
            allow_reference=allow_reference,
        )
        accepted_ids = {report.kernel_id for report in reports if report.accepted}
        for kernel_id in tuned_kernel_ids:
            if kernel_id in accepted_ids:
                return self._descriptors[kernel_id]

        accepted = [self._descriptors[kernel_id] for kernel_id in accepted_ids]
        if accepted:
            return sorted(
                accepted,
                key=lambda descriptor: (-descriptor.priority, descriptor.kernel_id),
            )[0]

        details = "; ".join(
            "%s: %s" % (report.kernel_id, ", ".join(report.reasons))
            for report in reports
        )
        if not details:
            details = "no descriptors registered for op"
        raise DispatchError(
            "no kernel for op=%s backend=%s (%s)"
            % (workload.op, str(backend), details)
        )

