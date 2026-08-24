"""Framework-level adapters for Attention implementations supplied by packages.

This SPI is intentionally separate from ``launcher_provider``.  The latter
resolves a selected low-level kernel artifact; this module only discovers an
installed operator package and publishes its capability profiles.  It performs
no package imports or operator calls by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple, runtime_checkable

from flashinfer_npu.runtime import Backend, SchemaError

from .capability import AttentionBackendCapabilityProfile
from .dispatch import AttentionDispatchReceipt


ATTENTION_OPERATOR_PROVIDER_VERSION = 1
CANN_ATTENTION_PROVIDER_ID = "cann"
FLASH_ATTENTION_NPU_PROVIDER_ID = "flash_attention_npu"

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class AttentionOperatorProviderSelectionError(RuntimeError):
    """Raised when a dispatch receipt has no available provider owner."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_provider_id(value: str) -> str:
    provider_id = str(value)
    if not _PROVIDER_ID.fullmatch(provider_id):
        raise SchemaError("invalid Attention operator provider_id")
    return provider_id


@dataclass(frozen=True)
class AttentionOperatorProviderProbe:
    """Side-effect-free availability result returned by one package adapter."""

    provider_id: str
    adapter_version: str
    available: bool
    package_versions: Tuple[Tuple[str, str], ...] = ()
    unavailable_reasons: Tuple[str, ...] = ()
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention operator provider version")
        object.__setattr__(self, "provider_id", _validate_provider_id(self.provider_id))
        if not str(self.adapter_version):
            raise SchemaError("Attention operator adapter_version must be non-empty")
        packages = tuple((str(name), str(version)) for name, version in self.package_versions)
        if any(not name or not version for name, version in packages):
            raise SchemaError("operator package names and versions must be non-empty")
        if len({name for name, _ in packages}) != len(packages):
            raise SchemaError("operator package names cannot contain duplicates")
        object.__setattr__(self, "package_versions", tuple(sorted(packages)))
        reasons = tuple(str(item) for item in self.unavailable_reasons)
        if any(not item for item in reasons) or len(set(reasons)) != len(reasons):
            raise SchemaError("operator provider rejection reasons must be unique")
        if bool(self.available) == bool(reasons):
            raise SchemaError(
                "available provider must have no rejection reasons and unavailable "
                "provider must explain why"
            )
        object.__setattr__(self, "unavailable_reasons", reasons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "adapter_version": self.adapter_version,
            "available": self.available,
            "package_versions": [list(item) for item in self.package_versions],
            "unavailable_reasons": list(self.unavailable_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionOperatorProviderProbe":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("AttentionOperatorProviderProbe fields are invalid")
        try:
            data["package_versions"] = tuple(tuple(item) for item in data["package_versions"])
            data["unavailable_reasons"] = tuple(data["unavailable_reasons"])
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("AttentionOperatorProviderProbe fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@runtime_checkable
class AttentionOperatorProvider(Protocol):
    """Minimal package-adapter SPI frozen before any real package integration."""

    provider_id: str

    def probe(self) -> AttentionOperatorProviderProbe:
        """Report availability without initializing a device or running an op."""

    def capability_profiles(self) -> Sequence[AttentionBackendCapabilityProfile]:
        """Declare exact Attention capabilities owned by this adapter."""


@dataclass(frozen=True)
class AttentionOperatorProviderRecord:
    """One validated discovery result and its capability ownership."""

    probe: AttentionOperatorProviderProbe
    profiles: Tuple[AttentionBackendCapabilityProfile, ...]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention operator provider record version")
        if not isinstance(self.probe, AttentionOperatorProviderProbe):
            raise TypeError("probe must be AttentionOperatorProviderProbe")
        profiles = tuple(self.profiles)
        if not profiles:
            raise SchemaError("operator provider must declare at least one profile")
        if len({item.profile_id for item in profiles}) != len(profiles):
            raise SchemaError("operator provider profile_id values must be unique")
        if any(item.backend == Backend.REFERENCE for item in profiles):
            raise SchemaError("operator provider cannot own reference profiles")
        object.__setattr__(self, "profiles", profiles)

    @property
    def provider_id(self) -> str:
        return self.probe.provider_id

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "probe_fingerprint": self.probe.fingerprint,
                "profile_fingerprints": [item.fingerprint for item in self.profiles],
            }
        )


class AttentionOperatorProviderRegistry:
    """Explicit, lazy registry for CANN and third-party Attention adapters."""

    def __init__(self, providers: Sequence[AttentionOperatorProvider] = ()) -> None:
        self._providers: Dict[str, AttentionOperatorProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: AttentionOperatorProvider) -> None:
        if not isinstance(provider, AttentionOperatorProvider):
            raise TypeError("provider must implement AttentionOperatorProvider")
        provider_id = _validate_provider_id(provider.provider_id)
        if provider_id in self._providers:
            raise SchemaError("duplicate Attention operator provider_id %r" % provider_id)
        self._providers[provider_id] = provider

    @property
    def provider_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._providers))

    def discover(self) -> Tuple[AttentionOperatorProviderRecord, ...]:
        """Probe registered adapters and validate global profile ownership.

        Registration alone has no probe/import side effect.  A real adapter's
        ``probe`` must remain device-initialization-free; selection and execution
        are deliberately outside checkpoint 004.
        """

        records = []
        owned_profile_ids = set()
        for provider_id in self.provider_ids:
            provider = self._providers[provider_id]
            probe = provider.probe()
            if not isinstance(probe, AttentionOperatorProviderProbe):
                raise TypeError("operator provider probe returned an invalid type")
            if probe.provider_id != provider_id:
                raise SchemaError("operator provider probe identity mismatch")
            record = AttentionOperatorProviderRecord(
                probe, tuple(provider.capability_profiles())
            )
            duplicates = owned_profile_ids.intersection(
                item.profile_id for item in record.profiles
            )
            if duplicates:
                raise SchemaError(
                    "Attention capability profile has multiple provider owners: %s"
                    % sorted(duplicates)[0]
                )
            owned_profile_ids.update(item.profile_id for item in record.profiles)
            records.append(record)
        return tuple(records)


@dataclass(frozen=True)
class AttentionOperatorProviderCandidateReport:
    """Explain why one discovered provider can or cannot own a dispatch."""

    provider_id: str
    probe_fingerprint: str
    profile_ids: Tuple[str, ...]
    accepted: bool
    reasons: Tuple[str, ...]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention operator candidate version")
        object.__setattr__(self, "provider_id", _validate_provider_id(self.provider_id))
        if len(self.probe_fingerprint) != 64 or any(
            item not in "0123456789abcdef" for item in self.probe_fingerprint
        ):
            raise SchemaError("operator provider probe fingerprint must be SHA-256")
        profile_ids = tuple(str(item) for item in self.profile_ids)
        if not profile_ids or any(not item for item in profile_ids):
            raise SchemaError("operator candidate profile_ids must be non-empty")
        if len(set(profile_ids)) != len(profile_ids):
            raise SchemaError("operator candidate profile_ids must be unique")
        object.__setattr__(self, "profile_ids", profile_ids)
        reasons = tuple(str(item) for item in self.reasons)
        if any(not item for item in reasons):
            raise SchemaError("operator candidate reasons must be non-empty strings")
        if self.accepted == bool(reasons):
            raise SchemaError("accepted operator provider must have no reasons")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class AttentionOperatorProviderBindingReport:
    """Provider-level explanation layered on an evidence-bearing receipt."""

    dispatch_receipt_fingerprint: str
    requested_provider: str
    candidates: Tuple[AttentionOperatorProviderCandidateReport, ...]
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention operator binding report version")
        if len(self.dispatch_receipt_fingerprint) != 64 or any(
            item not in "0123456789abcdef"
            for item in self.dispatch_receipt_fingerprint
        ):
            raise SchemaError("dispatch receipt fingerprint must be SHA-256")
        if self.requested_provider != "auto":
            _validate_provider_id(self.requested_provider)
        candidates = tuple(self.candidates)
        if len({item.provider_id for item in candidates}) != len(candidates):
            raise SchemaError("operator binding candidates must have unique provider_id")
        object.__setattr__(self, "candidates", candidates)

    @property
    def accepted(self) -> Tuple[AttentionOperatorProviderCandidateReport, ...]:
        return tuple(item for item in self.candidates if item.accepted)


@dataclass(frozen=True)
class AttentionOperatorProviderSelection:
    """Stable binding from an authorized kernel dispatch to its package adapter."""

    provider_id: str
    provider_probe_fingerprint: str
    provider_record_fingerprint: str
    dispatch_receipt_fingerprint: str
    profile_id: str
    profile_fingerprint: str
    backend: Backend
    requested_provider: str = "auto"
    schema_version: int = ATTENTION_OPERATOR_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTENTION_OPERATOR_PROVIDER_VERSION:
            raise SchemaError("unsupported Attention operator selection version")
        object.__setattr__(self, "provider_id", _validate_provider_id(self.provider_id))
        if self.requested_provider != "auto":
            _validate_provider_id(self.requested_provider)
            if self.requested_provider != self.provider_id:
                raise SchemaError("requested provider does not match selected provider")
        for name in (
            "provider_probe_fingerprint",
            "provider_record_fingerprint",
            "dispatch_receipt_fingerprint",
            "profile_fingerprint",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise SchemaError("%s must be lowercase SHA-256" % name)
        if not self.profile_id:
            raise SchemaError("operator selection profile_id must be non-empty")
        object.__setattr__(self, "backend", Backend(self.backend))
        if self.backend == Backend.REFERENCE:
            raise SchemaError("operator selection cannot bind the reference backend")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "provider_id": self.provider_id,
                "provider_probe_fingerprint": self.provider_probe_fingerprint,
                "provider_record_fingerprint": self.provider_record_fingerprint,
                "dispatch_receipt_fingerprint": self.dispatch_receipt_fingerprint,
                "profile_id": self.profile_id,
                "profile_fingerprint": self.profile_fingerprint,
                "backend": self.backend.value,
                "requested_provider": self.requested_provider,
            }
        )


def _validated_records(
    records: Sequence[AttentionOperatorProviderRecord],
) -> Tuple[AttentionOperatorProviderRecord, ...]:
    values = tuple(records)
    if any(not isinstance(item, AttentionOperatorProviderRecord) for item in values):
        raise TypeError("records must contain AttentionOperatorProviderRecord")
    if len({item.provider_id for item in values}) != len(values):
        raise SchemaError("operator provider records contain duplicate provider_id")
    profile_ids = [profile.profile_id for item in values for profile in item.profiles]
    if len(set(profile_ids)) != len(profile_ids):
        raise SchemaError("operator provider records contain duplicate profile ownership")
    return tuple(sorted(values, key=lambda item: item.provider_id))


def explain_attention_operator_provider_binding(
    records: Sequence[AttentionOperatorProviderRecord],
    receipt: AttentionDispatchReceipt,
    *,
    provider: str = "auto",
) -> AttentionOperatorProviderBindingReport:
    """Explain provider ownership after core plan-driven kernel dispatch.

    Core dispatch remains the authority for plan, capability evidence, kernel,
    workspace and ABI selection.  This layer only resolves the selected profile
    to an available package adapter and applies an optional provider policy.
    """

    if not isinstance(receipt, AttentionDispatchReceipt):
        raise TypeError("receipt must be AttentionDispatchReceipt")
    if provider != "auto":
        provider = _validate_provider_id(provider)
    candidates = []
    for record in _validated_records(records):
        reasons = []
        if not record.probe.available:
            reasons.extend(
                "provider unavailable: %s" % item
                for item in record.probe.unavailable_reasons
            )
        if provider != "auto" and record.provider_id != provider:
            reasons.append("excluded by provider policy %s" % provider)
        owned = next(
            (
                profile
                for profile in record.profiles
                if profile.profile_id == receipt.profile_id
            ),
            None,
        )
        if owned is None:
            reasons.append("does not own selected profile %s" % receipt.profile_id)
        else:
            if owned.fingerprint != receipt.profile_fingerprint:
                reasons.append("selected capability profile fingerprint is stale")
            if owned.backend != receipt.backend:
                reasons.append("selected capability profile backend is stale")
        candidates.append(
            AttentionOperatorProviderCandidateReport(
                provider_id=record.provider_id,
                probe_fingerprint=record.probe.fingerprint,
                profile_ids=tuple(item.profile_id for item in record.profiles),
                accepted=not reasons,
                reasons=tuple(reasons),
            )
        )
    return AttentionOperatorProviderBindingReport(
        dispatch_receipt_fingerprint=receipt.fingerprint,
        requested_provider=provider,
        candidates=tuple(candidates),
    )


def bind_attention_operator_provider(
    records: Sequence[AttentionOperatorProviderRecord],
    receipt: AttentionDispatchReceipt,
    *,
    provider: str = "auto",
) -> AttentionOperatorProviderSelection:
    """Bind exactly one available provider; never fall back to reference."""

    values = _validated_records(records)
    report = explain_attention_operator_provider_binding(
        values, receipt, provider=provider
    )
    if len(report.accepted) != 1:
        details = "; ".join(
            "%s: %s" % (item.provider_id, ", ".join(item.reasons))
            for item in report.candidates
        )
        raise AttentionOperatorProviderSelectionError(
            "no available Attention operator provider owns the dispatch (%s)"
            % (details or "no providers discovered")
        )
    candidate = report.accepted[0]
    record = next(item for item in values if item.provider_id == candidate.provider_id)
    profile = next(
        item for item in record.profiles if item.profile_id == receipt.profile_id
    )
    return AttentionOperatorProviderSelection(
        provider_id=record.provider_id,
        provider_probe_fingerprint=record.probe.fingerprint,
        provider_record_fingerprint=record.fingerprint,
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        backend=profile.backend,
        requested_provider=provider,
    )


__all__ = [
    "ATTENTION_OPERATOR_PROVIDER_VERSION",
    "CANN_ATTENTION_PROVIDER_ID",
    "FLASH_ATTENTION_NPU_PROVIDER_ID",
    "AttentionOperatorProvider",
    "AttentionOperatorProviderBindingReport",
    "AttentionOperatorProviderCandidateReport",
    "AttentionOperatorProviderProbe",
    "AttentionOperatorProviderRecord",
    "AttentionOperatorProviderRegistry",
    "AttentionOperatorProviderSelection",
    "AttentionOperatorProviderSelectionError",
    "bind_attention_operator_provider",
    "explain_attention_operator_provider_binding",
]
