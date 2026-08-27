"""Project diagnostics and manifest tooling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .attention import (
    AttentionAccuracyExpectationError,
    AttentionProtocolTrace,
    AttentionProtocolTraceCorpus,
    AttentionTrace,
    AttentionTraceCorpus,
    AttentionTraceMismatchError,
    DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS,
    build_framework_attention_corpus,
    build_attention_accuracy_corpus,
    audit_attention_public_interface,
    framework_attention_coverage_policy,
    load_attention_capability_manifest,
    validate_attention_kernel_bindings,
)
from .attention.json_envelope import decode_attention_json
from .config import format_config
from .parity import load_packaged_manifest
from .runtime import SchemaError, load_kernel_manifest


def packaged_kernel_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "kernels" / "registry.json"


def packaged_attention_capability_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "attention_capabilities.json"


def _show_config(_args: argparse.Namespace) -> int:
    print(format_config())
    return 0


def _parity_report(args: argparse.Namespace) -> int:
    manifest = load_packaged_manifest(args.scope)
    print(manifest.report())
    if args.require_complete and not manifest.is_complete:
        return 1
    return 0


def _list_kernels(_args: argparse.Namespace) -> int:
    descriptors = load_kernel_manifest(packaged_kernel_manifest_path())
    profiles = load_attention_capability_manifest(
        packaged_attention_capability_manifest_path()
    )
    validate_attention_kernel_bindings(profiles, descriptors)
    if not descriptors:
        print("No runnable kernels are registered.")
        return 0
    for descriptor in descriptors:
        print(
            "%s\t%s\t%s\t%s"
            % (
                descriptor.kernel_id,
                descriptor.op,
                descriptor.backend.value,
                (
                    descriptor.artifact.locator
                    if descriptor.artifact is not None
                    else "reference"
                ),
            )
        )
    return 0


def _attention_capabilities(_args: argparse.Namespace) -> int:
    profiles = load_attention_capability_manifest(
        packaged_attention_capability_manifest_path()
    )
    if not profiles:
        print("No Attention backend capability profiles are registered.")
        return 0
    for profile in profiles:
        print(
            "%s\t%s\t%s\t%s\t%d"
            % (
                profile.profile_id,
                profile.backend.value,
                profile.status.value,
                profile.environment.soc_version,
                len(profile.rules),
            )
        )
    return 0


def _attention_interface(_args: argparse.Namespace) -> int:
    try:
        report = audit_attention_public_interface()
    except SchemaError as error:
        print("attention interface audit failed: %s" % error, file=sys.stderr)
        return 2
    print(report.format())
    return 0


def _attention_replay(args: argparse.Namespace) -> int:
    path = Path(args.trace)
    try:
        size = path.stat().st_size
        if size > args.max_bytes:
            raise SchemaError(
                "Attention trace exceeds --max-bytes (%d)" % args.max_bytes
            )
        limits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS.with_max_bytes(
            args.max_bytes
        )
        trace = AttentionTrace.from_json(
            path.read_text(encoding="utf-8"), limits=limits
        )
        result = trace.replay(
            validate_expected=not args.no_validate,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (OSError, SchemaError, AttentionTraceMismatchError, ValueError) as error:
        print("attention replay failed: %s" % error, file=sys.stderr)
        return 2
    print("trace_fingerprint : %s" % trace.fingerprint)
    print("input_fingerprint : %s" % trace.input_fingerprint)
    print("mode              : %s" % trace.spec.mode.value)
    print(
        "output            : shape=%r dtype=%s"
        % (result.output.shape, result.output.dtype)
    )
    if result.lse is not None:
        print("lse               : shape=%r dtype=%s" % (result.lse.shape, result.lse.dtype))
    print(
        "validation        : %s"
        % ("skipped" if args.no_validate else "passed")
    )
    return 0


def _load_attention_corpus(path_value, max_bytes: int) -> AttentionTraceCorpus:
    if path_value is None:
        return build_framework_attention_corpus()
    path = Path(path_value)
    size = path.stat().st_size
    if size > max_bytes:
        raise SchemaError("Attention corpus exceeds --max-bytes (%d)" % max_bytes)
    limits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS.with_max_bytes(max_bytes)
    return AttentionTraceCorpus.from_json(
        path.read_text(encoding="utf-8"), limits=limits
    )


def _attention_coverage(args: argparse.Namespace) -> int:
    try:
        corpus = _load_attention_corpus(args.corpus, args.max_bytes)
        if args.replay:
            corpus.replay_all(atol=args.atol, rtol=args.rtol)
        report = framework_attention_coverage_policy().evaluate(corpus)
    except (OSError, SchemaError, AttentionTraceMismatchError, ValueError) as error:
        print("attention coverage failed: %s" % error, file=sys.stderr)
        return 2
    print(report.format())
    print("  replay: %s" % ("passed" if args.replay else "not requested"))
    if args.require_complete and not report.is_complete:
        return 1
    return 0


def _attention_corpus(_args: argparse.Namespace) -> int:
    corpus = build_framework_attention_corpus()
    print(corpus.to_json(indent=2 if _args.pretty else None))
    return 0


def _attention_accuracy(_args: argparse.Namespace) -> int:
    corpus = build_attention_accuracy_corpus()
    try:
        reports = corpus.replay_all()
    except (
        AttentionAccuracyExpectationError,
        AttentionTraceMismatchError,
        SchemaError,
        ValueError,
    ) as error:
        print("attention accuracy failed: %s" % error, file=sys.stderr)
        return 2
    accepted = sum(
        report.quantization_within_budget for _case_id, report in reports
    )
    rejected = len(reports) - accepted
    print("Attention quantization accuracy")
    print("  corpus:   %s" % corpus.name)
    print("  cases:    %d" % len(reports))
    print("  accepted: %d" % accepted)
    print("  rejected: %d" % rejected)
    print("  replay:   passed")
    print("  fingerprint: %s" % corpus.fingerprint)
    return 0


def _attention_accuracy_corpus(args: argparse.Namespace) -> int:
    corpus = build_attention_accuracy_corpus()
    print(corpus.to_json(indent=2 if args.pretty else None))
    return 0


def _attention_protocol_validate(args: argparse.Namespace) -> int:
    path = Path(args.protocol)
    try:
        size = path.stat().st_size
        if size > args.max_bytes:
            raise SchemaError(
                "Attention protocol input exceeds --max-bytes (%d)" % args.max_bytes
            )
        limits = DEFAULT_ATTENTION_JSON_ENVELOPE_LIMITS.with_max_bytes(args.max_bytes)
        decoded, _ = decode_attention_json(
            path.read_text(encoding="utf-8"), limits=limits
        )
        if not isinstance(decoded, dict):
            raise SchemaError("Attention protocol JSON must contain an object")
        if decoded.get("kind") == "attention_protocol_corpus":
            corpus = AttentionProtocolTraceCorpus.from_dict(decoded)
            print("kind        : corpus")
            print("fingerprint : %s" % corpus.fingerprint)
            print("corpus_id   : %s" % corpus.corpus_id)
            print("cases       : %d" % len(corpus.cases))
            print(
                "routes      : single_jit=%d provider=%d"
                % (
                    corpus.route_counts["single_jit"],
                    corpus.route_counts["provider"],
                )
            )
        else:
            trace = AttentionProtocolTrace.from_dict(decoded)
            print("kind        : trace")
            print("fingerprint : %s" % trace.fingerprint)
            print("trace_id    : %s" % trace.trace_id)
            print("route       : %s" % trace.route.value)
            print("events      : %d" % len(trace.events))
        print("validation  : passed")
    except (OSError, SchemaError, TypeError, ValueError) as error:
        print("attention protocol validation failed: %s" % error, file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashinfer-npu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-config", help="show local environment")
    show_config.set_defaults(handler=_show_config)

    parity = subparsers.add_parser(
        "parity-report", help="summarize FlashInfer API parity"
    )
    parity.add_argument(
        "--scope",
        choices=("attention", "all"),
        default="attention",
        help="API domain to report (default: attention)",
    )
    parity.add_argument(
        "--require-complete",
        action="store_true",
        help="return non-zero unless stable API parity is complete",
    )
    parity.set_defaults(handler=_parity_report)

    kernels = subparsers.add_parser(
        "list-kernels", help="list runtime-visible kernel descriptors"
    )
    kernels.set_defaults(handler=_list_kernels)

    capabilities = subparsers.add_parser(
        "attention-capabilities",
        help="list evidence-bearing Attention backend capability profiles",
    )
    capabilities.set_defaults(handler=_attention_capabilities)

    interface = subparsers.add_parser(
        "attention-interface",
        help="audit the live model-facing Attention call interface",
    )
    interface.set_defaults(handler=_attention_interface)

    replay = subparsers.add_parser(
        "attention-replay", help="replay a versioned Host Attention trace"
    )
    replay.add_argument("trace", help="path to an Attention trace JSON file")
    replay.add_argument("--atol", type=float, default=1e-6)
    replay.add_argument("--rtol", type=float, default=1e-6)
    replay.add_argument(
        "--no-validate",
        action="store_true",
        help="execute without comparing recorded oracle output",
    )
    replay.add_argument(
        "--max-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="reject trace files larger than this limit",
    )
    replay.set_defaults(handler=_attention_replay)

    coverage = subparsers.add_parser(
        "attention-coverage",
        help="report coverage cells for an Attention conformance corpus",
    )
    coverage.add_argument(
        "corpus",
        nargs="?",
        help="corpus JSON path; omit to use the built-in framework corpus",
    )
    coverage.add_argument("--replay", action="store_true")
    coverage.add_argument("--require-complete", action="store_true")
    coverage.add_argument("--atol", type=float, default=1e-6)
    coverage.add_argument("--rtol", type=float, default=1e-6)
    coverage.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    coverage.set_defaults(handler=_attention_coverage)

    corpus = subparsers.add_parser(
        "attention-corpus",
        help="emit the built-in versioned Attention conformance corpus",
    )
    corpus.add_argument("--pretty", action="store_true")
    corpus.set_defaults(handler=_attention_corpus)

    accuracy = subparsers.add_parser(
        "attention-accuracy",
        help="replay the built-in paired quantization-accuracy corpus",
    )
    accuracy.set_defaults(handler=_attention_accuracy)

    accuracy_corpus = subparsers.add_parser(
        "attention-accuracy-corpus",
        help="emit the built-in paired quantization-accuracy corpus",
    )
    accuracy_corpus.add_argument("--pretty", action="store_true")
    accuracy_corpus.set_defaults(handler=_attention_accuracy_corpus)

    protocol = subparsers.add_parser(
        "attention-protocol-validate",
        help="validate an Attention lifecycle trace or protocol corpus",
    )
    protocol.add_argument("protocol", help="path to protocol trace/corpus JSON")
    protocol.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    protocol.set_defaults(handler=_attention_protocol_validate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
