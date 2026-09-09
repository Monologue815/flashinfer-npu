# Provider call resource retention

## Scope

`attention.operator_retention` is a private framework building block for keeping
opaque call arguments and `retained_resources` alive until execution has finished.
It imports no device package, creates no NPU event and performs no synchronization.
Public wrappers do not configure event recording automatically. The existing result completion
receipt proves tensor metadata consistency, not asynchronous device completion.

The integrating runtime must own one `AttentionOperatorCallRetention` registry
across replans and executor replacement. Replacing a plan must not replace or
discard a registry that still contains pending calls. The registry's lifetime is
an explicit integration obligation; garbage collection is not a shutdown protocol.

## Invocation lifecycle

1. Retain the complete lowered call before an already-authorized executor can
   submit work. This includes the separate owner of a borrowed mask.
2. Assign a unique invocation token containing provider, operation and active-plan
   identity. Repeated runs of an identical plan receive different tokens.
3. After invocation, record an event covering the last queued resource use, on
   the actual execution stream. Also attempt this after an invocation failure,
   because some work may already have been submitted.
4. Bind that event to the exact invocation token. The same event object cannot
   be attached to two pending calls, and an attached event cannot be replaced.
5. Poll at a runtime collection point. Only a strict boolean `True` from the matching event releases
   the registry's references. Calls can complete out of order.

`execute_attention_retained_call(registry, call, invoke, recorder)` combines the
first four steps and returns the unchanged executor result plus the private token.
`invoke` must be an already-authorized execution function; this helper performs no
dispatch, package/signature authorization or operator selection. The token must
not become a model-facing `run()` parameter or change its return convention.

The injected recorder implements `record(token)`. Its event exposes the same
`token` and a non-blocking `query()` method. The implementation is responsible for
correct device/stream ordering and trustworthy completion proof. Matching Python
metadata alone cannot prove that an event covers the device work. Host synthetic
events can exercise the lifecycle but are not evidence of NPU execution safety.

## Failure and recovery

An invocation or recording failure raises `AttentionRetainedInvocationError`,
which exposes the retained token and preserves both errors when both occurred.
No execution retry or resource release happens implicitly. An invocation failure
with a successfully recorded event can still be polled normally. If recording
fails, the call remains retained without an event; the integration must later
supply a valid, correctly ordered recovery event through `bind_event()`.

Query failure, a non-boolean response, or event identity drift before or after
query leaves the call retained. Polling must never infer completion from elapsed
time, Python return, an output metadata receipt, or a new plan. There is no
force-release API. `close()` rejects any pending call and prevents new retention
after the registry becomes empty and closes successfully.

An unplanned runtime fork preserves the configured recorder but owns a fresh,
empty retention registry. This supports plan-equivalent workspace probes without
giving them control of the original runtime's pending calls. If an integration
executes calls through such a fork, it must drain that fork's registry separately;
the recorder must correctly record each supplied invocation token.

External queries run outside the registry lock so independent calls can progress.
Concurrent/reentrant polling of the same invocation is rejected. Completed tokens
are removed rather than retained indefinitely as history; polling a removed token
is an error. Diagnostic `pending_tokens` contains no tensor payloads or owners.

`collect_completed()` makes one bounded, non-waiting pass over the pending-token
snapshot. It skips calls with no event, calls already being queried, and tokens
another poll has removed. Calls added during the pass are left for a subsequent
pass. An ordinary query exception does not stop independent completed calls from
being released. The result reports released and remaining tokens, unrecorded
tokens, and per-token failure text. It retains no event, call, exception or traceback
objects. Interrupts are not converted into ordinary query failures.

## Remaining integration responsibilities

### Internal runtime connection

`AttentionOperatorRuntime` and `AttentionOperatorBatchRuntime` accept an optional
`completion_event_recorder` during internal construction. This is not a public
wrapper argument, a `plan()` option or a model-facing `run()` input. When supplied,
the runtime retains every successfully lowered call before invoking its selected,
already-bound executor. The helper's token stays internal; `run()` still returns
the original output or output/LSE convention. Result validation and success
receipt publication happen after event recording, so a result-validation failure
cannot remove the pending invocation.

Configured runtimes collect previously completed calls during `plan()` after
canonical plan validation and during `run()` before new submission. Successful
collection does not change the output/LSE return convention or expose tokens to
the model caller. Unfinished and unrecorded calls stay retained without waiting.
Query failures raise `AttentionCallCompletionCollectionError` with the collection
report and prevent that new plan/submission from proceeding. The active plan
remains usable after recovery; a failed `run()` attempt clears last-run success
diagnostics as usual. A failed `plan()` preserves them. Collection may still have
safely released other completed calls before reporting a query failure.

Each runtime owns a stable `call_retention` registry. Neither successful or failed
planning, executor replacement nor clearing last-run diagnostics replaces it.
Integrations use this internal registry to inspect pending tokens, poll events or
bind recovery proof. Polling does not rewrite numerical/result completion receipts.
Last-call diagnostics may retain an additional reference even after the registry
has released its own reference; this is safe extra retention, not event completion.

Without a recorder, existing calls with no `retained_resources` keep their prior
execution path. Calls carrying explicit retained owners are rejected before
execution, because the runtime has no proof that releasing those owners on Python
return is safe. This guard does not certify asynchronous lifetime safety for older
provider paths; it prevents new owner-dependent integrations from silently using
them. With a recorder configured, closing the empty registry prevents subsequent
submissions. Closing while pending calls exist remains an error.

### Device and public-wrapper integration

The runtime must preserve the registry, supply correctly ordered event recording,
and drain it before teardown. Entry-point collection is not background polling:
an idle wrapper can retain its final calls until an explicit collection or a
later `plan()`/`run()`. Teardown must not rely on another call arriving. It must
also retain returned/output allocations where
the provider's lifetime contract requires them. Registry removal releases only
its own references; other calls, diagnostics or exception tracebacks may still
hold references. This mechanism does not establish immutable mask contents,
allocator generations, device leases, graph-capture safety or numerical accuracy.
Those are separate contracts, and the public custom-mask provider path remains
closed until its full preparation and lifetime integration is complete.
