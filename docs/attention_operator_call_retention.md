# Provider call resource retention

## Scope

`attention.operator_retention` is a private framework building block for keeping
opaque call arguments and `retained_resources` alive until execution has finished.
It imports no device package, creates no NPU event and performs no synchronization.
Batch wrappers can obtain event recording from an explicitly installed bootstrap
factory; no device recorder is installed by default. The existing result
completion receipt proves tensor metadata consistency, not asynchronous device completion.

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

### Batch-wrapper bootstrap configuration

`install_attention_operator_runtime_resolvers()` accepts an optional internal
`batch_completion_event_recorder_factory`. Its `create(device=..., mode=...)`
method returns a recorder implementing `record(token)`. Installation validates
the factory interface but does not call it, import device packages, probe
operations or record an event. The factory reference is captured atomically with
the runtime registry generation and excluded from snapshot representation and
comparison; the snapshot is not a serialization format for executable factories.

Classic paged prefill, ragged prefill and paged decode wrappers, plus holistic
`BatchAttention`, call the captured factory during provider-wrapper construction.
The factory receives that wrapper's device and Attention mode. Its result must
provide a callable `record` method. Creation failure prevents construction before
any provider probe; it never falls back to untracked execution. The ordinary
public constructors, `plan()` parameters and `run()` return conventions do not
change. Host reference execution does not request a recorder.

Replanning uses the same recorder and retention registry. Installing a new
factory, or omitting it in a subsequent installation, affects only future
wrappers. It neither replaces existing recorders nor drains their pending calls.
Factory implementations remain process-local integration objects and must not
mutate their behavior incompatibly while captured wrappers are alive.

The same explicit hooks are accepted by the declared registry, provider bundle,
bootstrap-manifest and bootstrap-document installers. They are process-local
arguments, not JSON fields; each installation must explicitly supply them or
future wrappers receive the default unconfigured state. Single-request facades do
not use this batch-only hook: their short-lived runtime needs a separate owner
before asynchronous resource tracking can be connected safely. Neither this hook
nor a matching Python protocol proves stream ordering or authorizes a provider.
The integrating service must keep configured wrappers/runtimes alive and complete
the teardown handshake below. A recorder alone does not enable public masks;
batch prefill additionally requires explicit mask mappings and service ownership.

### Service ownership across wrapper disposal

The same internal installer accepts an optional `batch_runtime_owner`, an
`AttentionBatchRuntimeOwner` from `attention.operator_runtime_owner`. An owner
requires a completion recorder factory. Both references are captured atomically
in the bootstrap snapshot; neither becomes a model-facing constructor or plan
argument. Each provider batch wrapper adopts its newly constructed runtime into
the captured owner before the wrapper becomes available. The owner does not own
the wrapper, so disposing of a wrapper cannot discard its tracked runtime.

The service must hold a strong reference to this owner until `owner.close()`
succeeds. Registry replacement does not drain an old owner; the service remains
responsible for all generations it installed. There is no global immortal owner,
garbage-collection callback, implicit synchronization or background cleanup.
Untracked runtimes, duplicate adoption within one owner, and adoption after owner
shutdown begins are rejected. Integrations should assign one owner per runtime;
cross-owner lifecycle coordination is not provided.

`owner.close()` stops new adoption and makes one non-waiting close attempt on
each owned runtime. It releases a runtime only after its internal close has
established a closed state. Independent completed runtimes can be released even
when others fail. Remaining runtimes produce `AttentionRuntimeOwnerClosePending`
with opaque runtime identifiers and error text, not tensor objects or exception
tracebacks in the report. The service can use `get_runtime(id)` for internal
event recovery, then retry close. After successful shutdown, repeated close is
harmless and the owner cannot be reused. Concurrent or reentrant owner close is
rejected; planning, execution and service shutdown still require integration-level
serialization.

This is opt-in service shutdown, not a new public wrapper teardown method. An
owner keeps even idle runtimes until shutdown; it is not a cache-eviction policy.
Model callers retain ordinary `plan()` / `run()` usage. Single-request runtimes
and independently queued plan-time work remain separate integration
responsibilities. Borrowed batch prefill masks can use this owner when the
complete [mask bootstrap configuration](attention_plan_run_dispatch_design.md)
is installed.

### Device and public-wrapper integration

Internal `AttentionOperatorRuntime.close()` provides a non-waiting teardown
handshake for runtimes configured with a completion event recorder:

- The first attempt enters **closing**, rejecting new planning, submissions,
  workspace rebinding and unplanned forks.
- It makes one completion collection pass. Pending or unrecorded calls raise
  `AttentionCallRetentionClosePending` with the collection report. Query failures
  raise `AttentionCallCompletionCollectionError`. Both leave the runtime closing
  and retain the resources still needed by outstanding work.
- The integration may supply recovery events and retry `close()`. It must keep
  the runtime alive throughout this process; no timeout or error permits dropping
  the registry or forcing a release.
- Once no tracked calls remain, the registry closes and the runtime drops its
  active plan, executor, workspace binding and last-call diagnostics. The runtime
  becomes **closed**, cannot be reused and accepts repeated `close()` calls without
  further polling. Shared provider registrations are not unloaded.

An untracked legacy runtime cannot establish this completion proof, so its close
attempt is rejected without changing its state. Lifecycle checks also prevent
late candidate publication or submission when an injected preparation/lowering
callback has already closed the runtime. This is not a claim that arbitrary
concurrent use of one wrapper is thread-safe; integrations must serialize its
planning, execution entry and teardown operations.

The handshake covers tracked invocation resources. It does not prove completion
of independently queued plan-time materialization or other provider work; those
operations need their own lifetime and completion contracts. No Python destructor
or public wrapper teardown installs this handshake automatically yet.

The runtime must preserve the registry, supply correctly ordered event recording,
and drain it before teardown. Entry-point collection is not background polling:
an idle wrapper can retain its final calls until an explicit collection or a
later `plan()`/`run()`. Teardown must not rely on another call arriving. It must
also retain returned/output allocations where
the provider's lifetime contract requires them. Registry removal releases only
its own references; other calls, diagnostics or exception tracebacks may still
hold references. This mechanism does not establish immutable mask contents,
allocator generations, device leases, graph-capture safety or numerical accuracy.
Those are separate contracts. Public borrowed batch-mask execution is conditional
on explicit metadata mappings, event recording and runtime ownership; no real
provider or device event implementation is installed by default. Single-request
mask and provider graph paths remain closed.
