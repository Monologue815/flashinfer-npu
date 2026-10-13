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

After the executor returns, the registry also retains the returned object before
event recording and result validation. This covers provider-allocated output/LSE
that are not part of caller-supplied arguments. For tuple/list returns it takes a
shallow snapshot of the built-in sequence's member references, so clearing a
returned list cannot prematurely discard its original output allocations. It
does not copy tensors, read tensor values or invoke custom iteration hooks.
Keeping these references does not validate result arity, shape or numerical
contents; an invalid returned result still needs lifetime protection.

The invocation helper retains the exact execution callback before entering it.
A bound method keeps its executor alive; callable objects and closures keep their
own execution dependencies alive. Replanning may replace the active executor,
but cannot discard this reference while the original invocation is pending.
Recording failures and partial invocation failures preserve it as well. It is
released with the call only after that call's completion proof, and is excluded
from diagnostics. This does not prevent explicit provider unload or retain a
module lease that the callback does not own; providers must represent those
dependencies in their callable or explicit retained resources. Retention is not
permission to invoke the callback again or change its implementation in place.

Returned ownership binds once to the existing invocation token, before its event.
Result metadata failure, event-recording failure or query failure does not release
these references. The call and its returned owners are released together only on
that token's completion proof. A provider exception before returning gives the
framework no returned allocation handle; private allocations not exposed in the
call or result remain the provider's lifetime responsibility.

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

`owner.collect_completed()` makes one non-waiting pass across a snapshot of owned
runtimes while keeping the service open. Its metadata-only `collections` report
contains `(runtime_id, collection_report)` pairs; each collection lists released,
pending and unrecorded invocation tokens plus query failure text. An ordinary
query failure does not prevent independent calls or runtimes from being polled.
This allows idle returned outputs to be released without another model call or
service shutdown. It does not evict runtimes, clear active plans or release other
diagnostic references. Runtimes adopted during the pass wait for the next pass.
No background polling is started. The service schedules collection explicitly
and serializes it with planning/execution. Reentrant or concurrent collection and
close are rejected; once shutdown starts, use the retryable close handshake.

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

### Single-request ownership design (not connected)

Single prefill/decode currently construct an ephemeral runtime without the batch
recorder hook. Python function return is not device completion. The batch
retention guarantees above therefore must not be attributed to these single
facades. Their default provider registry remains empty; a future asynchronous
integration needs the following additional framework contract before enabling
borrowed resources such as single custom masks.

The public single-request signatures and return values stay unchanged. Ownership
is configured only in the process bootstrap snapshot, using a single-request
recorder factory and a service owner as one dependency pair. Supplying only one
must fail atomically. The single configuration is independent of the batch
configuration: enabling batch events must not silently change single execution.
No default events, package probing, operators or background workers are created
at installation time.

The proposed private lifecycle is:

```text
capture snapshot -> create tracked runtime -> adopt before planning
                 -> plan -> run -> seal submission (also on exception)
                 -> service collection -> completion proof -> close and remove
```

An adopted runtime starts in a **preparing** state. Collection skips preparing
runtimes, because a plan or call has not finished submitting yet. The facade
seals it in a `finally` path after the one-shot attempt, whether planning,
invocation, event recording or result validation succeeds or fails. Sealing is
not completion proof and must not discard the runtime. Only sealed runtimes
are eligible for per-request teardown. The service can keep the returned result
alive independently; closing a runtime drops only framework-owned references.

Unlike batch collection, single-request collection closes and removes each
completed sealed runtime while leaving the service owner open for new requests.
It makes one bounded, non-waiting pass. Pending or unrecorded calls and query
failures remain owned, with metadata-only diagnostics and private recovery access.
One failing request must not prevent independent completed requests from being
removed. A callback or invocation token is not a public executable plan handle.

Shutdown first stops new adoption. It cannot infer that preparing requests have
finished merely because their registries are empty. Preparing requests remain
owned until the facade seals them, and sealed pending requests retain the existing
event-recovery/close handshake. Retrying shutdown must not resubmit a provider
call, force-release a resource, or rely on a timeout. The service retains every
owner generation after bootstrap replacement until its shutdown completes.
Planning that independently queues device work is outside invocation tracking
and still needs its own provider lifetime contract.

Implementation acceptance must exercise the public single prefill and decode
facades with synthetic providers: completion after Python return; preparation
failure; partial submission and recording failure; invalid returned metadata;
collection during preparation; independent request release; bootstrap replacement;
and shutdown rejecting new work while preserving pending recovery. Weak-reference
checks must cover runtime, executor, inputs and provider-allocated outputs without
test fixtures or exception tracebacks accidentally keeping them alive. No real
device safety or numerical claim follows from these framework tests.

This section is a design boundary, not an available bootstrap API. Single custom
masks remain disabled until this ownership path and exact mask lowering are both
implemented; adding ownership alone must not advertise mask support.

The internal `AttentionSingleRuntimeOwner` primitive in
`attention.operator_single_owner` implements adoption, explicit `seal(id)`,
bounded collection and retryable shutdown. Its collection report separates
released, preparing and sealed-pending identifiers, with text-only failures.
It accepts only completion-tracked single prefill/decode runtimes. Sealing marks
collection eligibility; the integrating facade must prohibit further submissions
and call it in its `finally` path. Preparing requests can still be sealed after
shutdown starts. Collection snapshots eligibility, so requests sealed during a
pass wait for the next pass. This primitive is not yet connected to bootstrap or
public single functions, and does not change their current execution behavior.

### Service integration example

The following process-local example assumes the service already supplies a
reviewed `resolver_registry`, its matching `operation_catalog`, a
`completion_event_recorder_factory`, and `mask_integration` (or `None` for no
borrowed-mask support). These are integration objects, not model inputs. This
example neither constructs operators nor discovers or approves installed device
packages. Applications using declared/bundle installers pass the same lifetime
hooks to those installers instead of also installing this raw registry.

<!-- service-example: install -->
```python
from flashinfer_npu.attention import install_attention_operator_runtime_resolvers
from flashinfer_npu.attention.operator_runtime_owner import (
    AttentionBatchRuntimeOwner, AttentionRuntimeOwnerClosePending,
)

runtime_owner = AttentionBatchRuntimeOwner()
runtime_snapshot = install_attention_operator_runtime_resolvers(
    resolver_registry,
    operation_catalog=operation_catalog,
    batch_completion_event_recorder_factory=completion_event_recorder_factory,
    batch_runtime_owner=runtime_owner,
    batch_mask_integration=mask_integration,
)
```

Construct model wrappers only after installation. Keep `runtime_owner` strongly
owned by the service, including after registry replacement. Model calls continue
to use ordinary `plan()` and `run()`; they do not receive the owner. Schedule the
following pass explicitly in the service, serialized with model submissions:

<!-- service-example: collect -->
```python
collection = runtime_owner.collect_completed()
query_failures = tuple(
    (runtime_id, failure)
    for runtime_id, report in collection.collections
    for failure in report.failures
)
unrecorded_calls = tuple(
    (runtime_id, token)
    for runtime_id, report in collection.collections
    for token in report.unrecorded
)
```

Nonempty diagnostics require integration attention, not operator resubmission.
An unrecorded call needs a trustworthy event covering its original queued work;
never synthesize success or release it on timeout. Collection does not wait for
pending calls, and cannot establish permission to overwrite an output buffer.

Stop accepting new submissions before shutdown. Each execution of the next
block makes one attempt, without a busy loop or forced device synchronization:

<!-- service-example: close -->
```python
try:
    runtime_owner.close()
except AttentionRuntimeOwnerClosePending as error:
    shutdown_complete = False
    shutdown_report = error.report
else:
    shutdown_complete = True
    shutdown_report = None
```

When `shutdown_complete` is false, retain the owner and its dependencies, resolve
any recording/query failures through the integration, and retry this close block
on a later service lifecycle tick. Do not resume submissions or call the open
owner collection path after shutdown begins. Successful close is idempotent;
the closed owner cannot be reused. Reports contain metadata, not completion
events or tensors. Other unexpected exceptions are deliberately not swallowed.

### Runtime teardown contract

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
also account for private provider allocations not exposed in the call or result.
Returned output/LSE and caller-supplied output buffers are retained by the same
invocation registry. Registry removal releases only
its own references; other calls, diagnostics or exception tracebacks may still
hold references. This mechanism does not establish immutable mask contents,
allocator generations, device leases, graph-capture safety or numerical accuracy.
Those are separate contracts. Public borrowed batch-mask execution is conditional
on explicit metadata mappings, event recording and runtime ownership; no real
provider or device event implementation is installed by default. Single-request
mask and provider graph paths remain closed.
