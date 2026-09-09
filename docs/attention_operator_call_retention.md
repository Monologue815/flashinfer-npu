# Provider call resource retention

## Scope

`attention.operator_retention` is a private framework building block for keeping
opaque call arguments and `retained_resources` alive until execution has finished.
It imports no device package, creates no NPU event and performs no synchronization.
Public wrappers do not install it automatically. The existing result completion
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
5. Poll explicitly. Only a strict boolean `True` from the matching event releases
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

External queries run outside the registry lock so independent calls can progress.
Concurrent/reentrant polling of the same invocation is rejected. Completed tokens
are removed rather than retained indefinitely as history; polling a removed token
is an error. Diagnostic `pending_tokens` contains no tensor payloads or owners.

## Remaining integration responsibilities

The runtime must preserve the registry, arrange event recording and polling, and
drain it before teardown. It must also retain returned/output allocations where
the provider's lifetime contract requires them. Registry removal releases only
its own references; other calls, diagnostics or exception tracebacks may still
hold references. This mechanism does not establish immutable mask contents,
allocator generations, device leases, graph-capture safety or numerical accuracy.
Those are separate contracts, and the public custom-mask provider path remains
closed until its full preparation and lifetime integration is complete.
