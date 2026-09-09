import gc
import unittest
import weakref
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionOperatorBatchRuntime, AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorRuntime, AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry, AttentionStateError,
    build_attention_operator_runtime_resolvers,
)
from flashinfer_npu.attention.operator_retention import (
    AttentionCallRetentionError, AttentionRetainedInvocationError,
)
from flashinfer_npu.runtime import SchemaError
from tests.test_attention_capability import group_plan
from tests.test_checkpoint_019_package_runtime_integration import (
    FakeLogicalRunAdapter, build_components, framework_inputs, package_attention,
)
from tests.test_checkpoint_022_operator_runtime_bootstrap import bootstrap_components
from tests.test_checkpoint_056_provider_output_buffer_contract import buffer_tensor
from tests.test_checkpoint_062_runtime_completion_publication import (
    StrictResultPackageLoader, run_runtime, strict_calls, strict_results, valid_result,
)
from tests.test_checkpoint_129_call_retention import Event, Recorder


class Owner:
    pass


class OwningLogicalAdapter(FakeLogicalRunAdapter):
    def __init__(self):
        self.owners = []

    def lower(self, active, request):
        call = super().lower(active, request)
        owner = Owner()
        self.owners.append(weakref.ref(owner))
        return replace(call, retained_resources=(owner,))


def owning_runtime(recorder):
    values = build_components()
    adapter = OwningLogicalAdapter()
    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=100, package_resolver=values["implementation"]._package_resolver,
        plan_gate=values["gate"], authority_resolver=values["authority"],
        logical_factory=values["factory"], logical_run_adapter=adapter,
        tensor_materializer=values["materializer"])
    registry = AttentionOperatorRuntimeResolverRegistry(
        (("npu", AttentionOperatorRuntimeImplementationRegistry((implementation,))),))
    runtime = AttentionOperatorBatchRuntime(
        "npu:0", registry, values["catalog"], completion_event_recorder=recorder)
    runtime.plan(*framework_inputs())
    return values, adapter, runtime


def retained_strict_runtime(recorder):
    values = bootstrap_components()
    values["loader"] = StrictResultPackageLoader(values["events"])
    spec = replace(values["spec"], validate_provider_results=True)
    registry = build_attention_operator_runtime_resolvers(
        (spec,), operation_catalog=values["catalog"], package_loader=values["loader"])
    plan = group_plan()
    runtime = AttentionOperatorRuntime(
        "npu:0", registry, values["catalog"], mode=plan.spec.mode,
        completion_event_recorder=recorder)
    runtime.plan(plan.spec, plan.metadata)
    return runtime


class RuntimeCallRetentionCheckpoint(unittest.TestCase):
    def setUp(self):
        package_attention.calls[:] = []
        strict_calls[:] = []
        strict_results[:] = []

    def test_missing_recorder_rejects_owned_calls_before_execution(self):
        _, _, runtime = owning_runtime(None)
        with self.assertRaisesRegex(AttentionStateError, "require a completion event recorder"):
            runtime.run("q", ("k", "v"), return_lse=False)
        self.assertEqual(package_attention.calls, [])
        self.assertEqual(runtime.call_retention.pending_tokens, ())
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_lowered_call

    def test_replan_preserves_registry_and_old_owner_until_event_completion(self):
        recorder = Recorder()
        _, adapter, runtime = owning_runtime(recorder)
        registry = runtime.call_retention
        self.assertEqual(runtime.run("q1", ("k", "v"), return_lse=False), "package-output:q1")
        first = registry.pending_tokens[0]
        first_owner = adapter.owners[0]
        old_executor = runtime._executor
        runtime.plan(*framework_inputs(kv_length=64))
        self.assertIs(runtime.call_retention, registry)
        self.assertIsNot(runtime._executor, old_executor)
        gc.collect()
        self.assertIsNotNone(first_owner())
        self.assertEqual(runtime.run("q2", ("k", "v"), return_lse=True),
                         ("package-output:q2", "package-lse:0.25"))
        second = registry.pending_tokens[1]
        self.assertNotEqual(first.active_plan_fingerprint, second.active_plan_fingerprint)
        second_owner = adapter.owners[1]
        # Another plan clears last-call diagnostics without clearing pending calls.
        runtime.plan(*framework_inputs(kv_length=32))
        recorder.events[1].ready = True
        self.assertTrue(registry.poll(second))
        gc.collect()
        self.assertIsNone(second_owner())
        self.assertIsNotNone(first_owner())
        recorder.events[0].ready = True
        registry.poll(first)
        gc.collect()
        self.assertIsNone(first_owner())

    def test_failed_replan_preserves_pending_calls_and_old_executable(self):
        recorder = Recorder()
        values, adapter, runtime = owning_runtime(recorder)
        runtime.run("q", ("k", "v"), return_lse=False)
        active = runtime.operator_session.active_plan
        tokens = runtime.call_retention.pending_tokens
        values["authority"].fail = True
        with self.assertRaisesRegex(RuntimeError, "authority failure"):
            runtime.plan(*framework_inputs(kv_length=64))
        self.assertIs(runtime.operator_session.active_plan, active)
        self.assertEqual(runtime.call_retention.pending_tokens, tokens)
        self.assertIsNotNone(adapter.owners[0]())
        self.assertEqual(runtime.run("again", ("k", "v"), return_lse=False), "package-output:again")
        self.assertEqual(len(runtime.call_retention.pending_tokens), 2)

    def test_output_validation_failure_cannot_release_submitted_inputs(self):
        recorder = Recorder()
        runtime = retained_strict_runtime(recorder)
        strict_results.append(valid_result(runtime))
        expected = strict_results[0]
        actual = run_runtime(runtime)
        self.assertIs(actual[0], expected[0])
        self.assertIs(actual[1], expected[1])
        self.assertIsNotNone(runtime.last_completion_receipt)
        plan = runtime.plan_state
        strict_results.append((
            buffer_tensor("wrong-shape", plan.expected_output_shape[:-1]
                          + (plan.expected_output_shape[-1] + 1,), plan.spec.o_dtype),
            buffer_tensor("lse", plan.expected_lse_shape, "float32"),
        ))
        with self.assertRaisesRegex(SchemaError, "output result shape"):
            run_runtime(runtime)
        self.assertEqual(len(runtime.call_retention.pending_tokens), 2)
        self.assertEqual(len(recorder.events), 2)
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_completion_receipt
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_run_receipt
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_lowered_call
        for event in recorder.events:
            event.ready = True
            runtime.call_retention.poll(event.token)
        self.assertEqual(runtime.call_retention.pending_tokens, ())

    def test_executor_failure_records_an_event_but_publishes_no_success(self):
        recorder = Recorder()
        runtime = retained_strict_runtime(recorder)
        # The synthetic callable records its invocation, then fails when no
        # synthetic result is available. No device operator is involved.
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            run_runtime(runtime)
        self.assertIsInstance(caught.exception.invocation_error, IndexError)
        self.assertEqual(len(strict_calls), 1)
        self.assertEqual(runtime.call_retention.pending_tokens, (caught.exception.token,))
        self.assertEqual(len(recorder.events), 1)
        with self.assertRaises(AttentionStateError):
            _ = runtime.last_lowered_call

    def test_record_failure_survives_replan_and_supports_exact_token_recovery(self):
        recorder = Recorder()
        _, adapter, runtime = owning_runtime(recorder)
        recorder.failure = RuntimeError("event recorder unavailable")
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            runtime.run("q", ("k", "v"), return_lse=False)
        token = caught.exception.token
        self.assertEqual(len(package_attention.calls), 1)
        runtime.plan(*framework_inputs(kv_length=64))
        self.assertIsNotNone(adapter.owners[0]())
        self.assertEqual(runtime.call_retention.pending_tokens, (token,))
        with self.assertRaisesRegex(AttentionCallRetentionError, "still retain"):
            runtime.call_retention.close()
        runtime.call_retention.bind_event(token, Event(token, True))
        self.assertTrue(runtime.call_retention.poll(token))
        self.assertEqual(runtime.call_retention.pending_tokens, ())

    def test_preflight_failure_creates_no_pending_call_or_event(self):
        recorder = Recorder()
        runtime = retained_strict_runtime(recorder)
        with self.assertRaisesRegex(SchemaError, "query and kv_cache must be provided"):
            runtime.run(None, None, return_lse=True)
        self.assertEqual(runtime.call_retention.pending_tokens, ())
        self.assertEqual(recorder.events, [])
        self.assertEqual(strict_calls, [])

    def test_closed_registry_blocks_next_submission(self):
        recorder = Recorder()
        _, _, runtime = owning_runtime(recorder)
        runtime.call_retention.close()
        with self.assertRaisesRegex(AttentionCallRetentionError, "closed"):
            runtime.run("q", ("k", "v"), return_lse=False)
        self.assertEqual(package_attention.calls, [])
        self.assertEqual(recorder.events, [])
