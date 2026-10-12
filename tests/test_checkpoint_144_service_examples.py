"""Run the documented service lifecycle around public synthetic batch calls."""

import re
import unittest
from pathlib import Path

from flashinfer_npu.attention import AttentionStateError
from flashinfer_npu.attention.operator_retention import AttentionRetainedInvocationError
from tests.test_checkpoint_129_call_retention import Event
from tests import test_checkpoint_138_batch_runtime_owner as owner_fixtures
from tests.test_checkpoint_043_provider_workspace_reset import runtime_registry


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "attention_operator_call_retention.md"


class ServiceExamplesCheckpoint(unittest.TestCase):
    setUp = owner_fixtures.BatchRuntimeOwnerCheckpoint.setUp
    tearDown = owner_fixtures.BatchRuntimeOwnerCheckpoint.tearDown
    wrapper = owner_fixtures.BatchRuntimeOwnerCheckpoint.wrapper

    def examples(self):
        blocks = re.findall(r"<!-- service-example: ([a-z_]+) -->\n```python\n(.*?)\n```", GUIDE.read_text(), re.S)
        self.assertEqual([name for name, _ in blocks], ["install", "collect", "close"])
        return {name: compile(source, str(GUIDE) + ":" + name, "exec") for name, source in blocks}

    def install_example(self):
        self.blocks = self.examples()
        namespace = dict(resolver_registry=runtime_registry(self.values),
                         operation_catalog=self.values["catalog"],
                         completion_event_recorder_factory=self.factory,
                         mask_integration=None)
        exec(self.blocks["install"], namespace)
        self.owner = namespace["runtime_owner"]
        self.assertIs(namespace["runtime_snapshot"].batch_runtime_owner, self.owner)
        self.assertEqual(self.factory.calls, [])
        return namespace

    def test_documented_collection_keeps_public_wrapper_usable_until_shutdown(self):
        namespace = self.install_example()
        wrapper = self.wrapper()
        wrapper.run("q", ("k", "v"))
        event = self.factory.recorders[0].events[0]
        exec(self.blocks["collect"], namespace)
        self.assertEqual(namespace["query_failures"], ())
        self.assertEqual(len(namespace["collection"].collections[0][1].pending), 1)
        event.ready = True
        exec(self.blocks["collect"], namespace)
        self.assertEqual(len(namespace["collection"].collections[0][1].released), 1)
        wrapper.run("next", ("k", "v"))
        exec(self.blocks["close"], namespace)
        self.assertFalse(namespace["shutdown_complete"])
        self.assertEqual(namespace["shutdown_report"].remaining_runtime_ids, self.owner.owned_runtime_ids)
        with self.assertRaises(AttentionStateError):
            wrapper.run("late", ("k", "v"))
        with self.assertRaises(AttentionStateError):
            exec(self.blocks["collect"], namespace)
        self.factory.recorders[0].events[1].ready = True
        exec(self.blocks["close"], namespace)
        self.assertTrue(namespace["shutdown_complete"])
        self.assertIsNone(namespace["shutdown_report"])
        exec(self.blocks["close"], namespace)
        self.assertTrue(namespace["shutdown_complete"])

    def test_documented_query_diagnostics_do_not_hide_pending_work(self):
        namespace = self.install_example()
        wrapper = self.wrapper()
        wrapper.run("q", ("k", "v"))
        event = self.factory.recorders[0].events[0]
        def fail():
            raise RuntimeError("synthetic query fault")
        event.action = fail
        exec(self.blocks["collect"], namespace)
        runtime_id, failure = namespace["query_failures"][0]
        self.assertEqual(failure.message, "synthetic query fault")
        self.assertIs(self.owner.get_runtime(runtime_id), wrapper._operator_runtime)
        exec(self.blocks["close"], namespace)
        self.assertFalse(namespace["shutdown_complete"])
        event.action, event.ready = None, True
        exec(self.blocks["close"], namespace)
        self.assertTrue(namespace["shutdown_complete"])

    def test_documented_unrecorded_diagnostics_require_exact_recovery(self):
        namespace = self.install_example()
        wrapper = self.wrapper()
        self.factory.recorders[0].failure = RuntimeError("synthetic record fault")
        with self.assertRaises(AttentionRetainedInvocationError):
            wrapper.run("q", ("k", "v"))
        exec(self.blocks["collect"], namespace)
        runtime_id, token = namespace["unrecorded_calls"][0]
        exec(self.blocks["close"], namespace)
        self.assertFalse(namespace["shutdown_complete"])
        runtime = self.owner.get_runtime(runtime_id)
        runtime.call_retention.bind_event(token, Event(token, ready=True))
        exec(self.blocks["close"], namespace)
        self.assertTrue(namespace["shutdown_complete"])
