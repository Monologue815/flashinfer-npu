"""Public quantized prefill replanning with synthetic execution and events."""

import gc
import unittest
import weakref

from flashinfer_npu.attention.operator_retention import AttentionRetainedInvocationError
from flashinfer_npu.runtime import SchemaError
from tests import test_checkpoint_140_usage_examples as usage
from tests.test_checkpoint_129_call_retention import Event
from tests.test_checkpoint_135_quantized_mask_composition import result_pair, results
from tests.test_checkpoint_139_public_batch_mask import mask


class PublicReplanRetentionCheckpoint(unittest.TestCase):
    setUp = usage.UsageExamplesCheckpoint.setUp
    tearDown = usage.UsageExamplesCheckpoint.tearDown
    install = usage.UsageExamplesCheckpoint.install
    inputs = usage.UsageExamplesCheckpoint.inputs

    def start(self):
        self.install()
        namespace = self.inputs()
        results.append(result_pair(self.runtime, "first"))
        exec(self.blocks["paged_quant_mask"], namespace)
        return namespace, namespace["attention"]

    def replan(self, namespace, wrapper, *, packed=None):
        wrapper.plan(
            namespace["qo_indptr"], namespace["paged_kv_indptr"],
            namespace["paged_kv_indices"], namespace["paged_kv_last_page_len"],
            namespace["num_qo_heads"], namespace["num_kv_heads"],
            namespace["head_dim_qk"], namespace["page_size"],
            head_dim_vo=namespace["head_dim_vo"], q_data_type=namespace["q_dtype"],
            kv_data_type=namespace["quant_spec"], o_data_type=namespace["o_dtype"],
            custom_mask=namespace["custom_mask"], packed_custom_mask=packed,
        )

    def run_next(self, namespace, wrapper):
        results.append(result_pair(wrapper._operator_runtime, "next"))
        return wrapper.run(namespace["q_next"], namespace["kv_cache"], return_lse=True)

    def test_mask_replan_keeps_old_executor_until_its_own_completion(self):
        namespace, wrapper = self.start()
        runtime = wrapper._operator_runtime
        old = weakref.ref(runtime._executor)
        first_plan = wrapper.plan_state
        first_token = runtime.call_retention.pending_tokens[0]
        self.replan(namespace, wrapper, packed=mask(True, numel=2))
        self.assertIsNot(wrapper.plan_state, first_plan)
        self.assertIsNot(runtime._executor, old())
        gc.collect()
        self.assertIsNotNone(old())
        self.run_next(namespace, wrapper)
        recorder = self.factory.recorders[0]
        recorder.events[1].ready = True
        self.owner.collect_completed()
        self.assertEqual(runtime.call_retention.pending_tokens, (first_token,))
        gc.collect()
        self.assertIsNotNone(old())
        recorder.events[0].ready = True
        self.owner.collect_completed()
        gc.collect()
        self.assertIsNone(old())
        # The newly planned wrapper remains usable after collecting the old call.
        self.run_next(namespace, wrapper)

    def test_rejected_replan_preserves_executor_and_original_mask_plan(self):
        namespace, wrapper = self.start()
        runtime = wrapper._operator_runtime
        original = runtime._executor
        plan = wrapper.plan_state
        with self.assertRaises(SchemaError):
            self.replan(namespace, wrapper, packed=mask(True, numel=1))
        self.assertIs(runtime._executor, original)
        self.assertIs(wrapper.plan_state, plan)
        self.run_next(namespace, wrapper)
        self.assertEqual(len(runtime.call_retention.pending_tokens), 2)
        self.assertEqual(len(self.factory.recorders[0].events), 2)

    def test_unrecorded_old_call_survives_replan_until_explicit_recovery(self):
        namespace, wrapper = self.start()
        recorder = self.factory.recorders[0]
        recorder.events[0].ready = True
        self.owner.collect_completed()
        runtime = wrapper._operator_runtime
        old = weakref.ref(runtime._executor)
        recorder.failure = RuntimeError("synthetic event recording failure")
        with self.assertRaises(AttentionRetainedInvocationError) as caught:
            self.run_next(namespace, wrapper)
        token = caught.exception.token
        del caught
        recorder.failure = None
        self.replan(namespace, wrapper, packed=mask(True, numel=2))
        gc.collect()
        self.assertIsNotNone(old())
        report = self.owner.collect_completed().collections[0][1]
        self.assertEqual(report.unrecorded, (token,))
        self.run_next(namespace, wrapper)
        runtime.call_retention.bind_event(token, Event(token, ready=True))
        self.owner.collect_completed()
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual(len(runtime.call_retention.pending_tokens), 1)
