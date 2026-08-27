import inspect
import unittest

from flashinfer_npu.attention import BatchAttention, AttentionStateError
from flashinfer_npu.decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    single_decode_with_kv_cache,
)
from flashinfer_npu.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
)
from tests.test_checkpoint_062_runtime_completion_publication import (
    run_runtime,
    strict_results,
    strict_runtime,
    valid_result,
)


class PublicAttentionRunEvidenceTests(unittest.TestCase):
    """Batch facades expose evidence without exposing executable handles."""

    def setUp(self):
        strict_results[:] = []

    def successful_runtime(self):
        _, runtime = strict_runtime()
        strict_results.append(valid_result(runtime))
        run_runtime(runtime)
        return runtime

    def test_holistic_facade_proxies_read_only_atomic_run_receipt(self):
        runtime = self.successful_runtime()
        wrapper = BatchAttention(kv_layout="NHD", device="cpu")
        wrapper._operator_runtime = runtime

        self.assertIs(wrapper.last_run_receipt, runtime.last_run_receipt)
        with self.assertRaises(AttributeError):
            wrapper.last_run_receipt = object()

    def test_all_batch_wrappers_share_the_same_receipt_property(self):
        runtime = self.successful_runtime()
        for wrapper_type in (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
            BatchDecodeWithPagedKVCacheWrapper,
        ):
            with self.subTest(wrapper_type=wrapper_type.__name__):
                wrapper = object.__new__(wrapper_type)
                wrapper._operator_runtime = runtime
                self.assertIs(wrapper.last_run_receipt, runtime.last_run_receipt)

    def test_reference_route_fails_explicitly_instead_of_fabricating_receipt(self):
        wrapper = BatchAttention(kv_layout="NHD", device="cpu")

        with self.assertRaisesRegex(
            AttentionStateError, "reference.*do not publish"
        ):
            _ = wrapper.last_run_receipt

    def test_stateless_single_functions_keep_flashinfer_return_surface(self):
        for callable_value in (
            single_prefill_with_kv_cache,
            single_decode_with_kv_cache,
        ):
            parameters = inspect.signature(callable_value).parameters
            for forbidden in (
                "provider",
                "runtime",
                "plan_handle",
                "run_receipt",
                "return_receipt",
            ):
                self.assertNotIn(forbidden, parameters)

    def test_batch_public_signatures_expose_no_runtime_or_provider_handle(self):
        for wrapper_type in (
            BatchAttention,
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
            BatchDecodeWithPagedKVCacheWrapper,
        ):
            for callable_value in (
                wrapper_type,
                wrapper_type.plan,
                wrapper_type.run,
            ):
                parameters = inspect.signature(callable_value).parameters
                self.assertNotIn("provider", parameters)
                self.assertNotIn("runtime", parameters)
                self.assertNotIn("plan_handle", parameters)


if __name__ == "__main__":
    unittest.main()
