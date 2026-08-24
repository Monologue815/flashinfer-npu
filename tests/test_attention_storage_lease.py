import unittest
from dataclasses import replace

from flashinfer_npu.attention import (
    AttentionAddressBinding,
    AttentionLaunchLeaseContract,
    AttentionHostBufferLease,
    AttentionLeaseRegistry,
    AttentionLeaseState,
    AttentionStorageLease,
    AttentionStorageLeaseError,
    AttentionStorageLifetime,
    AttentionRunTensorContract,
    KVCacheView,
    PagedKVCacheSpec,
    StreamContext,
    TensorView,
)
from flashinfer_npu.runtime import SchemaError


def tensor_view(
    storage_id="storage-1",
    *,
    shape=(4,),
    storage_offset=0,
    storage_nbytes=16,
    writable=False,
    alignment=64,
):
    return TensorView(
        shape=shape,
        strides=(1,),
        dtype="float32",
        device="npu:0",
        storage_id=storage_id,
        storage_nbytes=storage_nbytes,
        storage_offset=storage_offset,
        data_ptr_alignment=alignment,
        writable=writable,
    )


def storage_lease(
    storage_id="storage-1",
    *,
    lease_id="lease-1",
    base_address=0x1000,
    capacity_bytes=16,
    generation=1,
    lifetime=AttentionStorageLifetime.RUN,
    writable=False,
):
    return AttentionStorageLease(
        lease_id=lease_id,
        storage_id=storage_id,
        device="npu:0",
        base_address=base_address,
        capacity_bytes=capacity_bytes,
        alignment=64,
        allocator_id="torch_npu_allocator:0",
        allocation_generation=generation,
        lifetime=lifetime,
        writable=writable,
    )


def binding(role="q", *, writable=False, **lease_changes):
    storage_id = lease_changes.pop("storage_id", "storage-1")
    capacity = lease_changes.get("capacity_bytes", 16)
    view = tensor_view(
        storage_id,
        storage_nbytes=capacity,
        writable=writable,
    )
    lease = storage_lease(
        storage_id,
        writable=writable,
        **lease_changes,
    )
    return AttentionAddressBinding(role, view, lease)


def contract(*bindings, graph_enabled=False):
    return AttentionLaunchLeaseContract(
        execution_identity_fingerprint="1" * 64,
        dispatch_receipt_fingerprint="2" * 64,
        stream_device="npu:0",
        stream_id="aclrt-stream:7",
        bindings=bindings or (binding(),),
        graph_enabled=graph_enabled,
    )


class AttentionStorageLeaseSchemaTests(unittest.TestCase):
    def test_host_buffer_lease_is_versioned_and_separate_from_device_storage(self):
        host = AttentionHostBufferLease(
            "host-lease",
            0x8000,
            256,
            64,
            "host-arena",
            1,
            AttentionStorageLifetime.RUN,
            True,
            pinned=True,
        )
        self.assertEqual(AttentionHostBufferLease.from_dict(host.to_dict()), host)
        self.assertNotEqual(
            host.fingerprint,
            replace(host, allocation_generation=2).fingerprint,
        )
        with self.assertRaisesRegex(SchemaError, "non-zero address"):
            replace(host, base_address=0)

    def test_binding_round_trip_freezes_absolute_address_and_generation(self):
        original = contract(
            binding("q"),
            binding(
                "out",
                storage_id="storage-out",
                lease_id="lease-out",
                base_address=0x2000,
                writable=True,
            ),
        )
        restored = AttentionLaunchLeaseContract.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)
        self.assertEqual(original.bindings[0].data_address, 0x1000)

        changed_lease = replace(
            original.bindings[0].lease,
            allocation_generation=2,
        )
        changed = replace(
            original,
            bindings=(
                replace(original.bindings[0], lease=changed_lease),
                original.bindings[1],
            ),
        )
        with self.assertRaisesRegex(AttentionStorageLeaseError, "stale"):
            original.validate_reuse(changed)

    def test_view_storage_device_capacity_write_and_alignment_are_bound(self):
        view = tensor_view(writable=True)
        readonly = storage_lease(writable=False)
        with self.assertRaisesRegex(SchemaError, "writable storage"):
            AttentionAddressBinding("out", view, readonly)
        with self.assertRaisesRegex(SchemaError, "identity"):
            AttentionAddressBinding(
                "q",
                tensor_view(storage_nbytes=16),
                storage_lease(capacity_bytes=32),
            )
        offset_view = tensor_view(
            shape=(1,), storage_offset=1, storage_nbytes=16, alignment=64
        )
        with self.assertRaisesRegex(SchemaError, "violates TensorView alignment"):
            AttentionAddressBinding("q", offset_view, storage_lease())

    def test_writable_overlap_and_graph_run_lifetime_are_rejected(self):
        shared_lease = storage_lease(writable=True)
        left = AttentionAddressBinding(
            "left",
            tensor_view(shape=(2,), writable=True),
            shared_lease,
        )
        right = AttentionAddressBinding(
            "right",
            tensor_view(shape=(2,)),
            shared_lease,
        )
        with self.assertRaisesRegex(SchemaError, "overlapping"):
            contract(left, right)
        with self.assertRaisesRegex(SchemaError, "persistent/capture"):
            contract(binding(), graph_enabled=True)

        persistent = binding(
            lifetime=AttentionStorageLifetime.CAPTURE,
        )
        self.assertTrue(contract(persistent, graph_enabled=True).graph_enabled)

    def test_empty_storage_can_use_null_but_nonempty_cannot(self):
        empty = AttentionStorageLease(
            "empty",
            "empty-storage",
            "npu:0",
            0,
            0,
            64,
            "allocator",
            1,
            AttentionStorageLifetime.RUN,
            True,
        )
        self.assertEqual(empty.address_interval, (0, 0))
        with self.assertRaisesRegex(SchemaError, "non-zero address"):
            replace(empty, capacity_bytes=1)

    def test_launch_roles_and_views_exactly_bind_runtime_tensor_contract(self):
        q = tensor_view("q", shape=(1,), storage_nbytes=4)
        key = TensorView((1, 1, 1, 1), (1, 1, 1, 1), "float32", "npu:0", "key", 4)
        value = TensorView((1, 1, 1, 1), (1, 1, 1, 1), "float32", "npu:0", "value", 4)
        kv = KVCacheView(
            PagedKVCacheSpec(
                1, 1, 1, 1, 1, "float32", structure="separate", device="npu:0"
            ),
            key,
            value,
        )
        tensors = AttentionRunTensorContract(
            q, kv, StreamContext("npu:0", "aclrt-stream:7")
        )

        def exact(role, item, address):
            lease = AttentionStorageLease(
                "lease-" + role,
                item.storage_id,
                item.device,
                address,
                item.storage_nbytes,
                64,
                "torch_npu_allocator:0",
                1,
                AttentionStorageLifetime.RUN,
                item.writable,
            )
            return AttentionAddressBinding(role, item, lease)

        launch = contract(
            exact("q", q, 0x1000),
            exact("kv.key_storage", key, 0x2000),
            exact("kv.value_storage", value, 0x3000),
        )
        launch.validate_tensor_contract(tensors, execution_identity_fingerprint="1" * 64)
        with self.assertRaisesRegex(SchemaError, "roles do not match"):
            replace(launch, bindings=launch.bindings[:2]).validate_tensor_contract(tensors)


class AttentionLeaseRegistryTests(unittest.TestCase):
    def test_submit_completion_and_release_are_event_owned(self):
        registry = AttentionLeaseRegistry()
        launch = contract(binding("out", writable=True))
        token = registry.acquire(launch)
        self.assertEqual(registry.state(token), AttentionLeaseState.ACQUIRED)
        registry.submit(token, "event-1")
        self.assertEqual(registry.state(token), AttentionLeaseState.SUBMITTED)
        self.assertEqual(registry.completion_event_id(token), "event-1")
        with self.assertRaisesRegex(AttentionStorageLeaseError, "before completion"):
            registry.release(token)
        with self.assertRaisesRegex(AttentionStorageLeaseError, "does not own"):
            registry.complete(token, "event-other")
        registry.complete(token, "event-1")
        self.assertEqual(registry.state(token), AttentionLeaseState.COMPLETED)
        registry.release(token)
        self.assertEqual(registry.state(token), AttentionLeaseState.RELEASED)

    def test_active_write_interval_blocks_alias_and_completed_can_be_reused(self):
        registry = AttentionLeaseRegistry()
        first = contract(binding("out", writable=True))
        token = registry.acquire(first)
        second = contract(
            binding(
                "out2",
                storage_id="new-storage-id",
                lease_id="new-lease-id",
                generation=2,
                writable=True,
            )
        )
        with self.assertRaisesRegex(AttentionStorageLeaseError, "already leased"):
            registry.acquire(second)
        registry.submit(token, "event")
        registry.complete(token, "event")
        second_token = registry.acquire(second)
        self.assertEqual(registry.state(second_token), AttentionLeaseState.ACQUIRED)
        with self.assertRaisesRegex(AttentionStorageLeaseError, "already leased"):
            registry.submit(token, "event-replay")

    def test_contract_change_is_rejected_before_submit(self):
        registry = AttentionLeaseRegistry()
        launch = contract(binding())
        token = registry.acquire(launch)
        changed = replace(launch, stream_id="another-stream")
        with self.assertRaisesRegex(AttentionStorageLeaseError, "stale"):
            registry.validate_contract(token, changed)

    def test_different_streams_cannot_bypass_writable_interval_ownership(self):
        registry = AttentionLeaseRegistry()
        first = contract(binding("out", writable=True))
        second = replace(first, stream_id="aclrt-stream:8")
        token = registry.acquire(first)
        with self.assertRaisesRegex(AttentionStorageLeaseError, "already leased"):
            registry.acquire(second)
        registry.release(token)

        readonly = contract(binding("q", writable=False))
        readonly_other_stream = replace(readonly, stream_id="aclrt-stream:9")
        left = registry.acquire(readonly)
        right = registry.acquire(readonly_other_stream)
        self.assertEqual(registry.state(left), AttentionLeaseState.ACQUIRED)
        self.assertEqual(registry.state(right), AttentionLeaseState.ACQUIRED)
        registry.release(left)
        registry.release(right)


if __name__ == "__main__":
    unittest.main()
