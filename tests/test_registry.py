import hashlib
import unittest

from flashinfer_npu.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    Backend,
    DeviceCapability,
    DispatchError,
    KernelConstraints,
    KernelArgumentABI,
    KernelArgumentDirection,
    KernelArgumentPassing,
    KernelBinaryABI,
    KernelDescriptor,
    KernelErrorABI,
    KernelErrorCodeABI,
    KernelLaunchABI,
    KernelRegistry,
    WorkloadSpec,
    WorkspaceFormula,
    build_plan,
)


def file_artifact(locator, target_soc="Ascend910B"):
    payload = ("synthetic:" + locator).encode("utf-8")
    return ArtifactRef(
        kind=ArtifactKind.FILE,
        format=ArtifactFormat.ASCENDC_OBJECT,
        locator=locator,
        digest=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        target_soc=target_soc,
        build_id="synthetic-registry-test",
    )


def builtin_artifact(locator="builtin:aclnn:rmsnorm"):
    return ArtifactRef(
        kind=ArtifactKind.BUILTIN,
        format=ArtifactFormat.ACLNN_BUILTIN,
        locator=locator,
        digest=hashlib.sha256(locator.encode("utf-8")).hexdigest(),
        target_soc="Ascend910B",
        build_id="synthetic-provider-contract",
    )


def launch_abi(name="flashinfer_npu.rmsnorm.v1"):
    return KernelLaunchABI(
        abi_name=name,
        entry_point="rmsnorm_entry",
        argument_names=("x", "weight", "out", "workspace", "stream"),
        mutable_arguments=("out", "workspace"),
        stream_argument="stream",
    )


def binary_abi(logical):
    return KernelBinaryABI(
        abi_name="flashinfer_npu.rmsnorm.binary.v1",
        arguments=tuple(
            KernelArgumentABI(
                name,
                (
                    KernelArgumentPassing.OPAQUE_HANDLE
                    if name == logical.stream_argument
                    else KernelArgumentPassing.POINTER
                ),
                (
                    KernelArgumentDirection.INOUT
                    if name in logical.mutable_arguments
                    else KernelArgumentDirection.INPUT
                ),
                nullable=(name == "workspace"),
            )
            for name in logical.argument_names
        ),
        error_abi=KernelErrorABI(
            "flashinfer_npu.test_error.v1",
            (KernelErrorCodeABI("success", 0),),
        ),
    )


class RegistryTests(unittest.TestCase):
    def setUp(self):
        constraints = KernelConstraints(
            supported_socs=("Ascend910B",),
            dtype_signatures=(("float16", "float16"),),
            required_features=("vector",),
        )
        aot_launch = launch_abi()
        self.aot = KernelDescriptor(
            kernel_id="rmsnorm_910b_aot_v1",
            op="rmsnorm",
            backend=Backend.ASCENDC_AOT,
            constraints=constraints,
            workspace=WorkspaceFormula(constant_bytes=33),
            artifact=file_artifact("artifacts/ascend910b/rmsnorm.o"),
            launch_abi=aot_launch,
            binary_abi=binary_abi(aot_launch),
            priority=20,
        )
        aclnn_launch = launch_abi("flashinfer_npu.rmsnorm.aclnn.v1")
        self.aclnn = KernelDescriptor(
            kernel_id="rmsnorm_aclnn_v1",
            op="rmsnorm",
            backend=Backend.ACLNN,
            constraints=constraints,
            artifact=builtin_artifact(),
            launch_abi=aclnn_launch,
            binary_abi=binary_abi(aclnn_launch),
            priority=10,
        )
        self.reference = KernelDescriptor(
            kernel_id="rmsnorm_reference_v1",
            op="rmsnorm",
            backend=Backend.REFERENCE,
            constraints=constraints,
            priority=100,
        )
        self.registry = KernelRegistry((self.reference, self.aclnn, self.aot))
        self.workload = WorkloadSpec(
            op="rmsnorm", dtypes=("float16", "float16")
        )
        self.capability = DeviceCapability(
            soc_version="Ascend910B",
            supported_dtypes=("float16", "bfloat16"),
            features=("vector",),
        )

    def test_auto_dispatch_does_not_silently_choose_reference(self):
        selected = self.registry.select(self.workload, self.capability)
        self.assertEqual(selected.kernel_id, self.aot.kernel_id)

    def test_tuning_record_overrides_heuristic_priority(self):
        selected = self.registry.select(
            self.workload,
            self.capability,
            tuned_kernel_ids=(self.aclnn.kernel_id,),
        )
        self.assertEqual(selected.kernel_id, self.aclnn.kernel_id)

    def test_forced_backend_is_honored(self):
        selected = self.registry.select(
            self.workload, self.capability, backend=Backend.ACLNN
        )
        self.assertEqual(selected.kernel_id, self.aclnn.kernel_id)

    def test_reference_requires_opt_in(self):
        registry = KernelRegistry((self.reference,))
        with self.assertRaisesRegex(DispatchError, "explicit opt-in"):
            registry.select(self.workload, self.capability)
        selected = registry.select(
            self.workload,
            self.capability,
            backend=Backend.REFERENCE,
            allow_reference=True,
        )
        self.assertEqual(selected.kernel_id, self.reference.kernel_id)

    def test_explain_reports_missing_feature(self):
        capability = DeviceCapability(
            soc_version="Ascend910B", supported_dtypes=("float16",)
        )
        reports = self.registry.explain(self.workload, capability)
        self.assertTrue(any("missing features" in reason for reason in reports[0].reasons))

    def test_plan_fingerprints_and_aligns_workspace(self):
        plan = build_plan(self.registry, self.workload, self.capability)
        self.assertEqual(plan.kernel_id, self.aot.kernel_id)
        self.assertEqual(plan.workspace_bytes, 64)
        self.assertEqual(plan.int_workspace_bytes, 0)
        plan.validate_runtime(self.workload, self.capability)

    def test_duplicate_kernel_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate kernel_id"):
            KernelRegistry((self.aot, self.aot))


if __name__ == "__main__":
    unittest.main()
