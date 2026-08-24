# Kernel manifests

`manifest.schema.json` defines the runtime-visible kernel descriptor format.
The packaged registry is `flashinfer_npu/data/kernels/registry.json` and is
currently empty because no Ascend C artifact has passed the Phase 0 vertical
slice yet. A kernel must not be registered before its artifact and capability
constraints are real and tested.

