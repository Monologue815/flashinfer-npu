"""Optional lazy Torch adapter for the framework-independent Attention views.

This module deliberately imports no torch symbols at module import time.  A
default adapter resolves ``torch.Tensor`` lazily; tests may inject a strict
tensor protocol type without pretending that a real Torch runtime was tested.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from flashinfer_npu.runtime import QuantSpec, SchemaError

from .planner import AttentionFrameworkPlan
from .schema import KVCacheSpec
from .tensor_contract import (
    AttentionRunTensorContract,
    AttentionTensorAccessPolicy,
    KVCacheView,
    QuantizedTensorView,
    StreamContext,
    TensorView,
    dtype_itemsize,
)


class TorchAdapterUnavailableError(ImportError):
    """Raised when the optional Torch frontend is requested without Torch."""


@dataclass(frozen=True)
class TorchQuantizedTensorInput:
    logical_shape: Tuple[int, ...]
    storage: object
    scale: object
    quant_spec: QuantSpec
    zero_point: Optional[object] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_shape", tuple(int(dim) for dim in self.logical_shape)
        )


def _call_or_value(value):
    return value() if callable(value) else value


def _bool_tensor_attribute(tensor, name: str, default=False) -> bool:
    if not hasattr(tensor, name):
        return bool(default)
    value = _call_or_value(getattr(tensor, name))
    if not isinstance(value, bool):
        raise SchemaError("Torch tensor %s must be boolean" % name)
    return value


def _canonical_torch_dtype(value) -> str:
    name = str(value)
    if name.startswith("torch."):
        name = name[6:]
    aliases = {
        "float": "float32",
        "double": "float64",
        "half": "float16",
        "long": "int64",
        "int": "int32",
        "short": "int16",
        "char": "int8",
        "byte": "uint8",
    }
    return aliases.get(name, name)


def _pointer_alignment(pointer: int) -> int:
    pointer = int(pointer)
    if pointer < 0:
        raise SchemaError("Torch data pointer cannot be negative")
    if pointer == 0:
        return 1
    return pointer & -pointer


def _opaque_storage_id(device: str, pointer: int, nbytes: int) -> str:
    payload = "%s:%d:%d" % (device, int(pointer), int(nbytes))
    return "torch-storage:" + hashlib.sha256(
        payload.encode("ascii")
    ).hexdigest()


class TorchTensorViewAdapter:
    """Map actual or protocol-compatible Torch tensors without copying them."""

    def __init__(
        self,
        *,
        tensor_type=None,
        torch_module=None,
        stream_resolver: Optional[Callable[[str], str]] = None,
    ) -> None:
        if tensor_type is None:
            if torch_module is None:
                try:
                    import torch as torch_module  # type: ignore
                except (ImportError, ModuleNotFoundError) as error:
                    raise TorchAdapterUnavailableError(
                        "Torch is not installed; install a supported Torch runtime "
                        "before constructing the Torch adapter"
                    ) from error
            tensor_type = getattr(torch_module, "Tensor", None)
            if tensor_type is None:
                raise TorchAdapterUnavailableError(
                    "the imported torch module does not expose torch.Tensor"
                )
        self._tensor_type = tensor_type
        self._torch_module = torch_module
        self._stream_resolver = stream_resolver

    def to_view(
        self,
        tensor,
        *,
        name: str,
        writable: bool = False,
    ) -> TensorView:
        self._require_tensor(tensor, name)
        self._validate_tensor_state(tensor, name)
        device = str(tensor.device)
        if device.split(":", 1)[0] == "meta":
            raise SchemaError("%s cannot use the meta device" % name)
        dtype = _canonical_torch_dtype(tensor.dtype)
        itemsize = dtype_itemsize(dtype)
        try:
            reported_itemsize = int(tensor.element_size())
        except (AttributeError, TypeError, ValueError) as error:
            raise SchemaError("%s must expose element_size()" % name) from error
        if reported_itemsize != itemsize:
            raise SchemaError(
                "%s element_size does not match canonical dtype %s" % (name, dtype)
            )
        try:
            storage = tensor.untyped_storage()
            storage_nbytes = int(_call_or_value(storage.nbytes))
            storage_pointer = int(storage.data_ptr())
            tensor_pointer = int(tensor.data_ptr())
            shape = tuple(int(dim) for dim in tensor.shape)
            strides = tuple(int(stride) for stride in tensor.stride())
            storage_offset = int(tensor.storage_offset())
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise SchemaError(
                "%s does not expose a regular strided untyped storage" % name
            ) from error
        if storage_nbytes < 0 or storage_pointer < 0:
            raise SchemaError("%s storage metadata cannot be negative" % name)
        storage_identity_pointer = (
            storage_pointer if storage_pointer != 0 else id(storage)
        )
        return TensorView(
            shape=shape,
            strides=strides,
            dtype=dtype,
            device=device,
            storage_id=_opaque_storage_id(
                device, storage_identity_pointer, storage_nbytes
            ),
            storage_nbytes=storage_nbytes,
            storage_offset=storage_offset,
            data_ptr_alignment=_pointer_alignment(tensor_pointer),
            writable=bool(writable),
        )

    def to_quantized_view(
        self,
        value: TorchQuantizedTensorInput,
        *,
        name: str,
    ) -> QuantizedTensorView:
        return QuantizedTensorView(
            logical_shape=value.logical_shape,
            storage=self.to_view(value.storage, name="%s.storage" % name),
            scale=self.to_view(value.scale, name="%s.scale" % name),
            zero_point=(
                self.to_view(value.zero_point, name="%s.zero_point" % name)
                if value.zero_point is not None
                else None
            ),
            quant_spec=value.quant_spec,
        )

    def to_kv_view(self, value, spec: KVCacheSpec) -> KVCacheView:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise TypeError("separate Torch KV input must be a (K, V) pair")
            key, val = value
            if isinstance(key, TorchQuantizedTensorInput) or isinstance(
                val, TorchQuantizedTensorInput
            ):
                if not isinstance(key, TorchQuantizedTensorInput) or not isinstance(
                    val, TorchQuantizedTensorInput
                ):
                    raise TypeError("quantized Torch K and V must be provided together")
                return KVCacheView(
                    spec,
                    self.to_quantized_view(key, name="kv.key"),
                    self.to_quantized_view(val, name="kv.value"),
                )
            return KVCacheView(
                spec,
                self.to_view(key, name="kv.key"),
                self.to_view(val, name="kv.value"),
            )
        packed = self.to_view(value, name="kv.packed")
        return KVCacheView(spec, packed, packed, packed=True)

    def stream_context(self, device: str) -> StreamContext:
        device_type = str(device).split(":", 1)[0]
        if device_type == "cpu":
            return StreamContext(str(device), "torch-cpu-synchronous")
        if self._stream_resolver is None:
            raise SchemaError(
                "accelerator tensor requires an explicit current-stream resolver"
            )
        try:
            stream_id = self._stream_resolver(str(device))
        except Exception as error:
            raise SchemaError("current-stream resolution failed") from error
        if not isinstance(stream_id, str) or not stream_id:
            raise SchemaError("stream resolver must return a non-empty opaque string")
        return StreamContext(str(device), stream_id)

    def build_run_contract(
        self,
        *,
        q,
        kv_data,
        kv_spec: KVCacheSpec,
        plan: AttentionFrameworkPlan,
        out=None,
        lse=None,
        workspace_float=None,
        workspace_int=None,
        policy: Optional[AttentionTensorAccessPolicy] = None,
    ) -> AttentionRunTensorContract:
        q_view = self.to_view(q, name="q")
        contract = AttentionRunTensorContract(
            q=q_view,
            kv=self.to_kv_view(kv_data, kv_spec),
            stream=self.stream_context(q_view.device),
            out=(
                self.to_view(out, name="out", writable=True)
                if out is not None
                else None
            ),
            lse=(
                self.to_view(lse, name="lse", writable=True)
                if lse is not None
                else None
            ),
            workspace_float=(
                self.to_view(
                    workspace_float, name="workspace_float", writable=True
                )
                if workspace_float is not None
                else None
            ),
            workspace_int=(
                self.to_view(workspace_int, name="workspace_int", writable=True)
                if workspace_int is not None
                else None
            ),
        )
        contract.validate(policy or AttentionTensorAccessPolicy(), plan=plan)
        return contract

    def _require_tensor(self, value, name: str) -> None:
        if not isinstance(value, self._tensor_type):
            raise TypeError("%s must be torch.Tensor" % name)

    def _validate_tensor_state(self, tensor, name: str) -> None:
        layout = str(getattr(tensor, "layout", ""))
        if layout not in {"torch.strided", "strided"}:
            raise SchemaError("%s must use torch.strided layout" % name)
        if _bool_tensor_attribute(tensor, "requires_grad"):
            raise SchemaError("%s requires_grad is unsupported for inference" % name)
        for attribute in ("is_conj", "is_neg", "is_sparse", "is_nested"):
            if _bool_tensor_attribute(tensor, attribute):
                raise SchemaError("%s %s state is unsupported" % (name, attribute))
        if _bool_tensor_attribute(tensor, "is_quantized"):
            raise SchemaError(
                "%s native Torch quantized tensor is unsupported; provide explicit "
                "storage/scale/zero views" % name
            )


__all__ = [
    "TorchAdapterUnavailableError",
    "TorchQuantizedTensorInput",
    "TorchTensorViewAdapter",
]
