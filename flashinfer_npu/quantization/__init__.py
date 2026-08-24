"""Public quantization contracts.

Kernel implementations will be added behind this API without changing the
versioned QuantSpec contract.
"""

from flashinfer_npu.runtime.schema import QuantSpec

__all__ = ["QuantSpec"]

