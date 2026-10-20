"""Fixed-point numerical operations and hardware bitshift utilities for SNNs."""

from typing import Union
import torch
from torch import Tensor

INT16_MIN: int = -32768
INT16_MAX: int = 32767


def float_to_int16(x: Tensor, scale: float) -> Tensor:
    """Quantize floating point tensor to signed 16-bit integers.

    Args:
        x: Floating-point input tensor.
        scale: Floating-point quantization step size.

    Returns:
        Tensor of type torch.int16 clamped to [-32768, 32767].
    """
    scale_val = max(float(scale), 1e-12)
    q = torch.clamp(torch.round(x / scale_val), INT16_MIN, INT16_MAX)
    return q.to(torch.int16)


def int16_to_float(q: Tensor, scale: float) -> Tensor:
    """De-quantize signed 16-bit integer tensor to floating point.

    Args:
        q: Integer tensor (int16 or integer-valued float).
        scale: Floating-point quantization step size.

    Returns:
        Tensor of type torch.float32 representing dequantized values.
    """
    return q.to(torch.float32) * float(scale)


def compute_scale(tensor: Tensor, bits: int = 16) -> float:
    """Compute symmetric quantization scale factor from tensor dynamic range.

    Args:
        tensor: Tensor whose values determine the quantization range.
        bits: Bit width of target signed integer (default: 16).

    Returns:
        Scale factor = max(|tensor|) / (2^(bits-1) - 1).
    """
    q_max = (1 << (bits - 1)) - 1
    max_val = float(torch.max(torch.abs(tensor)).item())
    if max_val == 0.0:
        return 1.0 / q_max
    return max_val / q_max


def count_saturations(x: Tensor, scale: float) -> int:
    """Count how many elements in tensor would exceed the int16 dynamic range.

    Args:
        x: Floating-point input tensor.
        scale: Quantization step size.

    Returns:
        Integer count of saturating values (< -32768 or > 32767).
    """
    scale_val = max(float(scale), 1e-12)
    unclamped = torch.round(x / scale_val)
    saturations = (unclamped < INT16_MIN) | (unclamped > INT16_MAX)
    return int(saturations.sum().item())


def beta_shift_int16(membrane_q: Tensor, beta: float) -> Tensor:
    """Simulate hardware arithmetic bitshift operations for membrane potential decay.

    Hardware mapping:
        beta = 0.25: membrane_q >> 2 (floor division by 4)
        beta = 0.50: membrane_q >> 1 (floor division by 2)
        beta = 0.75: membrane_q - (membrane_q >> 2) (i.e. x - floor(x / 4))

    Args:
        membrane_q: Integer or integer-valued membrane tensor.
        beta: Decay factor in {0.25, 0.5, 0.75}.

    Returns:
        torch.int16 tensor representing shifted/decayed integer membrane.
    """
    # Use int32 intermediate representation to prevent overflow during arithmetic
    if torch.is_floating_point(membrane_q):
        int_vals = torch.round(membrane_q).to(torch.int32)
    else:
        int_vals = membrane_q.to(torch.int32)

    if abs(beta - 0.25) < 1e-5:
        shifted = torch.div(int_vals, 4, rounding_mode="floor")
    elif abs(beta - 0.50) < 1e-5:
        shifted = torch.div(int_vals, 2, rounding_mode="floor")
    elif abs(beta - 0.75) < 1e-5:
        shifted = int_vals - torch.div(int_vals, 4, rounding_mode="floor")
    else:
        raise ValueError(
            f"Unsupported beta {beta} for bitshift operation. "
            "FPGA bitshift requires beta in {0.25, 0.5, 0.75}."
        )

    clamped = torch.clamp(shifted, INT16_MIN, INT16_MAX)
    return clamped.to(torch.int16)
