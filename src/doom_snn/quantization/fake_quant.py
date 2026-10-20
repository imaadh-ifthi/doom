"""Fake quantization modules implementing Straight-Through Estimators (STE) for INT16."""

import logging
from typing import Any, Optional
import torch
import torch.nn as nn
from torch import Tensor

from doom_snn.quantization.fixed_point import (
    INT16_MAX,
    INT16_MIN,
    compute_scale,
)

logger = logging.getLogger(__name__)


class FakeQuantInt16Function(torch.autograd.Function):
    """Autograd function implementing Straight-Through Estimator (STE) for INT16 quantization."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        scale: Tensor,
        int_min: int,
        int_max: int,
    ) -> Tensor:
        q = torch.clamp(torch.round(x / scale), int_min, int_max)
        x_hat = q * scale
        ctx.save_for_backward(x, scale, q)
        return x_hat

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Optional[Tensor], None, None]:
        x, scale, q = ctx.saved_tensors
        # Straight-through: pass upstream gradient through as identity
        grad_x = grad_output.clone()
        grad_scale = None
        if ctx.needs_input_grad[1]:
            # LSQ gradient for learnable quantization step size
            grad_scale = ((q - (x / scale)) * grad_output).sum()
        return grad_x, grad_scale, None, None


class FakeQuantInt16(nn.Module):
    """Simulates signed 16-bit integer quantization during training (QAT).

    Quantizes continuous tensor values to integers in range [-32768, 32767]
    and dequantizes back to floats, applying a Straight-Through Estimator (STE)
    gradient so that the model remains end-to-end differentiable.

    Args:
        scale: Fixed quantization scale factor. If None, scale is dynamically
               estimated from input tensor dynamic range.
        learnable_scale: Whether the scale parameter should be learned via
                         backpropagation (default: False).
    """

    def __init__(
        self,
        scale: Optional[float] = None,
        learnable_scale: bool = False,
    ) -> None:
        super().__init__()
        self.learnable_scale = learnable_scale
        self._last_saturation_count: int = 0
        self._last_scale: Optional[float] = float(scale) if scale is not None else None

        if scale is not None:
            scale_t = torch.tensor(float(scale), dtype=torch.float32)
            if learnable_scale:
                self.scale: Optional[nn.Parameter] = nn.Parameter(scale_t)
            else:
                self.register_buffer("scale", scale_t)
        else:
            self.scale = None

    @property
    def last_saturation_count(self) -> int:
        """Return the number of saturated values in the most recent forward pass."""
        return self._last_saturation_count

    @property
    def last_scale(self) -> Optional[float]:
        """Return the quantization scale used in the most recent forward pass."""
        return self._last_scale

    def forward(self, x: Tensor) -> Tensor:
        """Apply fake INT16 quantization to input tensor x.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Fake-quantized tensor x_hat with identical shape and dtype.
        """
        # Determine scale: static, dynamic, or initialize learnable
        if self.scale is None:
            scale_val = compute_scale(x, bits=16)
            scale_tensor = torch.tensor(scale_val, device=x.device, dtype=x.dtype)
            if self.learnable_scale:
                self.scale = nn.Parameter(scale_tensor)
                current_scale = self.scale
            else:
                current_scale = scale_tensor
        else:
            current_scale = self.scale.to(device=x.device, dtype=x.dtype)

        # Ensure scale is strictly positive
        current_scale = torch.clamp(current_scale, min=1e-12)

        # Track saturation statistics
        with torch.no_grad():
            unclamped = torch.round(x / current_scale)
            saturations = (unclamped < INT16_MIN) | (unclamped > INT16_MAX)
            sat_count = int(saturations.sum().item())
            self._last_saturation_count = sat_count
            self._last_scale = float(current_scale.detach().cpu().item())
            if self.training and sat_count > 0:
                logger.debug(
                    "FakeQuantInt16: %d elements saturated outside INT16 range [-32768, 32767].",
                    sat_count,
                )

        return FakeQuantInt16Function.apply(x, current_scale, INT16_MIN, INT16_MAX)
