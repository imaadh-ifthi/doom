"""Surrogate gradient functions for non-differentiable spiking neurons."""

from typing import Any, Union
import torch
from torch import Tensor


class FastSigmoidSurrogate(torch.autograd.Function):
    """Autograd function implementing fast-sigmoid surrogate gradient for Heaviside step.

    Forward:
        s = 1.0 if membrane >= threshold else 0.0

    Backward:
        ds/du = 1.0 / (1.0 + 2.0 * |u - V_th|)^2
    """

    @staticmethod
    def forward(ctx: Any, membrane: Tensor, threshold: Union[float, Tensor]) -> Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = (
            float(threshold) if isinstance(threshold, (int, float)) else float(threshold.item())
        )
        return (membrane >= threshold).to(dtype=membrane.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        diff = torch.abs(membrane - threshold)
        grad = 1.0 / ((1.0 + 2.0 * diff) ** 2)
        grad_membrane = grad_output * grad
        return grad_membrane, None


def fast_sigmoid_spike(
    membrane: Tensor,
    threshold: Union[float, Tensor] = 0.4,
) -> Tensor:
    """Compute binary spikes with fast-sigmoid surrogate derivative for backpropagation.

    Args:
        membrane: Sub-threshold membrane potential tensor.
        threshold: Neuron firing threshold potential (default: 0.4).

    Returns:
        Binary spike tensor (0.0 or 1.0) with surrogate gradient attached.
    """
    return FastSigmoidSurrogate.apply(membrane, threshold)
