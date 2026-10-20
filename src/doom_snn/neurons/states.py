"""Neuron state representations and initialization for LIF layers."""

from dataclasses import dataclass
from typing import Any, Optional
import torch
from torch import Tensor


@dataclass
class LIFState:
    """State of a Leaky Integrate-and-Fire neuron layer.

    Attributes:
        membrane: Sub-threshold membrane potential tensor.
        spikes: Spike activations from the most recent timestep.
    """

    membrane: Tensor
    spikes: Tensor

    def detach(self) -> "LIFState":
        """Detach membrane potential and spike tensors from computation graph.

        Useful for truncated backpropagation through time (BPTT).
        """
        return LIFState(
            membrane=self.membrane.detach(),
            spikes=self.spikes.detach(),
        )

    def clone(self) -> "LIFState":
        """Return a deep copy of the state tensors."""
        return LIFState(
            membrane=self.membrane.clone(),
            spikes=self.spikes.clone(),
        )

    def to(self, *args: Any, **kwargs: Any) -> "LIFState":
        """Cast and/or move state tensors to a target device/dtype."""
        return LIFState(
            membrane=self.membrane.to(*args, **kwargs),
            spikes=self.spikes.to(*args, **kwargs),
        )


def init_lif_state(
    batch_size: int,
    num_neurons: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LIFState:
    """Initialize LIF neuron state with zeros.

    Args:
        batch_size: Batch dimension size.
        num_neurons: Number of neurons in the layer.
        device: Target torch device (CPU or CUDA).
        dtype: Data type for tensors (default: torch.float32).

    Returns:
        LIFState initialized with zero membrane potential and zero spikes.
    """
    if dtype is None:
        dtype = torch.float32
    membrane = torch.zeros(batch_size, num_neurons, device=device, dtype=dtype)
    spikes = torch.zeros(batch_size, num_neurons, device=device, dtype=dtype)
    return LIFState(membrane=membrane, spikes=spikes)
