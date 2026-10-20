"""Leaky Integrate-and-Fire (LIF) and Recurrent LIF neuron layers.

Implements discrete-time leaky integrate-and-fire dynamics with hardware-aware
decay factors (beta in {0.25, 0.5, 0.75}), configurable reset modes (soft or hard),
and fast-sigmoid surrogate gradients for backpropagation through time.
"""

from typing import Optional, Union
import warnings
import torch
import torch.nn as nn
from torch import Tensor

from doom_snn.neurons.states import LIFState, init_lif_state
from doom_snn.neurons.surrogate import fast_sigmoid_spike

RECOMMENDED_BETAS = {0.25, 0.5, 0.75}


class LIFLayer(nn.Module):
    """Feedforward Leaky Integrate-and-Fire (LIF) neural layer.

    Args:
        input_size: Number of input features.
        hidden_size: Number of LIF neurons.
        beta: Membrane potential decay factor. Recommended values for FPGA
              bitshift compatibility: 0.25 (>> 2), 0.5 (>> 1), 0.75 (x - (x >> 2)).
        threshold: Firing threshold potential (default: 0.4).
        reset_mode: Membrane potential reset mode ('soft' or 'hard', default: 'soft').
        surrogate: Surrogate gradient function name (default: 'fast_sigmoid').
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        beta: float = 0.75,
        threshold: float = 0.4,
        reset_mode: str = "soft",
        surrogate: str = "fast_sigmoid",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.reset_mode = reset_mode
        self.surrogate = surrogate

        if self.beta not in RECOMMENDED_BETAS:
            warnings.warn(
                f"Beta {self.beta} is not in recommended hardware-friendly set {RECOMMENDED_BETAS}. "
                "Hardware bitshift optimizations require beta in {0.25, 0.5, 0.75}.",
                UserWarning,
                stacklevel=2,
            )

        if self.reset_mode not in ("soft", "hard"):
            raise ValueError(
                f"Unsupported reset_mode '{reset_mode}'. Expected 'soft' or 'hard'."
            )

        if self.surrogate != "fast_sigmoid":
            raise ValueError(
                f"Unsupported surrogate '{surrogate}'. Expected 'fast_sigmoid'."
            )

        self.fc = nn.Linear(input_size, hidden_size, bias=True)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LIFState:
        """Initialize zeroed LIFState for this layer.

        Args:
            batch_size: Number of items in batch.
            device: Device for state tensors (defaults to fc weight device).
            dtype: Tensor dtype (defaults to fc weight dtype).

        Returns:
            Zero-initialized LIFState.
        """
        if device is None:
            device = self.fc.weight.device
        if dtype is None:
            dtype = self.fc.weight.dtype
        return init_lif_state(
            batch_size=batch_size,
            num_neurons=self.hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        state: Optional[LIFState] = None,
    ) -> tuple[Tensor, Tensor, LIFState]:
        """Compute one step of feedforward LIF neuron dynamics.

        Args:
            x: Input tensor of shape [batch, input_size].
            state: Optional previous LIFState. If None, state is initialized to zeros.

        Returns:
            Tuple of:
                - spikes: Binary spike tensor of shape [batch, hidden_size].
                - membrane: Updated post-reset membrane potential tensor of shape [batch, hidden_size].
                - new_state: LIFState containing updated membrane potential and spikes.
        """
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        elif state.membrane.device != x.device or state.membrane.dtype != x.dtype:
            state = state.to(device=x.device, dtype=x.dtype)

        current = self.fc(x)
        membrane = self.beta * state.membrane + current
        spikes = fast_sigmoid_spike(membrane, self.threshold)

        if self.reset_mode == "soft":
            membrane = membrane - self.threshold * spikes
        elif self.reset_mode == "hard":
            membrane = membrane * (1.0 - spikes)

        new_state = LIFState(membrane=membrane, spikes=spikes)
        return spikes, membrane, new_state


class RecurrentLIFLayer(nn.Module):
    """Recurrent Leaky Integrate-and-Fire (RLIF) neural layer.

    Includes feedforward synaptic inputs and learned recurrent lateral feedback
    from spikes generated in the preceding simulation step.

    Args:
        input_size: Number of feedforward input features.
        hidden_size: Number of recurrent LIF neurons.
        beta: Membrane potential decay factor. Recommended values for FPGA
              bitshift compatibility: 0.25 (>> 2), 0.5 (>> 1), 0.75 (x - (x >> 2)).
        threshold: Firing threshold potential (default: 0.4).
        reset_mode: Membrane potential reset mode ('soft' or 'hard', default: 'soft').
        surrogate: Surrogate gradient function name (default: 'fast_sigmoid').
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        beta: float = 0.75,
        threshold: float = 0.4,
        reset_mode: str = "soft",
        surrogate: str = "fast_sigmoid",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.reset_mode = reset_mode
        self.surrogate = surrogate

        if self.beta not in RECOMMENDED_BETAS:
            warnings.warn(
                f"Beta {self.beta} is not in recommended hardware-friendly set {RECOMMENDED_BETAS}. "
                "Hardware bitshift optimizations require beta in {0.25, 0.5, 0.75}.",
                UserWarning,
                stacklevel=2,
            )

        if self.reset_mode not in ("soft", "hard"):
            raise ValueError(
                f"Unsupported reset_mode '{reset_mode}'. Expected 'soft' or 'hard'."
            )

        if self.surrogate != "fast_sigmoid":
            raise ValueError(
                f"Unsupported surrogate '{surrogate}'. Expected 'fast_sigmoid'."
            )

        self.fc_input = nn.Linear(input_size, hidden_size, bias=True)
        self.fc_recurrent = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LIFState:
        """Initialize zeroed LIFState for this recurrent layer.

        Args:
            batch_size: Number of items in batch.
            device: Device for state tensors (defaults to fc_input weight device).
            dtype: Tensor dtype (defaults to fc_input weight dtype).

        Returns:
            Zero-initialized LIFState.
        """
        if device is None:
            device = self.fc_input.weight.device
        if dtype is None:
            dtype = self.fc_input.weight.dtype
        return init_lif_state(
            batch_size=batch_size,
            num_neurons=self.hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        state: Optional[LIFState] = None,
    ) -> tuple[Tensor, Tensor, LIFState]:
        """Compute one step of recurrent LIF neuron dynamics.

        Args:
            x: Feedforward input tensor of shape [batch, input_size].
            state: Optional previous LIFState containing membrane and lateral spikes.
                   If None, state is initialized to zeros.

        Returns:
            Tuple of:
                - spikes: Binary spike tensor of shape [batch, hidden_size].
                - membrane: Updated post-reset membrane potential tensor of shape [batch, hidden_size].
                - new_state: LIFState containing updated membrane potential and lateral spikes.
        """
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        elif state.membrane.device != x.device or state.membrane.dtype != x.dtype:
            state = state.to(device=x.device, dtype=x.dtype)

        current = self.fc_input(x) + self.fc_recurrent(state.spikes)
        membrane = self.beta * state.membrane + current
        spikes = fast_sigmoid_spike(membrane, self.threshold)

        if self.reset_mode == "soft":
            membrane = membrane - self.threshold * spikes
        elif self.reset_mode == "hard":
            membrane = membrane * (1.0 - spikes)

        new_state = LIFState(membrane=membrane, spikes=spikes)
        return spikes, membrane, new_state
