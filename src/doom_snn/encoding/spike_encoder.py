"""Spike encoder implementations for SNN DOOM pipeline.

Provides deterministic accumulator-based rate encoding and stochastic Poisson
encoding to convert real-valued observation frames into binary spike trains.
"""

from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor


class SpikeStats(float):
    """Container for spike statistics inheriting from float for full compatibility.

    Supports numeric operations directly as a float (mean spike rate), as well
    as dictionary-like access (`stats['mean_spike_rate']`) or attribute access.
    """

    def __getitem__(self, key: str) -> float:
        if key in ("mean_spike_rate", "spike_rate", "rate", "mean"):
            return float(self)
        raise KeyError(f"Invalid key '{key}'. Available keys: 'mean_spike_rate'")

    def get(self, key: str, default: Any = None) -> Any:
        if key in ("mean_spike_rate", "spike_rate", "rate", "mean"):
            return float(self)
        return default

    @property
    def mean_spike_rate(self) -> float:
        return float(self)

    def to_dict(self) -> dict[str, float]:
        return {"mean_spike_rate": float(self)}

    def __repr__(self) -> str:
        return f"SpikeStats(mean_spike_rate={float(self):.4f})"


class SpikeEncoder(nn.Module):
    """Encodes continuous frame observations into temporal spike trains.

    Supports deterministic accumulator encoding (hardware-friendly) and
    stochastic Poisson rate encoding over T simulation timesteps.

    Args:
        input_channels: Number of continuous input features per sample (default: 768).
        T: Number of temporal simulation steps per frame (default: 8).
        method: Spike generation method: 'accumulator' or 'poisson' (default: 'accumulator').
        scale: Optional scaling factor applied to inputs during accumulation (default: 1.0).
    """

    def __init__(
        self,
        input_channels: int = 768,
        T: int = 8,
        method: str = "accumulator",
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if method not in ("accumulator", "poisson"):
            raise ValueError(
                f"Unknown encoding method '{method}'. Supported methods: 'accumulator', 'poisson'."
            )
        if T <= 0:
            raise ValueError(f"Timesteps T must be a positive integer, got {T}.")
        if input_channels <= 0:
            raise ValueError(
                f"input_channels must be a positive integer, got {input_channels}."
            )

        self.input_channels = input_channels
        self.T = T
        self.method = method
        self.scale = float(scale)

        self.accumulator: Optional[Tensor] = None
        self._last_spikes: Optional[Tensor] = None

    def reset_state(self) -> None:
        """Reset internal accumulator state to zeros."""
        if self.accumulator is not None:
            self.accumulator.zero_()
        self._last_spikes = None

    def forward(self, x: Tensor) -> Tensor:
        """Encode batch of normalized continuous inputs into binary spike tensors.

        Args:
            x: Input tensor of shape [batch, input_channels] with values in [0, 1].
               Also accepts 1D tensor [input_channels] which will be unbatched.

        Returns:
            Spikes tensor of shape [T, batch, input_channels] containing binary
            floats (0.0 or 1.0).
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(
                f"Expected input tensor of shape [batch, input_channels], got shape {list(x.shape)}."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.input_channels}, got {x.shape[1]}."
            )

        # Ensure inputs are safely clamped within normalized bounds [0, 1]
        x_clamped = torch.clamp(x, 0.0, 1.0)

        if self.method == "accumulator":
            if (
                self.accumulator is None
                or self.accumulator.shape != x_clamped.shape
                or self.accumulator.device != x_clamped.device
                or self.accumulator.dtype != x_clamped.dtype
            ):
                self.accumulator = torch.zeros_like(x_clamped)

            spikes_list = []
            acc = self.accumulator
            scaled_input = x_clamped * self.scale

            for _ in range(self.T):
                acc = acc + scaled_input
                spike = (acc >= 1.0).to(dtype=x.dtype)
                acc = acc - spike * 1.0
                spikes_list.append(spike)

            self.accumulator = acc.detach()
            spikes = torch.stack(spikes_list, dim=0)

        elif self.method == "poisson":
            spikes_list = []
            for _ in range(self.T):
                rand_draw = torch.rand_like(x_clamped)
                spike = (rand_draw < x_clamped).to(dtype=x.dtype)
                spikes_list.append(spike)
            spikes = torch.stack(spikes_list, dim=0)
        else:
            raise RuntimeError(f"Unhandled method: {self.method}")

        if was_1d:
            spikes = spikes.squeeze(1)

        self._last_spikes = spikes.detach()
        return spikes

    def get_spike_stats(self) -> SpikeStats:
        """Return the mean spike rate of the most recent forward pass.

        Returns:
            SpikeStats instance (inherits from float) representing the mean
            firing rate in range [0, 1], with dictionary-like accessor compatibility.
        """
        if self._last_spikes is None:
            return SpikeStats(0.0)
        mean_rate = float(self._last_spikes.mean().item())
        return SpikeStats(mean_rate)
