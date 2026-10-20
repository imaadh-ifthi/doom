"""Track B: Feed-Forward Spike-DQN model for value-based neuromorphic control."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from doom_snn.encoding.spike_encoder import SpikeEncoder
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.neurons.lif import LIFLayer


class SpikeDQN(SNNPolicy):
    """Feed-forward Spiking Deep Q-Network for ViZDoom control.

    Maps spike-encoded observations directly to action Q-values via continuous
    membrane potential readout.
    """

    def __init__(
        self,
        input_channels: int = 768,
        hidden_size: int = 96,
        num_actions: int = 6,
        beta: float = 0.75,
        threshold: float = 0.4,
        T: int = 8,
        reset_mode: str = "soft",
        readout: str = "membrane",
        quantization_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.core_neurons = hidden_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.beta = beta
        self.threshold = threshold
        self.T = T
        self.reset_mode = reset_mode
        self.readout_mode = readout
        self.is_recurrent = False
        self.is_actor_critic = False
        self.track = "b_spike_dqn"

        # 1. Deterministic temporal rate encoder
        self.encoder = SpikeEncoder(
            input_channels=input_channels,
            T=T,
            method="accumulator",
        )

        # 2. Feedforward LIF core
        self.core = LIFLayer(
            input_size=input_channels,
            hidden_size=hidden_size,
            beta=beta,
            threshold=threshold,
            reset_mode=reset_mode,
            surrogate="fast_sigmoid",
        )

        # 3. Readout head: linear regression to Q(s, a)
        self.q_head = nn.Linear(hidden_size, num_actions, bias=True)

        self.last_input_spike_rate: float = 0.0
        self.last_hidden_spike_rate: float = 0.0

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.core.fc.weight, gain=1.0)
        nn.init.constant_(self.core.fc.bias, 0.0)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.constant_(self.q_head.bias, 0.0)

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset spike encoder accumulator state."""
        self.encoder.reset_state()

    def forward(
        self,
        obs: Tensor,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Forward pass computing Q-values across T spiking timesteps."""
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        in_spikes = self.encoder(obs)  # [T, batch, input_channels]
        self.last_input_spike_rate = float(in_spikes.mean().item())

        layer_state = self.core.init_state(batch_size, device=device, dtype=obs.dtype)
        spikes_list = []

        for t in range(self.T):
            s_t, u_t, layer_state = self.core(in_spikes[t], layer_state)
            spikes_list.append(s_t)

        all_spikes = torch.stack(spikes_list, dim=0)
        self.last_hidden_spike_rate = float(all_spikes.mean().item())

        final_membrane = layer_state.membrane

        if self.readout_mode == "membrane":
            readout = final_membrane
        elif self.readout_mode == "spike_count":
            readout = all_spikes.mean(dim=0)
        else:
            readout = final_membrane

        q_values = self.q_head(readout)

        return {
            "q_values": q_values,
            "spikes": all_spikes,
            "membrane": final_membrane,
            "readout": readout,
            "mean_input_spike_rate": self.last_input_spike_rate,
            "mean_hidden_spike_rate": self.last_hidden_spike_rate,
        }
