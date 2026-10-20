"""Track B-RNN: Recurrent Spike-DQN model for sequential value estimation (Experimental)."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor

from doom_snn.encoding.spike_encoder import SpikeEncoder
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.neurons.lif import RecurrentLIFLayer
from doom_snn.neurons.states import LIFState


class RecurrentSpikeDQN(SNNPolicy):
    """Recurrent Spiking Deep Q-Network for ViZDoom control.

    Note:
        Recurrent value-based learning can exhibit training instability.
        Marked as an exploratory extension behind optional flags.
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
        self.is_recurrent = True
        self.is_actor_critic = False
        self.track = "b_recurrent_spike_dqn"

        self.encoder = SpikeEncoder(
            input_channels=input_channels,
            T=T,
            method="accumulator",
        )

        self.core = RecurrentLIFLayer(
            input_size=input_channels,
            hidden_size=hidden_size,
            beta=beta,
            threshold=threshold,
            reset_mode=reset_mode,
            surrogate="fast_sigmoid",
        )

        self.q_head = nn.Linear(hidden_size, num_actions, bias=True)

        self._state: Optional[LIFState] = None
        self.last_input_spike_rate: float = 0.0
        self.last_hidden_spike_rate: float = 0.0

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.core.fc_input.weight, gain=1.0)
        nn.init.constant_(self.core.fc_input.bias, 0.0)
        nn.init.orthogonal_(self.core.fc_recurrent.weight, gain=0.2)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.constant_(self.q_head.bias, 0.0)

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset internal recurrent state."""
        self._state = None
        self.encoder.reset_state()

    def forward(
        self,
        obs: Tensor,
        state: Optional[LIFState] = None,
    ) -> Dict[str, Any]:
        """Forward pass computing Q-values with recurrent state retention."""
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        in_spikes = self.encoder(obs)
        self.last_input_spike_rate = float(in_spikes.mean().item())

        if state is not None:
            curr_state = state
        elif self._state is not None and self._state.membrane.shape[0] == batch_size and self._state.membrane.device == device:
            curr_state = self._state
        else:
            curr_state = self.core.init_state(batch_size, device=device, dtype=obs.dtype)

        spikes_list = []
        for t in range(self.T):
            s_t, u_t, curr_state = self.core(in_spikes[t], curr_state)
            spikes_list.append(s_t)

        self._state = LIFState(
            membrane=curr_state.membrane.detach(),
            spikes=curr_state.spikes.detach(),
        )

        all_spikes = torch.stack(spikes_list, dim=0)
        self.last_hidden_spike_rate = float(all_spikes.mean().item())
        final_membrane = curr_state.membrane

        readout = final_membrane if self.readout_mode == "membrane" else all_spikes.mean(dim=0)
        q_values = self.q_head(readout)

        return {
            "q_values": q_values,
            "spikes": all_spikes,
            "membrane": final_membrane,
            "readout": readout,
            "state": self._state,
            "mean_input_spike_rate": self.last_input_spike_rate,
            "mean_hidden_spike_rate": self.last_hidden_spike_rate,
        }
