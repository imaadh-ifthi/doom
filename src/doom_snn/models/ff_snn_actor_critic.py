"""Track D: Feed-Forward Spiking Neural Network (FF-SNN) Actor-Critic model.

Scientific ablation isolating the performance and memory contribution of recurrent
synaptic feedback in neuromorphic DOOM control.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor

from doom_snn.encoding.spike_encoder import SpikeEncoder
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.neurons.lif import LIFLayer
from doom_snn.neurons.states import LIFState


class FFSNNActorCritic(SNNPolicy):
    """Feed-Forward Spiking Neural Network Actor-Critic for ViZDoom control.

    Contains no recurrent lateral feedback weights ($W_{rec} = 0$).
    Serves as an architectural control to measure the value of temporal recurrence.
    """

    def __init__(
        self,
        input_channels: int = 768,
        hidden_sizes: Union[int, List[int]] = 96,
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
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes]
        else:
            self.hidden_sizes = list(hidden_sizes)

        self.core_neurons = self.hidden_sizes[-1]
        self.num_actions = num_actions
        self.beta = beta
        self.threshold = threshold
        self.T = T
        self.reset_mode = reset_mode
        self.readout_mode = readout
        self.is_recurrent = False
        self.is_actor_critic = True
        self.track = "d_ff_snn"

        # 1. Deterministic temporal rate encoder
        self.encoder = SpikeEncoder(
            input_channels=input_channels,
            T=T,
            method="accumulator",
        )

        # 2. Feedforward LIF hidden layer(s)
        self.layers = nn.ModuleList()
        in_dim = input_channels
        for h_dim in self.hidden_sizes:
            self.layers.append(
                LIFLayer(
                    input_size=in_dim,
                    hidden_size=h_dim,
                    beta=beta,
                    threshold=threshold,
                    reset_mode=reset_mode,
                    surrogate="fast_sigmoid",
                )
            )
            in_dim = h_dim

        # For resource accounting compatibility
        self.core = self.layers[0]

        # 3. Readout heads
        final_dim = self.hidden_sizes[-1]
        if self.readout_mode == "membrane_spike_concat":
            readout_dim = final_dim * 2
        else:
            readout_dim = final_dim

        self.actor = nn.Linear(readout_dim, num_actions, bias=True)
        self.critic = nn.Linear(readout_dim, 1, bias=True)

        self.last_input_spike_rate: float = 0.0
        self.last_hidden_spike_rate: float = 0.0

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.layers:
            nn.init.orthogonal_(layer.fc.weight, gain=1.0)
            nn.init.constant_(layer.fc.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset spike encoder state (feed-forward core has no persistent decision state)."""
        self.encoder.reset_state()

    def forward(
        self,
        obs: Tensor,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Forward pass over T timesteps without decision-to-decision recurrence."""
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        # 1. Temporal spike encoding: [T, batch, input_channels]
        in_spikes = self.encoder(obs)
        self.last_input_spike_rate = float(in_spikes.mean().item())

        # Initialize per-decision states for each feed-forward layer
        layer_states = [
            layer.init_state(batch_size, device=device, dtype=obs.dtype)
            for layer in self.layers
        ]

        final_spikes_list = []
        final_membrane = None

        # Simulate T timesteps
        for t in range(self.T):
            current_spikes = in_spikes[t]
            for l_idx, layer in enumerate(self.layers):
                s_out, u_out, new_state = layer(current_spikes, layer_states[l_idx])
                layer_states[l_idx] = new_state
                current_spikes = s_out

            final_spikes_list.append(current_spikes)

        final_membrane = layer_states[-1].membrane
        all_final_spikes = torch.stack(final_spikes_list, dim=0)  # [T, batch, core_neurons]
        self.last_hidden_spike_rate = float(all_final_spikes.mean().item())

        # Readout
        if self.readout_mode == "membrane":
            readout = final_membrane
        elif self.readout_mode == "spike_count":
            readout = all_final_spikes.mean(dim=0)
        elif self.readout_mode == "membrane_spike_concat":
            readout = torch.cat([final_membrane, all_final_spikes.mean(dim=0)], dim=-1)
        else:
            readout = final_membrane

        action_logits = self.actor(readout)
        value = self.critic(readout)

        return {
            "action_logits": action_logits,
            "value": value,
            "spikes": all_final_spikes,
            "membrane": final_membrane,
            "readout": readout,
            "mean_input_spike_rate": self.last_input_spike_rate,
            "mean_hidden_spike_rate": self.last_hidden_spike_rate,
        }
