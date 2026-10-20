"""Track C: Primary Recurrent Spiking Neural Network (RSNN) Actor-Critic model.

Evolved from Dakshayani Ramanesh's RSNN lineage, upgraded with continuous
membrane potential readout, 16-bit fixed-point compliance, and surrogate gradients.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from doom_snn.encoding.spike_encoder import SpikeEncoder
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.neurons.lif import RecurrentLIFLayer
from doom_snn.neurons.states import LIFState


class RSNNActorCritic(SNNPolicy):
    """Recurrent Spiking Neural Network Actor-Critic for autonomous ViZDoom gameplay.

    Architecture:
        - Input: 768 grayscale pixels (32 x 24).
        - Spike Encoder: Deterministic accumulator rate coding over T=8 steps.
        - Core: Recurrent LIF layer with N hidden neurons (default N=96, beta in {0.25, 0.5, 0.75}).
        - Readout: Continuous membrane-potential readout u_T.
        - Actor Head: Linear(N, num_actions).
        - Critic Head: Linear(N, 1).
    """

    def __init__(
        self,
        input_channels: int = 768,
        core_neurons: int = 96,
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
        self.core_neurons = core_neurons
        self.num_actions = num_actions
        self.beta = beta
        self.threshold = threshold
        self.T = T
        self.reset_mode = reset_mode
        self.readout_mode = readout
        self.is_recurrent = True
        self.is_actor_critic = True
        self.track = "c_rsnn_ac"

        # 1. Deterministic temporal spike encoder
        self.encoder = SpikeEncoder(
            input_channels=input_channels,
            T=T,
            method="accumulator",
        )

        # 2. Recurrent LIF core layer
        self.core = RecurrentLIFLayer(
            input_size=input_channels,
            hidden_size=core_neurons,
            beta=beta,
            threshold=threshold,
            reset_mode=reset_mode,
            surrogate="fast_sigmoid",
        )

        # 3. Readout dimension
        if self.readout_mode == "membrane_spike_concat":
            readout_dim = core_neurons * 2
        else:
            readout_dim = core_neurons

        # 4. Continuous readout heads
        self.actor = nn.Linear(readout_dim, num_actions, bias=True)
        self.critic = nn.Linear(readout_dim, 1, bias=True)

        # Persistent episode recurrent state
        self._state: Optional[LIFState] = None

        # Firing activity telemetry
        self.last_input_spike_rate: float = 0.0
        self.last_hidden_spike_rate: float = 0.0

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear synaptic weights orthogonally."""
        nn.init.orthogonal_(self.core.fc_input.weight, gain=1.0)
        nn.init.constant_(self.core.fc_input.bias, 0.0)
        nn.init.orthogonal_(self.core.fc_recurrent.weight, gain=0.2)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset internal recurrent state at episode boundary."""
        self._state = None
        self.encoder.reset_state()

    def forward(
        self,
        obs: Tensor,
        state: Optional[LIFState] = None,
    ) -> Dict[str, Any]:
        """Forward pass over T internal simulation timesteps.

        Args:
            obs: Observation tensor [batch, 768] in range [0, 1].
            state: Optional external LIFState. If None, internal persistent state is used.

        Returns:
            Dict[str, Any] with action_logits, value, spikes, membrane, readout, and state.
        """
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        # 1. Deterministic accumulator rate encoding over T timesteps
        # Spikes shape: [T, batch, input_channels]
        in_spikes = self.encoder(obs)
        self.last_input_spike_rate = float(in_spikes.mean().item())

        # 2. Initialize or restore recurrent state
        if state is not None:
            curr_state = state
        elif self._state is not None and self._state.membrane.shape[0] == batch_size and self._state.membrane.device == device:
            curr_state = self._state
        else:
            curr_state = self.core.init_state(batch_size, device=device, dtype=obs.dtype)

        spikes_list = []
        u = curr_state.membrane
        s = curr_state.spikes

        # 3. Simulate T neural timesteps through recurrent LIF core
        for t in range(self.T):
            x_t = in_spikes[t]
            s_t, u_t, curr_state = self.core(x_t, curr_state)
            spikes_list.append(s_t)

        # 4. Detach recurrent state to bound BPTT history between decisions
        self._state = LIFState(
            membrane=curr_state.membrane.detach(),
            spikes=curr_state.spikes.detach(),
        )

        all_spikes = torch.stack(spikes_list, dim=0)  # [T, batch, core_neurons]
        self.last_hidden_spike_rate = float(all_spikes.mean().item())

        final_membrane = curr_state.membrane

        # 5. Extract continuous readout
        if self.readout_mode == "membrane":
            readout = final_membrane
        elif self.readout_mode == "spike_count":
            readout = all_spikes.mean(dim=0)
        elif self.readout_mode == "membrane_spike_concat":
            readout = torch.cat([final_membrane, all_spikes.mean(dim=0)], dim=-1)
        else:
            readout = final_membrane

        # 6. Compute actor logits and critic scalar value
        action_logits = self.actor(readout)
        value = self.critic(readout)

        return {
            "action_logits": action_logits,
            "value": value,
            "spikes": all_spikes,
            "membrane": final_membrane,
            "readout": readout,
            "state": self._state,
            "mean_input_spike_rate": self.last_input_spike_rate,
            "mean_hidden_spike_rate": self.last_hidden_spike_rate,
        }
