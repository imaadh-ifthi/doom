"""Track A: ANN baseline and ANN-to-SNN conversion modules."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from doom_snn.encoding.spike_encoder import SpikeEncoder
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.neurons.lif import LIFLayer


class ANNBaseline(nn.Module):
    """Standard artificial multi-layer perceptron (ANN) baseline.

    Operates under the identical assumed SNN benchmark:
    - 768 grayscale inputs (32 x 24).
    - 6 discrete actions.
    - Linear(768, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 6).
    """

    def __init__(
        self,
        input_channels: int = 768,
        hidden_sizes: Tuple[int, ...] = (128, 64),
        num_actions: int = 6,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_sizes = hidden_sizes
        self.num_actions = num_actions
        self.track = "a_ann_baseline"

        layers = []
        in_dim = input_channels
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def act(self, obs: Union[Tensor, Any], greedy: bool = True) -> int:
        self.eval()
        with torch.no_grad():
            if not isinstance(obs, Tensor):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
            else:
                obs_t = obs.to(dtype=torch.float32)
            device = next(self.parameters()).device
            obs_t = obs_t.to(device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            logits = self.forward(obs_t)
            if greedy:
                return int(torch.argmax(logits, dim=-1)[0].item())
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                return int(dist.sample()[0].item())


class ConvertedSNN(SNNPolicy):
    """SNN policy converted directly from trained ANN weights.

    Replaces ReLU activation layers with integrate-and-fire LIF neuron dynamics
    evaluated across T internal spiking steps.
    """

    def __init__(
        self,
        ann_model: ANNBaseline,
        T: int = 8,
        beta: float = 0.75,
        threshold: float = 0.4,
        reset_mode: str = "soft",
        readout: str = "membrane",
    ) -> None:
        super().__init__()
        self.input_channels = ann_model.input_channels
        self.hidden_sizes = list(ann_model.hidden_sizes)
        self.core_neurons = self.hidden_sizes[-1]
        self.num_actions = ann_model.num_actions
        self.T = T
        self.beta = beta
        self.threshold = threshold
        self.reset_mode = reset_mode
        self.readout_mode = readout
        self.is_recurrent = False
        self.is_actor_critic = False
        self.track = "a_converted_snn"

        # Rate encoder
        self.encoder = SpikeEncoder(
            input_channels=self.input_channels,
            T=T,
            method="accumulator",
        )

        # Extract linear layers from ANN
        ann_linears = [m for m in ann_model.net if isinstance(m, nn.Linear)]

        self.lif_layers = nn.ModuleList()
        for i in range(len(self.hidden_sizes)):
            ann_l = ann_linears[i]
            lif_l = LIFLayer(
                input_size=ann_l.in_features,
                hidden_size=ann_l.out_features,
                beta=beta,
                threshold=threshold,
                reset_mode=reset_mode,
                surrogate="fast_sigmoid",
            )
            # Copy trained weights and biases
            with torch.no_grad():
                lif_l.fc.weight.copy_(ann_l.weight)
                if ann_l.bias is not None and lif_l.fc.bias is not None:
                    lif_l.fc.bias.copy_(ann_l.bias)
            self.lif_layers.append(lif_l)

        self.core = self.lif_layers[0]

        # Output linear head
        last_ann_l = ann_linears[-1]
        self.head = nn.Linear(last_ann_l.in_features, last_ann_l.out_features)
        with torch.no_grad():
            self.head.weight.copy_(last_ann_l.weight)
            if last_ann_l.bias is not None and self.head.bias is not None:
                self.head.bias.copy_(last_ann_l.bias)

        self.last_input_spike_rate: float = 0.0
        self.last_hidden_spike_rate: float = 0.0

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.encoder.reset_state()

    def forward(
        self,
        obs: Tensor,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        in_spikes = self.encoder(obs)
        self.last_input_spike_rate = float(in_spikes.mean().item())

        states = [
            layer.init_state(batch_size, device=device, dtype=obs.dtype)
            for layer in self.lif_layers
        ]

        final_spikes = []
        for t in range(self.T):
            current = in_spikes[t]
            for l_idx, layer in enumerate(self.lif_layers):
                s_out, u_out, new_st = layer(current, states[l_idx])
                states[l_idx] = new_st
                current = s_out
            final_spikes.append(current)

        all_final_spikes = torch.stack(final_spikes, dim=0)
        self.last_hidden_spike_rate = float(all_final_spikes.mean().item())

        readout = states[-1].membrane if self.readout_mode == "membrane" else all_final_spikes.mean(dim=0)
        logits = self.head(readout)

        return {
            "action_logits": logits,
            "q_values": logits,
            "spikes": all_final_spikes,
            "membrane": states[-1].membrane,
            "readout": readout,
            "mean_input_spike_rate": self.last_input_spike_rate,
            "mean_hidden_spike_rate": self.last_hidden_spike_rate,
        }


def convert_ann_to_snn(
    ann: ANNBaseline,
    T: int = 8,
    beta: float = 0.75,
    threshold: float = 0.4,
) -> ConvertedSNN:
    """Convenience factory function converting trained ANN to ConvertedSNN."""
    return ConvertedSNN(ann_model=ann, T=T, beta=beta, threshold=threshold)
