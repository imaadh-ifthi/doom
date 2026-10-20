"""Base abstract interface for Spiking Neural Network policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from doom_snn.hardware.resource_accounting import compute_resource_report


class SNNPolicy(nn.Module, ABC):
    """Abstract base class for all hardware-aware SNN game-playing agents."""

    @abstractmethod
    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset internal recurrent membrane and spike states at episode boundary."""
        pass

    @abstractmethod
    def forward(
        self,
        obs: Tensor,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Perform forward inference over T internal spiking timesteps.

        Args:
            obs: Observation tensor [batch, input_channels] or [input_channels].
            state: Optional state override.

        Returns:
            Dict[str, Any] containing at least:
                - 'action_logits' (for actor-critic) or 'q_values' (for DQN)
                - 'spikes': hidden layer spikes
                - 'membrane': sub-threshold membrane potential
                - 'readout': tensor passed into the readout head
        """
        pass

    def act(
        self,
        obs: Union[Tensor, Any],
        state: Optional[Any] = None,
        greedy: bool = True,
        epsilon: float = 0.0,
    ) -> Tuple[int, float, float, Dict[str, Any]]:
        """High-level action selection method for environment interaction.

        Args:
            obs: Input observation array or tensor.
            state: Optional recurrent hidden state.
            greedy: If True, select argmax action; if False, sample from categorical.
            epsilon: Optional epsilon-greedy exploration probability (used in DQN).

        Returns:
            Tuple of:
                - action: int (discrete action index in [0, num_actions - 1])
                - log_prob: float (log-probability of selected action, or 0.0 for DQN)
                - value: float (critic state-value estimation, or max Q-value)
                - info: dict with auxiliary telemetry (spikes, membrane, latency)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(obs, Tensor):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
            else:
                obs_t = obs.to(dtype=torch.float32)

            device = next(self.parameters()).device
            obs_t = obs_t.to(device)

            was_1d = obs_t.dim() == 1
            if was_1d:
                obs_t = obs_t.unsqueeze(0)

            out = self.forward(obs_t, state=state)

            if "action_logits" in out:
                logits = out["action_logits"]
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)

                if greedy:
                    action_idx = int(torch.argmax(logits, dim=-1)[0].item())
                else:
                    action_idx = int(dist.sample()[0].item())

                log_prob = float(dist.log_prob(torch.tensor(action_idx, device=device))[0].item())
                value = float(out.get("value", torch.tensor([[0.0]]))[0, 0].item())

            elif "q_values" in out:
                q_vals = out["q_values"]
                if not greedy and torch.rand(1).item() < epsilon:
                    num_actions = q_vals.shape[-1]
                    action_idx = int(torch.randint(0, num_actions, (1,)).item())
                else:
                    action_idx = int(torch.argmax(q_vals, dim=-1)[0].item())

                log_prob = 0.0
                value = float(q_vals[0, action_idx].item())

            else:
                raise KeyError("Model output missing both 'action_logits' and 'q_values'")

            info = {
                "mean_hidden_spike_rate": getattr(self, "last_hidden_spike_rate", 0.0),
                "mean_input_spike_rate": getattr(self, "last_input_spike_rate", 0.0),
            }

            return action_idx, log_prob, value, info

    def get_resource_summary(self) -> Dict[str, Any]:
        """Compute hardware and parameter resource summary."""
        return compute_resource_report(self)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
