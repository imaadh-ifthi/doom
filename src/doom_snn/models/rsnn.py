"""Recurrent Spiking Neural Network (RSNN) architecture for ViZDoom control."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast-sigmoid surrogate gradient for spiking LIF neuron."""

    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float = 0.4, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        ctx.alpha = alpha
        return (u >= threshold).to(dtype=u.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (u,) = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        # Derivative: 1 / (1 + alpha * |u - V_th|)^2
        grad = 1.0 / ((1.0 + alpha * torch.abs(u - threshold)) ** 2)
        return grad_output * grad, None, None


def surrogate_spike(u: torch.Tensor, threshold: float = 0.4, alpha: float = 2.0) -> torch.Tensor:
    return FastSigmoidSurrogate.apply(u, threshold, alpha)


class RSNNModel(nn.Module):
    """Recurrent Spiking Neural Network (RSNN v2) for DOOM gameplay.

    Matches hardware constraints:
    - 768 input intensity values (32x24 grayscale frame)
    - 96 recurrent Leaky Integrate-and-Fire (LIF) hidden neurons
    - Beta decay: 0.5 (allowed values: 0.25, 0.5, 0.75)
    - Threshold: 0.4
    - Timesteps: T = 8
    - 6-action actor readout
    - 1-value scalar critic readout (for training)
    """

    def __init__(
        self,
        input_size: int = 768,
        core_neurons: int = 96,
        num_actions: int = 6,
        beta: float = 0.5,
        threshold: float = 0.4,
        T: int = 8,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.core_neurons = core_neurons
        self.num_actions = num_actions
        self.beta = beta
        self.threshold = threshold
        self.T = T
        self.is_recurrent = recurrent
        self.is_actor_critic = True

        # Synaptic layers
        self.fc_in = nn.Linear(input_size, core_neurons, bias=True)
        if recurrent:
            self.w_rec = nn.Linear(core_neurons, core_neurons, bias=False)
        else:
            self.w_rec = None

        # Readout heads
        self.actor = nn.Linear(core_neurons, num_actions, bias=True)
        self.critic = nn.Linear(core_neurons, 1, bias=True)

        # Persistent state within episode
        self.membrane: Optional[torch.Tensor] = None
        self.prev_spikes: Optional[torch.Tensor] = None

        # Activity recording
        self.last_input_spike_rate: float = 0.0
        self.last_hidden_spike_rate: float = 0.0

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.fc_in.weight, gain=1.0)
        nn.init.constant_(self.fc_in.bias, 0.0)
        if self.w_rec is not None:
            nn.init.orthogonal_(self.w_rec.weight, gain=0.2)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def reset_state(self) -> None:
        """Reset internal membrane and spike buffers at episode boundary."""
        self.membrane = None
        self.prev_spikes = None

    def reset(self) -> None:
        """Alias for reset_state."""
        self.reset_state()

    def encode_input(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Deterministic rate-to-spike encoding over T steps.

        Args:
            x: Tensor of shape (batch, input_size) with values in [0, 1].
            T: Number of timesteps.
        Returns:
            Tensor of shape (T, batch, input_size) with binary spikes {0, 1}.
        """
        # Threshold schedule: deterministic phase-distributed thresholds
        batch_size = x.size(0)
        # thresholds: shape (T, 1, 1)
        thresholds = torch.linspace(1.0 / (2 * T), 1.0 - 1.0 / (2 * T), T, device=x.device).view(T, 1, 1)
        # x expanded: shape (1, batch, input_size)
        x_exp = x.unsqueeze(0).expand(T, batch_size, -1)
        input_spikes = (x_exp >= thresholds).to(dtype=x.dtype)
        return input_spikes

    def forward(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single decision step across T internal neural timesteps.

        Args:
            x: Observation tensor of shape (batch, 768) or (batch, 1, 24, 32).
            T: Optional override for number of timesteps.
        Returns:
            Tuple of (logits, value):
                logits: (batch, num_actions)
                value: (batch, 1)
        """
        if T is None:
            T = self.T

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        batch_size = x.size(0)
        device = x.device

        # Initialize membrane potential and prev spikes if not initialized or batch size mismatch
        if self.membrane is None or self.membrane.size(0) != batch_size or self.membrane.device != device:
            self.membrane = torch.zeros(batch_size, self.core_neurons, device=device)
            self.prev_spikes = torch.zeros(batch_size, self.core_neurons, device=device)

        # Generate rate-encoded input spikes over T timesteps
        input_spikes = self.encode_input(x, T)  # (T, batch, input_size)
        self.last_input_spike_rate = float(input_spikes.mean().item())

        hidden_spike_list = []
        u = self.membrane
        s = self.prev_spikes

        for t in range(T):
            in_syn = self.fc_in(input_spikes[t])  # (batch, core_neurons)
            if self.w_rec is not None:
                rec_syn = self.w_rec(s)
            else:
                rec_syn = 0.0

            # LIF membrane update
            u = self.beta * u + in_syn + rec_syn
            s = surrogate_spike(u, threshold=self.threshold)
            # Hard reset: membrane drops to 0 after firing
            u = u * (1.0 - s)
            hidden_spike_list.append(s)

        # Retain state for next decision within this episode (detached to prevent unbounded graph)
        self.membrane = u.detach()
        self.prev_spikes = s.detach()

        # Stack spikes across timesteps: (T, batch, core_neurons)
        all_hidden_spikes = torch.stack(hidden_spike_list, dim=0)
        self.last_hidden_spike_rate = float(all_hidden_spikes.mean().item())

        # Readout: mean spike count over T steps
        mean_spikes = all_hidden_spikes.mean(dim=0)  # (batch, core_neurons)
        logits = self.actor(mean_spikes)
        value = self.critic(mean_spikes)

        return logits, value

    def act(self, obs: torch.Tensor, greedy: bool = True) -> int:
        """High-level action selection helper."""
        self.eval()
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            logits, _ = self.forward(obs)
            if greedy:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item())
        return action
