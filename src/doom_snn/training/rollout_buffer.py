"""On-policy trajectory rollout buffer with Generalized Advantage Estimation (GAE)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch import Tensor


class RolloutBuffer:
    """Stores sequential transitions for on-policy Advantage Actor-Critic (A2C)."""

    def __init__(self, rollout_length: int = 8, obs_dim: int = 768) -> None:
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim

        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.logits: List[Tensor] = []

        self.returns: Optional[Tensor] = None
        self.advantages: Optional[Tensor] = None

    def add(
        self,
        obs: Union[np.ndarray, Tensor],
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        logits: Optional[Tensor] = None,
    ) -> None:
        """Add a step transition into the rollout buffer."""
        if isinstance(obs, Tensor):
            obs_np = obs.detach().cpu().numpy().flatten()
        else:
            obs_np = np.asarray(obs, dtype=np.float32).flatten()

        self.observations.append(obs_np)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        if logits is not None:
            self.logits.append(logits.detach().cpu())

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and discounted targets."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = float(last_value)
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_val = float(self.values[t + 1])
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        self.advantages = torch.from_numpy(advantages)
        self.returns = torch.from_numpy(returns)

    def get_batch(self, device: torch.device) -> Dict[str, Tensor]:
        """Convert collected lists into tensors resident on target compute device."""
        if self.returns is None or self.advantages is None:
            raise RuntimeError("Must call compute_returns_and_advantages before get_batch()")

        # Normalize advantages for training stability
        adv = self.advantages.clone()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch: Dict[str, Tensor] = {
            "obs": torch.from_numpy(np.array(self.observations, dtype=np.float32)).to(device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
            "returns": self.returns.to(device),
            "advantages": adv.to(device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
        }

        if self.logits:
            batch["logits"] = torch.stack(self.logits).to(device)

        return batch

    def clear(self) -> None:
        """Reset buffer contents."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.logits.clear()
        self.returns = None
        self.advantages = None

    def __len__(self) -> int:
        return len(self.rewards)
