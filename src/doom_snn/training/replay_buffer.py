"""Off-policy experience replay buffer for value-based Spike-DQN."""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for experience transitions."""

    def __init__(self, capacity: int = 50000, obs_dim: int = 768) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros(self.capacity, dtype=np.int64)
        self.rewards_buf = np.zeros(self.capacity, dtype=np.float32)
        self.dones_buf = np.zeros(self.capacity, dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add(
        self,
        obs: Union[np.ndarray, Tensor],
        action: int,
        reward: float,
        next_obs: Union[np.ndarray, Tensor],
        done: bool,
    ) -> None:
        """Store a single transition into the circular buffer."""
        if isinstance(obs, Tensor):
            obs = obs.detach().cpu().numpy().flatten()
        if isinstance(next_obs, Tensor):
            next_obs = next_obs.detach().cpu().numpy().flatten()

        self.obs_buf[self.idx] = obs
        self.next_obs_buf[self.idx] = next_obs
        self.actions_buf[self.idx] = int(action)
        self.rewards_buf[self.idx] = float(reward)
        self.dones_buf[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        """Uniformly sample a batch of transitions."""
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {self.size}")

        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.from_numpy(self.obs_buf[indices]).to(device),
            "next_obs": torch.from_numpy(self.next_obs_buf[indices]).to(device),
            "actions": torch.from_numpy(self.actions_buf[indices]).to(device),
            "rewards": torch.from_numpy(self.rewards_buf[indices]).to(device),
            "dones": torch.from_numpy(self.dones_buf[indices]).to(device),
        }

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has accumulated sufficient samples."""
        return self.size >= min_size

    def __len__(self) -> int:
        return self.size
