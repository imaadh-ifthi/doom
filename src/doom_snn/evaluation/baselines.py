"""Baseline agents for DOOM SNN evaluation and benchmarking."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import numpy as np
import torch


class BaselineAgent(ABC):
    """Abstract base class for baseline policies in the ViZDoom environment."""

    @abstractmethod
    def act(self, obs: Union[np.ndarray, torch.Tensor, Any]) -> int:
        """Select an action index given an observation.

        Args:
            obs: Game observation, either numpy array, torch tensor, or raw buffer.

        Returns:
            int: Discrete action index (0 to 5 in 6-action ViZDoom space).
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return human-readable identifier for this baseline agent."""
        pass

    def reset(self) -> None:
        """Reset internal agent state at the beginning of an episode."""
        pass

    def seed(self, seed: Optional[int]) -> None:
        """Seed agent's random number generator if applicable."""
        pass

    def __call__(self, obs: Union[np.ndarray, torch.Tensor, Any]) -> int:
        return self.act(obs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name()}')"


class RandomAgent(BaselineAgent):
    """Uniform random baseline policy across available actions [0, 5]."""

    def __init__(self, num_actions: int = 6, seed: Optional[int] = None) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Union[np.ndarray, torch.Tensor, Any]) -> int:
        return int(self.rng.integers(0, self.num_actions))

    def name(self) -> str:
        return "random"


class AlwaysShootAgent(BaselineAgent):
    """Agent that persistently fires weapon (action 3: SHOOT / ATTACK)."""

    def __init__(self) -> None:
        super().__init__()

    def act(self, obs: Union[np.ndarray, torch.Tensor, Any]) -> int:
        return 3

    def name(self) -> str:
        return "always_shoot"


class LeftShootAgent(BaselineAgent):
    """Agent that moves left and continuously shoots (action 4: LEFT + SHOOT)."""

    def __init__(self) -> None:
        super().__init__()

    def act(self, obs: Union[np.ndarray, torch.Tensor, Any]) -> int:
        return 4

    def name(self) -> str:
        return "left_shoot"


def get_all_baselines() -> List[BaselineAgent]:
    """Instantiate and return all standard DOOM baseline agents.

    Returns:
        List[BaselineAgent]: List containing [RandomAgent, AlwaysShootAgent, LeftShootAgent].
    """
    return [RandomAgent(), AlwaysShootAgent(), LeftShootAgent()]
