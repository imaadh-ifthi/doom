"""Unified trainer factory dispatcher."""

from __future__ import annotations

import logging
from typing import Any, Union
import torch

from doom_snn.configs import RunConfig
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.models.spike_dqn import SpikeDQN
from doom_snn.training.a2c_gae import A2CGAETrainer
from doom_snn.training.dqn import DQNTrainer


def create_trainer(
    model: SNNPolicy,
    env: Any,
    config: RunConfig,
    logger: logging.Logger,
    device: Union[str, torch.device],
) -> Union[A2CGAETrainer, DQNTrainer]:
    """Create appropriate reinforcement learning trainer based on configuration.

    Args:
        model: SNN policy or value network.
        env: Active environment instance (VizDoomEnv or DummyEnv).
        config: Experiment RunConfig.
        logger: Logger instance.
        device: Torch execution device.

    Returns:
        Configured trainer instance (A2CGAETrainer or DQNTrainer).
    """
    algo = config.training.algorithm.lower()

    if algo in ("a2c", "a2c_gae", "actor_critic"):
        return A2CGAETrainer(
            model=model,
            env=env,
            config=config,
            logger_instance=logger,
            device=device,
        )
    elif algo in ("dqn", "spike_dqn"):
        if not isinstance(model, SpikeDQN):
            raise TypeError(f"DQN trainer requires SpikeDQN model, got {type(model)}")
        return DQNTrainer(
            model=model,
            env=env,
            config=config,
            logger_instance=logger,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported training algorithm: '{algo}'")
