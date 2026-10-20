"""Value-based deep reinforcement learning trainer for Spike-DQN."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from doom_snn.configs import RunConfig
from doom_snn.models.spike_dqn import SpikeDQN
from doom_snn.training.replay_buffer import ReplayBuffer
from doom_snn.utils.checkpointing import save_checkpoint
from doom_snn.utils.csv_logger import CSVLogger

logger = logging.getLogger(__name__)


class DQNTrainer:
    """Trainer implementing deep Q-learning for spiking neural networks (Spike-DQN)."""

    def __init__(
        self,
        model: SpikeDQN,
        env: Any,
        config: RunConfig,
        logger_instance: Optional[logging.Logger] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.target_model.eval()

        self.env = env
        self.config = config
        self.log = logger_instance or logger

        self.lr = config.dqn.lr
        self.gamma = config.dqn.gamma
        self.buffer_size = config.dqn.buffer_size
        self.batch_size = config.dqn.batch_size
        self.target_update_interval = config.dqn.target_update_interval
        self.min_history = config.dqn.min_history

        self.epsilon = config.dqn.epsilon_start
        self.epsilon_start = config.dqn.epsilon_start
        self.epsilon_end = config.dqn.epsilon_end
        self.epsilon_decay_steps = max(1, config.dqn.epsilon_decay_steps)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_dim=config.obs.input_channels,
        )

        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_logger = CSVLogger(self.output_dir / "dqn_metrics.csv")

        self.total_steps = 0
        self.best_reward = -float("inf")
        self.best_kill_rate = -1.0

    def sync_target_network(self) -> None:
        """Copy weights from online model to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self) -> float:
        """Sample a batch from replay buffer and perform one gradient descent step."""
        batch = self.buffer.sample(self.batch_size, device=self.device)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"].unsqueeze(-1)
        rewards = batch["rewards"]
        dones = batch["dones"]

        self.model.train()

        # Online network Q-values for chosen actions
        q_pred = self.model(obs)["q_values"].gather(1, actions).squeeze(-1)

        # Target network max next Q-value
        with torch.no_grad():
            next_q = self.target_model(next_obs)["q_values"].max(dim=-1)[0]
            q_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.huber_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Execute Spike-DQN training loop."""
        episodes_to_run = num_episodes if num_episodes is not None else self.config.training.episodes
        self.log.info(
            f"Starting Spike-DQN training for {episodes_to_run} episodes on device '{self.device}'"
        )

        episode_rewards = []
        episode_kills = []
        start_time = time.time()

        for ep in range(1, episodes_to_run + 1):
            ep_seed = self.config.seed + ep
            obs = self.env.reset(seed=ep_seed)
            self.model.reset_state(batch_size=1, device=self.device)

            done = False
            ep_reward = 0.0
            ep_steps = 0
            losses = []

            while not done:
                # Epsilon-greedy action selection
                action, _, _, _ = self.model.act(
                    obs,
                    greedy=False,
                    epsilon=self.epsilon,
                )

                step_res = self.env.step(action)
                if len(step_res) == 5:
                    next_obs, reward, terminated, truncated, env_info = step_res
                    done = terminated or truncated
                else:
                    next_obs, reward, done, env_info = step_res

                ep_reward += float(reward)
                ep_steps += 1
                self.total_steps += 1

                self.buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                )

                obs = next_obs

                # Training step
                if self.buffer.is_ready(self.min_history):
                    loss_val = self.train_step()
                    losses.append(loss_val)

                    # Update target network
                    if self.total_steps % self.target_update_interval == 0:
                        self.sync_target_network()

                # Decay epsilon
                if self.total_steps < self.epsilon_decay_steps:
                    decay_ratio = self.total_steps / self.epsilon_decay_steps
                    self.epsilon = self.epsilon_start - decay_ratio * (
                        self.epsilon_start - self.epsilon_end
                    )
                else:
                    self.epsilon = self.epsilon_end

            kill_success = bool(env_info.get("kill_success", ep_reward >= 50.0))
            episode_rewards.append(ep_reward)
            episode_kills.append(int(kill_success))

            mean_10_rew = float(np.mean(episode_rewards[-10:]))
            mean_10_kill = float(np.mean(episode_kills[-10:]))
            mean_loss = float(np.mean(losses)) if losses else 0.0

            row = {
                "episode": ep,
                "reward": ep_reward,
                "kill_success": int(kill_success),
                "decisions": ep_steps,
                "epsilon": round(self.epsilon, 4),
                "loss": round(mean_loss, 4),
                "mean_10_reward": round(mean_10_rew, 2),
                "mean_10_kill_rate": round(mean_10_kill, 2),
            }
            self.csv_logger.log(row)

            if ep % self.config.logging.log_interval == 0 or ep == 1:
                self.log.info(
                    f"Ep {ep}/{episodes_to_run} | Rew: {ep_reward:.1f} | "
                    f"Kill: {kill_success} | Epsilon: {self.epsilon:.3f} | Loss: {mean_loss:.4f}"
                )

            # Checkpoint
            if ep % self.config.training.eval_every == 0 or ep == episodes_to_run:
                save_checkpoint(
                    path=self.output_dir / "latest.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    config=self.config,
                    metrics={"reward": ep_reward, "mean_10_reward": mean_10_rew},
                    epoch=ep,
                )

                if mean_10_rew > self.best_reward:
                    self.best_reward = mean_10_rew
                    save_checkpoint(
                        path=self.output_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.config,
                        metrics={"best_reward": self.best_reward},
                        epoch=ep,
                    )

        total_time = time.time() - start_time
        return {
            "episodes": episodes_to_run,
            "total_time_sec": total_time,
            "best_reward": self.best_reward,
        }
