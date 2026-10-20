"""Advantage Actor-Critic (A2C) with GAE and surrogate gradient BPTT."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from doom_snn.configs import RunConfig
from doom_snn.evaluation.evaluator import Evaluator
from doom_snn.models.base_snn import SNNPolicy
from doom_snn.training.rollout_buffer import RolloutBuffer
from doom_snn.utils.checkpointing import save_checkpoint
from doom_snn.utils.csv_logger import CSVLogger

logger = logging.getLogger(__name__)


class A2CGAETrainer:
    """Trainer implementing on-policy Advantage Actor-Critic (A2C) with GAE.

    Tailored for spiking neural networks:
    - Truncated backpropagation through time over rollout chunks.
    - Preserves recurrent membrane/spike state between rollout boundaries.
    - Surrogate gradients for Heaviside spike step functions.
    - Continuous membrane potential readout for actor/critic heads.
    """

    def __init__(
        self,
        model: SNNPolicy,
        env: Any,
        config: RunConfig,
        logger_instance: Optional[logging.Logger] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.env = env
        self.config = config
        self.log = logger_instance or logger

        self.lr = config.training.lr
        self.gamma = config.training.gamma
        self.gae_lambda = config.training.gae_lambda
        self.rollout_length = config.training.rollout_length
        self.value_coef = config.training.value_coef
        self.entropy_coef = config.training.entropy_coef
        self.grad_clip = config.training.grad_clip

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.buffer = RolloutBuffer(
            rollout_length=self.rollout_length,
            obs_dim=config.obs.input_channels,
        )

        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_logger = CSVLogger(self.output_dir / "training_metrics.csv")

        self.best_reward = -float("inf")
        self.best_kill_rate = -1.0

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute actor policy loss, critic value loss, and entropy regularization."""
        obs = batch["obs"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Run forward pass through policy to obtain computational graph for BPTT
        out = self.model(obs)
        logits = out["action_logits"]
        values = out["value"].squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy loss with advantage weighting
        policy_loss = -(log_probs * advantages).mean()

        # Value loss: Huber loss for robustness against reward spikes
        value_loss = F.huber_loss(values, returns)

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        metrics = {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

        return total_loss, metrics

    def update_policy(self, last_value: float) -> Dict[str, float]:
        """Compute advantages, perform optimizer step, and clear rollout buffer."""
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        batch = self.buffer.get_batch(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        loss, loss_metrics = self.compute_loss(batch)
        loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.buffer.clear()

        return loss_metrics

    def evaluate(self, num_episodes: int = 20, seeds: Optional[list[int]] = None) -> Dict[str, Any]:
        """Run validation evaluation on policy."""
        if seeds is None:
            seeds = [self.config.seed + 100]

        evaluator = Evaluator(
            model_or_baseline=self.model,
            env_factory=lambda: self.env,
            config=self.config,
            device=self.device,
            logger=self.log,
        )
        return evaluator.evaluate(num_episodes=num_episodes, seeds=seeds, greedy=True)

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Execute complete A2C training loop.

        Args:
            num_episodes: Total number of training episodes.

        Returns:
            Dict containing final summary training outcomes.
        """
        episodes_to_run = num_episodes if num_episodes is not None else self.config.training.episodes
        self.log.info(
            f"Starting A2C+GAE training for {episodes_to_run} episodes on device '{self.device}'"
        )

        episode_rewards = []
        episode_kills = []
        episode_decisions = []
        start_time = time.time()

        for ep in range(1, episodes_to_run + 1):
            ep_seed = self.config.seed + ep
            obs = self.env.reset(seed=ep_seed)
            self.model.reset_state(batch_size=1, device=self.device)

            done = False
            ep_reward = 0.0
            ep_steps = 0
            ep_loss_metrics = []

            while not done:
                # Act in environment
                action, log_prob, value, info = self.model.act(obs, greedy=False)

                step_res = self.env.step(action)
                if len(step_res) == 5:
                    next_obs, reward, terminated, truncated, env_info = step_res
                    done = terminated or truncated
                else:
                    next_obs, reward, done, env_info = step_res

                ep_reward += float(reward)
                ep_steps += 1

                self.buffer.add(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=done,
                )

                obs = next_obs

                # Update policy if buffer reached rollout length
                if len(self.buffer) >= self.rollout_length or done:
                    last_val = 0.0
                    if not done:
                        with torch.no_grad():
                            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                            last_val = float(self.model(obs_t)["value"].item())

                    loss_m = self.update_policy(last_value=last_val)
                    ep_loss_metrics.append(loss_m)

            kill_success = bool(env_info.get("kill_success", ep_reward >= 50.0))
            episode_rewards.append(ep_reward)
            episode_kills.append(int(kill_success))
            episode_decisions.append(ep_steps)

            mean_10_rew = float(np.mean(episode_rewards[-10:]))
            mean_10_kill = float(np.mean(episode_kills[-10:]))

            # Logging row
            row = {
                "episode": ep,
                "reward": ep_reward,
                "kill_success": int(kill_success),
                "decisions": ep_steps,
                "mean_10_reward": round(mean_10_rew, 2),
                "mean_10_kill_rate": round(mean_10_kill, 2),
                "mean_input_spike_rate": getattr(self.model, "last_input_spike_rate", 0.0),
                "mean_hidden_spike_rate": getattr(self.model, "last_hidden_spike_rate", 0.0),
            }
            if ep_loss_metrics:
                row["loss"] = round(float(np.mean([m["loss"] for m in ep_loss_metrics])), 4)
                row["policy_loss"] = round(float(np.mean([m["policy_loss"] for m in ep_loss_metrics])), 4)
                row["value_loss"] = round(float(np.mean([m["value_loss"] for m in ep_loss_metrics])), 4)
                row["entropy"] = round(float(np.mean([m["entropy"] for m in ep_loss_metrics])), 4)

            self.csv_logger.log(row)

            if ep % self.config.logging.log_interval == 0 or ep == 1:
                self.log.info(
                    f"Ep {ep}/{episodes_to_run} | Rew: {ep_reward:.1f} (avg10: {mean_10_rew:.1f}) | "
                    f"Kill: {kill_success} (avg10: {mean_10_kill:.0%}) | Steps: {ep_steps}"
                )

            # Periodic evaluation & checkpointing
            if ep % self.config.training.eval_every == 0 or ep == episodes_to_run:
                # Save latest checkpoint
                save_checkpoint(
                    path=self.output_dir / "latest.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    config=self.config,
                    metrics={"reward": ep_reward, "mean_10_reward": mean_10_rew, "mean_10_kill": mean_10_kill},
                    epoch=ep,
                )

                if mean_10_rew > self.best_reward or mean_10_kill > self.best_kill_rate:
                    self.best_reward = max(self.best_reward, mean_10_rew)
                    self.best_kill_rate = max(self.best_kill_rate, mean_10_kill)
                    save_checkpoint(
                        path=self.output_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.config,
                        metrics={"best_reward": self.best_reward, "best_kill_rate": self.best_kill_rate},
                        epoch=ep,
                    )
                    self.log.info(f"==> New best model saved at episode {ep} (reward: {mean_10_rew:.1f})")

        total_time = time.time() - start_time
        self.log.info(f"Training completed in {total_time:.1f}s.")

        return {
            "episodes": episodes_to_run,
            "total_time_sec": total_time,
            "best_reward": self.best_reward,
            "best_kill_rate": self.best_kill_rate,
            "final_10_reward": float(np.mean(episode_rewards[-10:])),
            "final_10_kill_rate": float(np.mean(episode_kills[-10:])),
        }
