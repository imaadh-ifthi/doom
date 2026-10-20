"""Robustness and perturbation evaluation suite for DOOM SNN policies."""

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from doom_snn.evaluation.evaluator import Evaluator
from doom_snn.evaluation.metrics import compute_aggregate_metrics

logger = logging.getLogger(__name__)


class TransformedEnvWrapper:
    """Wraps an environment to apply arbitrary observation perturbations."""

    def __init__(self, base_env: Any, transform_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.base_env = base_env
        self.transform_fn = transform_fn

    def reset(self, *args, **kwargs) -> Any:
        res = self.base_env.reset(*args, **kwargs)
        if isinstance(res, tuple):
            obs, info = res
            return self.transform_fn(obs), info
        return self.transform_fn(res)

    def step(self, action: int) -> Any:
        res = self.base_env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            return self.transform_fn(obs), reward, terminated, truncated, info
        obs, reward, done, info = res
        return self.transform_fn(obs), reward, done, info

    def close(self) -> None:
        if hasattr(self.base_env, "close"):
            self.base_env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)


class RobustnessEvaluator:
    """Systematic perturbation and ablation evaluator for neuromorphic DOOM agents."""

    def __init__(
        self,
        model: Union[nn.Module, Any],
        env_factory: Callable[[], Any],
        config: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.env_factory = env_factory
        self.config = config or {}
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    def _eval_with_transform(
        self,
        transform_fn: Callable[[np.ndarray], np.ndarray],
        num_episodes: int,
        seeds: List[int],
    ) -> Dict[str, Any]:
        """Helper to run evaluation with an observation transform."""
        wrapped_factory = lambda: TransformedEnvWrapper(self.env_factory(), transform_fn)
        evaluator = Evaluator(
            model_or_baseline=self.model,
            env_factory=wrapped_factory,
            config=self.config,
            device=self.device,
        )
        return evaluator.evaluate(num_episodes=num_episodes, seeds=seeds)

    def evaluate_brightness(
        self,
        scales: Optional[List[float]] = None,
        num_episodes: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent under visual luminance/brightness attenuation.

        Args:
            scales: Multipliers on raw pixel intensity (default: [1.0, 0.75, 0.5]).
            num_episodes: Episodes per seed.
            seeds: Seeds list (default: [6201]).

        Returns:
            Dict mapping scale factor to evaluation results and relative drop.
        """
        if scales is None:
            scales = [1.0, 0.75, 0.5]
        if seeds is None:
            seeds = [6201]

        results: Dict[str, Any] = {}
        baseline_kill_rate: Optional[float] = None

        for scale in scales:
            transform = lambda obs, s=scale: np.clip(obs * s, 0.0, 1.0)
            eval_res = self._eval_with_transform(transform, num_episodes, seeds)
            agg = eval_res["aggregate"]

            if baseline_kill_rate is None and abs(scale - 1.0) < 1e-4:
                baseline_kill_rate = agg["kill_rate"]

            drop = round(baseline_kill_rate - agg["kill_rate"], 4) if baseline_kill_rate is not None else 0.0

            results[f"scale_{scale}"] = {
                "scale": scale,
                "kill_rate": agg["kill_rate"],
                "mean_reward": agg["mean_reward"],
                "mean_decisions": agg["mean_decisions"],
                "kill_rate_drop": drop,
                "full_aggregate": agg,
            }

        return results

    def evaluate_noise(
        self,
        noise_stds: Optional[List[float]] = None,
        num_episodes: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent under additive zero-mean Gaussian sensor noise.

        Args:
            noise_stds: Standard deviations of Gaussian noise (default: [0.0, 0.05, 0.1]).
            num_episodes: Episodes per seed.
            seeds: Seeds list (default: [6201]).
        """
        if noise_stds is None:
            noise_stds = [0.0, 0.05, 0.1]
        if seeds is None:
            seeds = [6201]

        results: Dict[str, Any] = {}
        baseline_kill_rate: Optional[float] = None

        for std in noise_stds:
            if std == 0.0:
                transform = lambda obs: obs
            else:
                transform = lambda obs, s=std: np.clip(
                    obs + np.random.normal(0.0, s, size=obs.shape).astype(np.float32),
                    0.0,
                    1.0,
                )

            eval_res = self._eval_with_transform(transform, num_episodes, seeds)
            agg = eval_res["aggregate"]

            if baseline_kill_rate is None and std == 0.0:
                baseline_kill_rate = agg["kill_rate"]

            drop = round(baseline_kill_rate - agg["kill_rate"], 4) if baseline_kill_rate is not None else 0.0

            results[f"noise_std_{std}"] = {
                "noise_std": std,
                "kill_rate": agg["kill_rate"],
                "mean_reward": agg["mean_reward"],
                "mean_decisions": agg["mean_decisions"],
                "kill_rate_drop": drop,
                "full_aggregate": agg,
            }

        return results

    def evaluate_shuffled_input(
        self,
        num_episodes: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent under spatial pixel permutation (shuffled visual input)."""
        if seeds is None:
            seeds = [6201]

        # Fix a deterministic permutation over 768 pixels
        perm_rng = np.random.default_rng(6201)
        perm = perm_rng.permutation(768)

        def transform(obs: np.ndarray) -> np.ndarray:
            flat = obs.flatten()
            if flat.size == 768:
                return flat[perm]
            return obs

        eval_res = self._eval_with_transform(transform, num_episodes, seeds)
        agg = eval_res["aggregate"]

        return {
            "test": "shuffled_input",
            "kill_rate": agg["kill_rate"],
            "mean_reward": agg["mean_reward"],
            "mean_decisions": agg["mean_decisions"],
            "action_distribution": agg["action_distribution"],
            "full_aggregate": agg,
        }

    def evaluate_constant_input(
        self,
        value: float = 0.5,
        num_episodes: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate policy when observation is clamped to a static uniform value."""
        if seeds is None:
            seeds = [6201]

        transform = lambda obs: np.full_like(obs, fill_value=value, dtype=np.float32)
        eval_res = self._eval_with_transform(transform, num_episodes, seeds)
        agg = eval_res["aggregate"]

        return {
            "test": "constant_input",
            "constant_value": value,
            "kill_rate": agg["kill_rate"],
            "mean_reward": agg["mean_reward"],
            "mean_decisions": agg["mean_decisions"],
            "action_distribution": agg["action_distribution"],
            "full_aggregate": agg,
        }

    def evaluate_recurrence_disabled(
        self,
        model: Optional[nn.Module] = None,
        num_episodes: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Ablation: temporarily zero out recurrent synaptic weights and measure performance."""
        target_model = model or self.model
        if seeds is None:
            seeds = [6201]

        if not isinstance(target_model, nn.Module):
            return {
                "test": "recurrence_disabled",
                "error": "Model is not an nn.Module; recurrence ablation skipped.",
                "kill_rate": 0.0,
            }

        # Identify recurrent weight parameter
        recurrent_params = []
        for name, param in target_model.named_parameters():
            if any(term in name.lower() for term in ["rec", "w_rec", "weight_hh"]):
                recurrent_params.append((name, param))

        if not recurrent_params:
            # Check for square matrix in hidden layer
            for name, param in target_model.named_parameters():
                if param.dim() == 2 and param.size(0) == param.size(1) and param.size(0) > 10:
                    recurrent_params.append((name, param))
                    break

        if not recurrent_params:
            return {
                "test": "recurrence_disabled",
                "warning": "No recurrent weight parameter found in model (feedforward architecture).",
                "kill_rate": 0.0,
            }

        # Backup original weights
        saved_weights = [(param, param.data.clone()) for _, param in recurrent_params]

        try:
            # Zero out recurrent weights
            with torch.no_grad():
                for param, _ in saved_weights:
                    param.data.zero_()

            evaluator = Evaluator(
                model_or_baseline=target_model,
                env_factory=self.env_factory,
                config=self.config,
                device=self.device,
            )
            eval_res = evaluator.evaluate(num_episodes=num_episodes, seeds=seeds)
            agg = eval_res["aggregate"]

            return {
                "test": "recurrence_disabled",
                "ablated_params": [name for name, _ in recurrent_params],
                "kill_rate": agg["kill_rate"],
                "mean_reward": agg["mean_reward"],
                "mean_decisions": agg["mean_decisions"],
                "action_distribution": agg["action_distribution"],
                "full_aggregate": agg,
            }

        finally:
            # Restore original weights
            with torch.no_grad():
                for param, orig_data in saved_weights:
                    param.data.copy_(orig_data)

    def evaluate_timestep_sweep(
        self,
        T_values: Optional[List[int]] = None,
        num_episodes: int = 30,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Sweep internal simulation timesteps T and evaluate inference sensitivity."""
        if T_values is None:
            T_values = [4, 8, 16]
        if seeds is None:
            seeds = [6201]

        results: Dict[str, Any] = {}
        orig_T = getattr(self.model, "T", None)

        try:
            for t_val in T_values:
                if hasattr(self.model, "T"):
                    self.model.T = t_val
                if hasattr(self.model, "num_steps"):
                    self.model.num_steps = t_val

                evaluator = Evaluator(
                    model_or_baseline=self.model,
                    env_factory=self.env_factory,
                    config=self.config,
                    device=self.device,
                )
                eval_res = evaluator.evaluate(num_episodes=num_episodes, seeds=seeds)
                agg = eval_res["aggregate"]

                results[f"T_{t_val}"] = {
                    "T": t_val,
                    "kill_rate": agg["kill_rate"],
                    "mean_reward": agg["mean_reward"],
                    "mean_latency_ms": agg["mean_latency_ms"],
                    "mean_spike_rate": agg["mean_spike_rate"],
                    "full_aggregate": agg,
                }

        finally:
            if orig_T is not None and hasattr(self.model, "T"):
                self.model.T = orig_T

        return results

    def full_robustness_report(self, num_episodes: int = 20) -> Dict[str, Any]:
        """Run complete battery of robustness evaluations and compile comprehensive report."""
        report: Dict[str, Any] = {
            "model_name": getattr(self.model, "name", lambda: self.model.__class__.__name__)(),
            "episodes_per_test": num_episodes,
        }

        logger.info("Executing brightness robustness battery...")
        report["brightness"] = self.evaluate_brightness(num_episodes=num_episodes)

        logger.info("Executing Gaussian noise robustness battery...")
        report["noise"] = self.evaluate_noise(num_episodes=num_episodes)

        logger.info("Executing spatial pixel shuffle test...")
        report["shuffled_input"] = self.evaluate_shuffled_input(num_episodes=num_episodes)

        logger.info("Executing constant input baseline test...")
        report["constant_input"] = self.evaluate_constant_input(num_episodes=num_episodes)

        logger.info("Executing recurrence ablation test...")
        report["recurrence_ablation"] = self.evaluate_recurrence_disabled(num_episodes=num_episodes)

        logger.info("Executing timestep sweep...")
        report["timestep_sweep"] = self.evaluate_timestep_sweep(num_episodes=num_episodes)

        return report
