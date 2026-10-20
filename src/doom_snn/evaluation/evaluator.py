"""Comprehensive evaluation engine for DOOM SNN policies and baselines."""

import csv
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from doom_snn.evaluation.baselines import BaselineAgent
from doom_snn.evaluation.metrics import (
    EpisodeMetrics,
    compute_aggregate_metrics,
)

logger = logging.getLogger(__name__)

# Standard 6-action discrete mapping for ViZDoom Basic scenario
# Buttons: [MOVE_LEFT, MOVE_RIGHT, ATTACK]
VIZDOOM_ACTIONS_6 = [
    [False, False, False],  # 0: noop
    [True, False, False],   # 1: left
    [False, True, False],   # 2: right
    [False, False, True],   # 3: shoot
    [True, False, True],    # 4: left_shoot
    [False, True, True],    # 5: right_shoot
]


def seed_everything(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DummyDoomEnv:
    """Deterministic simulated Doom environment for tests and headless verification.

    Mimics ViZDoom basic.wad dynamics:
    - Target starts at a random angle offset.
    - Observation is 32x24 grayscale frame normalized to [0.0, 1.0] (768 floats).
    - Player has 6 actions: noop, left, right, shoot, left_shoot, right_shoot.
    - Shooting while approximately facing target scores a kill (+100 reward, done=True).
    - Episode timeout is 50 decisions (200 game tics with frame_skip=4).
    """

    def __init__(self, seed: Optional[int] = None, frame_skip: int = 4) -> None:
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)
        self.max_decisions = 50
        self.decisions = 0
        self.target_pos = 0  # -10 (far left) to +10 (far right)
        self.player_aim = 0
        self.is_done = False

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self.decisions = 0
        self.is_done = False
        self.target_pos = int(self.rng.integers(-5, 6))
        self.player_aim = 0
        obs = self._render_obs()
        return obs, {"kill_success": False, "game_tics": 0}

    def _render_obs(self) -> np.ndarray:
        # Render a simple 24x32 synthetic frame with a 4x4 target patch
        frame = np.full((24, 32), 0.15, dtype=np.float32)
        # Target x position mapped to column [4, 28]
        col = int(np.clip(16 + (self.target_pos - self.player_aim) * 2, 4, 28))
        frame[10:14, col - 2 : col + 2] = 0.85
        return frame.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.is_done:
            raise RuntimeError("Step called on terminated environment. Call reset() first.")

        self.decisions += 1
        reward = -1.0  # Living penalty per decision
        kill_success = False

        # Action effects:
        # 0: noop
        # 1: left
        # 2: right
        # 3: shoot
        # 4: left_shoot
        # 5: right_shoot
        if action in (1, 4):
            self.player_aim -= 1
        elif action in (2, 5):
            self.player_aim += 1

        is_shooting = action in (3, 4, 5)
        aim_error = abs(self.target_pos - self.player_aim)

        if is_shooting and aim_error <= 1:
            # Hit target
            kill_success = True
            reward += 100.0
            self.is_done = True
        elif self.decisions >= self.max_decisions:
            self.is_done = True

        obs = self._render_obs()
        game_tics = self.decisions * self.frame_skip
        info = {
            "kill_success": kill_success,
            "game_tics": game_tics,
            "decisions": self.decisions,
        }
        return obs, reward, self.is_done, False, info

    def close(self) -> None:
        pass


def make_vizdoom_env(
    visible: bool = False,
    scenario_name: str = "basic.wad",
    frame_skip: int = 4,
) -> Any:
    """Create a ViZDoom game instance configured for RSNN experiments."""
    try:
        import vizdoom as vzd
    except ImportError as e:
        logger.warning(f"vizdoom not installed: {e}. Falling back to DummyDoomEnv.")
        return DummyDoomEnv(frame_skip=frame_skip)

    try:
        game = vzd.DoomGame()
        wad_path = Path(vzd.scenarios_path) / scenario_name
        if not wad_path.is_file():
            wad_path = Path("scenarios") / scenario_name

        if not wad_path.is_file():
            logger.warning(f"Scenario wad not found at {wad_path}. Falling back to DummyDoomEnv.")
            return DummyDoomEnv(frame_skip=frame_skip)

        game.set_doom_scenario_path(str(wad_path))
        game.set_doom_map("map01")
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_available_buttons([
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK,
        ])
        game.set_available_game_variables([
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.HEALTH,
        ])
        game.set_episode_timeout(200)
        game.set_episode_start_time(10)
        game.set_living_reward(-1.0)
        game.set_render_hud(False)
        game.set_window_visible(visible)
        game.set_mode(vzd.Mode.PLAYER)
        game.init()
        return game
    except Exception as e:
        logger.warning(f"Failed to initialize ViZDoom game ({e}). Using DummyDoomEnv.")
        return DummyDoomEnv(frame_skip=frame_skip)


class Evaluator:
    """Rigorous evaluation runner for SNN models and baseline agents."""

    def __init__(
        self,
        model_or_baseline: Union[nn.Module, BaselineAgent, Any],
        env_factory: Optional[Callable[[], Any]] = None,
        config: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize Evaluator.

        Args:
            model_or_baseline: PyTorch neural model or BaselineAgent instance.
            env_factory: Factory callable returning an environment instance or None.
            config: Optional experiment/hyperparameter configuration object or dict.
            device: Torch execution device ('cpu' or 'cuda').
            logger: Optional logging instance.
        """
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        self.is_baseline = isinstance(model_or_baseline, BaselineAgent)
        self.model = model_or_baseline
        if not self.is_baseline and isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

        if env_factory is not None:
            self.env_factory = env_factory
        else:
            self.env_factory = lambda: DummyDoomEnv()

        # Cache model/agent name
        if hasattr(self.model, "name") and callable(self.model.name):
            self.name = self.model.name()
        elif hasattr(self.model, "__class__"):
            self.name = self.model.__class__.__name__
        else:
            self.name = "UnknownAgent"

    def _preprocess_frame(self, screen_buffer: Any) -> np.ndarray:
        """Preprocess visual frame into normalized 32x24 grayscale array (768 floats)."""
        arr = np.asarray(screen_buffer, dtype=np.float32)

        # Already preprocessed 1D vector (e.g. from DummyEnv or VizDoomEnv)
        if arr.ndim == 1 and arr.size == 768:
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr.astype(np.float32)

        # Handle ViZDoom raw buffer shapes:
        # e.g., (120, 160) for GRAY8 or (1, 120, 160) or (3, 120, 160)
        if arr.ndim == 3:
            if arr.shape[0] == 3:
                # Convert RGB to grayscale: 0.2989 R + 0.5870 G + 0.1140 B
                arr = 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
            elif arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[2] == 3:
                arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

        # Resample to 24 rows x 32 columns if needed
        h, w = arr.shape[-2], arr.shape[-1]
        if (h, w) != (24, 32):
            step_y = max(1, h // 24)
            step_x = max(1, w // 32)
            arr = arr[::step_y, ::step_x][:24, :32]

        if arr.shape != (24, 32):
            # Fallback zero-pad or trim
            canvas = np.zeros((24, 32), dtype=np.float32)
            sh = min(24, arr.shape[0])
            sw = min(32, arr.shape[1])
            canvas[:sh, :sw] = arr[:sh, :sw]
            arr = canvas

        # Normalize pixel values to [0.0, 1.0]
        if arr.max() > 1.0:
            arr = arr / 255.0

        return arr.flatten()

    def _reset_agent_state(self) -> None:
        """Reset internal recurrent neural states or baseline state."""
        if hasattr(self.model, "reset_state") and callable(self.model.reset_state):
            self.model.reset_state()
        elif hasattr(self.model, "reset") and callable(self.model.reset):
            self.model.reset()

    def _select_action(self, obs: np.ndarray, greedy: bool = True) -> Tuple[int, float, float]:
        """Perform policy forward inference on single observation.

        Returns:
            Tuple of (action_idx, spike_rate, inference_time_ms)
        """
        t0 = time.perf_counter()

        if self.is_baseline:
            action = int(self.model.act(obs))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return action, 0.0, latency_ms

        # Neural policy forward inference
        spike_rate = 0.0
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Shape: (1, 768)

            if hasattr(self.model, "act") and callable(self.model.act):
                act_res = self.model.act(x, greedy=greedy)
                action = int(act_res[0]) if isinstance(act_res, tuple) else int(act_res)
            else:
                out = self.model(x)
                if isinstance(out, tuple):
                    logits = out[0]
                elif isinstance(out, dict):
                    logits = out["logits"]
                else:
                    logits = out

                if greedy:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())

            # Retrieve hidden spike rate if reported by model
            if hasattr(self.model, "last_hidden_spike_rate"):
                spike_rate = float(self.model.last_hidden_spike_rate)
            elif hasattr(self.model, "get_spike_rate") and callable(self.model.get_spike_rate):
                spike_rate = float(self.model.get_spike_rate())

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return action, spike_rate, latency_ms

    def evaluate_single_episode(
        self,
        env: Any = None,
        seed: Optional[int] = None,
        greedy: bool = True,
        frame_skip: int = 4,
    ) -> EpisodeMetrics:
        """Run a single evaluation episode, measuring timing and gameplay metrics.

        Args:
            env: Active environment instance. If None, created from env_factory.
            seed: Optional episode seed.
            greedy: Whether to choose greedy deterministic actions.
            frame_skip: Number of game tics per decision step.

        Returns:
            EpisodeMetrics: Captured episode outcomes.
        """
        owned_env = False
        if env is None:
            env = self.env_factory()
            owned_env = True

        self._reset_agent_state()
        action_counts = [0] * 6
        decisions = 0
        total_reward = 0.0
        spike_rates: List[float] = []
        latencies: List[float] = []
        kill_success = False
        game_tics = 0

        try:
            # Check if env is a ViZDoom DoomGame instance
            if hasattr(env, "is_episode_finished"):
                if seed is not None:
                    try:
                        env.set_seed(seed)
                    except Exception:
                        pass
                env.new_episode()
                prev_kills = env.get_game_variable(type(env).get_game_variable.__annotations__.get("KILLCOUNT", 0)) if hasattr(env, "get_game_variable") else 0

                while not env.is_episode_finished():
                    state = env.get_state()
                    if state is None:
                        break
                    obs = self._preprocess_frame(state.screen_buffer)
                    action_idx, s_rate, lat_ms = self._select_action(obs, greedy=greedy)
                    action_counts[action_idx % len(action_counts)] += 1
                    spike_rates.append(s_rate)
                    latencies.append(lat_ms)
                    decisions += 1

                    viz_action = VIZDOOM_ACTIONS_6[action_idx % len(VIZDOOM_ACTIONS_6)]
                    step_reward = float(env.make_action(viz_action, frame_skip))
                    total_reward += step_reward

                game_tics = decisions * frame_skip
                try:
                    import vizdoom as vzd
                    final_kills = env.get_game_variable(vzd.GameVariable.KILLCOUNT)
                    kill_success = (final_kills > 0) or (total_reward >= 50.0)
                except Exception:
                    kill_success = total_reward >= 50.0

            else:
                # Gym / Gymnasium / DummyDoomEnv interface
                reset_res = env.reset(seed=seed) if seed is not None else env.reset()
                obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
                obs = self._preprocess_frame(obs)
                done = False

                while not done:
                    action_idx, s_rate, lat_ms = self._select_action(obs, greedy=greedy)
                    action_counts[action_idx % len(action_counts)] += 1
                    spike_rates.append(s_rate)
                    latencies.append(lat_ms)
                    decisions += 1

                    step_res = env.step(action_idx)
                    if len(step_res) == 5:
                        next_obs, step_reward, terminated, truncated, info = step_res
                        done = terminated or truncated
                    else:
                        next_obs, step_reward, done, info = step_res

                    total_reward += float(step_reward)
                    if info.get("kill_success", False):
                        kill_success = True
                    if "game_tics" in info:
                        game_tics = int(info["game_tics"])
                    obs = self._preprocess_frame(next_obs)

                if game_tics == 0:
                    game_tics = decisions * frame_skip

                if not kill_success and total_reward >= 50.0:
                    kill_success = True

        finally:
            if owned_env and hasattr(env, "close"):
                env.close()

        mean_spike_rate = float(np.mean(spike_rates)) if spike_rates else 0.0
        mean_latency_ms = float(np.mean(latencies)) if latencies else 0.0

        return EpisodeMetrics(
            kill_success=kill_success,
            reward=round(total_reward, 2),
            decisions=decisions,
            game_tics=game_tics,
            action_counts=action_counts,
            spike_rate=round(mean_spike_rate, 4),
            latency_ms=round(mean_latency_ms, 3),
        )

    def evaluate(
        self,
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None,
        greedy: bool = True,
    ) -> Dict[str, Any]:
        """Run multi-seed evaluation protocol.

        Args:
            num_episodes: Number of episodes to evaluate per seed.
            seeds: List of random seeds (defaults to [6201, 6202, 6203]).
            greedy: Whether to choose greedy deterministic actions.

        Returns:
            Dict[str, Any] containing per_seed breakdowns and full aggregate metrics.
        """
        if seeds is None:
            seeds = [6201, 6202, 6203]

        all_episodes: List[EpisodeMetrics] = []
        per_seed_results: Dict[int, Dict[str, Any]] = {}

        # Instantiate persistent environment across episodes
        env = self.env_factory()

        try:
            for seed in seeds:
                seed_everything(seed)
                if hasattr(self.model, "seed") and callable(self.model.seed):
                    self.model.seed(seed)

                seed_episodes: List[EpisodeMetrics] = []
                self.logger.info(f"Starting evaluation on seed {seed} ({num_episodes} episodes)...")

                for ep_idx in range(num_episodes):
                    ep_seed = seed * 1000 + ep_idx
                    metric = self.evaluate_single_episode(
                        env=env,
                        seed=ep_seed,
                        greedy=greedy,
                    )
                    seed_episodes.append(metric)
                    all_episodes.append(metric)

                per_seed_results[seed] = {
                    "metrics": compute_aggregate_metrics(seed_episodes),
                    "episodes": [ep.to_dict() for ep in seed_episodes],
                }

        finally:
            if hasattr(env, "close"):
                env.close()

        aggregate_metrics = compute_aggregate_metrics(all_episodes)

        return {
            "model_name": self.name,
            "seeds": seeds,
            "episodes_per_seed": num_episodes,
            "total_episodes": len(all_episodes),
            "per_seed": per_seed_results,
            "aggregate": aggregate_metrics,
            "all_episodes": [ep.to_dict() for ep in all_episodes],
        }

    def save_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
        """Save results.json, results.csv, and action_distribution.csv to output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. results.json
        json_path = out_path / "results.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. results.csv (flat episode-level data)
        csv_path = out_path / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "seed",
                "episode_idx",
                "kill_success",
                "reward",
                "decisions",
                "game_tics",
                "spike_rate",
                "latency_ms",
                "action_0_noop",
                "action_1_left",
                "action_2_right",
                "action_3_shoot",
                "action_4_left_shoot",
                "action_5_right_shoot",
            ]
            writer.writerow(header)

            for seed, seed_data in results.get("per_seed", {}).items():
                for idx, ep in enumerate(seed_data.get("episodes", [])):
                    act_counts = ep.get("action_counts", [0] * 6)
                    while len(act_counts) < 6:
                        act_counts.append(0)
                    row = [
                        seed,
                        idx + 1,
                        int(ep.get("kill_success", False)),
                        ep.get("reward", 0.0),
                        ep.get("decisions", 0),
                        ep.get("game_tics", 0),
                        ep.get("spike_rate", 0.0),
                        ep.get("latency_ms", 0.0),
                        act_counts[0],
                        act_counts[1],
                        act_counts[2],
                        act_counts[3],
                        act_counts[4],
                        act_counts[5],
                    ]
                    writer.writerow(row)

        # 3. action_distribution.csv
        dist_path = out_path / "action_distribution.csv"
        with dist_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "group",
                "kill_rate",
                "mean_reward",
                "action_0_noop_pct",
                "action_1_left_pct",
                "action_2_right_pct",
                "action_3_shoot_pct",
                "action_4_left_shoot_pct",
                "action_5_right_shoot_pct",
            ])

            # Write aggregate row
            agg = results.get("aggregate", {})
            agg_dist = agg.get("action_distribution", [0.0] * 6)
            while len(agg_dist) < 6:
                agg_dist.append(0.0)
            writer.writerow([
                "aggregate",
                agg.get("kill_rate", 0.0),
                agg.get("mean_reward", 0.0),
                agg_dist[0],
                agg_dist[1],
                agg_dist[2],
                agg_dist[3],
                agg_dist[4],
                agg_dist[5],
            ])

            # Write per-seed rows
            for seed, sdata in results.get("per_seed", {}).items():
                s_metrics = sdata.get("metrics", {})
                s_dist = s_metrics.get("action_distribution", [0.0] * 6)
                while len(s_dist) < 6:
                    s_dist.append(0.0)
                writer.writerow([
                    f"seed_{seed}",
                    s_metrics.get("kill_rate", 0.0),
                    s_metrics.get("mean_reward", 0.0),
                    s_dist[0],
                    s_dist[1],
                    s_dist[2],
                    s_dist[3],
                    s_dist[4],
                    s_dist[5],
                ])

        self.logger.info(f"Results successfully saved to {out_path}")
