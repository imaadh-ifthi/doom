"""Evaluation metrics computation for DOOM SNN policies."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics recorded from a single evaluation episode."""

    kill_success: bool
    reward: float
    decisions: int
    game_tics: int
    action_counts: List[int] = field(default_factory=lambda: [0] * 6)
    spike_rate: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert EpisodeMetrics instance to a dictionary."""
        return asdict(self)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> List[float]:
    """Calculate Wilson score interval for binomial proportion."""
    if total <= 0:
        return [0.0, 0.0]
    # Default z for 95% confidence
    z = 1.95996398454
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return [round(lower, 4), round(upper, 4)]


def compute_aggregate_metrics(episodes: List[EpisodeMetrics]) -> Dict[str, Any]:
    """Compute aggregate statistical metrics across a collection of episodes.

    Args:
        episodes: List of EpisodeMetrics from one or more seeds.

    Returns:
        Dict[str, Any] containing:
            - kill_rate: fraction of successful kills in [0.0, 1.0]
            - mean_reward: average episodic reward
            - median_reward: median episodic reward
            - std_reward: standard deviation of episodic rewards
            - mean_decisions: average decision steps per episode
            - mean_tics: average game tics per episode
            - action_distribution (%): list of percentage allocations per action
            - mean_spike_rate: average spiking rate per decision step
            - mean_latency_ms: average inference latency in milliseconds
            - total_episodes: total number of evaluated episodes
    """
    total = len(episodes)
    if total == 0:
        return {
            "kill_rate": 0.0,
            "mean_reward": 0.0,
            "median_reward": 0.0,
            "std_reward": 0.0,
            "mean_decisions": 0.0,
            "mean_tics": 0.0,
            "action_distribution": [0.0] * 6,
            "action_distribution_pct": [0.0] * 6,
            "mean_spike_rate": 0.0,
            "mean_latency_ms": 0.0,
            "total_episodes": 0,
            "total_kills": 0,
            "kill_rate_wilson_95": [0.0, 0.0],
            "min_reward": 0.0,
            "max_reward": 0.0,
        }

    kills = sum(1 for ep in episodes if ep.kill_success)
    kill_rate = float(kills / total)

    rewards = np.array([ep.reward for ep in episodes], dtype=np.float64)
    decisions = np.array([ep.decisions for ep in episodes], dtype=np.float64)
    tics = np.array([ep.game_tics for ep in episodes], dtype=np.float64)
    spike_rates = np.array([ep.spike_rate for ep in episodes], dtype=np.float64)
    latencies = np.array([ep.latency_ms for ep in episodes], dtype=np.float64)

    # Determine maximum number of actions observed across episodes (default 6)
    max_actions = max((len(ep.action_counts) for ep in episodes), default=6)
    max_actions = max(max_actions, 6)

    total_action_counts = np.zeros(max_actions, dtype=np.int64)
    for ep in episodes:
        for idx, count in enumerate(ep.action_counts):
            if idx < max_actions:
                total_action_counts[idx] += count

    total_actions_sum = int(np.sum(total_action_counts))
    if total_actions_sum > 0:
        action_distribution_pct = [
            round(float((count / total_actions_sum) * 100.0), 2)
            for count in total_action_counts
        ]
    else:
        action_distribution_pct = [0.0] * max_actions

    return {
        "kill_rate": round(kill_rate, 4),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "median_reward": round(float(np.median(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_decisions": round(float(np.mean(decisions)), 2),
        "mean_tics": round(float(np.mean(tics)), 2),
        "action_distribution": action_distribution_pct,
        "action_distribution_pct": action_distribution_pct,
        "action_totals": [int(c) for c in total_action_counts],
        "mean_spike_rate": round(float(np.mean(spike_rates)), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "total_episodes": total,
        "total_kills": kills,
        "kill_rate_wilson_95": wilson_score_interval(kills, total),
        "min_reward": round(float(np.min(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
    }


def compute_fixed_point_degradation(
    float_metrics: Dict[str, Any],
    quant_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute degradation caused by fixed-point quantization.

    Compares floating-point software model metrics with quantized/fixed-point metrics.

    Args:
        float_metrics: Metric dictionary from full-precision model evaluation.
        quant_metrics: Metric dictionary from quantized/fixed-point model evaluation.

    Returns:
        Dict[str, Any] containing drops and changes in key gameplay metrics.
    """
    f_kill = float(float_metrics.get("kill_rate", 0.0))
    q_kill = float(quant_metrics.get("kill_rate", 0.0))
    kill_rate_drop = round(f_kill - q_kill, 4)

    f_rew = float(float_metrics.get("mean_reward", 0.0))
    q_rew = float(quant_metrics.get("mean_reward", 0.0))
    reward_drop = round(f_rew - q_rew, 2)

    f_dec = float(float_metrics.get("mean_decisions", 0.0))
    q_dec = float(quant_metrics.get("mean_decisions", 0.0))
    decisions_change = round(q_dec - f_dec, 2)

    # Relative drops in percent
    rel_kill_drop_pct = (
        round(((f_kill - q_kill) / f_kill) * 100.0, 2) if f_kill > 1e-6 else 0.0
    )
    rel_reward_drop_pct = (
        round(((f_rew - q_rew) / abs(f_rew)) * 100.0, 2) if abs(f_rew) > 1e-6 else 0.0
    )

    return {
        "kill_rate_drop": kill_rate_drop,
        "reward_drop": reward_drop,
        "decisions_change": decisions_change,
        "relative_kill_rate_drop_pct": rel_kill_drop_pct,
        "relative_reward_drop_pct": rel_reward_drop_pct,
        "float_kill_rate": f_kill,
        "quant_kill_rate": q_kill,
        "float_mean_reward": f_rew,
        "quant_mean_reward": q_rew,
        "float_mean_decisions": f_dec,
        "quant_mean_decisions": q_dec,
    }
