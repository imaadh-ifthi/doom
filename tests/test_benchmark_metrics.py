"""Unit tests for evaluation metrics aggregation and fixed-point degradation."""

import pytest
from doom_snn.evaluation.metrics import (
    EpisodeMetrics,
    compute_aggregate_metrics,
    compute_fixed_point_degradation,
)


def test_episode_metrics_creation():
    ep = EpisodeMetrics(
        kill_success=True,
        reward=95.0,
        decisions=18,
        game_tics=72,
        action_counts=[1, 2, 2, 8, 3, 2],
        spike_rate=0.25,
        latency_ms=2.5,
    )
    d = ep.to_dict()
    assert d["kill_success"] is True
    assert d["reward"] == 95.0
    assert d["decisions"] == 18


def test_aggregate_metrics():
    episodes = [
        EpisodeMetrics(kill_success=True, reward=100.0, decisions=10, game_tics=40, action_counts=[1, 1, 1, 5, 1, 1], spike_rate=0.2, latency_ms=1.0),
        EpisodeMetrics(kill_success=False, reward=-10.0, decisions=50, game_tics=200, action_counts=[10, 10, 10, 10, 5, 5], spike_rate=0.1, latency_ms=1.2),
    ]

    agg = compute_aggregate_metrics(episodes)
    assert agg["total_episodes"] == 2
    assert agg["kill_rate"] == 0.5  # 1 out of 2
    assert agg["mean_reward"] == 45.0  # (100 - 10) / 2
    assert agg["mean_decisions"] == 30.0  # (10 + 50) / 2


def test_fixed_point_degradation():
    float_metrics = {"kill_rate": 0.80, "mean_reward": 50.0, "mean_decisions": 16.0}
    quant_metrics = {"kill_rate": 0.78, "mean_reward": 48.0, "mean_decisions": 17.0}

    deg = compute_fixed_point_degradation(float_metrics, quant_metrics)
    assert abs(deg["kill_rate_drop"] - 0.02) < 1e-4
    assert abs(deg["reward_drop"] - 2.0) < 1e-4
    assert abs(deg["decisions_change"] - 1.0) < 1e-4
