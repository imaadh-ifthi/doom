"""Unit tests for offline DummyEnv test fixture."""

import numpy as np
import pytest
from doom_snn.configs import RunConfig
from doom_snn.env.dummy_env import DummyEnv


def test_dummy_env_reset():
    cfg = RunConfig()
    env = DummyEnv(cfg)
    obs = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (768,)
    assert obs.dtype == np.float32
    assert env.decisions_count == 0


def test_dummy_env_step():
    cfg = RunConfig()
    env = DummyEnv(cfg)
    _ = env.reset(seed=123)

    next_obs, reward, terminated, truncated, info = env.step(3)  # shoot action
    assert next_obs.shape == (768,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "kill_success" in info
    assert "episode_reward" in info
    assert "game_tics" in info


def test_dummy_env_episode_terminates():
    cfg = RunConfig()
    env = DummyEnv(cfg, max_steps=5)
    _ = env.reset(seed=1)

    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
        if steps > 10:
            break

    assert done is True
    assert steps <= 5
