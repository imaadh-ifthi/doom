"""Shared pytest fixtures for DOOM SNN test suite."""

import os
from pathlib import Path
import sys
import pytest
import torch

# Ensure OpenMP works cleanly on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure src is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from doom_snn.configs import RunConfig
from doom_snn.env.dummy_env import DummyEnv


@pytest.fixture
def default_config() -> RunConfig:
    """Return default experiment configuration."""
    cfg = RunConfig()
    cfg.env.dummy_env = True
    cfg.device = "cpu"
    return cfg


@pytest.fixture
def dummy_env(default_config) -> DummyEnv:
    """Return an instantiated DummyEnv for testing."""
    return DummyEnv(default_config)


@pytest.fixture
def sample_obs() -> torch.Tensor:
    """Return a single synthetic batch of 768 continuous observation values in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand((2, 768), dtype=torch.float32)
