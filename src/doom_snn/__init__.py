"""Doom SNN package for hardware-aware Spiking Neural Networks playing ViZDoom."""

import os

# Prevent OpenMP runtime conflict on Windows platforms
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"

from doom_snn.configs import (
    ActionConfig,
    DQNTrainingConfig,
    EnvConfig,
    EvalConfig,
    LoggingConfig,
    ModelConfig,
    NeuronConfig,
    ObsConfig,
    QuantizationConfig,
    RunConfig,
    TrainingConfig,
    config_hash,
    load_config,
    save_config,
)
from doom_snn.device import resolve_device
from doom_snn.seeding import get_reproducibility_info, seed_everything

__all__ = [
    "__version__",
    "ObsConfig",
    "ActionConfig",
    "NeuronConfig",
    "ModelConfig",
    "TrainingConfig",
    "DQNTrainingConfig",
    "EvalConfig",
    "QuantizationConfig",
    "LoggingConfig",
    "EnvConfig",
    "RunConfig",
    "load_config",
    "save_config",
    "config_hash",
    "resolve_device",
    "seed_everything",
    "get_reproducibility_info",
]
