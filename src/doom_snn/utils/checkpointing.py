"""Model and optimizer checkpoint serialization utilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer],
    config: Any,
    metrics: Dict[str, Any],
    epoch: int,
) -> None:
    """Save PyTorch model weights, optimizer state, configuration, and metrics.

    Args:
        path: Output file path for checkpoint (.pt or .pth).
        model: PyTorch neural network module.
        optimizer: PyTorch optimizer or None.
        config: Dataclass or dictionary representing run configuration.
        metrics: Dictionary of metric values at checkpoint time.
        epoch: Current epoch or episode index.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_data = asdict(config) if is_dataclass(config) else config

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "config": cfg_data,
    }

    torch.save(checkpoint, ckpt_path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load model weights and optimizer state from checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Target PyTorch model to populate with weights.
        optimizer: Optional optimizer to restore state into.
        device: Target compute device or map_location string (default: 'cpu').

    Returns:
        Dictionary containing 'config', 'metrics', and 'epoch'.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        KeyError: If required keys are missing in checkpoint dictionary.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    target_device = torch.device(device) if isinstance(device, str) else device
    checkpoint = torch.load(ckpt_path, map_location=target_device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint at {ckpt_path} missing 'model_state_dict'")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "config": checkpoint.get("config"),
        "metrics": checkpoint.get("metrics", {}),
        "epoch": checkpoint.get("epoch", 0),
    }
