"""PyTorch device resolution and hardware acceleration management."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve compute device supporting CPU and CUDA with fallback semantics.

    Args:
        requested: Requested device string ('auto', 'cpu', 'cuda', 'cuda:0', etc.).
            If 'auto', the environment variable DOOM_SNN_DEVICE is checked as fallback.

    Returns:
        Resolved torch.device instance.

    Raises:
        RuntimeError: If CUDA is explicitly requested but not available.
        ValueError: If device string cannot be parsed.
    """
    normalized = requested.strip().lower() if requested else "auto"

    if normalized == "auto":
        env_device = os.environ.get("DOOM_SNN_DEVICE", "").strip().lower()
        if env_device:
            normalized = env_device

    if normalized == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif normalized == "cpu":
        device = torch.device("cpu")
    elif normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{normalized}' was requested, but torch.cuda.is_available() is False."
            )
        try:
            device = torch.device(normalized)
        except Exception as e:
            raise ValueError(f"Failed to initialize CUDA device '{normalized}': {e}") from e
    else:
        try:
            device = torch.device(normalized)
        except Exception as e:
            raise ValueError(f"Unknown device specification '{requested}': {e}") from e

    logger.debug("Resolved device '%s' to %s", requested, device)
    return device
