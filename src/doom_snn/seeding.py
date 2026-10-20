"""Reproducibility and random seeding utilities for deterministic training runs."""

from __future__ import annotations

import datetime
import os
import platform
import random
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python built-in random, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_reproducibility_info() -> Dict[str, Any]:
    """Collect git commit, environment versioning, and hardware info.

    Returns:
        Dictionary containing reproducibility metadata.
    """
    git_commit = "unknown"
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        pass

    cuda_avail = torch.cuda.is_available()
    cuda_ver: Optional[str] = torch.version.cuda if cuda_avail else None
    active_device = "cuda" if cuda_avail else "cpu"

    return {
        "git_commit": git_commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "device": active_device,
    }
