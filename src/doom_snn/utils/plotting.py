"""Training curves and metric visualization utilities."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def plot_training_curve(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    metrics: Optional[List[str]] = None,
    title: str = "Training Progress",
) -> bool:
    """Plot training metrics from CSV log file.

    Args:
        csv_path: Path to CSV log file.
        output_path: Destination image path (.png, .svg).
        metrics: List of column names to plot. Defaults to ['reward', 'kill_rate'].
        title: Plot title string.

    Returns:
        bool: True if plotting succeeded, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot generation.")
        return False

    csv_p = Path(csv_path)
    if not csv_p.exists():
        logger.warning(f"CSV file not found for plotting: {csv_p}")
        return False

    if metrics is None:
        metrics = ["reward", "kill_rate"]

    data: dict[str, list[float]] = {}
    with open(csv_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    pass

    if not data:
        return False

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    available_metrics = [m for m in metrics if m in data and len(data[m]) > 0]
    if not available_metrics:
        # Fall back to plotting whatever numerical series are present
        available_metrics = [k for k, v in data.items() if len(v) > 0 and k != "episode"][:3]

    if not available_metrics:
        return False

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(8, 3 * len(available_metrics)), sharex=True)
    if len(available_metrics) == 1:
        axes = [axes]

    episodes = data.get("episode", list(range(1, len(data[available_metrics[0]]) + 1)))

    for ax, m in zip(axes, available_metrics):
        ax.plot(episodes, data[m], label=m, color="#1f77b4", linewidth=1.5)
        ax.set_ylabel(m)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Episode")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_p, dpi=150)
    plt.close(fig)
    return True
