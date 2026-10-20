"""Structured logging setup and configuration dump utilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union


def setup_logging(output_dir: Union[str, Path], run_name: str) -> logging.Logger:
    """Configure logger with timestamped file handler and stdout stream handler.

    Args:
        output_dir: Directory where log file will be saved.
        run_name: Name of the experiment run.

    Returns:
        Configured logging.Logger instance.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_path / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger(f"doom_snn.{run_name}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Initialized logger for run '%s'. Logging to %s", run_name, log_file)
    return logger


def log_config(
    logger: logging.Logger,
    config: Any,
    repro_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Log full experiment configuration and reproducibility metadata.

    Args:
        logger: Logger instance to output to.
        config: Dataclass or dictionary representing experiment configuration.
        repro_info: Optional dictionary containing reproducibility info.
    """
    cfg_dict = asdict(config) if is_dataclass(config) else config
    cfg_json = json.dumps(cfg_dict, indent=2, default=str)
    logger.info("=== Run Configuration ===\n%s", cfg_json)

    if repro_info:
        repro_json = json.dumps(repro_info, indent=2, default=str)
        logger.info("=== Reproducibility Metadata ===\n%s", repro_json)
