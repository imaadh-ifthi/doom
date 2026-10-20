"""Track A: ANN training, ANN-to-SNN conversion, and surrogate fine-tuning script."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

# Prevent OpenMP runtime conflict on Windows platforms
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

import torch

from doom_snn.configs import RunConfig, load_config
from doom_snn.device import resolve_device
from doom_snn.env.dummy_env import DummyEnv
from doom_snn.env.vizdoom_env import VizDoomEnv
from doom_snn.evaluation.evaluator import Evaluator
from doom_snn.models.ann_to_snn import ANNBaseline, convert_ann_to_snn
from doom_snn.utils.checkpointing import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ANN to SNN and evaluate conversion loss")
    parser.add_argument("--config", type=str, default="configs/track_a_ann_to_snn.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/track_a_conversion")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--dummy-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("ann_conversion")

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config) if Path(args.config).exists() else RunConfig()
    if args.dummy_env:
        config.env.dummy_env = True

    def env_factory():
        if config.env.dummy_env:
            return DummyEnv(config)
        try:
            return VizDoomEnv(config)
        except Exception:
            return DummyEnv(config)

    logger.info("Initializing reference ANN baseline...")
    ann = ANNBaseline(input_channels=config.obs.input_channels, num_actions=config.actions.num_actions).to(device)

    # Evaluate ANN baseline
    ann_evaluator = Evaluator(ann, env_factory, config, device, logger)
    ann_results = ann_evaluator.evaluate(num_episodes=args.episodes, seeds=[6201], greedy=True)
    logger.info(f"ANN Baseline -> Kill Rate: {ann_results['aggregate']['kill_rate']:.1%}, Reward: {ann_results['aggregate']['mean_reward']:.1f}")

    # Convert to SNN
    logger.info("Converting ANN to SNN with accumulator rate coding (T=8)...")
    snn = convert_ann_to_snn(ann, T=8, beta=config.neuron.beta, threshold=config.neuron.threshold).to(device)

    snn_evaluator = Evaluator(snn, env_factory, config, device, logger)
    snn_results = snn_evaluator.evaluate(num_episodes=args.episodes, seeds=[6201], greedy=True)
    logger.info(f"Converted SNN (T=8) -> Kill Rate: {snn_results['aggregate']['kill_rate']:.1%}, Reward: {snn_results['aggregate']['mean_reward']:.1f}")

    # Save converted checkpoint
    save_checkpoint(
        path=out_dir / "converted_snn.pt",
        model=snn,
        optimizer=None,
        config=config,
        metrics=snn_results["aggregate"],
        epoch=0,
    )
    logger.info(f"Converted SNN model saved to {out_dir / 'converted_snn.pt'}")


if __name__ == "__main__":
    main()
