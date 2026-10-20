"""Evaluation CLI entrypoint for trained SNN checkpoints and baselines."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Prevent OpenMP runtime conflict on Windows platforms
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

import torch

from doom_snn.configs import RunConfig, load_config
from doom_snn.device import resolve_device
from doom_snn.env.dummy_env import DummyEnv
from doom_snn.env.vizdoom_env import VizDoomEnv
from doom_snn.evaluation.baselines import get_all_baselines
from doom_snn.evaluation.evaluator import Evaluator
from doom_snn.evaluation.robustness import RobustnessEvaluator
from doom_snn.models.ann_to_snn import ANNBaseline
from doom_snn.models.ff_snn_actor_critic import FFSNNActorCritic
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic
from doom_snn.models.spike_dqn import SpikeDQN
from doom_snn.quantization.qat import insert_fake_quant
from doom_snn.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SNN checkpoints and baselines")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per seed (default: 100)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[6201, 6202, 6203], help="Evaluation seeds")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save eval outputs")
    parser.add_argument("--baselines", action="store_true", help="Also evaluate standard baselines")
    parser.add_argument("--robustness", action="store_true", help="Run full perturbation robustness battery")
    parser.add_argument("--dummy-env", action="store_true", help="Force synthetic DummyEnv")
    return parser.parse_args()


def build_model_from_config(config: RunConfig) -> torch.nn.Module:
    track = config.model.track.lower()
    if track in ("c", "c_rsnn_ac", "rsnn"):
        model = RSNNActorCritic(
            input_channels=config.obs.input_channels,
            core_neurons=config.model.core_neurons,
            num_actions=config.actions.num_actions,
            beta=config.neuron.beta,
            threshold=config.neuron.threshold,
            T=config.neuron.T,
            reset_mode=config.neuron.reset,
            readout=config.model.readout,
        )
    elif track in ("d", "d_ff_snn", "ff_snn"):
        model = FFSNNActorCritic(
            input_channels=config.obs.input_channels,
            hidden_sizes=config.model.core_neurons,
            num_actions=config.actions.num_actions,
            beta=config.neuron.beta,
            threshold=config.neuron.threshold,
            T=config.neuron.T,
            reset_mode=config.neuron.reset,
            readout=config.model.readout,
        )
    elif track in ("b", "b_spike_dqn", "spike_dqn"):
        model = SpikeDQN(
            input_channels=config.obs.input_channels,
            hidden_size=config.model.core_neurons,
            num_actions=config.actions.num_actions,
            beta=config.neuron.beta,
            threshold=config.neuron.threshold,
            T=config.neuron.T,
            reset_mode=config.neuron.reset,
            readout=config.model.readout,
        )
    elif track in ("a", "a_ann_baseline", "ann"):
        model = ANNBaseline(
            input_channels=config.obs.input_channels,
            hidden_sizes=(128, 64),
            num_actions=config.actions.num_actions,
        )
    else:
        raise ValueError(f"Unknown track: {track}")

    if config.quantization.enabled or config.quantization.qat:
        model = insert_fake_quant(model, {"weight_bits": config.quantization.weight_bits})

    return model


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("eval")

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config) if args.config else RunConfig()
    if args.dummy_env:
        config.env.dummy_env = True

    # 1. Environment factory
    def env_factory():
        if config.env.dummy_env:
            return DummyEnv(config)
        try:
            return VizDoomEnv(config)
        except Exception:
            return DummyEnv(config)

    # 2. Evaluate Baselines if requested or if no checkpoint provided
    if args.baselines or args.checkpoint is None:
        logger.info("Evaluating heuristic baselines...")
        for baseline in get_all_baselines():
            evaluator = Evaluator(
                model_or_baseline=baseline,
                env_factory=env_factory,
                config=config,
                device=device,
                logger=logger,
            )
            res = evaluator.evaluate(
                num_episodes=args.episodes,
                seeds=args.seeds,
                greedy=True,
            )
            evaluator.save_results(res, out_dir / baseline.name())
            agg = res["aggregate"]
            logger.info(
                f"Baseline [{baseline.name()}] -> Kill Rate: {agg['kill_rate']:.1%}, "
                f"Reward: {agg['mean_reward']:.1f}, Steps: {agg['mean_decisions']:.1f}"
            )

    # 3. Evaluate Neural Model Checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        logger.info(f"Loading checkpoint: {ckpt_path}")

        model = build_model_from_config(config)
        load_checkpoint(ckpt_path, model, device=device)

        evaluator = Evaluator(
            model_or_baseline=model,
            env_factory=env_factory,
            config=config,
            device=device,
            logger=logger,
        )
        res = evaluator.evaluate(
            num_episodes=args.episodes,
            seeds=args.seeds,
            greedy=True,
        )
        model_out = out_dir / ckpt_path.stem
        evaluator.save_results(res, model_out)

        agg = res["aggregate"]
        logger.info(
            f"Model [{ckpt_path.name}] -> Kill Rate: {agg['kill_rate']:.1%}, "
            f"Reward: {agg['mean_reward']:.1f}, Steps: {agg['mean_decisions']:.1f}"
        )

        # 4. Run Robustness Battery if requested
        if args.robustness:
            logger.info("Executing robustness perturbation battery...")
            rob_evaluator = RobustnessEvaluator(
                model=model,
                env_factory=env_factory,
                config=config,
                device=device,
            )
            rob_report = rob_evaluator.full_robustness_report(num_episodes=min(20, args.episodes))
            rob_path = model_out / "robustness_report.json"
            with open(rob_path, "w", encoding="utf-8") as f:
                json.dump(rob_report, f, indent=2)
            logger.info(f"Robustness report saved to {rob_path}")


if __name__ == "__main__":
    main()
