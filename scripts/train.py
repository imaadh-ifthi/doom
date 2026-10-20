"""Main training CLI entrypoint for DOOM SNN models."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

# Prevent OpenMP runtime conflict on Windows platforms
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add src to python path for direct execution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

import torch

from doom_snn.configs import RunConfig, load_config, save_config
from doom_snn.device import resolve_device
from doom_snn.env.dummy_env import DummyEnv
from doom_snn.env.vizdoom_env import VizDoomEnv
from doom_snn.logging_utils import log_config, setup_logging
from doom_snn.models.ann_to_snn import ANNBaseline
from doom_snn.models.ff_snn_actor_critic import FFSNNActorCritic
from doom_snn.models.recurrent_spike_dqn import RecurrentSpikeDQN
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic
from doom_snn.models.spike_dqn import SpikeDQN
from doom_snn.quantization.qat import insert_fake_quant
from doom_snn.seeding import get_reproducibility_info, seed_everything
from doom_snn.training.trainer import create_trainer
from doom_snn.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SNN agents on ViZDoom BASIC")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--episodes", type=int, default=None, help="Total episodes override")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluation interval override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file path to resume from")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-episode smoke test")
    parser.add_argument("--dummy-env", action="store_true", help="Force synthetic DummyEnv (no ViZDoom needed)")
    parser.add_argument("--track", type=str, default=None, help="Track override (c, d, b, a)")
    parser.add_argument("--beta", type=float, default=None, help="Membrane decay beta override (0.25, 0.5, 0.75)")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization override (none, qat, int16)")
    return parser.parse_args()


def build_model(config: RunConfig) -> torch.nn.Module:
    """Build SNN model according to track and configuration."""
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
    elif track in ("b_recurrent", "recurrent_spike_dqn"):
        model = RecurrentSpikeDQN(
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
        raise ValueError(f"Unknown model track: '{track}'")

    # Quantization-Aware Training (QAT) insertion if enabled
    if config.quantization.enabled or config.quantization.qat:
        model = insert_fake_quant(model, {"weight_bits": config.quantization.weight_bits})

    return model


def main() -> None:
    args = parse_args()

    # 1. Load base or specific config
    config = load_config(args.config)

    # 2. Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.episodes is not None:
        config.training.episodes = args.episodes
    if args.eval_every is not None:
        config.training.eval_every = args.eval_every
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.dummy_env:
        config.env.dummy_env = True
    if args.track:
        track_map = {
            "c": "c_rsnn_ac",
            "d": "d_ff_snn",
            "b": "b_spike_dqn",
            "a": "a_ann_baseline",
        }
        config.model.track = track_map.get(args.track.lower(), args.track)
    if args.beta is not None:
        config.neuron.beta = args.beta
    if args.quantization:
        if args.quantization.lower() in ("qat", "true", "int16"):
            config.quantization.enabled = True
            config.quantization.qat = True

    # 3. Handle smoke test configuration
    if args.smoke:
        config.training.episodes = 2
        config.training.rollout_length = 2
        config.training.eval_every = 2
        config.env.dummy_env = True
        config.logging.log_interval = 1

    # 4. Resolve compute device and seed all PRNGs
    device = resolve_device(config.device)
    seed_everything(config.seed)

    # 5. Initialize logging and telemetry
    out_dir = Path(config.output_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, run_name=config.name)
    repro_info = get_reproducibility_info()
    log_config(logger, config, repro_info)
    save_config(config, out_dir / "resolved_config.yaml")

    # 6. Initialize Environment
    if config.env.dummy_env:
        logger.info("Initializing synthetic DummyEnv (offline mode)...")
        env = DummyEnv(config)
    else:
        try:
            logger.info("Initializing ViZDoom headless environment...")
            env = VizDoomEnv(config)
        except Exception as e:
            logger.warning(f"Failed to launch ViZDoom ({e}). Falling back to DummyEnv.")
            env = DummyEnv(config)

    # 7. Build Model
    model = build_model(config)
    logger.info(f"Initialized model '{model.__class__.__name__}' (Track: {config.model.track})")

    # 8. Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming model weights from checkpoint: {args.resume}")
        load_checkpoint(args.resume, model, device=device)

    # 9. Create Trainer and execute training loop
    trainer = create_trainer(
        model=model,
        env=env,
        config=config,
        logger=logger,
        device=device,
    )

    results = trainer.train(num_episodes=config.training.episodes)
    logger.info(f"Training successfully completed! Summary: {results}")

    env.close()


if __name__ == "__main__":
    main()
