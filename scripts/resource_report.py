"""Hardware resource reporting CLI tool for SNN models."""

from __future__ import annotations

import argparse
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

from doom_snn.configs import RunConfig, load_config
from doom_snn.hardware.resource_accounting import (
    compute_resource_report,
    format_resource_table,
    save_resource_report,
)
from doom_snn.models.ann_to_snn import ANNBaseline
from doom_snn.models.ff_snn_actor_critic import FFSNNActorCritic
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic
from doom_snn.models.spike_dqn import SpikeDQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SNAP-V hardware resource accounting report")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration")
    parser.add_argument("--output-dir", type=str, default="hardware_reports", help="Directory to save report outputs")
    parser.add_argument("--neurons", type=int, default=None, help="Core neuron override")
    parser.add_argument("--track", type=str, default="c_rsnn_ac", help="Track to report (c_rsnn_ac, d_ff_snn, b_spike_dqn)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else RunConfig()

    if args.neurons:
        config.model.core_neurons = args.neurons
    if args.track:
        config.model.track = args.track

    track = config.model.track.lower()
    if "rsnn" in track or "c" in track:
        model = RSNNActorCritic(
            input_channels=config.obs.input_channels,
            core_neurons=config.model.core_neurons,
            beta=config.neuron.beta,
            T=config.neuron.T,
        )
    elif "ff" in track or "d" in track:
        model = FFSNNActorCritic(
            input_channels=config.obs.input_channels,
            hidden_sizes=config.model.core_neurons,
            beta=config.neuron.beta,
            T=config.neuron.T,
        )
    elif "dqn" in track or "b" in track:
        model = SpikeDQN(
            input_channels=config.obs.input_channels,
            hidden_size=config.model.core_neurons,
            beta=config.neuron.beta,
            T=config.neuron.T,
        )
    else:
        model = ANNBaseline(input_channels=config.obs.input_channels)

    report = compute_resource_report(model, config)
    out_dir = Path(args.output_dir)
    save_resource_report(report, out_dir)

    print("\n================== NEUROMORPHIC RESOURCE REPORT ==================")
    print(format_resource_table([report]))
    print("\nConstraint Status:")
    for k, v in report["constraints"].items():
        if k in ("warnings", "errors"):
            if v:
                print(f"  {k.upper()}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"\nReport files written to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
