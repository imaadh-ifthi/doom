"""Benchmark comparison script across SNN tracks, ANN baselines, and heuristic agents."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
import yaml

# Prevent OpenMP runtime conflict on Windows platforms
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from doom_snn.configs import RunConfig, load_config
from doom_snn.device import resolve_device
from doom_snn.env.dummy_env import DummyEnv
from doom_snn.env.vizdoom_env import VizDoomEnv
from doom_snn.evaluation.baselines import AlwaysShootAgent, LeftShootAgent, RandomAgent
from doom_snn.evaluation.evaluator import Evaluator
from doom_snn.hardware.resource_accounting import compute_resource_report
from doom_snn.models.ann_to_snn import ANNBaseline
from doom_snn.models.ff_snn_actor_critic import FFSNNActorCritic
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic
from doom_snn.models.spike_dqn import SpikeDQN
from doom_snn.quantization.qat import insert_fake_quant
from doom_snn.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark comparison suite")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml", help="Path to benchmark YAML")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--output-dir", type=str, default="benchmark_output", help="Output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per seed override")
    parser.add_argument("--dummy-env", action="store_true", help="Force synthetic DummyEnv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("benchmark")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Benchmark config not found: {cfg_path}")
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        bench_data = yaml.safe_load(f).get("benchmark", {})

    episodes = args.episodes or bench_data.get("episodes_per_seed", 50)
    seeds = bench_data.get("seeds", [6201, 6202, 6203])
    models_spec = bench_data.get("models", [])

    base_config = RunConfig()
    if args.dummy_env:
        base_config.env.dummy_env = True

    def env_factory():
        if base_config.env.dummy_env:
            return DummyEnv(base_config)
        try:
            return VizDoomEnv(base_config)
        except Exception:
            return DummyEnv(base_config)

    benchmark_rows = []
    full_results = {}

    for spec in models_spec:
        name = spec.get("name")
        m_type = spec.get("type", "baseline")
        logger.info(f"Running benchmark entry: {name} (type: {m_type})...")

        agent = None
        resource_info = {}

        if m_type == "baseline":
            b_name = spec.get("baseline", "random")
            if b_name == "random":
                agent = RandomAgent()
            elif b_name == "always_shoot":
                agent = AlwaysShootAgent()
            elif b_name == "left_shoot":
                agent = LeftShootAgent()
            else:
                agent = RandomAgent()

        elif m_type == "checkpoint":
            ckpt_path = Path(spec.get("checkpoint", ""))
            cfg_file = spec.get("config")
            m_config = load_config(cfg_file) if cfg_file else base_config

            track = m_config.model.track.lower()
            if "rsnn" in track or "c" in track:
                agent = RSNNActorCritic(
                    input_channels=m_config.obs.input_channels,
                    core_neurons=m_config.model.core_neurons,
                    beta=m_config.neuron.beta,
                    threshold=m_config.neuron.threshold,
                    T=m_config.neuron.T,
                )
            elif "ff" in track or "d" in track:
                agent = FFSNNActorCritic(
                    input_channels=m_config.obs.input_channels,
                    hidden_sizes=m_config.model.core_neurons,
                    beta=m_config.neuron.beta,
                    threshold=m_config.neuron.threshold,
                    T=m_config.neuron.T,
                )
            elif "dqn" in track or "b" in track:
                agent = SpikeDQN(
                    input_channels=m_config.obs.input_channels,
                    hidden_size=m_config.model.core_neurons,
                    beta=m_config.neuron.beta,
                    threshold=m_config.neuron.threshold,
                    T=m_config.neuron.T,
                )
            else:
                agent = ANNBaseline(input_channels=m_config.obs.input_channels)

            if m_config.quantization.enabled or m_config.quantization.qat:
                agent = insert_fake_quant(agent, {"weight_bits": 16})

            if ckpt_path.exists():
                try:
                    load_checkpoint(ckpt_path, agent, device=device)
                    logger.info(f"Loaded weights for {name} from {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint for {name}: {e}. Evaluating initialized weights.")
            else:
                logger.warning(f"Checkpoint not found at {ckpt_path}; evaluating initialized model.")

            resource_info = compute_resource_report(agent)

        evaluator = Evaluator(
            model_or_baseline=agent,
            env_factory=env_factory,
            config=base_config,
            device=device,
            logger=logger,
        )
        res = evaluator.evaluate(num_episodes=episodes, seeds=seeds, greedy=True)
        agg = res["aggregate"]
        full_results[name] = {"eval": res, "resource": resource_info}

        row = {
            "name": name,
            "type": m_type,
            "kill_rate": agg["kill_rate"],
            "mean_reward": agg["mean_reward"],
            "median_reward": agg["median_reward"],
            "std_reward": agg["std_reward"],
            "mean_decisions": agg["mean_decisions"],
            "mean_latency_ms": agg["mean_latency_ms"],
            "mean_spike_rate": agg["mean_spike_rate"],
            "core_neurons": resource_info.get("core_neurons", "-"),
            "total_weights": resource_info.get("total_weights", "-"),
            "int16_weight_mb": resource_info.get("int16_weight_mb", "-"),
            "neuron_budget_ok": resource_info.get("neuron_budget_ok", "-"),
        }
        benchmark_rows.append(row)

    # 1. Save benchmark_results.json
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # 2. Save benchmark_results.csv
    csv_file = out_dir / "benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        if benchmark_rows:
            writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
            writer.writeheader()
            writer.writerows(benchmark_rows)

    # 3. Save benchmark_summary.md
    md_file = out_dir / "benchmark_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Assumed Effective SNN Benchmark Comparison Summary\n\n")
        f.write(f"- **Evaluated Episodes per Seed:** {episodes}\n")
        f.write(f"- **Seeds:** {seeds}\n")
        f.write(f"- **Device:** {device}\n\n")

        headers = ["Model / Policy", "Type", "Kill Rate", "Mean Reward", "Decisions", "Latency (ms)", "Neurons", "INT16 Weight MB"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for r in benchmark_rows:
            line = [
                str(r["name"]),
                str(r["type"]),
                f"{r['kill_rate']:.1%}",
                f"{r['mean_reward']:.1f}",
                f"{r['mean_decisions']:.1f}",
                f"{r['mean_latency_ms']:.2f}",
                str(r["core_neurons"]),
                f"{r['int16_weight_mb']}" if r["int16_weight_mb"] != "-" else "-",
            ]
            f.write("| " + " | ".join(line) + " |\n")
        f.write("\n")

    logger.info(f"Benchmark run complete! Results written to {out_dir}")


if __name__ == "__main__":
    main()
