"""Hardware resource and synaptic memory accounting for neuromorphic deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from doom_snn.hardware.snapv_constraints import check_constraints


def compute_resource_report(
    model: nn.Module,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute detailed hardware resource utilization report for an SNN model.

    Evaluates neuron count, synaptic weights, fixed-point memory footprint,
    and computational MAC complexity under PeraMorphIQ / SNAP-V constraints.

    Args:
        model: PyTorch SNN policy model.
        config: Optional configuration object or dict.

    Returns:
        Dict[str, Any] containing comprehensive resource breakdown.
    """
    # Extract model attributes with resilient fallbacks
    core_neurons = getattr(model, "core_neurons", None)
    if core_neurons is None:
        core_neurons = getattr(model, "hidden_size", 96)
    core_neurons = int(core_neurons)

    input_channels = getattr(model, "input_channels", None)
    if input_channels is None:
        input_channels = getattr(model, "input_size", 768)
    input_channels = int(input_channels)

    is_recurrent = getattr(model, "is_recurrent", False)
    if not is_recurrent and hasattr(model, "core"):
        # Check if core is RecurrentLIFLayer or has fc_recurrent
        if hasattr(model.core, "fc_recurrent") and getattr(model.core, "fc_recurrent", None) is not None:
            is_recurrent = True

    num_actions = getattr(model, "num_actions", 6)
    is_actor_critic = getattr(model, "is_actor_critic", True)
    if hasattr(model, "actor") and hasattr(model, "critic"):
        is_actor_critic = True
    elif hasattr(model, "q_head"):
        is_actor_critic = False

    T = getattr(model, "T", 8)
    beta = getattr(model, "beta", 0.75)
    if hasattr(model, "core") and hasattr(model.core, "beta"):
        beta = model.core.beta

    # 1. Synaptic connection counts
    input_weights = input_channels * core_neurons
    recurrent_weights = (core_neurons * core_neurons) if is_recurrent else 0
    actor_weights = (core_neurons * num_actions) if is_actor_critic else 0
    critic_weights = (core_neurons * 1) if is_actor_critic else 0
    q_weights = (core_neurons * num_actions) if not is_actor_critic else 0

    total_weights = input_weights + recurrent_weights + actor_weights + critic_weights + q_weights

    # Bias counts
    input_bias = core_neurons
    actor_bias = num_actions if is_actor_critic else 0
    critic_bias = 1 if is_actor_critic else 0
    q_bias = num_actions if not is_actor_critic else 0
    total_biases = input_bias + actor_bias + critic_bias + q_bias

    total_parameters = total_weights + total_biases

    # 2. Memory accounting (Assuming INT16 = 2 bytes per weight/bias)
    int16_weight_memory_bytes = total_weights * 2
    int16_total_memory_bytes = total_parameters * 2
    int16_weight_mb = int16_weight_memory_bytes / (1024 * 1024)
    int16_total_mb = int16_total_memory_bytes / (1024 * 1024)

    # State memory
    membrane_state_bytes = core_neurons * 2  # INT16 membrane per batch elem
    spike_state_bytes = core_neurons  # 1 byte (or 1 bit) per neuron
    encoder_accumulator_bytes = input_channels * 4  # FP32 or INT32 accumulator

    # 3. Neuron accounting
    # Default: raw input pixels are external channels, only LIF core neurons count
    fallback_total_neurons = core_neurons + input_channels
    neuron_budget_ok = core_neurons < 1024
    fallback_neuron_budget_ok = fallback_total_neurons < 1024

    # 4. Compute / MAC complexity
    dense_macs_per_timestep = input_weights + recurrent_weights + actor_weights + critic_weights + q_weights
    dense_macs_per_decision = T * dense_macs_per_timestep

    # Check hardware constraints
    constraints_input = {
        "core_neurons": core_neurons,
        "beta": beta,
        "T": T,
        "fallback_total_neurons": fallback_total_neurons,
    }
    constraint_status = check_constraints(constraints_input)

    track_name = getattr(model, "track", "c_rsnn_ac" if is_recurrent else "d_ff_snn")

    report: Dict[str, Any] = {
        "track": track_name,
        "model_class": model.__class__.__name__,
        "core_neurons": core_neurons,
        "input_channels": input_channels,
        "is_recurrent": is_recurrent,
        "is_actor_critic": is_actor_critic,
        "beta": float(beta),
        "T": int(T),
        "input_weights": input_weights,
        "recurrent_weights": recurrent_weights,
        "actor_weights": actor_weights,
        "critic_weights": critic_weights,
        "q_weights": q_weights,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total_parameters": total_parameters,
        "int16_weight_memory_bytes": int16_weight_memory_bytes,
        "int16_weight_mb": round(int16_weight_mb, 4),
        "int16_total_memory_bytes": int16_total_memory_bytes,
        "int16_total_mb": round(int16_total_mb, 4),
        "membrane_state_bytes": membrane_state_bytes,
        "spike_state_bytes": spike_state_bytes,
        "encoder_accumulator_bytes": encoder_accumulator_bytes,
        "fallback_total_neurons": fallback_total_neurons,
        "neuron_budget_ok": neuron_budget_ok,
        "fallback_neuron_budget_ok": fallback_neuron_budget_ok,
        "dense_macs_per_timestep": dense_macs_per_timestep,
        "dense_macs_per_decision": dense_macs_per_decision,
        "constraints": constraint_status,
    }

    return report


def compute_activity_report(
    model: nn.Module,
    obs_sample: torch.Tensor,
    T: int = 8,
) -> Dict[str, Any]:
    """Measure empirical spike activity and estimate event-driven computational savings.

    Args:
        model: SNN model to evaluate.
        obs_sample: Observation sample tensor [batch, input_channels].
        T: Spiking timesteps per decision.

    Returns:
        Dict[str, Any] containing firing rates and synaptic event estimates.
    """
    model.eval()
    with torch.no_grad():
        if obs_sample.dim() == 1:
            obs_sample = obs_sample.unsqueeze(0)

        # Trigger forward pass
        if hasattr(model, "forward"):
            _ = model(obs_sample)

        input_spike_rate = getattr(model, "last_input_spike_rate", 0.0)
        hidden_spike_rate = getattr(model, "last_hidden_spike_rate", 0.0)

        resource = compute_resource_report(model)
        input_synapses = resource["input_weights"]
        rec_synapses = resource["recurrent_weights"]

        # Event-driven synaptic updates per decision
        input_events = input_spike_rate * input_synapses * T
        rec_events = hidden_spike_rate * rec_synapses * T
        total_synaptic_events = input_events + rec_events

        dense_macs = resource["dense_macs_per_decision"]
        sparsity_savings_pct = (
            (1.0 - (total_synaptic_events / dense_macs)) * 100.0 if dense_macs > 0 else 0.0
        )

        return {
            "mean_input_spike_rate": round(float(input_spike_rate), 4),
            "mean_hidden_spike_rate": round(float(hidden_spike_rate), 4),
            "input_synaptic_events_per_decision": round(float(input_events), 1),
            "recurrent_synaptic_events_per_decision": round(float(rec_events), 1),
            "total_synaptic_events_per_decision": round(float(total_synaptic_events), 1),
            "worst_case_dense_macs": dense_macs,
            "event_driven_sparsity_savings_pct": round(float(sparsity_savings_pct), 2),
        }


def format_resource_table(reports: List[Dict[str, Any]]) -> str:
    """Format multiple model resource reports into a markdown comparison table.

    Args:
        reports: List of resource report dictionaries.

    Returns:
        Markdown table string.
    """
    headers = [
        "Track",
        "Model",
        "Beta",
        "T",
        "Core Neurons",
        "Input Weights",
        "Recurrent Weights",
        "Total Weights",
        "INT16 Weight MB",
        "Neuron Budget OK",
        "Fallback Budget OK",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for r in reports:
        row = [
            str(r.get("track", "-")),
            str(r.get("model_class", "-")),
            str(r.get("beta", "-")),
            str(r.get("T", "-")),
            str(r.get("core_neurons", "-")),
            f"{r.get('input_weights', 0):,}",
            f"{r.get('recurrent_weights', 0):,}",
            f"{r.get('total_weights', 0):,}",
            f"{r.get('int16_weight_mb', 0.0):.3f} MB",
            "Yes" if r.get("neuron_budget_ok") else "No",
            "Yes" if r.get("fallback_neuron_budget_ok") else "Exceeds",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def save_resource_report(report: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """Save resource report to resource.json and resource.md.

    Args:
        report: Single resource report or dictionary of multiple reports.
        output_dir: Destination directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "resource.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_path / "resource.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Neuromorphic Hardware Resource Report\n\n")
        if "reports" in report and isinstance(report["reports"], list):
            f.write(format_resource_table(report["reports"]))
        else:
            f.write(format_resource_table([report]))
        f.write("\n")
