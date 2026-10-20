"""Unit tests for neuromorphic hardware and parameter accounting."""

import pytest
from doom_snn.hardware.resource_accounting import compute_resource_report
from doom_snn.hardware.snapv_constraints import check_constraints
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic


def test_96_neuron_rsnn_counts():
    model = RSNNActorCritic(input_channels=768, core_neurons=96, num_actions=6, beta=0.5, T=8)
    report = compute_resource_report(model)

    assert report["core_neurons"] == 96
    assert report["input_weights"] == 73728  # 768 * 96
    assert report["recurrent_weights"] == 9216  # 96 * 96
    assert report["actor_weights"] == 576  # 96 * 6
    assert report["critic_weights"] == 96  # 96 * 1
    assert report["total_weights"] == 83616
    assert report["total_parameters"] == 83719  # weights + 96 (in_bias) + 6 (act_bias) + 1 (crit_bias)


def test_fallback_accounting():
    model = RSNNActorCritic(input_channels=768, core_neurons=96, num_actions=6)
    report = compute_resource_report(model)

    # If the 768 rate encoder channels are mapped to core spiking neurons:
    assert report["fallback_total_neurons"] == 864  # 96 + 768
    assert report["fallback_neuron_budget_ok"] is True  # 864 < 1024


def test_neuron_budget_check():
    # 96-neuron model satisfies SNAP-V budget (<1024 and <=964)
    res_96 = check_constraints({"core_neurons": 96, "beta": 0.75, "T": 8})
    assert res_96["neuron_budget_ok"] is True
    assert res_96["neurons_under_operating_point"] is True
    assert res_96["overall_compliant"] is True

    # 1050-neuron model exceeds budget
    res_large = check_constraints({"core_neurons": 1050, "beta": 0.75, "T": 8})
    assert res_large["neuron_budget_ok"] is False
    assert res_large["overall_compliant"] is False


def test_128_neuron_counts():
    model = RSNNActorCritic(input_channels=768, core_neurons=128, num_actions=6)
    report = compute_resource_report(model)

    assert report["core_neurons"] == 128
    assert report["input_weights"] == 768 * 128
    assert report["recurrent_weights"] == 128 * 128
    assert report["int16_weight_mb"] > 0.0
