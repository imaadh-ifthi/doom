"""Unit tests verifying forward tensor shapes and state semantics across all tracks."""

import pytest
import torch
from doom_snn.models.ann_to_snn import ANNBaseline, convert_ann_to_snn
from doom_snn.models.ff_snn_actor_critic import FFSNNActorCritic
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic
from doom_snn.models.spike_dqn import SpikeDQN


def test_rsnn_ac_forward_shape():
    model = RSNNActorCritic(input_channels=768, core_neurons=96, num_actions=6, T=8)
    x = torch.rand((2, 768))

    out = model(x)
    assert out["action_logits"].shape == (2, 6)
    assert out["value"].shape == (2, 1)
    assert out["spikes"].shape == (8, 2, 96)
    assert out["membrane"].shape == (2, 96)


def test_ff_snn_forward_shape():
    model = FFSNNActorCritic(input_channels=768, hidden_sizes=96, num_actions=6, T=8)
    x = torch.rand((3, 768))

    out = model(x)
    assert out["action_logits"].shape == (3, 6)
    assert out["value"].shape == (3, 1)
    assert out["spikes"].shape == (8, 3, 96)
    assert out["membrane"].shape == (3, 96)


def test_spike_dqn_forward_shape():
    model = SpikeDQN(input_channels=768, hidden_size=96, num_actions=6, T=8)
    x = torch.rand((4, 768))

    out = model(x)
    assert "q_values" in out
    assert out["q_values"].shape == (4, 6)
    assert out["spikes"].shape == (8, 4, 96)


def test_ann_baseline_shape():
    ann = ANNBaseline(input_channels=768, hidden_sizes=(128, 64), num_actions=6)
    x = torch.rand((2, 768))
    out = ann(x)
    assert out.shape == (2, 6)


def test_converted_snn_shape():
    ann = ANNBaseline(input_channels=768, hidden_sizes=(128, 64), num_actions=6)
    snn = convert_ann_to_snn(ann, T=8)
    x = torch.rand((2, 768))
    out = snn(x)
    assert out["action_logits"].shape == (2, 6)


def test_recurrent_state_persistence():
    model = RSNNActorCritic(input_channels=768, core_neurons=96, num_actions=6, T=8)
    x = torch.rand((1, 768))

    # First decision
    _ = model(x)
    assert model._state is not None
    st1_mem = model._state.membrane.clone()

    # Second decision: should update persistent state
    _ = model(x)
    st2_mem = model._state.membrane.clone()
    assert not torch.equal(st1_mem, st2_mem)

    # Reset state at episode boundary
    model.reset_state()
    assert model._state is None


def test_ff_no_recurrence():
    model = FFSNNActorCritic(input_channels=768, hidden_sizes=96, num_actions=6, T=8)
    assert model.is_recurrent is False
    assert not hasattr(model, "fc_recurrent")
