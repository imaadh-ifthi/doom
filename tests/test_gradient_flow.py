"""Unit tests verifying surrogate gradient propagation through temporal spiking unrolling."""

import pytest
import torch
from doom_snn.models.ff_snn_actor_critic import FFSNNActorCritic
from doom_snn.models.rsnn_actor_critic import RSNNActorCritic
from doom_snn.models.spike_dqn import SpikeDQN
from doom_snn.quantization.qat import insert_fake_quant


def test_rsnn_gradients_flow():
    model = RSNNActorCritic(input_channels=768, core_neurons=96, num_actions=6, T=8)
    x = torch.rand((2, 768))

    out = model(x)
    loss = out["action_logits"].sum() + out["value"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} had non-finite gradients"


def test_ff_snn_gradients_flow():
    model = FFSNNActorCritic(input_channels=768, hidden_sizes=96, num_actions=6, T=8)
    x = torch.rand((2, 768))

    out = model(x)
    loss = out["action_logits"].sum() + out["value"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} missing gradients"
            assert torch.all(torch.isfinite(param.grad))


def test_spike_dqn_gradients_flow():
    model = SpikeDQN(input_channels=768, hidden_size=96, num_actions=6, T=8)
    x = torch.rand((2, 768))

    out = model(x)
    loss = out["q_values"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))


def test_qat_gradients_flow():
    model = RSNNActorCritic(input_channels=768, core_neurons=96, num_actions=6, T=8)
    model = insert_fake_quant(model, {"weight_bits": 16})

    x = torch.rand((2, 768))
    out = model(x)
    loss = out["action_logits"].sum() + out["value"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} missing grad in QAT"
            assert torch.all(torch.isfinite(param.grad))
