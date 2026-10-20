"""Unit tests for discrete-time LIF neurons and surrogate gradients."""

import pytest
import warnings
import torch
from doom_snn.neurons.lif import LIFLayer, RecurrentLIFLayer
from doom_snn.neurons.states import LIFState
from doom_snn.neurons.surrogate import fast_sigmoid_spike


def test_output_shape():
    layer = LIFLayer(input_size=10, hidden_size=20, beta=0.75, threshold=0.4)
    x = torch.rand((4, 10))
    spikes, membrane, state = layer(x)

    assert spikes.shape == (4, 20)
    assert membrane.shape == (4, 20)
    assert isinstance(state, LIFState)
    assert state.membrane.shape == (4, 20)


def test_spike_threshold():
    # Large input to force membrane way above threshold 0.4
    layer = LIFLayer(input_size=1, hidden_size=1, beta=0.5, threshold=0.4)
    with torch.no_grad():
        layer.fc.weight.fill_(5.0)
        layer.fc.bias.fill_(0.0)

    x = torch.tensor([[1.0]])
    spikes, membrane, _ = layer(x)
    assert spikes.item() == 1.0


def test_soft_reset():
    layer = LIFLayer(input_size=1, hidden_size=1, beta=0.0, threshold=0.4, reset_mode="soft")
    with torch.no_grad():
        layer.fc.weight.fill_(1.0)
        layer.fc.bias.fill_(0.0)

    x = torch.tensor([[1.0]])
    # current = 1.0, membrane before reset = 1.0 >= 0.4 -> spike = 1.0
    # soft reset: membrane = 1.0 - 0.4 = 0.6
    spikes, membrane, _ = layer(x)
    assert spikes.item() == 1.0
    assert abs(membrane.item() - 0.6) < 1e-5


def test_hard_reset():
    layer = LIFLayer(input_size=1, hidden_size=1, beta=0.0, threshold=0.4, reset_mode="hard")
    with torch.no_grad():
        layer.fc.weight.fill_(1.0)
        layer.fc.bias.fill_(0.0)

    x = torch.tensor([[1.0]])
    spikes, membrane, _ = layer(x)
    assert spikes.item() == 1.0
    assert abs(membrane.item() - 0.0) < 1e-5


def test_valid_betas():
    for beta in (0.25, 0.5, 0.75):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            _ = LIFLayer(input_size=10, hidden_size=10, beta=beta)
            assert len(recorded) == 0, f"Expected no warning for compliant beta={beta}"


def test_invalid_beta_warning():
    with pytest.warns(UserWarning, match="recommended hardware-friendly set"):
        _ = LIFLayer(input_size=10, hidden_size=10, beta=0.875)


def test_surrogate_gradient():
    u = torch.tensor([0.35, 0.45], requires_grad=True)
    spikes = fast_sigmoid_spike(u, threshold=0.4)
    loss = spikes.sum()
    loss.backward()

    assert u.grad is not None
    assert torch.all(torch.isfinite(u.grad))
    assert torch.all(u.grad > 0.0)


def test_recurrent_lif():
    rlayer = RecurrentLIFLayer(input_size=8, hidden_size=16, beta=0.5, threshold=0.4)
    x = torch.rand((2, 8))

    state = rlayer.init_state(batch_size=2)
    s1, u1, state1 = rlayer(x, state)
    s2, u2, state2 = rlayer(x, state1)

    assert s1.shape == (2, 16)
    assert s2.shape == (2, 16)
    assert state2.spikes.shape == (2, 16)
