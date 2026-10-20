"""Unit tests for INT16 fixed-point simulation, bitshifts, and QAT straight-through estimators."""

import pytest
import torch
from doom_snn.quantization.fake_quant import FakeQuantInt16
from doom_snn.quantization.fixed_point import (
    INT16_MAX,
    INT16_MIN,
    beta_shift_int16,
    compute_scale,
    count_saturations,
    float_to_int16,
    int16_to_float,
)
from doom_snn.quantization.qat import QuantizedLIFDynamics, QuantizedLinear


def test_int16_clamp():
    scale = 1.0
    large_vals = torch.tensor([-50000.0, 50000.0])
    q = float_to_int16(large_vals, scale)

    assert q[0].item() == INT16_MIN
    assert q[1].item() == INT16_MAX


def test_fake_quant_roundtrip():
    x = torch.linspace(-1.0, 1.0, 100)
    scale = compute_scale(x, bits=16)

    q = float_to_int16(x, scale)
    x_hat = int16_to_float(q, scale)

    max_err = torch.max(torch.abs(x - x_hat)).item()
    assert max_err < (scale + 1e-6), f"Roundtrip error {max_err} exceeds step size {scale}"


def test_saturation_detection():
    x = torch.tensor([10.0, 40000.0, -40000.0])
    scale = 1.0
    sats = count_saturations(x, scale)
    assert sats == 2


def test_beta_shift_025():
    u = torch.tensor([100, 200, 4])
    decayed = beta_shift_int16(u, beta=0.25)
    # 100 >> 2 = 25, 200 >> 2 = 50, 4 >> 2 = 1
    assert torch.equal(decayed, torch.tensor([25, 50, 1], dtype=torch.int16))


def test_beta_shift_050():
    u = torch.tensor([100, 200, 5])
    decayed = beta_shift_int16(u, beta=0.50)
    # 100 >> 1 = 50, 200 >> 1 = 100, 5 >> 1 = 2
    assert torch.equal(decayed, torch.tensor([50, 100, 2], dtype=torch.int16))


def test_beta_shift_075():
    u = torch.tensor([100, 200, 8])
    decayed = beta_shift_int16(u, beta=0.75)
    # 100 - (100 >> 2) = 100 - 25 = 75
    # 200 - (200 >> 2) = 200 - 50 = 150
    # 8 - 2 = 6
    assert torch.equal(decayed, torch.tensor([75, 150, 6], dtype=torch.int16))


def test_beta_shift_negative_values():
    u = torch.tensor([-100])
    decayed = beta_shift_int16(u, beta=0.5)
    # -100 // 2 = -50
    assert decayed.item() == -50


def test_qat_backward_finite():
    fq = FakeQuantInt16()
    x = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
    x_hat = fq(x)
    loss = (x_hat ** 2).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_quantized_lif_dynamics():
    qlif = QuantizedLIFDynamics(beta=0.75, threshold=0.4, membrane_scale=1e-3, current_scale=1e-3)
    curr = torch.tensor([[0.5, 0.1]])
    spikes, membrane, state = qlif(curr)

    assert spikes.shape == (1, 2)
    assert membrane.shape == (1, 2)
    assert spikes[0, 0].item() == 1.0  # 0.5 >= 0.4 fires
    assert spikes[0, 1].item() == 0.0  # 0.1 < 0.4 doesn't fire
