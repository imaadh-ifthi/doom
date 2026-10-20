"""Unit tests for deterministic accumulator and Poisson spike encoders."""

import pytest
import torch
from doom_snn.encoding.spike_encoder import SpikeEncoder


def test_output_shape():
    batch = 4
    channels = 768
    T = 8
    encoder = SpikeEncoder(input_channels=channels, T=T, method="accumulator")

    x = torch.rand((batch, channels))
    spikes = encoder(x)

    assert spikes.shape == (T, batch, channels)


def test_binary_outputs():
    encoder = SpikeEncoder(input_channels=768, T=8, method="accumulator")
    x = torch.rand((2, 768))
    spikes = encoder(x)

    unique_vals = torch.unique(spikes).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0), f"Spikes contained non-binary value: {v}"


def test_deterministic():
    encoder1 = SpikeEncoder(input_channels=768, T=8, method="accumulator")
    encoder2 = SpikeEncoder(input_channels=768, T=8, method="accumulator")

    x = torch.rand((2, 768))
    s1 = encoder1(x)
    s2 = encoder2(x)

    assert torch.equal(s1, s2), "Deterministic accumulator encoder produced non-identical outputs"


def test_brighter_more_spikes():
    encoder = SpikeEncoder(input_channels=768, T=8, method="accumulator")

    dark_img = torch.full((1, 768), 0.1)
    bright_img = torch.full((1, 768), 0.9)

    encoder.reset_state()
    dark_spikes = encoder(dark_img)
    encoder.reset_state()
    bright_spikes = encoder(bright_img)

    assert bright_spikes.sum() > dark_spikes.sum(), (
        f"Brighter input should produce strictly more spikes: {bright_spikes.sum()} vs {dark_spikes.sum()}"
    )


def test_accumulator_reset():
    encoder = SpikeEncoder(input_channels=768, T=8, method="accumulator")
    x = torch.full((1, 768), 0.35)

    _ = encoder(x)
    assert encoder.accumulator is not None
    assert encoder.accumulator.sum() > 0

    encoder.reset_state()
    assert encoder.accumulator is not None
    assert torch.equal(encoder.accumulator, torch.zeros_like(encoder.accumulator))


def test_batch_execution():
    encoder = SpikeEncoder(input_channels=768, T=8, method="accumulator")
    x = torch.rand((8, 768))
    spikes = encoder(x)
    assert spikes.shape == (8, 8, 768)
    assert float(encoder.get_spike_stats()) >= 0.0
