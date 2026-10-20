# Agent Implementation Plan: Hardware-Aware SNN Pipeline for ViZDoom

## 1. Executive Summary
This document provides the formal software systems engineering plan for implementing, verifying, and benchmarking a hardware-aware Spiking Neural Network (SNN) pipeline targeting autonomous gameplay in ViZDoom BASIC under PeraMorphIQ / SNAP-V neuromorphic processor constraints.

## 2. Implementation Phases & Architecture Mapping

### Phase 0: Repository Inspection & Context Preservation
- **Existing CNN:** Preserved untouched in `CNN-build/` as an autonomous vision-control reference.
- **Existing RSNN Submodule:** `SNN-build/` analyzed; pinned commit lacked updated RSNN v2 source. Upgraded modular implementation developed directly in `src/doom_snn/` with full provenance tracking.
- **Documentation:** Updated roadmap and architectural specifications maintained in `docs/`.

### Phase 1: Core Modular Infrastructure
- `src/doom_snn/configs.py`: Strongly typed dataclass hierarchy covering Observation, Action, Neuron, Model, Training, DQN, Evaluation, Quantization, and Environment settings with YAML and CLI overrides.
- `src/doom_snn/device.py`: Cross-platform compute device selector with environment variable support (`DOOM_SNN_DEVICE`).
- `src/doom_snn/seeding.py`: Multi-library deterministic seed synchronizer (Python, NumPy, PyTorch CPU/CUDA).
- `src/doom_snn/env/`: ViZDoom wrapper with auto-discovery and headless mode, complemented by synthetic `DummyEnv` for headless offline CI verification.

### Phase 2: Neuromorphic Encoding, Neurons, & Surrogate Gradients
- `src/doom_snn/encoding/spike_encoder.py`: Deterministic accumulator rate coding across $T=8$ steps.
- `src/doom_snn/neurons/lif.py`: `LIFLayer` and `RecurrentLIFLayer` with hardware-compliant decay $\beta \in \{0.25, 0.5, 0.75\}$ and configurable soft/hard resets.
- `src/doom_snn/neurons/surrogate.py`: Fast-sigmoid surrogate derivative function:
  $$\frac{\partial s}{\partial u} \approx \frac{1}{(1 + 2|u - V_{th}|)^2}$$

### Phase 3: 16-Bit Fixed-Point Simulation & Quantization-Aware Training (QAT)
- `src/doom_snn/quantization/fixed_point.py`: Signed 16-bit integer representation `[-32768, 32767]`, dynamic scale estimation, saturation logging, and bitshift-based $\beta$ decay.
- `src/doom_snn/quantization/fake_quant.py`: Straight-Through Estimator (STE) fake-quantization module.
- `src/doom_snn/quantization/qat.py`: Quantized linear layers and graph insertion utility.

### Phase 4: Model Suite (Tracks A, B, C, D)
- **Track C (Primary):** `RSNNActorCritic` — 96 recurrent LIF hidden neurons, accumulator rate coding, continuous membrane potential readout $u_T$, A2C+GAE learning.
- **Track D (Scientific Control):** `FFSNNActorCritic` — Feed-forward LIF core without lateral recurrence.
- **Track B (Value-Based):** `SpikeDQN` and optional `RecurrentSpikeDQN`.
- **Track A (Conversion):** `ANNBaseline` (768 -> 128 -> 64 -> 6) and `ConvertedSNN`.

### Phase 5: Hardware Resource Accounting & SNAP-V Verification
- `src/doom_snn/hardware/snapv_constraints.py`: Hard boundary checking ($<1024$ neurons, $\beta \in \{0.25, 0.5, 0.75\}$, $T=8$).
- `src/doom_snn/hardware/resource_accounting.py`: Synaptic count, INT16 memory footprint, state memory, and MAC operations.

### Phase 6: Assumed Effective Benchmark Protocol & Reporting
- `scripts/benchmark.py`: Cross-model evaluation under standardized conditions (32×24 grayscale, 6 actions, frame skip 4, seeds 6201–6203).
- Full suite of shell scripts and unit tests in `tests/`.
