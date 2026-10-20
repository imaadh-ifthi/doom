# 🎮 projectDOOM: Autonomous ViZDoom Gameplay via Hardware-Aware Spiking Neural Networks & CNNs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware Constraints](https://img.shields.io/badge/SNAP--V-Compliant-brightgreen.svg)](docs/HARDWARE_CONSTRAINTS.md)
[![Tests Passing](https://img.shields.io/badge/tests-44%2F44%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, reproducible, hardware-aware machine learning pipeline for autonomous gameplay in **ViZDoom BASIC**. This repository hosts two complementary tracks:

1. **Neuromorphic SNN Track (`src/doom_snn/`)**: A hardware-constrained Spiking Neural Network (SNN) pipeline designed for FPGA / neuromorphic hardware deployment (PeraMorphIQ / SNAP-V architecture). Supports Recurrent SNNs (RSNN), Quantization-Aware Training (QAT), 16-bit fixed-point simulation, surrogate gradient BPTT, and comprehensive hardware resource accounting.
2. **Convolutional ANN Track (`CNN-build/`)**: Sujeevan's baseline convolutional neural network system featuring imitation learning from privileged teacher agents, DQN training, and brightness robustness evaluations.

---

## 📑 Table of Contents

- [1. Repository Structure](#1-repository-structure)
- [2. Neuromorphic Hardware Constraints (SNAP-V)](#2-neuromorphic-hardware-constraints-snap-v)
- [3. SNN Training Tracks (A, B, C, D)](#3-snn-training-tracks-a-b-c-d)
- [4. Benchmark Protocol & Baselines](#4-benchmark-protocol--baselines)
- [5. Hardware Resource Accounting](#5-hardware-resource-accounting)
- [6. Quickstart Guide](#6-quickstart-guide)
  - [Installation](#installation)
  - [Running Unit & Smoke Tests](#running-unit--smoke-tests)
  - [Training Models](#training-models)
  - [Evaluation & Baselines](#evaluation--baselines)
  - [Cross-Track Benchmarking](#cross-track-benchmarking)
  - [Hardware Resource Reporting](#hardware-resource-reporting)
- [7. Documentation Map](#7-documentation-map)
- [8. Heritage & Attribution](#8-heritage--attribution)

---

## 1. Repository Structure

```text
projectDOOM/
├── src/                        # Core doom_snn neuromorphic package
│   └── doom_snn/
│       ├── configs.py          # Typed dataclass config system (YAML + CLI)
│       ├── device.py           # Device resolver (auto CPU / CUDA detection)
│       ├── seeding.py          # Multi-library deterministic seed manager
│       ├── logging_utils.py    # Structured console and file logger
│       ├── encoding/           # Rate coding & deterministic accumulator
│       ├── env/                # ViZDoom wrapper & offline DummyEnv
│       ├── evaluation/         # Metrics, baselines, and robustness suites
│       ├── hardware/           # SNAP-V constraints & resource accounting
│       ├── models/             # RSNN, FF-SNN, Spike-DQN, ANN-to-SNN
│       ├── neurons/            # LIF & Recurrent LIF neurons, surrogate gradients
│       ├── quantization/       # 16-bit fixed-point math, STE, and QAT
│       ├── training/           # A2C+GAE and Spike-DQN trainers
│       └── utils/              # Checkpointing, CSV logging, plotting
│
├── configs/                    # Experiment profiles and training configs
│   ├── base.yaml               # Shared configuration template
│   ├── laptop_cpu.yaml         # Lightweight profile for CPU development
│   ├── server_gpu.yaml         # High-throughput GPU profile
│   ├── track_c_*.yaml          # Primary RSNN Actor-Critic configs (beta=0.5, 0.75, QAT)
│   ├── track_d_ff_snn.yaml     # Feed-forward SNN ablation config
│   ├── track_b_spike_dqn_ff.yaml# Feed-forward Spike-DQN config
│   ├── track_a_ann_to_snn.yaml # ANN baseline & conversion config
│   └── benchmark.yaml          # Multi-model benchmarking configuration
│
├── scripts/                    # Command-line entry points
│   ├── train.py                # Main training script (all tracks, QAT, smoke tests)
│   ├── eval.py                 # Multi-seed evaluation with baselines & robustness
│   ├── benchmark.py            # Cross-model benchmark comparison runner
│   ├── resource_report.py      # SNAP-V hardware resource accounting tool
│   ├── convert_ann_to_snn.py   # ANN-to-SNN weight transfer CLI
│   ├── run_smoke_tests.bat/.sh # Automated verification scripts
│   ├── run_laptop_quick.bat/.sh# Quick training launcher for laptops
│   └── run_server_full.bat/.sh # Multi-seed training pipeline for servers
│
├── tests/                      # Pytest suite (44 unit & integration tests)
├── docs/                       # Technical specifications and architectural docs
├── CNN-build/                  # Untouched CNN imitation learning & DQN codebase
└── SNN-build/                  # Git submodule reference to original RSNN lineage
```

---

## 2. Neuromorphic Hardware Constraints (SNAP-V)

The SNN pipeline is engineered to adhere strictly to the **PeraMorphIQ / SNAP-V** neuromorphic accelerator specifications defined by the hardware team:

| Constraint | Specification | Pipeline Implementation |
|---|---|---|
| **Numerical Precision** | 16-bit signed fixed point (`int16`, $[-32768, 32767]$) | `FakeQuantInt16` with Straight-Through Estimator (STE); int32 intermediate accumulation with saturation clamping. |
| **Membrane Decay ($\beta$)** | Strictly $\beta \in \{0.25, 0.5, 0.75\}$ | Exact bitshift simulation: $\beta=0.25 \rightarrow (\mathbf{u} \gg 2)$, $\beta=0.5 \rightarrow (\mathbf{u} \gg 1)$, $\beta=0.75 \rightarrow \mathbf{u} - (\mathbf{u} \gg 2)$. |
| **Spike Resolution ($T$)** | $T = 8$ internal spiking timesteps / decision | Fixed temporal dimension $T=8$ per environment decision. |
| **Action Repeat** | 4 game tics per agent decision | Frame skip of 4 in `VizDoomEnv` (consistent with literature). |
| **Neuron Budget** | Core hidden neurons $< 1024$ (operating point $\le 964$) | Default operating point: 96 recurrent LIF hidden neurons. Conservative fallback: $768\text{ input} + 96\text{ core} = 864$ total neurons ($< 1024$). |
| **Neuron Reset** | Soft reset default ($u \leftarrow u - v_{th}$) | Soft reset preserves graded sub-threshold energy; hard reset ($u \leftarrow 0$) supported via config. |
| **Readout Mode** | Continuous membrane-potential readout ($u_T$) | Avoids rate-coding discretization bottleneck at decision heads. |

For detailed documentation, see [`docs/HARDWARE_CONSTRAINTS.md`](docs/HARDWARE_CONSTRAINTS.md) and [`docs/FIXED_POINT_QAT.md`](docs/FIXED_POINT_QAT.md).

---

## 3. SNN Training Tracks (A, B, C, D)

To systematically isolate neuromorphic advantages and hardware trade-offs, the repository defines four distinct training tracks:

```text
               ┌────────────────────────────────────────────────────────┐
               │              ViZDoom Game Observations                │
               │            (32x24 Grayscale = 768 Inputs)              │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │  Deterministic Spike Accumulator (T = 8)   │
                   └──────────────────────┬──────────────────────┘
                                          │
       ┌──────────────────┬───────────────┴───────────────┬──────────────────┐
       │                  │                               │                  │
┌──────▼───────┐   ┌──────▼───────┐                ┌──────▼───────┐   ┌──────▼───────┐
│   Track C    │   │   Track D    │                │   Track B    │   │   Track A    │
│  Recurrent   │   │ Feed-Forward │                │  Spike-DQN   │   │  ANN-to-SNN  │
│ Actor-Critic │   │ Actor-Critic │                │ Feed-Forward │   │  Conversion  │
│  (PRIMARY)   │   │  (Ablation)  │                │(Alternative) │   │ (Reference)  │
│  96 Rec-LIF  │   │  96 FF-LIF   │                │  96 FF-LIF   │   │  Pretrained  │
│   A2C+GAE    │   │   A2C+GAE    │                │ Q-Learning   │   │  MLP Weights │
└──────┬───────┘   └──────┬───────┘                └──────┬───────┘   └──────┬───────┘
       │                  │                               │                  │
       └──────────────────┴───────────────┬───────────────┴──────────────────┘
                                          │
                   ┌──────────────────────▼──────────────────────┐
                   │    Continuous Membrane Readout (u_T)        │
                   │       16-bit Fixed-Point / QAT Mode         │
                   └──────────────────────┬──────────────────────┘
                                          │
                   ┌──────────────────────▼──────────────────────┐
                   │       Compound Action Output (0 to 5)       │
                   └─────────────────────────────────────────────┘
```

* **Track C — Recurrent Actor-Critic RSNN (Primary Focus)**
  * Evolved from Dakshayani Ramanesh's RSNN v2 lineage (reported 81.3% kill rate).
  * 96 recurrent LIF hidden neurons ($96 \times 96$ recurrent synaptic matrix).
  * Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE).
  * Differentiable surrogate gradient: Fast-sigmoid derivative $\frac{1}{(1 + 2|u - v_{th}|)^2}$.
  * Available in standard float32 and 16-bit Quantization-Aware Training (QAT).
* **Track D — Feed-Forward SNN Actor-Critic (Ablation)**
  * Identical to Track C but with recurrent connections disabled ($W_{\text{rec}} = 0$).
  * Serves as the scientific control for quantifying the temporal memory contribution of recurrent spiking dynamics.
* **Track B — Hardware-Native Spike-DQN (Alternative Paradigm)**
  * Feed-forward SNN predicting $Q$-values directly with experience replay and target networks.
  * Includes an optional experimental recurrent Spike-DQN model.
* **Track A — ANN Baseline & ANN-to-SNN Conversion (Historical Reference)**
  * Standard MLP (768 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 6) trained with ReLU.
  * Converted to spiking dynamics at $T=8$ via threshold balancing and weight rescaling.

---

## 4. Benchmark Protocol & Baselines

To eliminate evaluation variance and prevent misleading reporting, all models are evaluated against an assumed, standardized protocol:

* **Evaluation Seeds:** Exactly 3 canonical seeds: `[6201, 6202, 6203]`.
* **Sample Size:** 100 episodes per seed (300 episodes total).
* **Policy Mode:** Strictly greedy (deterministic argmax over logits or $Q$-values).
* **Primary Metrics:**
  * **Kill Success Rate (%)**: Reported with Wilson score 95% confidence intervals.
  * **Mean & Median Episode Reward**: Net game score.
  * **Mean Decision Steps**: Number of decisions taken before episode termination.
  * **Mean Spiking Activity**: Average spikes per neuron per decision step.
  * **Inference Latency (ms)**: Per-decision wall-clock execution time.
* **Reference Baselines:**
  * `RandomAgent`: Uniform random action selection ($a \in \{0, \dots, 5\}$).
  * `AlwaysShootAgent`: Static weapon firing ($a = 3$, attack).
  * `LeftShootAgent`: Strafe left while shooting ($a = 4$, left + attack).

---

## 5. Hardware Resource Accounting

Hardware resource demands under Imaadh's SNAP-V specifications are computed dynamically for each model:

| Parameter | Track C (RSNN AC) | Track D (FF-SNN AC) | Track B (Spike-DQN) | Track A (Converted) |
|---|---|---|---|---|
| **Core LIF Neurons** | **96** | **96** | **96** | 192 (128+64) |
| **Total Neurons (w/ Encoder)** | **864** | **864** | **864** | 960 |
| **Neuron Budget (< 1024)** | **PASSED** | **PASSED** | **PASSED** | **PASSED** |
| Input Weights ($768 \times N$) | 73,728 | 73,728 | 73,728 | 98,304 |
| Recurrent Weights ($N \times N$) | 9,216 | 0 | 0 | 0 |
| Readout Weights | 672 | 672 | 576 | 1,222 |
| **Total Synaptic Weights** | **83,616** | **74,400** | **74,304** | **107,718** |
| **16-bit Weight Memory** | **163.31 KB** | **145.31 KB** | **145.13 KB** | **210.39 KB** |
| Dense MACs / Decision ($T=8$) | 669,792 | 596,064 | 595,296 | 864,832 |

---

## 6. Quickstart Guide

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/imaadh-ifthi/doom.git
cd doom
pip install -r requirements-doom-snn.txt
```

*(Note: ViZDoom requires system packages on Linux like `cmake`, `libboost-all-dev`, `libsdl2-dev`. On Windows, prebuilt wheels are installed automatically via pip.)*

### Running Unit & Smoke Tests

Verify that your local environment passes all 44 unit and integration tests:

```bash
# Run pytest test suite directly
pytest tests/ -v

# Windows automated smoke test
scripts\run_smoke_tests.bat

# Linux / Server automated smoke test
bash scripts/run_smoke_tests.sh
```

### Training Models

Train the primary Track C RSNN agent:

```bash
# Train Track C (RSNN, beta=0.75, Float32)
python scripts/train.py --config configs/track_c_rsnn_ac_beta075_float.yaml --device auto --episodes 500

# Train Track C with 16-bit Quantization-Aware Training (QAT)
python scripts/train.py --config configs/track_c_rsnn_ac_beta075_int16_qat.yaml --device auto --episodes 500

# Quick smoke training run (2 episodes using synthetic DummyEnv)
python scripts/train.py --config configs/base.yaml --smoke --dummy-env --device cpu
```

### Evaluation & Baselines

Evaluate a trained checkpoint across canonical seeds with the baseline comparison:

```bash
# Evaluate checkpoint
python scripts/eval.py --checkpoint runs/track_c_rsnn_ac_beta075_float/best.pt --episodes 100 --seeds 6201 6202 6203

# Run all reference baselines (Random, AlwaysShoot, LeftShoot)
python scripts/eval.py --baselines --episodes 100

# Evaluate with robustness stress tests (brightness, noise, pixel shuffle)
python scripts/eval.py --checkpoint runs/track_c_rsnn_ac_beta075_float/best.pt --robustness
```

### Cross-Track Benchmarking

Run an automated benchmark comparing all model tracks and baselines under the exact same protocol:

```bash
python scripts/benchmark.py --config configs/benchmark.yaml --output-dir benchmark_output
```

This outputs:
* `benchmark_output/benchmark_results.csv`: Raw episode data.
* `benchmark_output/benchmark_results.json`: Aggregated metrics with confidence intervals.
* `benchmark_output/benchmark_summary.md`: Publication-ready comparison table.

### Hardware Resource Reporting

Inspect the neuromorphic resource consumption for any architecture configuration:

```bash
python scripts/resource_report.py --neurons 96 --track c_rsnn_ac --output-dir hardware_reports
```

---

## 7. Documentation Map

Detailed technical documentation is available in the [`docs/`](docs/) directory:

* [`docs/HARDWARE_CONSTRAINTS.md`](docs/HARDWARE_CONSTRAINTS.md): Comprehensive SNAP-V hardware specification from Imaadh.
* [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md): Standardized multi-seed benchmark methodology and metrics.
* [`docs/TRACKS.md`](docs/TRACKS.md): Detailed architectural description of Tracks A, B, C, and D.
* [`docs/FIXED_POINT_QAT.md`](docs/FIXED_POINT_QAT.md): 16-bit fixed-point format, saturation handling, bitshift decay, and QAT math.
* [`docs/RESOURCE_ACCOUNTING.md`](docs/RESOURCE_ACCOUNTING.md): Theoretical derivations for neuron, synapse, memory, and MAC calculations.
* [`docs/RUNNING_ON_LAPTOP_AND_SERVER.md`](docs/RUNNING_ON_LAPTOP_AND_SERVER.md): Practical guide for training on CPU laptops vs. GPU servers.
* [`docs/REPO_FINDINGS_AND_CHANGES.md`](docs/REPO_FINDINGS_AND_CHANGES.md): Full engineering change log and repository audit.
* [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Transparent technical limitations and future roadmap.

---

## 8. Heritage & Attribution

This repository integrates and builds upon foundational research by the team:

* **RSNN Neuromorphic Foundation**: Evolved from Dakshayani Ramanesh's RSNN v2 architecture ([`SNN-build/`](SNN-build/)), upgrading it into a continuous-readout, 16-bit fixed-point, hardware-compliant system.
* **CNN Baseline & Methodology**: Preserves Sujeevan's imitation learning and DQN exploration suite ([`CNN-build/`](CNN-build/)).
* **Hardware Specifications**: Designed to satisfy the PeraMorphIQ / SNAP-V neuromorphic constraints specified by Imaadh.
