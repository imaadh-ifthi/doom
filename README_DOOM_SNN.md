# Hardware-Aware SNN Pipeline for ViZDoom Autonomous Gameplay

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware Constraints](https://img.shields.io/badge/SNAP--V-Compliant-brightgreen.svg)](docs/HARDWARE_CONSTRAINTS.md)

Modular, reproducible, hardware-aware Spiking Neural Network (SNN) pipeline in PyTorch for training and evaluating neuromorphic game-playing agents under PeraMorphIQ / SNAP-V hardware accelerator constraints.

---

## 1. Project Purpose
The pipeline enables training, quantizing, evaluating, and benchmarking spiking controllers capable of autonomous gameplay in ViZDoom BASIC. Models are constrained to strict hardware rules for eventual mapping to FPGA/neuromorphic hardware.

---

## 2. Official Hardware Constraints
- **Numerical Precision:** 16-bit signed fixed-point (`int16`, $[-32768, 32767]$).
- **Membrane Decay:** $\beta \in \{0.25, 0.5, 0.75\}$ (bitshift operations only: `u>>2`, `u>>1`, `u-(u>>2)`).
- **Temporal Resolution:** $T = 8$ internal spiking timesteps per game decision.
- **Action Repeat:** 4 ViZDoom game tics per decision.
- **Neuron Budget:** Core hidden neurons $< 1024$ (operating point $\le 964$).
- **Readout:** Continuous membrane-potential readout ($u_T$) for stable policy learning.

---

## 3. Experimental Tracks

- **Track C (Primary):** Recurrent Actor-Critic RSNN (96 LIF neurons, A2C+GAE, surrogate gradients). Evolved from Dakshayani Ramanesh's RSNN baseline lineage.
- **Track D (Scientific Control):** Feed-Forward SNN Actor-Critic ablation ($W_{\text{rec}} = 0$).
- **Track B (Alternative):** Hardware-native Spike-DQN with experience replay.
- **Track A (Reference):** ANN baseline and ANN-to-SNN conversion at $T=8$.

---

## 4. Quickstart

### Installation
```bash
pip install -r requirements-doom-snn.txt
```

### Run Tests
```bash
# Windows
scripts\run_smoke_tests.bat

# Linux / Server
bash scripts/run_smoke_tests.sh
```

### Train Track C RSNN
```bash
python scripts/train.py --config configs/track_c_rsnn_ac_beta075_float.yaml --device auto --episodes 500
```

### Evaluate Model
```bash
python scripts/eval.py --checkpoint runs/track_c_rsnn_ac_beta075_float/best.pt --episodes 100 --seeds 6201 6202 6203 --robustness
```

### Hardware Resource Report
```bash
python scripts/resource_report.py --neurons 96 --track c_rsnn_ac
```

---

## 5. Directory Structure

```text
projectDOOM/
├── CNN-build/              # Sujeevan's CNN baseline (reference, untouched)
├── SNN-build/              # SNN submodule reference
├── configs/                # YAML configuration files for all tracks
├── docs/                   # Engineering specifications and constraints
├── scripts/                # Training, evaluation, and benchmark CLIs
├── src/                    # Core doom_snn python package
│   └── doom_snn/
│       ├── configs.py
│       ├── device.py
│       ├── seeding.py
│       ├── encoding/       # Deterministic accumulator rate coding
│       ├── env/            # ViZDoom and DummyEnv wrappers
│       ├── evaluation/     # Metrics, baselines, robustness
│       ├── hardware/       # SNAP-V constraints & resource accounting
│       ├── models/         # RSNN, FF-SNN, Spike-DQN, ANN-to-SNN
│       ├── neurons/        # Discrete LIF, RecurrentLIF, surrogate
│       ├── quantization/   # Fixed-point simulation & QAT
│       └── training/       # A2C+GAE and DQN trainers
├── tests/                  # Pytest test suite (44 unit & integration tests)
└── README_DOOM_SNN.md
```
