# Experimental SNN Tracks & Architectural Priorities

## Strategic Overview
The project structures model development into four prioritized experimental tracks. The recurrent actor-critic RSNN (Track C) represents the primary candidate lineage based on established empirical baselines from Dakshayani's prior work.

---

## Track C: Recurrent Actor-Critic RSNN (Primary Highest-Value Track)
- **Role:** Main deliverable and primary candidate for FPGA hardware mapping.
- **Lineage:** Evolved from Dakshayani Ramanesh's RSNN v2.
- **Architecture:**
  - Input: 768 grayscale channels.
  - Encoder: Deterministic accumulator rate coding over $T=8$ steps.
  - Core: 96 Recurrent LIF neurons ($96 \times 96$ recurrent synaptic matrix).
  - Heads: Linear Actor (96 -> 6), Linear Critic (96 -> 1).
  - Readout: Continuous membrane potential $u_T$.
  - Surrogate Gradient: Fast-sigmoid derivative:
    $$\frac{\partial s}{\partial u} \approx \frac{1}{(1 + 2|u - V_{th}|)^2}$$
- **Experiments:**
  - `C1`: $\beta = 0.5$, FP32.
  - `C2`: $\beta = 0.75$, FP32.
  - `C3`: $\beta = 0.75$, Post-Training Quantization (PTQ) to INT16.
  - `C4`: $\beta = 0.75$, Quantization-Aware Training (QAT) in INT16.
  - `C5`: $\beta = 0.5$, INT16 QAT.
  - `C6`: Readout ablation (Spike-count vs. Membrane readout).
  - `C7`: Recurrent ablation (lateral weights zeroed).

---

## Track D: Feed-Forward SNN Ablation (High Scientific Priority)
- **Role:** Controlled ablation quantifying the exact performance and memory value of recurrent synaptic feedback.
- **Architecture:** 768 inputs -> 96 Feed-Forward LIF neurons -> Membrane readout -> Actor/Critic heads.
- **Key Difference:** $W_{\text{rec}} = 0$. Hidden state does not persist across decisions.
- **Hardware Advantage:** Saves 9,216 weights ($\sim 18.4$ KB) and eliminates recurrent state memory buffering.

---

## Track B: Hardware-Native Spike-DQN (Medium Priority)
- **Role:** Direct value-based reinforcement learning alternative.
- **Architecture:**
  - Feed-Forward Spike-DQN: 768 -> 96 LIF -> $Q(s, a)$ head (6 outputs).
  - Loss: Huber Temporal Difference (TD) loss.
  - Replay buffer: 50,000 transitions.
  - Inference advantage: Simple $\text{argmax}_a Q(s, a)$ on FPGA without sampling.
  - Risk: Temporal difference learning with surrogate gradients can exhibit instability compared to on-policy policy gradients.

---

## Track A: ANN-to-SNN Conversion (Medium Priority)
- **Role:** Safe initialization and conversion-loss measurement.
- **Architecture:** Train a compact MLP under the exact SNN benchmark -> Copy weights to LIF layers -> Evaluate at $T=8$ with accumulator rate encoding -> Fine-tune with surrogate gradients.
