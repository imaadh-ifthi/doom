# Assumed Effective Benchmark Protocol for ViZDoom SNN Control

## 1. Scope & Objective Statement
> "We target autonomous gameplay in the ViZDoom BASIC scenario, not full general DOOM. The BASIC scenario requires horizontal movement, aiming, and shooting, and is suitable for evaluating compact hardware-aware SNN controllers."

This document establishes the standardized benchmark protocol uniting all SNN experimental tracks (A, B, C, D) and reference baselines.

---

## 2. Standardized Evaluation Parameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Scenario** | `basic.cfg` (`basic.wad`, Map01) | Canonical single-enemy aiming and shooting benchmark. |
| **Input Format** | Grayscale $32 \times 24$ (768 normalized floats) | Minimizes dense input synapse footprint while retaining spatial monster location. |
| **Discrete Actions** | 6 compound actions | Standard discrete space: `noop`, `left`, `right`, `shoot`, `left_shoot`, `right_shoot`. |
| **Frame Repeat** | 4 game tics per decision | Matches $\sim 114$ ms decision cycle; standard in ViZDoom RL literature. |
| **Internal Timesteps** | $T = 8$ spiking steps | Hardware temporal budget per decision. |
| **Evaluation Seeds** | `[6201, 6202, 6203]` | Multi-seed verification to establish confidence intervals. |
| **Evaluation Budget** | 100 episodes per seed (300 total) | Statistically sound sample size; 20 eps per seed for laptop quick checks. |
| **Policy Mode** | Deterministic / Greedy ($\epsilon = 0.0$) | Evaluates true learned policy without exploration noise. |

---

## 3. Metric Definitions

1. **Kill Success Rate (%):** Percentage of episodes resulting in an enemy kill before agent timeout (maximum 50 decisions / 200 tics) or death. Accompanied by 95% Wilson score confidence intervals.
2. **Mean & Median Reward:** Episodic cumulative reward:
   $$R = \sum_{t=0}^{H} r_t$$
   Living penalty: $-1$ per decision; hit/kill bonus: $+100$.
3. **Decisions per Episode:** Average number of forward policy evaluations until episode termination.
4. **Action Frequency Distribution (%):** Percentage breakdown across the 6 actions to detect degenerative policies (e.g. infinite shooting without aiming).
5. **Mean Spike Rate:** Average firing probability across all hidden LIF neurons over $T=8$ steps:
   $$\bar{s} = \frac{1}{T \cdot N} \sum_{t=1}^{T} \sum_{i=1}^{N} s_i[t]$$
6. **Inference Latency Proxy (ms):** Wall-clock duration per single forward decision pass.
7. **Fixed-Point Degradation:** Difference between FP32 baseline and INT16 fixed-point performance:
   $$\Delta \text{Kill} = \text{Kill}_{\text{FP32}} - \text{Kill}_{\text{INT16}}$$

---

## 4. Evaluated Baselines

1. **Random Agent:** Chooses uniformly from $\{0, 1, 2, 3, 4, 5\}$.
2. **Always Shoot Agent:** Persistently executes action `3` (`ATTACK`).
3. **Left Shoot Agent:** Moves left while continuously firing (action `4`).
4. **ANN Reference Baseline:** Multi-layer perceptron (768 -> 128 -> 64 -> 6) evaluated under identical benchmark conditions.
5. **CNN Reference:** Sujeevan's imitation CNN (reported separately as a distinct paradigm reference, not directly compared due to input/action space mismatch).
