# Official Hardware Constraints (PeraMorphIQ / SNAP-V Specification)

## 1. Context and Authority
This document defines the authoritative, immutable hardware constraint set derived from Imaadh's specification for the SNAP-V / PeraMorphIQ neuromorphic accelerator architecture. All training tracks (A, B, C, D) and benchmark evaluation protocols implemented in this repository strictly enforce these parameters.

---

## 2. Constraint Inventory

### 2.1 Numerical Precision
- **Target Arithmetic:** 16-bit signed fixed-point integer (`int16`).
- **Dynamic Range:** $[-32,768, +32,767]$.
- **Spike Signals:** Pure binary activations $\{0, 1\}$ (1-bit representation).
- **Membrane Potentials and Synaptic Currents:** Simulated via 16-bit signed representation. Intermediate accumulation registers inside the hardware synaptic arithmetic unit use 32-bit (`int32`) width to guarantee zero overflow during high-fan-in integration, saturating cleanly back to signed 16-bit limits upon state writeback.
- **Quantization Policy:** Symmetric scale representation with Straight-Through Estimator (STE) during Quantization-Aware Training (QAT).
- **INT8 Status:** Non-default exploratory flag only. INT8 is *not* compliant with the primary SNAP-V hardware target.

### 2.2 Membrane Potential Decay ($\beta$)
In digital neuromorphic hardware, multiplying arbitrary floating-point numbers on every synaptic clock cycle requires costly DSP slices. To eliminate DSP multipliers, membrane decay must be computable solely through arithmetic bitshifts (`>>`) and additions/subtractions:

$$\beta \in \{0.25, 0.50, 0.75\}$$

| Value | Hardware Shift-and-Add Formulation | Logic Cost |
| :---: | :--- | :--- |
| **0.25** | `u >> 2` | 1 arithmetic right shift |
| **0.50** | `u >> 1` | 1 arithmetic right shift |
| **0.75** | `u - (u >> 2)` or `(u >> 1) + (u >> 2)` | 1 shift + 1 subtract / add |

> **Constraint Rule:** $\beta = 0.875$ is prohibited from default configurations. It may only be evaluated as an explicit exploratory ablation.

### 2.3 Temporal Constraints & Clocking
Two distinct clocks govern autonomous gameplay:
1. **Internal Spiking Clock:** $T = 8$ internal neural timesteps per game decision. The spiking core evaluates 8 discrete integration-and-fire cycles per visual frame.
2. **Environment Game Clock:** Action repeat is fixed to **4 ViZDoom game tics** per decision. The discrete action emitted after $T=8$ internal steps is held active in the game engine for 4 physics engine ticks (~114 ms of game time at 35 FPS).

### 2.4 Spiking Neuron Budget
- **Hard Upper Bound:** $< 1,024$ physical spiking neurons.
- **Recommended Operating Point:** $\le 964$ neurons (to provide routing margin and guarantee timing closure on FPGA fabric).
- **Default Core Interpretation:** Input image pixels (768 channels) and the deterministic accumulator spike encoder reside on external host pre-processing logic. Only core Leaky Integrate-and-Fire (LIF) neurons count against the accelerator neuron budget.
- **Fallback Interpretation:** If the hardware engineering team requires the 768 input rate encoders to be implemented on-chip as LIF neurons, the total neuron count becomes:
  $$N_{\text{total}} = N_{\text{core}} + 768$$
  - Under $N_{\text{core}} = 96$: $96 + 768 = 864 < 1,024$ (**Compliant**).
  - Under $N_{\text{core}} = 964$: $964 + 768 = 1,732 > 1,024$ (**Violates Budget**).

### 2.5 Synaptic Weight & Memory Realism
A dense layer connecting 768 inputs to 964 core neurons demands:
$$768 \times 964 = 740,352 \text{ synapses}$$
At 16 bits (2 bytes) per weight, input synapses alone consume:
$$740,352 \times 2 = 1,480,704 \text{ bytes} \approx 1.48 \text{ MB}$$
Adding a recurrent matrix of $964 \times 964$ connections adds:
$$929,296 \times 2 = 1,858,592 \text{ bytes} \approx 1.86 \text{ MB}$$
Total weight memory would exceed **3.34 MB**, which strains on-chip block RAM (BRAM/URAM).

**Engineering Recommendation:**
Retain compact core hidden sizes:
- **96 neurons:** 83,616 weights ($\sim 0.16$ MB INT16) — Matches baseline lineage.
- **128 neurons:** 115,584 weights ($\sim 0.22$ MB INT16).
- **192 neurons:** 185,664 weights ($\sim 0.35$ MB INT16).

### 2.6 Continuous Membrane Readout
Readout heads for policy action logits $\pi(a|s)$, value estimates $V(s)$, and Q-values $Q(s, a)$ consume the sub-threshold continuous membrane potential $u_T$ from the final simulation step:
$$\text{Readout} = u_T$$
Relying strictly on binary spike counts over $T=8$ steps quantizes outputs to only 9 discrete levels $\{0, 1, \dots, 8\}$, inducing severe variance and policy collapse during reinforcement learning.
