# Neuromorphic Hardware Resource & Memory Accounting

## 1. Accounting Policy

### Core vs. Fallback Neuron Accounting
- **Default Policy:** Input pixels (768 channels) are treated as external pre-processing channels. Only hidden LIF core neurons count against the 1,024 neuron ceiling.
- **Fallback Policy:** If the rate encoder is mapped to on-chip LIF neurons, total neurons = $N_{\text{core}} + 768$.

| Model Configuration | Core Neurons | Fallback Neurons | Budget Status (<1024) |
| :--- | :---: | :---: | :---: |
| 96-Neuron RSNN | 96 | 864 | ✅ Compliant (both) |
| 128-Neuron RSNN | 128 | 896 | ✅ Compliant (both) |
| 192-Neuron RSNN | 192 | 960 | ✅ Compliant (both) |
| 964-Neuron RSNN | 964 | 1,732 | ⚠️ Violates fallback |

---

## 2. Synaptic Parameter Breakdown (96-Neuron RSNN)

| Component | Dimensions | Connection Weights | Biases | Trainable Parameters | INT16 Footprint |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Input Synapses | $768 \times 96$ | 73,728 | 96 | 73,824 | 147.6 KB |
| Recurrent Synapses | $96 \times 96$ | 9,216 | 0 | 9,216 | 18.4 KB |
| Actor Readout | $96 \times 6$ | 576 | 6 | 582 | 1.16 KB |
| Critic Readout | $96 \times 1$ | 96 | 1 | 97 | 194 Bytes |
| **Total** | | **83,616** | **103** | **83,719** | **~167.4 KB** |

---

## 3. Computational Operations (MACs)
- **Worst-Case Dense MACs per Decision:**
  $$\text{Dense MACs} = T \times (\text{Input Synapses} + \text{Recurrent Synapses} + \text{Heads}) = 8 \times 83,616 = 668,928 \text{ MACs}$$
- **Event-Driven Synaptic Updates:**
  Neuromorphic hardware executes synaptic updates only when a spike arrives. At an average input spike rate of 15% and hidden spike rate of 10%:
  $$\text{Events} = 8 \times (0.15 \times 73,728 + 0.10 \times 9,216) \approx 95,846 \text{ synaptic events}$$
  This represents an empirical **85.7% computational sparsity saving**.
