# 16-Bit Fixed-Point Simulation & Quantization-Aware Training (QAT)

## 1. Mathematical Representation
All continuous activations, synaptic weights, biases, and membrane potentials are mapped to signed 16-bit integers (`int16`):
$$q \in [-32768, +32767]$$

Given a dynamic scale factor $S > 0$:
$$\text{Quantize}(x) = \text{clamp}\left(\left\lfloor \frac{x}{S} + 0.5 \right\rfloor, -32768, 32767\right)$$
$$\text{Dequantize}(q) = q \times S$$

---

## 2. Straight-Through Estimator (STE)
During backpropagation, rounding and clamping have zero derivative almost everywhere. We employ a Straight-Through Estimator where gradients pass through the quantization operation unchanged:
$$\frac{\partial \hat{x}}{\partial x} = 1.0 \quad \text{for } x \in [x_{\min}, x_{\max}]$$

---

## 3. Shift-and-Add Membrane Dynamics
In the quantized integer domain, the membrane potential update at timestep $t$ avoids floating-point multiplication:
$$u_{t}^{\text{int}} = \text{Shift}(u_{t-1}^{\text{int}}, \beta) + I_{t}^{\text{int}}$$

For compliant $\beta$ values:
- $\beta = 0.25$: `u >> 2`
- $\beta = 0.50$: `u >> 1`
- $\beta = 0.75$: `u - (u >> 2)`

Arithmetic right-shift preserves sign for negative potentials. Accumulation uses an internal 32-bit register to prevent intermediate overflow.

---

## 4. QAT Workflow
1. **Pre-training:** Train model in FP32 or initialize from baseline.
2. **Layer Replacement:** Call `insert_fake_quant(model)`, replacing linear layers with `QuantizedLinear`.
3. **Fine-Tuning:** Run A2C+GAE with fake-quantized weights.
4. **Validation:** Measure saturation counts and evaluate degradation against the FP32 baseline.
