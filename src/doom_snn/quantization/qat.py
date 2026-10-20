"""Quantization-Aware Training (QAT) modules and graph transformation utilities."""

import logging
from typing import Any, Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from doom_snn.neurons.states import LIFState, init_lif_state
from doom_snn.neurons.surrogate import fast_sigmoid_spike
from doom_snn.quantization.fake_quant import FakeQuantInt16
from doom_snn.quantization.fixed_point import (
    INT16_MAX,
    INT16_MIN,
    beta_shift_int16,
    float_to_int16,
    int16_to_float,
)

logger = logging.getLogger(__name__)

RECOMMENDED_BETAS = {0.25, 0.5, 0.75}


class QuantizedLinear(nn.Module):
    """Linear layer wrapper that applies INT16 fake quantization to weights during QAT.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to False, layer will not learn an additive bias (default: True).
        weight_bits: Bit-width for weight quantization (default: 16).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_quant = FakeQuantInt16()

    @property
    def weight(self) -> nn.Parameter:
        """Reference to underlying linear layer weight parameter."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        """Reference to underlying linear layer bias parameter."""
        return self.linear.bias

    def forward(self, x: Tensor) -> Tensor:
        """Apply fake-quantized linear transformation to incoming data.

        Args:
            x: Input tensor [*, in_features].

        Returns:
            Output tensor [*, out_features].
        """
        q_weight = self.weight_quant(self.linear.weight)
        return F.linear(x, q_weight, self.linear.bias)


class QuantizedLIFDynamics(nn.Module):
    """Simulates 16-bit fixed-point LIF neuron update dynamics for hardware matching.

    Simulates:
        - Membrane potential representation in INT16 domain.
        - Beta decay via hardware arithmetic bitshift operations.
        - Current accumulation using INT32 intermediate register with saturation to INT16.
        - Quantized domain threshold comparison with fast-sigmoid surrogate gradient.
        - Membrane reset in quantized domain.

    Args:
        beta: Membrane decay factor in {0.25, 0.5, 0.75} (default: 0.75).
        threshold: Firing threshold voltage (default: 0.4).
        membrane_scale: Voltage per INT16 step for membrane potential (default: 1e-4).
        current_scale: Current per INT16 step for synaptic current (default: 1e-4).
        reset_mode: Membrane reset mode ('soft' or 'hard', default: 'soft').
    """

    def __init__(
        self,
        beta: float = 0.75,
        threshold: float = 0.4,
        membrane_scale: float = 1e-4,
        current_scale: float = 1e-4,
        reset_mode: str = "soft",
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.membrane_scale = float(membrane_scale)
        self.current_scale = float(current_scale)
        self.reset_mode = reset_mode

        if self.beta not in RECOMMENDED_BETAS:
            warnings.warn(
                f"Beta {self.beta} is not in recommended hardware set {RECOMMENDED_BETAS}. "
                "Bitshift operations require beta in {0.25, 0.5, 0.75}.",
                UserWarning,
                stacklevel=2,
            )

        if self.reset_mode not in ("soft", "hard"):
            raise ValueError(
                f"Unsupported reset_mode '{reset_mode}'. Expected 'soft' or 'hard'."
            )

        self.last_saturation_count: int = 0
        self.total_saturations: int = 0

    def init_state(
        self,
        batch_size: int,
        num_neurons: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LIFState:
        """Initialize zeroed LIFState for quantized dynamics.

        Args:
            batch_size: Batch dimension size.
            num_neurons: Hidden neuron dimension size.
            device: Target torch device.
            dtype: Tensor dtype (default: torch.float32).

        Returns:
            Initialized zero LIFState.
        """
        return init_lif_state(
            batch_size=batch_size,
            num_neurons=num_neurons,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        current: Tensor,
        state: Optional[LIFState] = None,
    ) -> tuple[Tensor, Tensor, LIFState]:
        """Compute one step of quantized LIF dynamics with INT32 accumulation.

        Args:
            current: Synaptic input current tensor [batch, num_neurons].
            state: Optional previous LIFState. If None, initialized to zeros.

        Returns:
            Tuple of:
                - spikes: Binary spike activations [batch, num_neurons].
                - membrane: Updated post-reset membrane potential [batch, num_neurons].
                - new_state: LIFState containing updated state.
        """
        if state is None:
            state = self.init_state(
                current.shape[0],
                current.shape[1],
                device=current.device,
                dtype=current.dtype,
            )
        elif state.membrane.device != current.device or state.membrane.dtype != current.dtype:
            state = state.to(device=current.device, dtype=current.dtype)

        # 1. Convert floating membrane state into INT16 representation
        membrane_int = float_to_int16(state.membrane, self.membrane_scale)

        # 2. Apply decay factor beta via arithmetic bitshift operations
        decayed_int = beta_shift_int16(membrane_int, self.beta)

        # 3. Quantize input synaptic current and scale to membrane unit
        current_int = float_to_int16(current, self.current_scale)
        scale_ratio = self.current_scale / max(self.membrane_scale, 1e-12)
        if abs(scale_ratio - 1.0) < 1e-6:
            current_in_membrane = current_int.to(torch.int32)
        else:
            current_in_membrane = torch.round(current_int.to(torch.float32) * scale_ratio).to(torch.int32)

        # 4. INT32 intermediate accumulation to simulate hardware accumulator register
        accum_int32 = decayed_int.to(torch.int32) + current_in_membrane

        # 5. Check and record saturation against INT16 bounds
        sat_mask = (accum_int32 < INT16_MIN) | (accum_int32 > INT16_MAX)
        sat_count = int(sat_mask.sum().item())
        self.last_saturation_count = sat_count
        self.total_saturations += sat_count
        if sat_count > 0:
            logger.debug(
                "QuantizedLIFDynamics: %d activations saturated to INT16 limits.", sat_count
            )

        # Saturate back to signed 16-bit range
        accum_clamped = torch.clamp(accum_int32, INT16_MIN, INT16_MAX).to(torch.int16)

        # 6. Quantized threshold comparison with surrogate gradient for backprop
        accum_float = int16_to_float(accum_clamped, self.membrane_scale)
        spikes = fast_sigmoid_spike(accum_float, self.threshold)

        # 7. Apply reset mechanism in quantized integer domain
        if self.reset_mode == "soft":
            thresh_int = int(round(self.threshold / max(self.membrane_scale, 1e-12)))
            reset_accum = accum_clamped.to(torch.int32) - (spikes.to(torch.int32) * thresh_int)
            reset_clamped = torch.clamp(reset_accum, INT16_MIN, INT16_MAX).to(torch.int16)
        elif self.reset_mode == "hard":
            reset_clamped = (accum_clamped * (1.0 - spikes.to(torch.int16))).to(torch.int16)
        else:
            raise RuntimeError(f"Unhandled reset mode: {self.reset_mode}")

        post_reset_float = int16_to_float(reset_clamped, self.membrane_scale)
        new_state = LIFState(membrane=post_reset_float, spikes=spikes)
        return spikes, post_reset_float, new_state


# Alias QuantizedLIF to QuantizedLIFDynamics
QuantizedLIF = QuantizedLIFDynamics


def insert_fake_quant(model: nn.Module, config: Optional[dict[str, Any]] = None) -> nn.Module:
    """Recursively replace nn.Linear layers in model with QuantizedLinear.

    Preserves existing weights and bias values.

    Args:
        model: Target PyTorch model to modify in-place.
        config: Optional configuration dictionary (e.g. {"weight_bits": 16}).

    Returns:
        Modified model with QuantizedLinear layers.
    """
    if config is None:
        config = {}
    weight_bits = config.get("weight_bits", 16)

    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, QuantizedLinear):
            has_bias = child.bias is not None
            quant_layer = QuantizedLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=has_bias,
                weight_bits=weight_bits,
            )
            with torch.no_grad():
                quant_layer.linear.weight.copy_(child.weight)
                if has_bias and child.bias is not None:
                    quant_layer.linear.bias.copy_(child.bias)
            setattr(model, name, quant_layer)
        else:
            insert_fake_quant(child, config)

    return model


def qat_summary(model: nn.Module) -> dict[str, Any]:
    """Generate summary of quantization scales and saturation counts across all layers.

    Args:
        model: PyTorch model containing quantized layers.

    Returns:
        Dictionary containing layer summaries, scales, saturation counts, and totals.
    """
    summary: dict[str, Any] = {
        "layers": {},
        "scales": {},
        "saturation_counts": {},
        "total_saturations": 0,
    }

    total_sat = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            scale = module.weight_quant.last_scale
            sat_count = module.weight_quant.last_saturation_count
            summary["layers"][name] = {
                "type": "QuantizedLinear",
                "weight_scale": scale,
                "weight_saturation_count": sat_count,
            }
            summary["scales"][f"{name}.weight"] = scale
            summary["saturation_counts"][f"{name}.weight"] = sat_count
            total_sat += sat_count
        elif isinstance(module, FakeQuantInt16):
            scale = module.last_scale
            sat_count = module.last_saturation_count
            summary["layers"][name] = {
                "type": "FakeQuantInt16",
                "scale": scale,
                "saturation_count": sat_count,
            }
            summary["scales"][name] = scale
            summary["saturation_counts"][name] = sat_count
            total_sat += sat_count
        elif isinstance(module, QuantizedLIFDynamics):
            summary["layers"][name] = {
                "type": "QuantizedLIFDynamics",
                "membrane_scale": module.membrane_scale,
                "current_scale": module.current_scale,
                "saturation_count": module.last_saturation_count,
                "total_saturations": module.total_saturations,
            }
            summary["scales"][f"{name}.membrane"] = module.membrane_scale
            summary["scales"][f"{name}.current"] = module.current_scale
            summary["saturation_counts"][name] = module.last_saturation_count
            total_sat += module.last_saturation_count

    summary["total_saturations"] = total_sat
    return summary
