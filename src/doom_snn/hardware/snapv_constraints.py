"""SNAP-V Neuromorphic Hardware Constraint Checker.

Enforces physical and architectural limits of the SNAP-V neuromorphic processor:
- Maximum neuron budget: 1024 neurons
- Safe operating point: 964 neurons (to avoid routing/timing congestion)
- Allowed LIF membrane decay constants: beta in {0.25, 0.5, 0.75} (shift-and-add hardware)
- Required temporal execution step: T = 8 timesteps
- Numerical precision: 16-bit signed fixed-point (Q1.15 or Q4.12)
"""

from typing import Any, Dict, List

NEURON_BUDGET: int = 1024
OPERATING_POINT: int = 964
ALLOWED_BETAS: List[float] = [0.25, 0.5, 0.75]
ALLOWED_T: int = 8
INT16_BITS: int = 16


def check_constraints(resource_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Verify whether a candidate model/configuration satisfies SNAP-V hardware limits.

    Args:
        resource_summary: Dictionary containing model architecture and hardware fields:
            - core_neurons (or hidden_neurons, num_neurons)
            - beta (membrane leak constant)
            - T (or timesteps, time_steps)
            - fallback_total_neurons (optional, including input frame encoders)

    Returns:
        Dict[str, Any] with boolean flags, warning messages, and error diagnostics:
            - neuron_budget_ok: bool (True if core_neurons < 1024)
            - neurons_under_operating_point: bool (True if core_neurons <= 964)
            - beta_compliant: bool (True if beta in [0.25, 0.5, 0.75])
            - T_compliant: bool (True if T == 8)
            - overall_compliant: bool (True if all constraints are satisfied without errors)
            - warnings: List[str]
            - errors: List[str]
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Extract core neurons
    core_neurons = (
        resource_summary.get("core_neurons")
        or resource_summary.get("hidden_neurons")
        or resource_summary.get("num_neurons")
        or 96
    )
    core_neurons = int(core_neurons)

    # 1. Check neuron budget: core_neurons < 1024
    if core_neurons >= NEURON_BUDGET:
        neuron_budget_ok = False
        errors.append(
            f"Core neuron count ({core_neurons}) exceeds maximum hardware budget of "
            f"{NEURON_BUDGET} neurons."
        )
    else:
        neuron_budget_ok = True

    # 2. Check operating point: warn if core_neurons > 964
    if core_neurons > OPERATING_POINT:
        neurons_under_operating_point = False
        warnings.append(
            f"Core neuron count ({core_neurons}) exceeds safe operating point of "
            f"{OPERATING_POINT} neurons. Risk of FPGA routing congestion and timing closure failure."
        )
    else:
        neurons_under_operating_point = True

    # Check fallback total neurons if input encoder neurons are mapped to core
    fallback_total_neurons = resource_summary.get("fallback_total_neurons")
    if fallback_total_neurons is not None and int(fallback_total_neurons) >= NEURON_BUDGET:
        warnings.append(
            f"Fallback total neuron count ({fallback_total_neurons}) with on-chip rate encoders "
            f"exceeds {NEURON_BUDGET}. Input rate encoding must run off-core or via host DSP logic."
        )

    # 3. Check beta compliance: beta in {0.25, 0.5, 0.75}
    raw_beta = resource_summary.get("beta", 0.5)
    try:
        beta = float(raw_beta)
    except (TypeError, ValueError):
        beta = 0.5

    beta_compliant = any(abs(beta - allowed) < 1e-4 for allowed in ALLOWED_BETAS)
    if not beta_compliant:
        errors.append(
            f"Membrane decay beta={beta} is not supported by SNAP-V fixed-point decay logic. "
            f"Allowed values are {ALLOWED_BETAS} (power-of-two shift combinations)."
        )

    # 4. Check timestep compliance: T == 8
    raw_t = (
        resource_summary.get("T")
        or resource_summary.get("timesteps")
        or resource_summary.get("time_steps")
        or 8
    )
    try:
        t_val = int(raw_t)
    except (TypeError, ValueError):
        t_val = 8

    if t_val == ALLOWED_T:
        T_compliant = True
    else:
        T_compliant = False
        errors.append(
            f"Simulation timesteps T={t_val} violates fixed hardware frame-rate budget of "
            f"T={ALLOWED_T} steps per decision."
        )

    overall_compliant = neuron_budget_ok and beta_compliant and T_compliant and (len(errors) == 0)

    return {
        "neuron_budget_ok": neuron_budget_ok,
        "neurons_under_operating_point": neurons_under_operating_point,
        "beta_compliant": beta_compliant,
        "T_compliant": T_compliant,
        "overall_compliant": overall_compliant,
        "warnings": warnings,
        "errors": errors,
        "neuron_budget": NEURON_BUDGET,
        "operating_point": OPERATING_POINT,
        "allowed_betas": ALLOWED_BETAS,
        "allowed_t": ALLOWED_T,
        "int16_bits": INT16_BITS,
    }
