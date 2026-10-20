"""Hardware constraint and resource accounting package."""

from doom_snn.hardware.resource_accounting import (
    compute_activity_report,
    compute_resource_report,
    format_resource_table,
    save_resource_report,
)
from doom_snn.hardware.snapv_constraints import (
    ALLOWED_BETAS,
    ALLOWED_T,
    INT16_BITS,
    NEURON_BUDGET,
    OPERATING_POINT,
    check_constraints,
)

__all__ = [
    "NEURON_BUDGET",
    "OPERATING_POINT",
    "ALLOWED_BETAS",
    "ALLOWED_T",
    "INT16_BITS",
    "check_constraints",
    "compute_resource_report",
    "compute_activity_report",
    "format_resource_table",
    "save_resource_report",
]
