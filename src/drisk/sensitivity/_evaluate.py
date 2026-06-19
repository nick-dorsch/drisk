"""Deterministic evaluators for sensitivity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from drisk.distributions import Distribution
from drisk.models import MCModel

from ._inputs import DistributionKey, distribution_key


def evaluate_mc_model(
    model: MCModel,
    fixed_values: dict[DistributionKey, float],
) -> float:
    """Evaluate an MCModel with distribution leaves replaced by fixed values."""
    result = _eval_mc_operand(model, fixed_values=fixed_values)
    result_arr = np.asarray(result, dtype=float)
    if result_arr.size != 1:
        raise ValueError(
            "Sensitivity evaluation requires scalar model outputs after fixing "
            "distribution leaves"
        )
    return float(result_arr.reshape(-1)[0])


def _eval_mc_operand(
    operand: Any,
    *,
    fixed_values: dict[DistributionKey, float],
) -> np.ndarray:
    """Evaluate an MCModel operand with fixed distribution values."""
    if isinstance(operand, MCModel):
        values = [
            _eval_mc_operand(nested_operand, fixed_values=fixed_values)
            for nested_operand in operand.operands
        ]
        return np.asarray(operand.op.function(*values))

    if isinstance(operand, Distribution):
        key = distribution_key(operand)
        if key not in fixed_values:
            raise ValueError(
                f"No fixed value provided for distribution "
                f"{operand.name or operand.dist_type!r}"
            )
        return np.asarray(fixed_values[key])

    return np.asarray(operand)
