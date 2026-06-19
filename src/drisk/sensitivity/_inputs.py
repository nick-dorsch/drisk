"""Input helpers for sensitivity analysis."""

from __future__ import annotations

from typing import Any

from drisk.distributions import Distribution
from drisk.summary import percentile_label

DistributionKey = tuple[type[Any], int]


def distribution_key(distribution: Distribution) -> DistributionKey:
    """Return an identity key for a distribution object."""
    return (type(distribution), id(distribution))


def distribution_label(distribution: Distribution, index: int) -> str:
    """Return a stable user-facing label for a distribution leaf."""
    return distribution.name or f"{distribution.dist_type}_{index + 1}"


def distribution_percentile(
    distribution: Distribution,
    percentile: float | int,
) -> float:
    """Return a distribution value using Drisk descending percentile semantics."""
    quantile = 1 - float(percentile) / 100
    return float(distribution.ppf(quantile))


def percentile_scenario_values(
    distributions: tuple[Distribution, ...],
    percentile: float | int,
) -> dict[DistributionKey, float]:
    """Return fixed distribution values for one percentile scenario."""
    return {
        distribution_key(distribution): distribution_percentile(
            distribution, percentile
        )
        for distribution in distributions
    }


def percentile_column(percentile: float | int) -> str:
    """Return the display label for a sensitivity percentile."""
    return percentile_label(percentile)
