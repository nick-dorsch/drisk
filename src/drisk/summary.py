"""Summary helpers for Monte Carlo outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_PERCENTILES = (99, 90, 75, 50, 25, 10, 1)


def percentile_label(percentile: float | int) -> str:
    """Return a compact percentile label like ``p90`` or ``p99.5``."""
    percentile_value = float(percentile)
    if not 0 <= percentile_value <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

    label_value = f"{percentile_value:g}"
    return f"p{label_value}"


def descending_percentile_values(
    samples: np.ndarray,
    percentiles: Sequence[float | int],
) -> dict[str, float]:
    """
    Return summary percentile values using descending percentile semantics.

    Labels such as ``p90`` represent the value exceeded by 90% of samples, so
    ``p90`` is calculated from the 10th ascending percentile, ``p50`` from the
    median, and ``p10`` from the 90th ascending percentile.
    """
    ascending_percentiles = [100 - float(percentile) for percentile in percentiles]
    percentile_values = np.percentile(samples, ascending_percentiles)
    return {
        percentile_label(percentile): float(value)
        for percentile, value in zip(percentiles, percentile_values, strict=True)
    }


def apply_percentile_yaxis(
    ax: Any,
    percentiles: Sequence[float | int] = DEFAULT_PERCENTILES,
) -> None:
    """Apply descending percentile tick labels to a cumulative-probability y-axis."""
    tick_positions = [(100 - float(percentile)) / 100 for percentile in percentiles]
    tick_labels = [percentile_label(percentile) for percentile in percentiles]
    ax.set_yticks(tick_positions, labels=tick_labels)
    ax.set_ylabel("")
    ax.set_ylim(bottom=0, top=1)


def threshold_condition_label(threshold: float | int) -> str:
    """Return a compact label for an exceedance condition."""
    return f"> {float(threshold):g}"


def threshold_probability_label(threshold: float | int) -> str:
    """Return a compact label for probability of exceeding a threshold."""
    return f"p({threshold_condition_label(threshold)})"


def conditional_stat_label(label: str, threshold: float | int) -> str:
    """Return an explicit label for a statistic conditional on exceeding a threshold."""
    return f"{label} | {threshold_condition_label(threshold)}"
