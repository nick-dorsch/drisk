"""Summary helpers for Monte Carlo outputs."""

from __future__ import annotations


def percentile_label(percentile: float | int) -> str:
    """Return a compact percentile label like ``p90`` or ``p99.5``."""
    percentile_value = float(percentile)
    if not 0 <= percentile_value <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

    label_value = f"{percentile_value:g}"
    return f"p{label_value}"


def threshold_probability_label(threshold: float | int) -> str:
    """Return a compact label for probability of exceeding a threshold."""
    threshold_value = f"{float(threshold):g}"
    return f"p(> {threshold_value})"
