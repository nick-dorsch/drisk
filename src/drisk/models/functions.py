"""Convenience constructors for Monte Carlo model expressions."""

from typing import Any

from drisk.models.base import MCModel


def where(condition: Any, x: Any, y: Any, *, name: str | None = None) -> MCModel:
    """Build a lazy model expression equivalent to ``numpy.where``."""
    return MCModel.where(condition, x, y, name=name)
