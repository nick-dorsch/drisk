"""Outcome nodes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from drisk.random import SeedLike

from .._sampling import sample_value
from .._types import ValueLike
from .base import DTreeNode


class OutcomeNode(DTreeNode):
    """Terminal node containing a scalar, distribution, or Monte Carlo model value."""

    node_type: Literal["outcome"] = "outcome"
    value: ValueLike

    def __init__(
        self, value: ValueLike = 0, name: str | None = None, **data: Any
    ) -> None:
        if "value" not in data:
            data["value"] = value
        if name is not None and "name" not in data:
            data["name"] = name
        super().__init__(**data)

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        samples = np.ravel(
            sample_value(self.value, size=10_000 if size is None else size, seed=seed)
        )
        return float(np.mean(samples))

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        return sample_value(self.value, size=size, seed=seed)

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "node": self.name or "outcome",
                "node_type": self.node_type,
                "branch": None,
                "probability": None,
                "expected_value": self.expected_value(size=size, seed=seed),
                "selected": None,
            }
        ]
