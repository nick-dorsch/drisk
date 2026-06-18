"""Base classes for decision tree nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from drisk.random import SeedLike


class DTreeNode(BaseModel, ABC):
    """Abstract base class for decision tree nodes."""

    node_type: str
    name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        """Return the expected value of this node."""
        pass

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Generate outcome samples from this node."""
        pass

    @abstractmethod
    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        """Return tidy rollback rows for this node and descendants."""
        pass
