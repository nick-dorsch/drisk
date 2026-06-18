"""Base interfaces for probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from drisk.arithmetic import ArithmeticMixin
from drisk.distributions.types import ArrayLike
from drisk.random import SeedLike


class Distribution(ArithmeticMixin, BaseModel, ABC):
    """
    Top-level abstract base class for probability distributions.

    Combines Pydantic models for validation/serialization with abstract methods
    for the sampling interface used by Drisk models.
    """

    dist_type: str
    name: str | None = None
    elicitation_params: dict[str, Any] | None = None

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    @abstractmethod
    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples from the distribution."""
        pass

    def rvs(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Alias for :meth:`sample` for users familiar with SciPy naming."""
        return self.sample(size=size, seed=seed)

    @classmethod
    @abstractmethod
    def elicit(cls, **kwargs: Any) -> Self:
        """
        Construct a distribution from elicited parameters.

        Implementations should store the elicitation inputs on the returned
        object's ``elicitation_params`` attribute.
        """
        pass

    @classmethod
    @abstractmethod
    def fit(cls, data: ArrayLike, **kwargs: Any) -> Self:
        """Fit a distribution to observed data."""
        pass

    @abstractmethod
    def plot(self, **kwargs: Any) -> Any:
        """Create a quicklook plot for the distribution."""
        pass
