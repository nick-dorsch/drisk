"""Univariate distribution base classes."""

from abc import ABC, abstractmethod

import numpy as np

from drisk.distributions.base import Distribution


class UvDistribution(Distribution, ABC):
    """Base class for distributions over a single variable."""

    params: dict[str, float | int]

    @abstractmethod
    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        pass

    @abstractmethod
    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        pass

    @abstractmethod
    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        """Distribution support as ``(lower, upper)``."""
        pass

    @property
    @abstractmethod
    def x_range(self) -> tuple[float, float]:
        """Practical plotting/model-inspection range for the distribution."""
        pass

    @property
    def bounded(self) -> bool:
        """Whether the distribution has finite lower and upper support."""
        lower, upper = self.support
        return bool(np.isfinite(lower) and np.isfinite(upper))
