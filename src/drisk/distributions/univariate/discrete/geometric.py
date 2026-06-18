"""Geometric distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.discrete.base import UvCountDiscrete
from drisk.random import SeedLike, get_scipy_random_state


class Geometric(UvCountDiscrete):
    """
    Geometric distribution for the number of failures before first success.

    Parameterized by success probability ``p``. This implementation uses
    zero-based support ``{0, 1, 2, ...}``, while SciPy's ``geom`` uses one-based
    support for the number of trials until first success.
    """

    dist_type: Literal["geometric"] = "geometric"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``p`` success-probability parameter."""
        if "p" not in params:
            raise ValueError("Geometric requires 'p' parameter")

        p = params["p"]
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")

        params["p"] = float(p)
        return params

    @classmethod
    def elicit(cls, p: float, name: str | None = None) -> Self:
        """Elicit a geometric distribution from a success probability."""
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")

        return cls(
            name=name,
            params={"p": float(p)},
            elicitation_params={"p": p},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a geometric distribution to non-negative integer counts."""
        data_arr = np.asarray(data, dtype=float)
        if data_arr.size == 0:
            raise ValueError("Geometric requires at least one observation")
        if not np.all(np.isfinite(data_arr)):
            raise ValueError("Geometric data must be finite")
        if np.any(data_arr < 0):
            raise ValueError("Geometric data must be non-negative")
        if np.any(data_arr != np.floor(data_arr)):
            raise ValueError("Geometric data must be integer-valued")

        p = 1.0 / (float(np.mean(data_arr)) + 1.0)
        return cls(name=name, params={"p": p})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return (
            stats.geom.rvs(
                p=self.params["p"],
                size=size,
                random_state=get_scipy_random_state(seed),
            )
            - 1
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability mass function."""
        return stats.geom.pmf(np.asarray(x) + 1, p=self.params["p"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.geom.cdf(np.asarray(x) + 1, p=self.params["p"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.geom.ppf(q, p=self.params["p"]) - 1

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, np.inf)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        p = self.params["p"]
        mean = (1 - p) / p
        std = np.sqrt(1 - p) / p
        upper = max(10.0, mean + 5 * std)
        return (-0.5, float(upper) + 0.5)

    @property
    def mean(self) -> float:
        """Expected value."""
        p = self.params["p"]
        return float((1 - p) / p)

    @property
    def variance(self) -> float:
        """Variance."""
        p = self.params["p"]
        return float((1 - p) / p**2)
