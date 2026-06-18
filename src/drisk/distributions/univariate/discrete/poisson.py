"""Poisson distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.discrete.base import UvCountDiscrete
from drisk.random import SeedLike, get_scipy_random_state


class Poisson(UvCountDiscrete):
    """Poisson distribution parameterized by rate ``lam``."""

    dist_type: Literal["poisson"] = "poisson"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``lam`` rate parameter."""
        if "lam" not in params:
            raise ValueError("Poisson requires 'lam' parameter")

        lam = params["lam"]
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")

        params["lam"] = float(lam)
        return params

    @classmethod
    def elicit(cls, lam: float, name: str | None = None) -> Self:
        """Elicit a Poisson distribution from a non-negative rate parameter."""
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")

        return cls(
            name=name,
            params={"lam": float(lam)},
            elicitation_params={"lam": lam},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a Poisson distribution to non-negative integer counts."""
        data_arr = np.asarray(data, dtype=float)
        if data_arr.size == 0:
            raise ValueError("Poisson requires at least one observation")
        if not np.all(np.isfinite(data_arr)):
            raise ValueError("Poisson data must be finite")
        if np.any(data_arr < 0):
            raise ValueError("Poisson data must be non-negative")
        if np.any(data_arr != np.floor(data_arr)):
            raise ValueError("Poisson data must be integer-valued")

        lam = np.mean(data_arr)
        return cls(name=name, params={"lam": float(lam)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.poisson.rvs(
            mu=self.params["lam"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability mass function."""
        return stats.poisson.pmf(x, mu=self.params["lam"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.poisson.cdf(x, mu=self.params["lam"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.poisson.ppf(q, mu=self.params["lam"])

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, np.inf)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        lam = self.params["lam"]
        upper = max(10.0, lam + 5 * np.sqrt(lam))
        return (-0.5, float(upper) + 0.5)

    @property
    def mean(self) -> float:
        """Expected value."""
        return float(self.params["lam"])

    @property
    def variance(self) -> float:
        """Variance."""
        return float(self.params["lam"])
