"""Bernoulli distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.discrete.base import UvFiniteDiscrete
from drisk.random import SeedLike, get_scipy_random_state


class Bernoulli(UvFiniteDiscrete):
    """Bernoulli distribution for binary outcomes, parameterized by ``p``."""

    dist_type: Literal["bernoulli"] = "bernoulli"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``p`` success-probability parameter."""
        if "p" not in params:
            raise ValueError("Bernoulli requires 'p' parameter")

        p = params["p"]
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")

        params["p"] = float(p)
        return params

    @classmethod
    def elicit(cls, p: float, name: str | None = None) -> Self:
        """Elicit a Bernoulli distribution from a success probability."""
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")

        return cls(
            name=name,
            params={"p": float(p)},
            elicitation_params={"p": p},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a Bernoulli distribution to binary observations."""
        data_arr = np.asarray(data)
        if data_arr.size == 0:
            raise ValueError("Bernoulli requires at least one observation")
        if not np.all(np.isin(data_arr, [0, 1])):
            raise ValueError("Bernoulli data must contain only 0 and 1 values")

        p = np.mean(data_arr)
        return cls(name=name, params={"p": float(p)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.bernoulli.rvs(
            p=self.params["p"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability mass function."""
        return stats.bernoulli.pmf(x, p=self.params["p"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.bernoulli.cdf(x, p=self.params["p"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.bernoulli.ppf(q, p=self.params["p"])

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, 1.0)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (-0.5, 1.5)

    @property
    def mean(self) -> float:
        """Expected value."""
        return float(self.params["p"])

    @property
    def variance(self) -> float:
        """Variance."""
        p = self.params["p"]
        return float(p * (1 - p))
