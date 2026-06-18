"""Gamma distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvPositiveContinuous
from drisk.random import SeedLike, get_scipy_random_state


class Gamma(UvPositiveContinuous):
    """Gamma distribution parameterized by shape ``alpha`` and rate ``beta``."""

    dist_type: Literal["gamma"] = "gamma"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate ``alpha`` and ``beta`` parameters."""
        if "alpha" not in params or "beta" not in params:
            raise ValueError("Gamma requires 'alpha' and 'beta' parameters")

        alpha = params["alpha"]
        beta = params["beta"]
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        return params

    @classmethod
    def elicit(
        cls,
        k: float,
        rate: float,
        name: str | None = None,
    ) -> Self:
        """
        Elicit a gamma distribution from waiting-time parameters.

        Models the waiting time for ``k`` events to occur at ``rate`` events per
        unit time. The resulting distribution has mean ``k / rate`` and
        variance ``k / rate**2``.
        """
        if k <= 0:
            raise ValueError(f"k (number of events) must be positive, got {k}")
        if rate <= 0:
            raise ValueError(
                f"rate (events per unit time) must be positive, got {rate}"
            )

        return cls(
            name=name,
            params={"alpha": float(k), "beta": float(rate)},
            elicitation_params={"k": k, "rate": rate},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a gamma distribution to positive observations."""
        data_arr = np.asarray(data, dtype=float)
        if np.any(data_arr <= 0):
            raise ValueError("Gamma requires positive data")

        alpha, _, scale = stats.gamma.fit(data_arr, floc=0)
        beta = 1.0 / scale
        return cls(name=name, params={"alpha": float(alpha), "beta": float(beta)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.gamma.rvs(
            a=self.params["alpha"],
            scale=1.0 / self.params["beta"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        return stats.gamma.pdf(
            x,
            a=self.params["alpha"],
            scale=1.0 / self.params["beta"],
        )

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.gamma.cdf(
            x,
            a=self.params["alpha"],
            scale=1.0 / self.params["beta"],
        )

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.gamma.ppf(
            q,
            a=self.params["alpha"],
            scale=1.0 / self.params["beta"],
        )

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, np.inf)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (float(self.ppf(0.0001)), float(self.ppf(0.9999)))

    @property
    def mean(self) -> float:
        """Expected value."""
        return self.params["alpha"] / self.params["beta"]

    @property
    def variance(self) -> float:
        """Variance."""
        return self.params["alpha"] / (self.params["beta"] ** 2)
