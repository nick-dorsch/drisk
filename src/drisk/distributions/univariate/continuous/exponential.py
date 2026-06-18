"""Exponential distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvPositiveContinuous
from drisk.random import SeedLike, get_scipy_random_state


class Exponential(UvPositiveContinuous):
    """Exponential distribution parameterized by rate ``lam``."""

    dist_type: Literal["exponential"] = "exponential"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``lam`` rate parameter."""
        if "lam" not in params:
            raise ValueError("Exponential requires 'lam' parameter")

        lam = params["lam"]
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")

        return params

    @classmethod
    def elicit(
        cls,
        lam: float,
        name: str | None = None,
    ) -> Self:
        """
        Elicit an exponential distribution from a rate parameter.

        The resulting distribution has mean ``1 / lam`` and variance
        ``1 / lam**2``.
        """
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")

        return cls(
            name=name,
            params={"lam": float(lam)},
            elicitation_params={"lam": lam},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit an exponential distribution to positive observations."""
        data_arr = np.asarray(data, dtype=float)
        if np.any(data_arr <= 0):
            raise ValueError("Exponential requires positive data")

        lam_mle = 1.0 / np.mean(data_arr)
        return cls(name=name, params={"lam": float(lam_mle)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.expon.rvs(
            scale=1.0 / self.params["lam"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        return stats.expon.pdf(x, scale=1.0 / self.params["lam"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.expon.cdf(x, scale=1.0 / self.params["lam"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.expon.ppf(q, scale=1.0 / self.params["lam"])

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
        return 1.0 / self.params["lam"]

    @property
    def variance(self) -> float:
        """Variance."""
        return 1.0 / (self.params["lam"] ** 2)
