"""Lognormal distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvPositiveContinuous
from drisk.random import SeedLike, get_scipy_random_state


class LogNormal(UvPositiveContinuous):
    """Lognormal distribution parameterized by underlying normal ``mu`` and ``sigma``."""

    dist_type: Literal["lognormal"] = "lognormal"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate ``mu`` and ``sigma`` parameters."""
        if "mu" not in params or "sigma" not in params:
            raise ValueError("LogNormal requires 'mu' and 'sigma' parameters")

        sigma = params["sigma"]
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        return params

    @classmethod
    def elicit(
        cls,
        lower: float,
        upper: float,
        confidence: float = 0.8,
        name: str | None = None,
    ) -> Self:
        """Elicit a lognormal distribution from positive percentile bounds."""
        if lower <= 0 or upper <= 0:
            raise ValueError("LogNormal bounds must be positive")

        if lower >= upper:
            raise ValueError(f"lower ({lower}) must be < upper ({upper})")

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        log_lower = np.log(lower)
        log_upper = np.log(upper)

        tail_prob = (1 - confidence) / 2
        z_lower = stats.norm.ppf(tail_prob)
        z_upper = stats.norm.ppf(1 - tail_prob)

        mu = (log_lower + log_upper) / 2
        sigma = (log_upper - log_lower) / (z_upper - z_lower)

        return cls(
            name=name,
            params={"mu": float(mu), "sigma": float(sigma)},
            elicitation_params={
                "lower": lower,
                "upper": upper,
                "confidence": confidence,
            },
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a lognormal distribution to positive observations."""
        data_arr = np.asarray(data, dtype=float)
        if np.any(data_arr <= 0):
            raise ValueError("LogNormal requires positive data")

        mu, sigma = stats.norm.fit(np.log(data_arr))
        return cls(name=name, params={"mu": float(mu), "sigma": float(sigma)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.lognorm.rvs(
            s=self.params["sigma"],
            scale=np.exp(self.params["mu"]),
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        return stats.lognorm.pdf(
            x,
            s=self.params["sigma"],
            scale=np.exp(self.params["mu"]),
        )

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.lognorm.cdf(
            x,
            s=self.params["sigma"],
            scale=np.exp(self.params["mu"]),
        )

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.lognorm.ppf(
            q,
            s=self.params["sigma"],
            scale=np.exp(self.params["mu"]),
        )

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, np.inf)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (float(self.ppf(0.0001)), float(self.ppf(0.9999)))
