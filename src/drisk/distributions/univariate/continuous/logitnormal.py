"""Logit-normal distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats
from scipy.special import expit, logit

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvUnitBoundedContinuous
from drisk.random import SeedLike, get_scipy_random_state


class LogitNormal(UvUnitBoundedContinuous):
    """Logit-normal distribution for values in the unit interval."""

    dist_type: Literal["logitnormal"] = "logitnormal"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate ``mu`` and ``sigma`` parameters."""
        if "mu" not in params or "sigma" not in params:
            raise ValueError("LogitNormal requires 'mu' and 'sigma' parameters")

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
        """Elicit a logit-normal distribution from unit-interval percentile bounds."""
        if not (0 < lower < 1) or not (0 < upper < 1):
            raise ValueError("LogitNormal bounds must be in (0, 1)")

        if lower >= upper:
            raise ValueError(f"lower ({lower}) must be < upper ({upper})")

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        logit_lower = logit(lower)
        logit_upper = logit(upper)

        tail_prob = (1 - confidence) / 2
        z_lower = stats.norm.ppf(tail_prob)
        z_upper = stats.norm.ppf(1 - tail_prob)

        mu = (logit_lower + logit_upper) / 2
        sigma = (logit_upper - logit_lower) / (z_upper - z_lower)

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
        """Fit a logit-normal distribution to observations in ``(0, 1)``."""
        data_arr = np.asarray(data, dtype=float)
        if np.any((data_arr <= 0) | (data_arr >= 1)):
            raise ValueError("LogitNormal requires data in (0, 1)")

        mu, sigma = stats.norm.fit(logit(data_arr))
        return cls(name=name, params={"mu": float(mu), "sigma": float(sigma)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        normal_samples = stats.norm.rvs(
            loc=self.params["mu"],
            scale=self.params["sigma"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )
        return expit(normal_samples)

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        x_arr = np.asarray(x, dtype=float)
        result = np.zeros_like(x_arr, dtype=float)
        mask = (x_arr > 0) & (x_arr < 1)

        if np.any(mask):
            x_valid = x_arr[mask]
            normal_pdf = stats.norm.pdf(
                logit(x_valid),
                loc=self.params["mu"],
                scale=self.params["sigma"],
            )
            result[mask] = normal_pdf / (x_valid * (1 - x_valid))

        return result

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        x_arr = np.asarray(x, dtype=float)
        result = np.zeros_like(x_arr, dtype=float)
        result[x_arr >= 1] = 1.0
        mask = (x_arr > 0) & (x_arr < 1)

        if np.any(mask):
            result[mask] = stats.norm.cdf(
                logit(x_arr[mask]),
                loc=self.params["mu"],
                scale=self.params["sigma"],
            )

        return result

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        logit_quantile = stats.norm.ppf(
            q,
            loc=self.params["mu"],
            scale=self.params["sigma"],
        )
        return expit(logit_quantile)

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, 1.0)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (0.0, 1.0)
