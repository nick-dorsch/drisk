"""Normal distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvRealContinuous
from drisk.random import SeedLike, get_scipy_random_state


class Normal(UvRealContinuous):
    """Normal / Gaussian distribution parameterized by ``mu`` and ``sigma``."""

    dist_type: Literal["normal"] = "normal"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate ``mu`` and ``sigma`` parameters."""
        if "mu" not in params or "sigma" not in params:
            raise ValueError("Normal requires 'mu' and 'sigma' parameters")

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
        """Elicit a normal distribution from symmetric percentile bounds."""
        if lower >= upper:
            raise ValueError(f"lower ({lower}) must be < upper ({upper})")

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        tail_prob = (1 - confidence) / 2
        z_lower = stats.norm.ppf(tail_prob)
        z_upper = stats.norm.ppf(1 - tail_prob)

        mu = (lower + upper) / 2
        sigma = (upper - lower) / (z_upper - z_lower)

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
        """Fit a normal distribution to observations using maximum likelihood."""
        data_arr = np.asarray(data, dtype=float)
        mu, sigma = stats.norm.fit(data_arr)
        return cls(name=name, params={"mu": float(mu), "sigma": float(sigma)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.norm.rvs(
            loc=self.params["mu"],
            scale=self.params["sigma"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        return stats.norm.pdf(x, loc=self.params["mu"], scale=self.params["sigma"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.norm.cdf(x, loc=self.params["mu"], scale=self.params["sigma"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.norm.ppf(q, loc=self.params["mu"], scale=self.params["sigma"])

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (-np.inf, np.inf)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (float(self.ppf(0.0001)), float(self.ppf(0.9999)))
