"""Negative binomial distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.discrete.base import UvCountDiscrete
from drisk.random import SeedLike, get_scipy_random_state


class NegativeBinomial(UvCountDiscrete):
    """
    Negative binomial distribution for failures before ``r`` successes.

    Parameterized by positive integer number of successes ``r`` and per-trial
    success probability ``p``. The support is ``{0, 1, 2, ...}``, representing
    the number of failures observed before the ``r``-th success.
    """

    dist_type: Literal["negative_binomial"] = "negative_binomial"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``r`` and ``p`` parameters."""
        if "r" not in params or "p" not in params:
            raise ValueError("NegativeBinomial requires 'r' and 'p' parameters")

        r = params["r"]
        p = params["p"]

        if not isinstance(r, int) and r != int(r):
            raise ValueError(f"r must be integer, got {r}")

        r_int = int(r)
        if r_int <= 0:
            raise ValueError(f"r must be positive, got {r_int}")

        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")

        params["r"] = r_int
        params["p"] = float(p)
        return params

    @classmethod
    def elicit(cls, r: int | float, p: float, name: str | None = None) -> Self:
        """Elicit a negative binomial distribution from ``r`` and ``p``."""
        if not isinstance(r, int) and r != int(r):
            raise ValueError(f"r must be integer, got {r}")

        r_int = int(r)
        if r_int <= 0:
            raise ValueError(f"r must be positive, got {r_int}")

        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")

        return cls(
            name=name,
            params={"r": r_int, "p": float(p)},
            elicitation_params={"r": r, "p": p},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a negative binomial distribution to count data by moments."""
        data_arr = np.asarray(data, dtype=float)
        if data_arr.size == 0:
            raise ValueError("NegativeBinomial requires at least one observation")
        if not np.all(np.isfinite(data_arr)):
            raise ValueError("NegativeBinomial data must be finite")
        if np.any(data_arr < 0):
            raise ValueError("NegativeBinomial data must be non-negative")
        if np.any(data_arr != np.floor(data_arr)):
            raise ValueError("NegativeBinomial data must be integer-valued")

        mean = float(np.mean(data_arr))
        variance = float(np.var(data_arr, ddof=0))

        if variance <= mean:
            r = max(1, int(mean))
            p = 0.5
        else:
            p = mean / variance
            r = mean**2 / (variance - mean)
            p = max(0.001, min(0.999, p))
            r = max(1, int(r))

        return cls(name=name, params={"r": r, "p": float(p)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.nbinom.rvs(
            n=self.params["r"],
            p=self.params["p"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability mass function."""
        return stats.nbinom.pmf(x, n=self.params["r"], p=self.params["p"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.nbinom.cdf(x, n=self.params["r"], p=self.params["p"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.nbinom.ppf(q, n=self.params["r"], p=self.params["p"])

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, np.inf)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        r = self.params["r"]
        p = self.params["p"]
        mean = r * (1 - p) / p
        variance = r * (1 - p) / p**2
        upper = max(10.0, mean + 5 * np.sqrt(variance))
        return (-0.5, float(upper) + 0.5)

    @property
    def mean(self) -> float:
        """Expected value."""
        r = self.params["r"]
        p = self.params["p"]
        return float(r * (1 - p) / p)

    @property
    def variance(self) -> float:
        """Variance."""
        r = self.params["r"]
        p = self.params["p"]
        return float(r * (1 - p) / p**2)
