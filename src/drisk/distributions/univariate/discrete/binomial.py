"""Binomial distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.discrete.base import UvFiniteDiscrete
from drisk.random import SeedLike, get_scipy_random_state


class Binomial(UvFiniteDiscrete):
    """Binomial distribution parameterized by trials ``n`` and probability ``p``."""

    dist_type: Literal["binomial"] = "binomial"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``n`` and ``p`` parameters."""
        if "n" not in params or "p" not in params:
            raise ValueError("Binomial requires 'n' and 'p' parameters")

        n = params["n"]
        p = params["p"]

        if not isinstance(n, int) and n != int(n):
            raise ValueError(f"n must be integer, got {n}")

        n_int = int(n)
        if n_int <= 0:
            raise ValueError(f"n must be positive, got {n_int}")

        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")

        params["n"] = n_int
        params["p"] = float(p)
        return params

    @classmethod
    def elicit(cls, n: int | float, p: float, name: str | None = None) -> Self:
        """Elicit a binomial distribution from ``n`` and ``p``."""
        if not isinstance(n, int) and n != int(n):
            raise ValueError(f"n must be integer, got {n}")

        n_int = int(n)
        if n_int <= 0:
            raise ValueError(f"n must be positive, got {n_int}")

        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")

        return cls(
            name=name,
            params={"n": n_int, "p": float(p)},
            elicitation_params={"n": n, "p": p},
        )

    @classmethod
    def fit(cls, data: ArrayLike, n: int | float, name: str | None = None) -> Self:
        """Fit a binomial distribution with known ``n`` to count observations."""
        if not isinstance(n, int) and n != int(n):
            raise ValueError(f"n must be integer, got {n}")

        n_int = int(n)
        if n_int <= 0:
            raise ValueError(f"n must be positive, got {n_int}")

        data_arr = np.asarray(data, dtype=float)
        if data_arr.size == 0:
            raise ValueError("Binomial requires at least one observation")
        if not np.all(np.isfinite(data_arr)):
            raise ValueError("Binomial data must be finite")
        if np.any(data_arr < 0):
            raise ValueError("Binomial data must be non-negative")
        if np.any(data_arr != np.floor(data_arr)):
            raise ValueError("Binomial data must be integer-valued")
        if np.any(data_arr > n_int):
            raise ValueError(f"Binomial data values must be <= n ({n_int})")

        p = np.mean(data_arr) / n_int
        p = max(0.0, min(1.0, p))
        return cls(name=name, params={"n": n_int, "p": float(p)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.binom.rvs(
            n=self.params["n"],
            p=self.params["p"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability mass function."""
        return stats.binom.pmf(x, n=self.params["n"], p=self.params["p"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.binom.cdf(x, n=self.params["n"], p=self.params["p"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.binom.ppf(q, n=self.params["n"], p=self.params["p"])

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, float(self.params["n"]))

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (-0.5, float(self.params["n"]) + 0.5)

    @property
    def mean(self) -> float:
        """Expected value."""
        return float(self.params["n"] * self.params["p"])

    @property
    def variance(self) -> float:
        """Variance."""
        n = self.params["n"]
        p = self.params["p"]
        return float(n * p * (1 - p))
