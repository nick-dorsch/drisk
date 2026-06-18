"""Beta distribution."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvUnitBoundedContinuous
from drisk.random import SeedLike, get_scipy_random_state


class Beta(UvUnitBoundedContinuous):
    """Beta distribution for values in the unit interval."""

    dist_type: Literal["beta"] = "beta"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate ``alpha`` and ``beta`` shape parameters."""
        if "alpha" not in params or "beta" not in params:
            raise ValueError("Beta requires 'alpha' and 'beta' parameters")
        if params["alpha"] <= 0 or params["beta"] <= 0:
            raise ValueError("alpha and beta must be positive")
        return params

    @classmethod
    def elicit(
        cls,
        *,
        mode: float | None = None,
        concentration: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        name: str | None = None,
    ) -> Self:
        """
        Create a beta distribution from either intuitive or direct parameters.

        Provide exactly one of:
        - ``mode`` and ``concentration``
        - ``alpha`` and ``beta``
        """
        mode_given = mode is not None
        concentration_given = concentration is not None
        alpha_given = alpha is not None
        beta_given = beta is not None

        if mode_given and concentration_given and not (alpha_given or beta_given):
            if not (0 < mode < 1):
                raise ValueError(f"mode must be in (0, 1), got {mode}")
            if concentration <= 0:
                raise ValueError(f"concentration must be positive, got {concentration}")

            alpha_value = 1 + concentration * mode
            beta_value = 1 + concentration * (1 - mode)
            return cls(
                name=name,
                params={"alpha": float(alpha_value), "beta": float(beta_value)},
                elicitation_params={"mode": mode, "concentration": concentration},
            )

        if alpha_given and beta_given and not (mode_given or concentration_given):
            return cls(
                name=name,
                params={"alpha": float(alpha), "beta": float(beta)},
                elicitation_params={"alpha": alpha, "beta": beta},
            )

        raise ValueError(
            "Must provide exactly one of (mode, concentration) or (alpha, beta). "
            f"Got: mode={mode}, concentration={concentration}, alpha={alpha}, beta={beta}"
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a beta distribution to observations in ``(0, 1)``."""
        data_arr = np.asarray(data, dtype=float)
        if np.any((data_arr <= 0) | (data_arr >= 1)):
            raise ValueError("Beta requires data in (0, 1)")

        alpha, beta, _, _ = stats.beta.fit(data_arr, floc=0, fscale=1)
        return cls(name=name, params={"alpha": float(alpha), "beta": float(beta)})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.beta.rvs(
            a=self.params["alpha"],
            b=self.params["beta"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        return stats.beta.pdf(x, a=self.params["alpha"], b=self.params["beta"])

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.beta.cdf(x, a=self.params["alpha"], b=self.params["beta"])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.beta.ppf(q, a=self.params["alpha"], b=self.params["beta"])

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (0.0, 1.0)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return (0.0, 1.0)

    @property
    def mean(self) -> float:
        """Expected value."""
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        return alpha / (alpha + beta)

    @property
    def mode_value(self) -> float | None:
        """Mode when unique; otherwise ``None`` for flat or U-shaped cases."""
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        if alpha > 1 and beta > 1:
            return (alpha - 1) / (alpha + beta - 2)
        if alpha == 1 and beta == 1:
            return None
        if alpha < 1 and beta < 1:
            return None
        if alpha <= 1:
            return 0.0
        return 1.0

    @property
    def variance(self) -> float:
        """Variance."""
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
