"""Stretched beta and PERT distributions."""

from typing import Literal, Self

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.continuous.base import UvBoundedContinuous
from drisk.random import SeedLike, get_scipy_random_state


class StretchedBeta(UvBoundedContinuous):
    """Beta distribution scaled to arbitrary finite bounds."""

    dist_type: Literal["stretched_beta"] = "stretched_beta"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate stretched beta parameters."""
        required = ["min", "max", "alpha", "beta"]
        if not all(key in params for key in required):
            raise ValueError(f"StretchedBeta requires parameters: {required}")

        if params["min"] >= params["max"]:
            raise ValueError("min must be < max")
        if params["alpha"] <= 0 or params["beta"] <= 0:
            raise ValueError("alpha and beta must be positive")

        return params

    @classmethod
    def elicit(
        cls,
        min: float,
        mode: float,
        max: float,
        concentration: float = 4.0,
        name: str | None = None,
    ) -> Self:
        """Elicit a stretched beta distribution from min/mode/max values."""
        if not (min < mode < max):
            raise ValueError("Must satisfy: min < mode < max")
        if concentration <= 0:
            raise ValueError("concentration must be positive")

        alpha = 1 + concentration * (mode - min) / (max - min)
        beta = 1 + concentration * (max - mode) / (max - min)

        return cls(
            name=name,
            params={
                "min": float(min),
                "max": float(max),
                "alpha": float(alpha),
                "beta": float(beta),
                "mode": float(mode),
                "concentration": float(concentration),
            },
            elicitation_params={
                "min": min,
                "mode": mode,
                "max": max,
                "concentration": concentration,
            },
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a stretched beta distribution to finite observations."""
        data_arr = np.asarray(data, dtype=float)
        if data_arr.size == 0:
            raise ValueError("StretchedBeta requires at least one observation")
        if not np.all(np.isfinite(data_arr)):
            raise ValueError("StretchedBeta requires finite data")

        alpha, beta, loc, scale = stats.beta.fit(data_arr)
        min_value = float(loc)
        max_value = float(loc + scale)
        params: dict[str, float] = {
            "min": min_value,
            "max": max_value,
            "alpha": float(alpha),
            "beta": float(beta),
        }

        concentration = float(alpha + beta - 2)
        if alpha > 1 and beta > 1 and concentration > 0:
            mode = min_value + (max_value - min_value) * (alpha - 1) / concentration
            params["mode"] = float(mode)
            params["concentration"] = concentration

        return cls(name=name, params=params)

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples."""
        return stats.beta.rvs(
            a=self.params["alpha"],
            b=self.params["beta"],
            loc=self.params["min"],
            scale=self.params["max"] - self.params["min"],
            size=size,
            random_state=get_scipy_random_state(seed),
        )

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability density function."""
        return stats.beta.pdf(
            x,
            a=self.params["alpha"],
            b=self.params["beta"],
            loc=self.params["min"],
            scale=self.params["max"] - self.params["min"],
        )

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return stats.beta.cdf(
            x,
            a=self.params["alpha"],
            b=self.params["beta"],
            loc=self.params["min"],
            scale=self.params["max"] - self.params["min"],
        )

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        return stats.beta.ppf(
            q,
            a=self.params["alpha"],
            b=self.params["beta"],
            loc=self.params["min"],
            scale=self.params["max"] - self.params["min"],
        )

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        return (float(self.params["min"]), float(self.params["max"]))

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        return self.support


class PERT(StretchedBeta):
    """PERT distribution as a stretched beta with concentration fixed at 4."""

    dist_type: Literal["pert"] = "pert"  # type: ignore[assignment]

    @classmethod
    def elicit(
        cls,
        min: float,
        mode: float,
        max: float,
        name: str | None = None,
    ) -> Self:
        """Elicit a PERT distribution from min/mode/max values."""
        return super().elicit(
            min=min,
            mode=mode,
            max=max,
            concentration=4.0,
            name=name,
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """
        Fit a PERT distribution using sample bounds and the PERT mean relation.

        The concentration is fixed at 4, so the sample minimum and maximum define
        the bounds and the sample mean determines the implied mode via
        ``mean = (min + 4 * mode + max) / 6``.
        """
        data_arr = np.asarray(data, dtype=float)
        if data_arr.size == 0:
            raise ValueError("PERT requires at least one observation")
        if not np.all(np.isfinite(data_arr)):
            raise ValueError("PERT requires finite data")

        min_value = float(np.min(data_arr))
        max_value = float(np.max(data_arr))
        if min_value >= max_value:
            raise ValueError("PERT requires data with nonzero range")

        mean = float(np.mean(data_arr))
        mode = (6 * mean - min_value - max_value) / 4
        mode = float(np.clip(mode, min_value + 1e-12, max_value - 1e-12))

        return cls.elicit(min=min_value, mode=mode, max=max_value, name=name)
