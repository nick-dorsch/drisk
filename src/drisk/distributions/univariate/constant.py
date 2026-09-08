"""Constant / degenerate univariate distribution."""

from typing import Any, Literal, Self

import numpy as np
from pydantic import field_validator

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate.base import UvDistribution
from drisk.random import SeedLike
from drisk.summary import DEFAULT_PERCENTILES, apply_percentile_yaxis


class Constant(UvDistribution):
    """Degenerate distribution that always returns the same numeric value."""

    dist_type: Literal["constant"] = "constant"

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, float | int]) -> dict[str, float | int]:
        """Validate the ``value`` parameter."""
        if "value" not in params:
            raise ValueError("Constant requires 'value' parameter")

        value = params["value"]
        if not np.isfinite(value):
            raise ValueError(f"value must be finite, got {value}")

        return params

    @classmethod
    def elicit(cls, value: float | int, name: str | None = None) -> Self:
        """Elicit a constant distribution from its fixed value."""
        return cls(
            name=name,
            params={"value": value},
            elicitation_params={"value": value},
        )

    @classmethod
    def fit(cls, data: ArrayLike, name: str | None = None) -> Self:
        """Fit a constant distribution to identical observations."""
        data_arr = np.asarray(data)
        if data_arr.size == 0:
            raise ValueError("Constant requires at least one observation")

        first = data_arr.reshape(-1)[0]
        if not np.all(data_arr == first):
            raise ValueError("Constant data must contain identical values")

        return cls(name=name, params={"value": first.item()})

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate samples, all equal to the fixed value."""
        del seed
        return np.full(size, self.params["value"])

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Point-mass indicator: 1 at the fixed value and 0 elsewhere."""
        return np.where(np.asarray(x) == self.params["value"], 1.0, 0.0)

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return np.where(np.asarray(x) < self.params["value"], 0.0, 1.0)

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF."""
        q_arr = np.asarray(q)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("q must be in [0, 1]")
        return np.full_like(q_arr, self.params["value"], dtype=float)

    def plot(
        self,
        ax: Any = None,
        *,
        show: bool = False,
        cdf_kwargs: dict[str, Any] | None = None,
        marker_kwargs: dict[str, Any] | None = None,
        percentiles: list[float | int] | tuple[float | int, ...] = DEFAULT_PERCENTILES,
        **kwargs: Any,
    ) -> Any:
        """Plot the step CDF and mark the point mass."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        value = self.params["value"]
        x_min, x_max = self.x_range

        line_kwargs = {**(cdf_kwargs or {}), **kwargs}
        (cdf_line,) = ax.step(
            [x_min, value, x_max],
            [0.0, 1.0, 1.0],
            where="post",
            **line_kwargs,
        )

        point_kwargs = {
            "color": cdf_line.get_color(),
            "alpha": 0.4,
            "linewidth": 2,
            **(marker_kwargs or {}),
        }
        ax.axvline(value, ymin=0, ymax=1, **point_kwargs)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel(self.name or "x")
        apply_percentile_yaxis(ax, percentiles)
        ax.set_title(self.name or self.dist_type)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax

    @property
    def support(self) -> tuple[float, float]:
        """Distribution support."""
        value = float(self.params["value"])
        return (value, value)

    @property
    def x_range(self) -> tuple[float, float]:
        """Practical plotting range."""
        value = float(self.params["value"])
        margin = max(1.0, abs(value) * 0.1)
        return (value - margin, value + margin)

    @property
    def mean(self) -> float:
        """Expected value."""
        return float(self.params["value"])

    @property
    def variance(self) -> float:
        """Variance."""
        return 0.0
