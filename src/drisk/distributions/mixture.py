"""Univariate mixture distribution."""

from typing import Any, Literal, Self

import numpy as np
from pydantic import SerializeAsAny, model_validator

from drisk.distributions.types import ArrayLike
from drisk.distributions.univariate import UvDistribution
from drisk.random import SeedLike, get_rng


class UvMixture(UvDistribution):
    """Weighted mixture of one or more univariate component distributions."""

    dist_type: Literal["uv_mixture"] = "uv_mixture"
    components: tuple[SerializeAsAny[UvDistribution], ...]
    weights: tuple[float, ...]
    params: dict[str, float | int] = {}

    @model_validator(mode="after")
    def validate_mixture(self) -> Self:
        """Validate component and weight shapes, then normalize weights."""
        if len(self.components) == 0:
            raise ValueError("UvMixture requires at least one component distribution")

        if len(self.components) != len(self.weights):
            raise ValueError(
                "UvMixture requires the same number of components and weights"
            )

        weights = np.asarray(self.weights, dtype=float)
        if not np.all(np.isfinite(weights)):
            raise ValueError("UvMixture weights must be finite")
        if np.any(weights < 0):
            raise ValueError("UvMixture weights must be non-negative")

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            raise ValueError("At least one UvMixture weight must be positive")

        self.weights = tuple(float(weight / weight_sum) for weight in weights)
        return self

    @classmethod
    def elicit(
        cls,
        components: list[UvDistribution] | tuple[UvDistribution, ...],
        weights: list[float] | tuple[float, ...],
        name: str | None = None,
    ) -> Self:
        """Elicit a mixture directly from component distributions and weights."""
        return cls(
            name=name,
            components=tuple(components),
            weights=tuple(weights),
            elicitation_params={"weights": tuple(weights)},
        )

    @classmethod
    def fit(cls, data: ArrayLike, **kwargs: Any) -> Self:
        """Fit is not implemented for generic univariate mixtures."""
        raise NotImplementedError("Generic UvMixture.fit is not implemented")

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate samples by first choosing a component, then sampling from it."""
        shape = (size,) if isinstance(size, int) else size
        sample_count = int(np.prod(shape, dtype=int))
        rng = get_rng(seed)

        component_indexes = rng.choice(
            len(self.components), size=sample_count, p=np.asarray(self.weights)
        )
        samples = np.empty(sample_count, dtype=float)

        for component_index, component in enumerate(self.components):
            mask = component_indexes == component_index
            component_count = int(np.sum(mask))
            if component_count == 0:
                continue
            samples[mask] = np.asarray(
                component.sample(size=component_count, seed=rng)
            ).reshape(component_count)

        return samples.reshape(shape)

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        """Weighted component probability density/mass function."""
        x_arr = np.asarray(x)
        values = np.zeros_like(x_arr, dtype=float)
        for weight, component in zip(self.weights, self.components, strict=True):
            values += weight * component.pdf(x_arr)
        return values

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        """Weighted component cumulative distribution function."""
        x_arr = np.asarray(x)
        values = np.zeros_like(x_arr, dtype=float)
        for weight, component in zip(self.weights, self.components, strict=True):
            values += weight * component.cdf(x_arr)
        return values

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Percent point function / inverse CDF is not implemented for mixtures."""
        raise NotImplementedError("UvMixture.ppf is not implemented")

    def plot(
        self,
        ax: Any = None,
        *,
        n: int = 500,
        show: bool = False,
        cdf_kwargs: dict[str, Any] | None = None,
        pdf_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the mixture CDF with a low-alpha PDF/PMF fill."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        x_min, x_max = self.x_range
        x = np.linspace(x_min, x_max, n)
        cdf = self.cdf(x)
        pdf = self.pdf(x)

        line_kwargs = {**(cdf_kwargs or {}), **kwargs}
        (cdf_line,) = ax.plot(x, cdf, **line_kwargs)

        pdf_ax = ax.twinx()
        fill_kwargs = {
            "color": cdf_line.get_color(),
            "alpha": 0.2,
            "linewidth": 0,
            **(pdf_kwargs or {}),
        }
        pdf_ax.fill_between(x, 0, pdf, **fill_kwargs)

        ax.set_xlabel(self.name or "x")
        ax.set_ylabel("cumulative probability")
        ax.set_ylim(bottom=0, top=1)
        ax.set_title(self.name or self.dist_type)

        pdf_ax.set_ylim(bottom=0)
        pdf_ax.set_yticks([])
        pdf_ax.set_ylabel("")
        pdf_ax.spines["right"].set_visible(False)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax

    @property
    def support(self) -> tuple[float, float]:
        """Union of component supports."""
        lowers, uppers = zip(
            *(component.support for component in self.components), strict=True
        )
        return (float(np.min(lowers)), float(np.max(uppers)))

    @property
    def x_range(self) -> tuple[float, float]:
        """Union of component plotting ranges."""
        lowers, uppers = zip(
            *(component.x_range for component in self.components), strict=True
        )
        return (float(np.min(lowers)), float(np.max(uppers)))
