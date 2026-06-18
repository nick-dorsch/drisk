"""Univariate discrete distribution base classes."""

from abc import ABC
from typing import Any

import numpy as np

from drisk.distributions.univariate.base import UvDistribution
from drisk.summary import apply_percentile_yaxis


class UvDiscrete(UvDistribution, ABC):
    """Base class for univariate discrete distributions."""

    def pmf(self, x: float | np.ndarray) -> np.ndarray:
        """Probability mass function alias for :meth:`pdf`."""
        return self.pdf(x)

    def plot(
        self,
        ax: Any = None,
        *,
        show: bool = False,
        cdf_kwargs: dict[str, Any] | None = None,
        pmf_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot the CDF with low-alpha PMF bars on a secondary axis.

        Returns the primary Matplotlib ``Axes`` object. Importing Matplotlib is
        deferred so non-plotting use stays lightweight. Extra keyword arguments
        are passed to the CDF step line for convenient calls like
        ``dist.plot(color="steelblue")``.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        x_min, x_max = self.x_range
        x = np.arange(np.ceil(x_min), np.floor(x_max) + 1, dtype=int)
        lower, upper = self.support
        x = x[(x >= lower) & (x <= upper)]

        cdf = self.cdf(x)
        pmf = self.pmf(x)

        line_kwargs = {**(cdf_kwargs or {}), **kwargs}
        (cdf_line,) = ax.step(x, cdf, where="post", **line_kwargs)

        pmf_ax = ax.twinx()
        bar_kwargs = {
            "color": cdf_line.get_color(),
            "alpha": 0.2,
            "linewidth": 0,
            **(pmf_kwargs or {}),
        }
        pmf_ax.bar(x, pmf, width=0.8, **bar_kwargs)

        ax.set_xlabel(self.name or "x")
        apply_percentile_yaxis(ax)
        ax.set_title(self.name or self.dist_type)

        pmf_ax.set_ylim(bottom=0)
        pmf_ax.set_yticks([])
        pmf_ax.set_ylabel("")
        pmf_ax.spines["right"].set_visible(False)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax


class UvFiniteDiscrete(UvDiscrete, ABC):
    """Discrete distribution with finite support."""

    pass


class UvCountDiscrete(UvDiscrete, ABC):
    """Discrete distribution over non-negative integer counts."""

    pass
