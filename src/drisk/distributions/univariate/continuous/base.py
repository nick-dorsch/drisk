"""Univariate continuous distribution base classes."""

from abc import ABC
from typing import Any

import numpy as np

from drisk.distributions.univariate.base import UvDistribution
from drisk.summary import apply_percentile_yaxis


class UvContinuous(UvDistribution, ABC):
    """Base class for univariate continuous distributions."""

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
        """
        Plot the CDF with a low-alpha PDF fill on a secondary axis.

        Returns the primary Matplotlib ``Axes`` object. Importing Matplotlib is
        deferred so non-plotting use stays lightweight. Extra keyword arguments
        are passed to the CDF line for convenient calls like
        ``dist.plot(color="steelblue")``.
        """
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
        apply_percentile_yaxis(ax)
        ax.set_title(self.name or self.dist_type)

        pdf_ax.set_ylim(bottom=0)
        pdf_ax.set_yticks([])
        pdf_ax.set_ylabel("")
        pdf_ax.spines["right"].set_visible(False)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax


class UvRealContinuous(UvContinuous, ABC):
    """Continuous distribution with support over all real numbers."""

    pass


class UvPositiveContinuous(UvContinuous, ABC):
    """Continuous distribution with positive support."""

    pass


class UvBoundedContinuous(UvContinuous, ABC):
    """Continuous distribution with finite lower and upper support."""

    pass


class UvUnitBoundedContinuous(UvBoundedContinuous, ABC):
    """Continuous distribution with support on the unit interval."""

    pass
