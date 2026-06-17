"""Univariate continuous distribution base classes."""

from abc import ABC
from typing import Any

import numpy as np

from monte.distributions.univariate.base import UvDistribution


class UvContinuous(UvDistribution, ABC):
    """Base class for univariate continuous distributions."""

    def plot(
        self,
        ax: Any = None,
        *,
        n: int = 500,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Plot a quick probability-density look at the distribution.

        Returns the Matplotlib ``Axes`` object. Importing Matplotlib is deferred
        so non-plotting use stays lightweight.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        x_min, x_max = self.x_range
        x = np.linspace(x_min, x_max, n)
        ax.plot(x, self.pdf(x), **kwargs)
        ax.set_xlabel(self.name or "x")
        ax.set_ylabel("density")
        ax.set_title(self.name or self.dist_type)

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
