"""Gaussian copula."""

from typing import Literal

import numpy as np
from scipy import stats

from drisk.random import SeedLike, get_rng

from .base import Copula


class GaussianCopula(Copula):
    """Sample marginal distributions with dependence induced by a Gaussian copula."""

    copula_type: Literal["gaussian"] = "gaussian"

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Jointly sample marginals, returning an array shaped ``(dims, *size)``."""
        if isinstance(size, int):
            size = (size,)

        rng = get_rng(seed)
        normal_samples = rng.multivariate_normal(
            mean=np.zeros(self.dims),
            cov=self.corr_matrix.to_numpy(),
            size=size,
        )
        normal_samples = np.moveaxis(normal_samples, -1, 0)
        uniform_samples = stats.norm.cdf(normal_samples)

        samples = np.empty_like(uniform_samples)
        for i, dist in enumerate(self.distributions):
            samples[i, ...] = dist.ppf(uniform_samples[i, ...])

        return samples
