"""Student-t copula."""

from typing import Literal

import numpy as np
from pydantic import field_validator
from scipy import stats

from drisk.random import SeedLike, get_rng

from .base import Copula


class StudentTCopula(Copula):
    """Sample marginals with dependence induced by a Student-t copula."""

    copula_type: Literal["student_t"] = "student_t"
    nu: float = 4.0

    @field_validator("nu")
    @classmethod
    def validate_nu(cls, nu: float) -> float:
        """Validate degrees of freedom."""
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        return nu

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

        chi_square_samples = rng.chisquare(df=self.nu, size=size) / self.nu
        t_samples = normal_samples / np.sqrt(chi_square_samples)
        uniform_samples = stats.t.cdf(t_samples, df=self.nu)

        samples = np.empty_like(uniform_samples)
        for i, dist in enumerate(self.distributions):
            samples[i, ...] = dist.ppf(uniform_samples[i, ...])

        return samples
