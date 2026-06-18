"""Base interfaces for copula models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from drisk.correlations import CorrelationMatrix
from drisk.distributions import UvDistribution
from drisk.random import SeedLike


class Copula(BaseModel, ABC):
    """Base class for copulas that jointly sample marginal distributions."""

    distributions: Sequence[UvDistribution]
    corr_matrix: CorrelationMatrix

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def dims(self) -> int:
        """Number of marginal distributions."""
        return len(self.distributions)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Ensure the correlation matrix dimension matches the marginals."""
        n = len(self.distributions)
        matrix_n = len(self.corr_matrix.matrix)
        if matrix_n != n:
            raise ValueError(
                f"Correlation matrix size ({matrix_n}) does not match number of distributions ({n})."
            )
        return self

    @classmethod
    def from_distributions_and_correlation(
        cls,
        distributions: Sequence[UvDistribution],
        correlation: float,
        **kwargs: object,
    ) -> Self:
        """Create a copula from marginals and one shared pairwise correlation."""
        corr_matrix = CorrelationMatrix.from_n_corr(len(distributions), correlation)
        return cls(distributions=distributions, corr_matrix=corr_matrix, **kwargs)

    @abstractmethod
    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Jointly sample marginals, returning an array shaped ``(dims, *size)``."""
        pass

    def rvs(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Alias for :meth:`sample` for users familiar with SciPy naming."""
        return self.sample(size=size, seed=seed)
