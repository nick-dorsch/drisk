"""Base interfaces for copula models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from inspect import isabstract
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema

from drisk.correlations import CorrelationMatrix
from drisk.distributions.univariate import UvDistribution
from drisk.random import SeedLike


class Copula(BaseModel, ABC):
    """Base class for copulas that jointly sample marginal distributions."""

    distributions: tuple[UvDistribution, ...]
    corr_matrix: CorrelationMatrix

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Use ``copula_type`` to validate abstract copula-typed fields."""
        if not getattr(cls, "__pydantic_complete__", False) or not isabstract(cls):
            return handler(source_type)

        try:
            from drisk.copulas.registry import concrete_copula_types_for
        except ImportError:
            return handler(source_type)

        try:
            choices = {
                copula_cls.model_fields["copula_type"].default: handler.generate_schema(
                    copula_cls
                )
                for copula_cls in concrete_copula_types_for(cls)
            }
        except ImportError:
            return handler(source_type)

        if not choices:
            return handler(source_type)

        return core_schema.tagged_union_schema(
            choices=choices,
            discriminator="copula_type",
            from_attributes=True,
        )

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
