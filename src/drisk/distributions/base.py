"""Base interfaces for probability distributions."""

from abc import ABC, abstractmethod
from inspect import isabstract
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from drisk.arithmetic import ArithmeticMixin
from drisk.distributions.types import ArrayLike
from drisk.random import SeedLike


class Distribution(ArithmeticMixin, BaseModel, ABC):
    """
    Top-level abstract base class for probability distributions.

    Combines Pydantic models for validation/serialization with abstract methods
    for the sampling interface used by Drisk models.
    """

    dist_type: str
    name: str | None = None
    elicitation_params: dict[str, Any] | None = None

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Use ``dist_type`` to validate abstract distribution-typed fields.

        Concrete distribution classes keep Pydantic's normal model schema. Abstract
        distribution/domain classes behave like discriminated unions over the
        registered concrete distribution classes that subclass them, enabling
        annotations such as ``UvUnitBoundedContinuous`` to round-trip through JSON.
        """
        if not getattr(cls, "__pydantic_complete__", False) or not isabstract(cls):
            return handler(source_type)

        try:
            from drisk.distributions.registry import concrete_distribution_types_for
        except ImportError:
            return handler(source_type)

        try:
            choices = {
                distribution_cls.model_fields[
                    "dist_type"
                ].default: handler.generate_schema(distribution_cls)
                for distribution_cls in concrete_distribution_types_for(cls)
            }
        except ImportError:
            return handler(source_type)

        if not choices:
            return handler(source_type)

        return core_schema.tagged_union_schema(
            choices=choices,
            discriminator="dist_type",
            from_attributes=True,
        )

    @abstractmethod
    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Generate random samples from the distribution."""
        pass

    def rvs(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        """Alias for :meth:`sample` for users familiar with SciPy naming."""
        return self.sample(size=size, seed=seed)

    @classmethod
    @abstractmethod
    def elicit(cls, **kwargs: Any) -> Self:
        """
        Construct a distribution from elicited parameters.

        Implementations should store the elicitation inputs on the returned
        object's ``elicitation_params`` attribute.
        """
        pass

    @classmethod
    @abstractmethod
    def fit(cls, data: ArrayLike, **kwargs: Any) -> Self:
        """Fit a distribution to observed data."""
        pass

    @abstractmethod
    def plot(self, **kwargs: Any) -> Any:
        """Create a quicklook plot for the distribution."""
        pass
