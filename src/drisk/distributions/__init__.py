"""Probability distribution interfaces and implementations."""

from .base import Distribution
from .mixture import UvMixture
from .types import ArrayLike, DataFrameLike
from .univariate import (
    PERT,
    Bernoulli,
    Beta,
    Binomial,
    Constant,
    Exponential,
    Gamma,
    Geometric,
    LogitNormal,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StretchedBeta,
    UvBoundedContinuous,
    UvContinuous,
    UvCountDiscrete,
    UvDiscrete,
    UvDistribution,
    UvFiniteDiscrete,
    UvPositiveContinuous,
    UvRealContinuous,
    UvUnitBoundedContinuous,
)

# ``UvMixture.components`` is typed as ``UvDistribution``. During initial class
# construction the concrete distribution registry is not complete, so rebuild the
# model once all distribution implementations are imported.
UvMixture.model_rebuild(force=True)

__all__ = [
    "ArrayLike",
    "Bernoulli",
    "Beta",
    "Binomial",
    "Constant",
    "DataFrameLike",
    "Gamma",
    "Geometric",
    "Distribution",
    "Exponential",
    "LogitNormal",
    "LogNormal",
    "UvMixture",
    "NegativeBinomial",
    "Normal",
    "PERT",
    "Poisson",
    "StretchedBeta",
    "UvBoundedContinuous",
    "UvContinuous",
    "UvCountDiscrete",
    "UvDiscrete",
    "UvDistribution",
    "UvFiniteDiscrete",
    "UvPositiveContinuous",
    "UvRealContinuous",
    "UvUnitBoundedContinuous",
]
