"""Registry of concrete distribution implementations for Pydantic polymorphism."""

from functools import cache
from typing import cast

from drisk.distributions.base import Distribution


@cache
def concrete_distribution_types() -> tuple[type[Distribution], ...]:
    """Return all concrete distribution classes supported by Drisk."""
    from drisk.distributions.mixture import UvMixture
    from drisk.distributions.univariate import Constant
    from drisk.distributions.univariate.continuous import (
        PERT,
        Beta,
        Exponential,
        Gamma,
        LogitNormal,
        LogNormal,
        Normal,
        StretchedBeta,
    )
    from drisk.distributions.univariate.discrete import (
        Bernoulli,
        Binomial,
        Geometric,
        NegativeBinomial,
        Poisson,
    )

    return (
        Constant,
        Normal,
        LogNormal,
        LogitNormal,
        Exponential,
        Gamma,
        Beta,
        StretchedBeta,
        PERT,
        Bernoulli,
        Binomial,
        Geometric,
        NegativeBinomial,
        Poisson,
        UvMixture,
    )


def concrete_distribution_types_for[DistributionT: Distribution](
    base_cls: type[DistributionT],
) -> tuple[type[DistributionT], ...]:
    """Return concrete registered distributions that are subclasses of ``base_cls``."""
    return tuple(
        cast(type[DistributionT], distribution_cls)
        for distribution_cls in concrete_distribution_types()
        if issubclass(distribution_cls, base_cls)
    )
