"""Registry of concrete copula implementations for Pydantic polymorphism."""

from functools import cache
from typing import cast

from drisk.copulas.base import Copula


@cache
def concrete_copula_types() -> tuple[type[Copula], ...]:
    """Return all concrete copula classes supported by Drisk."""
    from drisk.copulas.gaussian import GaussianCopula
    from drisk.copulas.student_t import StudentTCopula

    return (GaussianCopula, StudentTCopula)


def concrete_copula_types_for[CopulaT: Copula](
    base_cls: type[CopulaT],
) -> tuple[type[CopulaT], ...]:
    """Return concrete registered copulas that are subclasses of ``base_cls``."""
    return tuple(
        cast(type[CopulaT], copula_cls)
        for copula_cls in concrete_copula_types()
        if issubclass(copula_cls, base_cls)
    )
