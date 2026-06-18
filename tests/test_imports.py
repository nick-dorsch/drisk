import drisk
from drisk import (
    Copula,
    CorrelationMatrix,
    Exponential,
    Gamma,
    GaussianCopula,
    StudentTCopula,
)
from drisk.distributions import Distribution, UvContinuous, UvDistribution


def test_package_imports() -> None:
    assert drisk.__name__ == "drisk"
    assert Distribution is not None
    assert Exponential is drisk.Exponential
    assert UvDistribution is not None
    assert UvContinuous is not None
    assert Copula is not None
    assert CorrelationMatrix is not None
    assert GaussianCopula is not None
    assert Gamma is drisk.Gamma
    assert StudentTCopula is not None
