from drisk.distributions import (
    UvBoundedContinuous,
    UvContinuous,
    UvPositiveContinuous,
    UvRealContinuous,
    UvUnitBoundedContinuous,
)
from drisk.distributions.univariate import UvDistribution


def test_uv_continuous_domain_hierarchy() -> None:
    assert issubclass(UvContinuous, UvDistribution)
    assert issubclass(UvRealContinuous, UvContinuous)
    assert issubclass(UvPositiveContinuous, UvContinuous)
    assert issubclass(UvBoundedContinuous, UvContinuous)
    assert issubclass(UvUnitBoundedContinuous, UvBoundedContinuous)
