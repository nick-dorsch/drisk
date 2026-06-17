import numpy as np
import pytest

from monte.distributions import (
    LogitNormal,
    LogNormal,
    Normal,
    UvPositiveContinuous,
    UvRealContinuous,
    UvUnitBoundedContinuous,
)


def test_normal_elicit_sample_and_fit() -> None:
    dist = Normal.elicit(lower=10, upper=20, confidence=0.8)

    assert isinstance(dist, UvRealContinuous)
    assert dist.elicitation_params == {"lower": 10, "upper": 20, "confidence": 0.8}
    assert dist.support == (-np.inf, np.inf)
    assert dist.sample(size=4, seed=1).shape == (4,)
    np.testing.assert_array_equal(dist.rvs(size=4, seed=1), dist.sample(size=4, seed=1))

    fitted = Normal.fit([1.0, 2.0, 3.0])
    assert fitted.params["mu"] == pytest.approx(2.0)


def test_lognormal_elicit_sample_and_fit() -> None:
    dist = LogNormal.elicit(lower=10, upper=100, confidence=0.8)

    assert isinstance(dist, UvPositiveContinuous)
    assert dist.support == (0.0, np.inf)
    assert np.all(dist.sample(size=10, seed=1) > 0)

    fitted = LogNormal.fit([1.0, 2.0, 4.0])
    assert fitted.params["sigma"] > 0

    with pytest.raises(ValueError, match="positive data"):
        LogNormal.fit([0.0, 1.0])


def test_logitnormal_elicit_sample_and_fit() -> None:
    dist = LogitNormal.elicit(lower=0.15, upper=0.25, confidence=0.8)

    assert isinstance(dist, UvUnitBoundedContinuous)
    assert dist.support == (0.0, 1.0)
    samples = dist.sample(size=10, seed=1)
    assert np.all((samples > 0) & (samples < 1))
    assert dist.pdf(np.array([-1.0, 0.5, 2.0])).tolist()[0] == 0.0
    assert dist.cdf(np.array([-1.0, 0.5, 2.0])).tolist()[2] == 1.0

    fitted = LogitNormal.fit([0.2, 0.4, 0.6])
    assert fitted.params["sigma"] > 0

    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        LogitNormal.fit([0.0, 0.5])
