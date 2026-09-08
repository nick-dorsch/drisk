import numpy as np
import pytest
from pydantic import BaseModel

import drisk as dr
from drisk.distributions import Constant, UvDistribution


class UvDistributionHolder(BaseModel):
    value: UvDistribution


def test_constant_elicit_sample_fit_and_functions() -> None:
    dist = Constant.elicit(3.5, name="fixed")

    assert dist.dist_type == "constant"
    assert dist.name == "fixed"
    assert dist.params == {"value": 3.5}
    assert dist.elicitation_params == {"value": 3.5}
    assert dist.support == (3.5, 3.5)
    assert dist.bounded is True
    assert dist.mean == pytest.approx(3.5)
    assert dist.variance == pytest.approx(0.0)
    assert dist.stdev == pytest.approx(0.0)
    assert dist.x_range[0] < 3.5 < dist.x_range[1]

    samples = dist.sample(size=(2, 3), seed=1)
    assert samples.shape == (2, 3)
    np.testing.assert_array_equal(samples, np.full((2, 3), 3.5))
    np.testing.assert_array_equal(dist.rvs(size=(2, 3), seed=2), samples)

    x = np.array([3.0, 3.5, 4.0])
    np.testing.assert_array_equal(dist.pdf(x), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(dist.cdf(x), np.array([0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(dist.ppf(np.array([0.0, 0.5, 1.0])), np.full(3, 3.5))

    fitted = Constant.fit([7, 7, 7])
    assert fitted.params == {"value": 7}
    assert fitted.elicitation_params is None


def test_constant_validation_and_serialization() -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        Constant.elicit(np.inf)
    with pytest.raises(ValueError, match="at least one observation"):
        Constant.fit([])
    with pytest.raises(ValueError, match="identical values"):
        Constant.fit([1, 1, 2])
    with pytest.raises(ValueError, match="q must be in"):
        Constant.elicit(1).ppf(1.1)

    original = Constant.elicit(4.2, name="answer")
    restored = Constant.model_validate_json(original.model_dump_json())
    assert restored.dist_type == "constant"
    assert restored.name == "answer"
    assert restored.params == {"value": 4.2}
    assert restored.elicitation_params == {"value": 4.2}


def test_constant_public_imports_and_polymorphic_round_trip() -> None:
    dist = dr.Constant.elicit(5)
    assert Constant is dr.Constant
    assert isinstance(dist, UvDistribution)

    holder = UvDistributionHolder(value=dist)
    restored = UvDistributionHolder.model_validate_json(holder.model_dump_json())

    assert type(restored.value) is Constant
    assert restored.model_dump(mode="json") == holder.model_dump(mode="json")


def test_constant_plot_returns_axes() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    returned_ax = Constant.elicit(2).plot(ax=ax)

    assert returned_ax is ax
    assert ax.get_ylim() == pytest.approx((0, 1))
    assert ax.get_title() == "constant"

    plt.close(fig)
