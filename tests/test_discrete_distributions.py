import numpy as np
import pytest
from scipy import stats

from drisk.distributions import (
    Bernoulli,
    Binomial,
    Geometric,
    NegativeBinomial,
    Poisson,
    UvCountDiscrete,
    UvDiscrete,
    UvFiniteDiscrete,
)


def test_discrete_domain_hierarchy() -> None:
    assert issubclass(UvFiniteDiscrete, UvDiscrete)
    assert issubclass(UvCountDiscrete, UvDiscrete)
    assert isinstance(Bernoulli.elicit(0.5), UvFiniteDiscrete)
    assert isinstance(Binomial.elicit(10, 0.5), UvFiniteDiscrete)
    assert isinstance(Geometric.elicit(0.5), UvCountDiscrete)
    assert isinstance(NegativeBinomial.elicit(5, 0.5), UvCountDiscrete)
    assert isinstance(Poisson.elicit(3.0), UvCountDiscrete)


def test_bernoulli_elicit_sample_fit_and_functions() -> None:
    dist = Bernoulli.elicit(p=0.3)

    assert dist.dist_type == "bernoulli"
    assert dist.params == {"p": 0.3}
    assert dist.elicitation_params == {"p": 0.3}
    assert dist.support == (0.0, 1.0)
    assert dist.x_range == (-0.5, 1.5)
    assert dist.bounded is True
    assert dist.mean == pytest.approx(0.3)
    assert dist.variance == pytest.approx(0.21)

    samples = dist.sample(size=10, seed=1)
    assert samples.shape == (10,)
    assert np.all(np.isin(samples, [0, 1]))
    np.testing.assert_array_equal(dist.rvs(size=10, seed=1), samples)

    x = np.array([-1, 0, 1, 2])
    np.testing.assert_allclose(dist.pdf(x), stats.bernoulli.pmf(x, 0.3))
    np.testing.assert_allclose(dist.pmf(x), dist.pdf(x))
    assert dist.cdf(0) == pytest.approx(0.7)
    assert dist.ppf(1.0) == pytest.approx(1.0)

    fitted = Bernoulli.fit([0, 1, 1, 0, 1])
    assert fitted.params["p"] == pytest.approx(0.6)
    assert fitted.elicitation_params is None


def test_bernoulli_validation_and_serialization() -> None:
    with pytest.raises(ValueError, match="p must be in"):
        Bernoulli.elicit(-0.1)
    with pytest.raises(ValueError, match="only 0 and 1"):
        Bernoulli.fit([0, 1, 2])

    original = Bernoulli.elicit(p=0.7, name="success")
    restored = Bernoulli.model_validate_json(original.model_dump_json())
    assert restored.dist_type == "bernoulli"
    assert restored.name == "success"
    assert restored.params == {"p": 0.7}
    assert restored.elicitation_params == {"p": 0.7}


def test_binomial_elicit_sample_fit_and_functions() -> None:
    dist = Binomial.elicit(n=10, p=0.3)

    assert dist.dist_type == "binomial"
    assert dist.params == {"n": 10, "p": 0.3}
    assert dist.elicitation_params == {"n": 10, "p": 0.3}
    assert dist.support == (0.0, 10.0)
    assert dist.x_range == (-0.5, 10.5)
    assert dist.bounded is True
    assert dist.mean == pytest.approx(3.0)
    assert dist.variance == pytest.approx(2.1)

    samples = dist.sample(size=(4, 5), seed=1)
    assert samples.shape == (4, 5)
    assert np.all((samples >= 0) & (samples <= 10))
    np.testing.assert_array_equal(dist.rvs(size=(4, 5), seed=1), samples)

    x = np.array([-1, 0, 5, 10, 11])
    np.testing.assert_allclose(dist.pdf(x), stats.binom.pmf(x, 10, 0.3))
    np.testing.assert_allclose(dist.pmf(x), dist.pdf(x))
    assert dist.cdf(-1) == pytest.approx(0.0)
    assert dist.ppf(1.0) == pytest.approx(10.0)

    fitted = Binomial.fit([0, 5, 10], n=10)
    assert fitted.params == {"n": 10, "p": 0.5}


def test_binomial_validation_and_serialization() -> None:
    assert Binomial.elicit(n=10.0, p=0.5).params["n"] == 10

    with pytest.raises(ValueError, match="n must be positive"):
        Binomial.elicit(0, 0.5)
    with pytest.raises(ValueError, match="n must be integer"):
        Binomial.elicit(10.5, 0.5)
    with pytest.raises(ValueError, match="p must be in"):
        Binomial.elicit(10, 1.1)
    with pytest.raises(ValueError, match="integer-valued"):
        Binomial.fit([0, 1.5], n=10)
    with pytest.raises(ValueError, match="<= n"):
        Binomial.fit([0, 11], n=10)

    original = Binomial.elicit(n=10, p=0.7, name="wins")
    restored = Binomial.model_validate_json(original.model_dump_json())
    assert restored.dist_type == "binomial"
    assert restored.name == "wins"
    assert restored.params == {"n": 10, "p": 0.7}
    assert restored.elicitation_params == {"n": 10, "p": 0.7}


def test_geometric_elicit_sample_fit_and_functions() -> None:
    dist = Geometric.elicit(p=0.3)

    assert dist.dist_type == "geometric"
    assert dist.params == {"p": 0.3}
    assert dist.elicitation_params == {"p": 0.3}
    assert dist.support == (0.0, np.inf)
    assert dist.bounded is False
    assert dist.mean == pytest.approx((1 - 0.3) / 0.3)
    assert dist.variance == pytest.approx((1 - 0.3) / 0.3**2)
    assert dist.x_range[0] == -0.5
    assert dist.x_range[1] > 10.0

    samples = dist.sample(size=(4, 5), seed=1)
    assert samples.shape == (4, 5)
    assert np.all(samples >= 0)
    assert np.all(samples == np.floor(samples))
    np.testing.assert_array_equal(dist.rvs(size=(4, 5), seed=1), samples)

    x = np.array([-1, 0, 1, 2, 5])
    np.testing.assert_allclose(dist.pdf(x), stats.geom.pmf(x + 1, 0.3))
    np.testing.assert_allclose(dist.pmf(x), dist.pdf(x))
    assert dist.pdf(0) == pytest.approx(0.3)
    assert dist.cdf(-1) == pytest.approx(0.0)
    assert dist.cdf(0) == pytest.approx(0.3)
    assert dist.ppf(0.0) == pytest.approx(-1.0)

    fitted = Geometric.fit([0, 1, 2, 3, 4])
    assert fitted.params == {"p": pytest.approx(1 / 3)}
    assert fitted.elicitation_params is None


def test_geometric_validation_and_serialization() -> None:
    certain = Geometric.elicit(p=1.0)
    assert np.all(certain.sample(size=10, seed=1) == 0)

    with pytest.raises(ValueError, match="p must be in"):
        Geometric.elicit(0.0)
    with pytest.raises(ValueError, match="non-negative"):
        Geometric.fit([-1, 0, 1])
    with pytest.raises(ValueError, match="integer-valued"):
        Geometric.fit([0, 1.5])

    original = Geometric.elicit(p=0.7, name="attempts")
    restored = Geometric.model_validate_json(original.model_dump_json())
    assert restored.dist_type == "geometric"
    assert restored.name == "attempts"
    assert restored.params == {"p": 0.7}
    assert restored.elicitation_params == {"p": 0.7}


def test_negative_binomial_elicit_sample_fit_and_functions() -> None:
    dist = NegativeBinomial.elicit(r=5, p=0.3)

    assert dist.dist_type == "negative_binomial"
    assert dist.params == {"r": 5, "p": 0.3}
    assert dist.elicitation_params == {"r": 5, "p": 0.3}
    assert dist.support == (0.0, np.inf)
    assert dist.bounded is False
    assert dist.mean == pytest.approx(5 * (1 - 0.3) / 0.3)
    assert dist.variance == pytest.approx(5 * (1 - 0.3) / 0.3**2)
    assert dist.x_range[0] == -0.5
    assert dist.x_range[1] > 10.0

    samples = dist.sample(size=(4, 5), seed=1)
    assert samples.shape == (4, 5)
    assert np.all(samples >= 0)
    assert np.all(samples == np.floor(samples))
    np.testing.assert_array_equal(dist.rvs(size=(4, 5), seed=1), samples)

    x = np.array([-1, 0, 1, 3, 5])
    np.testing.assert_allclose(dist.pdf(x), stats.nbinom.pmf(x, 5, 0.3))
    np.testing.assert_allclose(dist.pmf(x), dist.pdf(x))
    assert dist.cdf(-1) == pytest.approx(0.0)
    assert dist.ppf(0.0) == pytest.approx(-1.0)

    fitted = NegativeBinomial.fit([2, 4, 7, 9, 12])
    assert fitted.params["r"] >= 1
    assert 0 < fitted.params["p"] <= 1
    assert fitted.elicitation_params is None


def test_negative_binomial_validation_and_serialization() -> None:
    assert NegativeBinomial.elicit(r=10.0, p=0.5).params["r"] == 10

    certain = NegativeBinomial.elicit(r=5, p=1.0)
    assert np.all(certain.sample(size=10, seed=1) == 0)

    with pytest.raises(ValueError, match="r must be positive"):
        NegativeBinomial.elicit(0, 0.5)
    with pytest.raises(ValueError, match="r must be integer"):
        NegativeBinomial.elicit(5.5, 0.5)
    with pytest.raises(ValueError, match="p must be in"):
        NegativeBinomial.elicit(5, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        NegativeBinomial.fit([-1, 0, 1])
    with pytest.raises(ValueError, match="integer-valued"):
        NegativeBinomial.fit([0, 1.5])

    original = NegativeBinomial.elicit(r=5, p=0.7, name="failures")
    restored = NegativeBinomial.model_validate_json(original.model_dump_json())
    assert restored.dist_type == "negative_binomial"
    assert restored.name == "failures"
    assert restored.params == {"r": 5, "p": 0.7}
    assert restored.elicitation_params == {"r": 5, "p": 0.7}


def test_poisson_elicit_sample_fit_and_functions() -> None:
    dist = Poisson.elicit(lam=3.0)

    assert dist.dist_type == "poisson"
    assert dist.params == {"lam": 3.0}
    assert dist.elicitation_params == {"lam": 3.0}
    assert dist.support == (0.0, np.inf)
    assert dist.bounded is False
    assert dist.mean == pytest.approx(3.0)
    assert dist.variance == pytest.approx(3.0)
    assert dist.x_range[0] == -0.5
    assert dist.x_range[1] > 3.0

    samples = dist.sample(size=10, seed=1)
    assert samples.shape == (10,)
    assert np.all(samples >= 0)
    assert np.all(samples == np.floor(samples))
    np.testing.assert_array_equal(dist.rvs(size=10, seed=1), samples)

    x = np.array([-1, 0, 1, 3, 5])
    np.testing.assert_allclose(dist.pdf(x), stats.poisson.pmf(x, 3.0))
    np.testing.assert_allclose(dist.pmf(x), dist.pdf(x))
    assert dist.cdf(-1) == pytest.approx(0.0)
    assert dist.ppf(0.0) == pytest.approx(-1.0)

    fitted = Poisson.fit([0, 1, 2, 3, 4])
    assert fitted.params == {"lam": 2.0}


def test_poisson_validation_and_serialization() -> None:
    zero = Poisson.elicit(lam=0.0)
    assert np.all(zero.sample(size=10, seed=1) == 0)

    with pytest.raises(ValueError, match="lam must be non-negative"):
        Poisson.elicit(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        Poisson.fit([-1, 0, 1])
    with pytest.raises(ValueError, match="integer-valued"):
        Poisson.fit([0, 1.5])

    original = Poisson.elicit(lam=3.7, name="events")
    restored = Poisson.model_validate_json(original.model_dump_json())
    assert restored.dist_type == "poisson"
    assert restored.name == "events"
    assert restored.params == {"lam": 3.7}
    assert restored.elicitation_params == {"lam": 3.7}


def test_discrete_plot_returns_axes() -> None:
    import matplotlib.pyplot as plt

    dist = Poisson.elicit(lam=3.0)
    fig, ax = plt.subplots()

    returned_ax = dist.plot(ax=ax)

    assert returned_ax is ax
    assert ax.get_ylim() == pytest.approx((0, 1))
    np.testing.assert_allclose(ax.get_yticks(), [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "p99",
        "p90",
        "p75",
        "p50",
        "p25",
        "p10",
        "p1",
    ]
    assert ax.get_ylabel() == ""

    plt.close(fig)
