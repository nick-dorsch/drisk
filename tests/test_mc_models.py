import numpy as np
import pytest

import drisk as dr
from drisk.models import MCModel


def test_distribution_arithmetic_returns_mc_model() -> None:
    x = dr.Normal.elicit(-1.0, 1.0)
    y = dr.Normal.elicit(9.0, 11.0)

    model = x + y * 2 - 3

    assert isinstance(model, MCModel)
    assert model.sample(size=5, seed=123).shape == (5,)


def test_model_arithmetic_matches_numpy_with_shared_rng() -> None:
    x = dr.Normal.elicit(-1.0, 1.0)
    y = dr.LogNormal.elicit(8.0, 12.0)
    model = x * y + 2

    rng = np.random.default_rng(123)
    expected = x.sample(size=10, seed=rng) * y.sample(size=10, seed=rng) + 2

    np.testing.assert_allclose(model.sample(size=10, seed=123), expected)


def test_reused_distribution_object_uses_same_samples() -> None:
    x = dr.Normal.elicit(-1.0, 1.0)

    doubled = (x + x).sample(size=10, seed=123)
    expected = (2 * x).sample(size=10, seed=123)

    np.testing.assert_allclose(doubled, expected)


def test_model_reuses_cached_samples_across_report_methods() -> None:
    x = dr.Normal.elicit(-1.0, 1.0)
    model = x * 2

    samples = model.sample(size=1_000, seed=123)
    summary = model.summary(threshold=0)

    assert summary.loc["value", "mean"] == pytest.approx(float(np.mean(samples)))
    assert summary.loc["value", "p(> 0)"] == pytest.approx(float(np.mean(samples > 0)))


def test_model_distributions_returns_unique_leaves_in_expression_order() -> None:
    x = dr.Normal.elicit(-1.0, 1.0, name="x")
    y = dr.Normal.elicit(9.0, 11.0, name="y")

    model = x + y * x

    assert model.distributions() == (x, y)


def test_add_copula_uses_joint_samples_for_model_evaluation() -> None:
    x = dr.Normal(params={"mu": 0.0, "sigma": 1.0}, name="x")
    y = dr.Normal(params={"mu": 10.0, "sigma": 2.0}, name="y")
    model = x + y
    copula = dr.GaussianCopula.from_distributions_and_correlation([x, y], 0.75)

    returned = model.add_copula(copula)
    samples = model.sample(size=1_000, seed=123)
    joint_samples = copula.sample(size=1_000, seed=123)
    expected = joint_samples[0] + joint_samples[1]

    assert returned is model
    np.testing.assert_allclose(samples, expected)
    assert np.corrcoef(joint_samples)[0, 1] == pytest.approx(0.75, abs=0.08)


def test_add_copula_clears_cached_independent_samples() -> None:
    x = dr.Normal(params={"mu": 0.0, "sigma": 1.0})
    y = dr.Normal(params={"mu": 0.0, "sigma": 1.0})
    model = x + y

    independent = model.sample(size=100, seed=123)
    copula = dr.GaussianCopula.from_distributions_and_correlation([x, y], 0.5)
    correlated = model.add_copula(copula).sample(size=100, seed=123)

    assert not np.array_equal(independent, correlated)


def test_correlate_with_scalar_builds_gaussian_copula_in_distribution_order() -> None:
    x = dr.Normal(params={"mu": 0.0, "sigma": 1.0}, name="x")
    y = dr.Normal(params={"mu": 10.0, "sigma": 2.0}, name="y")
    z = dr.Normal(params={"mu": 20.0, "sigma": 3.0}, name="z")
    model = x + y * z

    returned = model.correlate(0.4)

    assert returned is model
    assert isinstance(model.copula, dr.GaussianCopula)
    assert tuple(model.copula.distributions) == (x, y, z)
    np.testing.assert_allclose(
        model.copula.corr_matrix.to_numpy(),
        np.array([[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]]),
    )


def test_correlate_with_matrix_builds_gaussian_copula() -> None:
    x = dr.Normal(params={"mu": 0.0, "sigma": 1.0}, name="x")
    y = dr.Normal(params={"mu": 10.0, "sigma": 2.0}, name="y")
    model = x + y
    matrix = [[1.0, 0.7], [0.7, 1.0]]

    model.correlate(matrix)
    samples = model.sample(size=1_000, seed=123)
    explicit_copula = dr.GaussianCopula(
        distributions=[x, y],
        corr_matrix=dr.CorrelationMatrix(matrix=matrix),
    )
    explicit_samples = explicit_copula.sample(size=1_000, seed=123)

    np.testing.assert_allclose(samples, explicit_samples[0] + explicit_samples[1])


def test_model_refresh_replaces_cached_samples() -> None:
    x = dr.Normal.elicit(-1.0, 1.0)
    model = x * 2

    first = model.sample(size=1_000, seed=123)
    refreshed = model.sample(size=1_000, seed=456, refresh=True)

    assert not np.array_equal(first, refreshed)
    np.testing.assert_array_equal(model.sample(size=1_000), refreshed)


def test_model_summary_returns_configurable_dataframe() -> None:
    x = dr.Normal.elicit(-1.0, 1.0)
    model = x * 2

    summary = model.summary(
        size=1_000,
        seed=123,
        threshold=0,
        percentiles=(99, 90, 75, 50, 25, 10, 1),
    )

    assert "value" in summary.index
    assert {"mean", "p(> 0)", "p99", "p50", "p1"}.issubset(summary.columns)
    assert summary.loc["value", "p(> 0)"] == pytest.approx(
        np.mean(model.sample(size=1_000, seed=123) > 0)
    )


def test_model_plot_returns_axes_and_uses_default_x_quantile_range() -> None:
    import matplotlib.pyplot as plt

    x = dr.Normal.elicit(-1.0, 1.0)
    model = x * 2
    fig, ax = plt.subplots()

    returned_ax = model.plot(ax=ax, size=100, seed=123)

    assert returned_ax is ax
    samples = np.ravel(model.sample(size=100, seed=123))
    assert ax.get_xlim() == pytest.approx(tuple(np.quantile(samples, (0.001, 0.999))))

    plt.close(fig)


def test_model_plot_accepts_custom_x_quantile_range() -> None:
    import matplotlib.pyplot as plt

    x = dr.Normal.elicit(0.0, 1.0)
    model = x + 1
    fig, ax = plt.subplots()

    model.plot(ax=ax, size=100, seed=123, x_quantile_range=(0.05, 0.95))

    samples = np.ravel(model.sample(size=100, seed=123))
    assert ax.get_xlim() == pytest.approx(tuple(np.quantile(samples, (0.05, 0.95))))

    plt.close(fig)


def test_model_plot_can_show_full_sampled_x_range() -> None:
    import matplotlib.pyplot as plt

    x = dr.Normal.elicit(0.0, 1.0)
    model = x + 1
    fig, ax = plt.subplots()

    model.plot(ax=ax, size=100, seed=123, x_quantile_range=None)

    samples = np.ravel(model.sample(size=100, seed=123))
    x_min, x_max = ax.get_xlim()
    assert x_min < np.min(samples)
    assert x_max > np.max(samples)

    plt.close(fig)


def test_reverse_and_unary_operations() -> None:
    x = dr.Normal.elicit(1.0, 3.0)

    rng = np.random.default_rng(123)
    x_samples = x.sample(size=8, seed=rng)
    expected = abs(-((100 / x_samples) ** 2))

    model = abs(-((100 / x) ** 2))

    np.testing.assert_allclose(model.sample(size=8, seed=123), expected)
