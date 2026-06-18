import numpy as np
import pytest

from drisk import CorrelationMatrix, GaussianCopula, Normal, StudentTCopula


def test_gaussian_copula_sample_shape_and_rvs_alias() -> None:
    distributions = [
        Normal(params={"mu": 0.0, "sigma": 1.0}),
        Normal(params={"mu": 10.0, "sigma": 2.0}),
    ]
    copula = GaussianCopula(
        distributions=distributions,
        corr_matrix=CorrelationMatrix.from_n_corr(2, 0.5),
    )

    samples = copula.sample(size=100, seed=42)
    assert samples.shape == (2, 100)
    np.testing.assert_array_equal(copula.rvs(size=100, seed=42), samples)

    samples_tup = copula.sample(size=(10, 5), seed=42)
    assert samples_tup.shape == (2, 10, 5)


def test_gaussian_copula_from_distributions_and_correlation() -> None:
    distributions = [
        Normal(params={"mu": 0.0, "sigma": 1.0}),
        Normal(params={"mu": 0.0, "sigma": 1.0}),
        Normal(params={"mu": 0.0, "sigma": 1.0}),
    ]

    copula = GaussianCopula.from_distributions_and_correlation(
        distributions=distributions,
        correlation=0.25,
    )

    assert copula.dims == 3
    np.testing.assert_allclose(
        copula.corr_matrix.to_numpy(),
        np.array([[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 1.0]]),
    )


def test_gaussian_copula_recovers_correlation_for_normal_marginals() -> None:
    distributions = [
        Normal(params={"mu": 0.0, "sigma": 1.0}),
        Normal(params={"mu": 10.0, "sigma": 2.0}),
    ]
    copula = GaussianCopula.from_distributions_and_correlation(distributions, 0.65)

    samples = copula.sample(size=30_000, seed=123)
    empirical_corr = np.corrcoef(samples)[0, 1]

    assert empirical_corr == pytest.approx(0.65, abs=0.03)


def test_student_t_copula_sample_shape_seed_and_nu() -> None:
    distributions = [
        Normal(params={"mu": 0.0, "sigma": 1.0}),
        Normal(params={"mu": 10.0, "sigma": 2.0}),
    ]
    copula = StudentTCopula.from_distributions_and_correlation(
        distributions=distributions,
        correlation=0.5,
        nu=5,
    )

    samples_a = copula.sample(size=100, seed=42)
    samples_b = copula.sample(size=100, seed=42)

    assert copula.nu == 5
    assert samples_a.shape == (2, 100)
    np.testing.assert_array_equal(samples_a, samples_b)

    samples_tup = copula.sample(size=(10, 5), seed=42)
    assert samples_tup.shape == (2, 10, 5)
