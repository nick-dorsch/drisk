import numpy as np
import pytest

from drisk.distributions import Normal, Poisson, UvMixture


def test_uv_mixture_normalizes_weights_and_samples_from_components() -> None:
    mixture = UvMixture(
        components=(Normal.elicit(0, 1), Normal.elicit(10, 11)),
        weights=(1, 3),
    )

    assert mixture.dist_type == "uv_mixture"
    assert mixture.weights == (0.25, 0.75)

    samples = mixture.sample(size=5000, seed=123)

    assert samples.shape == (5000,)
    assert 7.0 < float(np.mean(samples)) < 9.0


def test_uv_mixture_rvs_aliases_sample_with_reproducible_seed() -> None:
    mixture = UvMixture.elicit(
        components=(Poisson.elicit(1), Poisson.elicit(10)),
        weights=(0.5, 0.5),
    )

    np.testing.assert_array_equal(
        mixture.rvs(size=(4, 3), seed=123),
        mixture.sample(size=(4, 3), seed=123),
    )


def test_uv_mixture_elicit_sets_elicitation_params() -> None:
    components = (Normal.elicit(0, 1), Normal.elicit(2, 3))
    mixture = UvMixture.elicit(components=components, weights=(2, 1), name="test")

    assert mixture.name == "test"
    assert mixture.components == components
    np.testing.assert_allclose(mixture.weights, (2 / 3, 1 / 3))
    assert mixture.elicitation_params == {"weights": (2, 1)}


def test_uv_mixture_serializes_component_specific_fields() -> None:
    mixture = UvMixture.elicit(
        components=(Normal.elicit(0, 1), Poisson.elicit(3)),
        weights=(0.4, 0.6),
    )

    dumped = mixture.model_dump()

    assert dumped["components"][0]["dist_type"] == "normal"
    assert dumped["components"][0]["params"] == pytest.approx(
        mixture.components[0].params
    )
    assert dumped["components"][1]["dist_type"] == "poisson"
    assert dumped["components"][1]["params"] == {"lam": 3.0}


def test_uv_mixture_pdf_cdf_and_ppf_not_implemented() -> None:
    components = (Normal.elicit(0, 1), Normal.elicit(10, 11))
    mixture = UvMixture.elicit(components=components, weights=(0.25, 0.75))
    x = np.array([0.0, 5.0, 10.0])

    expected_pdf = 0.25 * components[0].pdf(x) + 0.75 * components[1].pdf(x)
    expected_cdf = 0.25 * components[0].cdf(x) + 0.75 * components[1].cdf(x)

    np.testing.assert_allclose(mixture.pdf(x), expected_pdf)
    np.testing.assert_allclose(mixture.cdf(x), expected_cdf)
    with pytest.raises(NotImplementedError, match="UvMixture.ppf"):
        mixture.ppf([0.1, 0.5, 0.9])


@pytest.mark.parametrize(
    ("components", "weights", "match"),
    [
        ((), (), "at least one"),
        ((Normal.elicit(0, 1), Normal.elicit(2, 3)), (0, 0), "positive"),
        ((Normal.elicit(0, 1),), (-1,), "non-negative"),
        ((Normal.elicit(0, 1),), (1, 2), "same number"),
    ],
)
def test_uv_mixture_validates_inputs(
    components: tuple[Normal, ...], weights: tuple[float, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        UvMixture(components=components, weights=weights)


def test_generic_uv_mixture_fit_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        UvMixture.fit([1, 2, 3])
