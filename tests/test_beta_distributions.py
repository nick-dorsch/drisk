import numpy as np
import pytest

import monte as mt


def test_beta_elicit_mode_concentration() -> None:
    dist = mt.Beta.elicit(mode=0.7, concentration=10)

    assert isinstance(dist, mt.UvUnitBoundedContinuous)
    assert dist.params == {"alpha": 8.0, "beta": 4.0}
    assert dist.elicitation_params == {"mode": 0.7, "concentration": 10}
    assert dist.support == (0.0, 1.0)
    assert dist.mean == pytest.approx(8 / 12)
    assert dist.mode_value == pytest.approx(0.7)
    assert np.all((dist.sample(10, seed=1) > 0) & (dist.sample(10, seed=1) < 1))


def test_beta_elicit_alpha_beta_and_fit() -> None:
    dist = mt.Beta.elicit(alpha=2, beta=5)

    assert dist.params == {"alpha": 2.0, "beta": 5.0}
    assert dist.elicitation_params == {"alpha": 2, "beta": 5}

    fitted = mt.Beta.fit([0.2, 0.3, 0.4, 0.5])
    assert fitted.params["alpha"] > 0
    assert fitted.params["beta"] > 0

    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        mt.Beta.fit([0.0, 0.5])


def test_stretched_beta_elicit_and_fit() -> None:
    dist = mt.StretchedBeta.elicit(min=10, mode=20, max=40, concentration=4)

    assert isinstance(dist, mt.UvBoundedContinuous)
    assert dist.support == (10.0, 40.0)
    assert dist.params["alpha"] == pytest.approx(1 + 4 * (20 - 10) / (40 - 10))
    assert dist.params["beta"] == pytest.approx(1 + 4 * (40 - 20) / (40 - 10))
    assert dist.elicitation_params == {
        "min": 10,
        "mode": 20,
        "max": 40,
        "concentration": 4,
    }
    samples = dist.sample(10, seed=1)
    assert np.all((samples >= 10) & (samples <= 40))

    fitted = mt.StretchedBeta.fit(samples)
    assert fitted.params["alpha"] > 0
    assert fitted.params["beta"] > 0
    assert fitted.support[0] < fitted.support[1]


def test_pert_is_fixed_concentration_stretched_beta() -> None:
    dist = mt.PERT.elicit(min=10, mode=20, max=40)

    assert isinstance(dist, mt.StretchedBeta)
    assert dist.dist_type == "pert"
    assert dist.params["concentration"] == 4.0
    assert dist.elicitation_params == {
        "min": 10,
        "mode": 20,
        "max": 40,
        "concentration": 4.0,
    }

    fitted = mt.PERT.fit([10, 15, 20, 30, 40])
    assert fitted.params["concentration"] == 4.0
    assert fitted.support == (10.0, 40.0)
