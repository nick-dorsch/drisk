import numpy as np

from drisk.distributions import ArrayLike, UvContinuous
from drisk.random import SeedLike


class ToyDistribution(UvContinuous):
    dist_type: str = "toy"
    params: dict[str, float | int] = {"value": 1.0}

    def sample(
        self, size: int | tuple[int, ...] = 1, *, seed: SeedLike = None
    ) -> np.ndarray:
        return np.full(size, self.params["value"])

    @classmethod
    def elicit(cls, *, value: float) -> "ToyDistribution":
        return cls(params={"value": value}, elicitation_params={"value": value})

    @classmethod
    def fit(cls, data: ArrayLike) -> "ToyDistribution":
        return cls(params={"value": float(np.mean(np.asarray(data)))})

    def pdf(self, x: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float)

    def plot(self, **kwargs: object) -> object:
        return None

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, 2.0)

    @property
    def x_range(self) -> tuple[float, float]:
        return self.support


def test_sample_replaces_rvs_and_rvs_aliases_sample() -> None:
    dist = ToyDistribution.elicit(value=3.0)

    np.testing.assert_array_equal(dist.sample(size=3), np.array([3.0, 3.0, 3.0]))
    np.testing.assert_array_equal(dist.rvs(size=3), dist.sample(size=3))


def test_elicitation_params_are_nullable_and_serialized() -> None:
    assert ToyDistribution.fit(np.array([1.0, 2.0])).elicitation_params is None
    assert ToyDistribution.elicit(value=4.0).elicitation_params == {"value": 4.0}
