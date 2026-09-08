from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

import drisk as dr


def assert_model_json_round_trip(obj: BaseModel) -> None:
    """Require a Pydantic model to survive a JSON dump/validate round trip."""
    restored = type(obj).model_validate_json(obj.model_dump_json())

    assert type(restored) is type(obj)
    assert restored.model_dump(mode="json") == obj.model_dump(mode="json")


@pytest.mark.parametrize(
    "dist",
    [
        dr.Constant.elicit(5, name="constant"),
        dr.Normal.elicit(10, 20, name="normal"),
        dr.LogNormal.elicit(10, 100, name="lognormal"),
        dr.LogitNormal.elicit(0.1, 0.3, name="logitnormal"),
        dr.Exponential.elicit(2.5, name="exponential"),
        dr.Gamma.elicit(3, 2.5, name="gamma"),
        dr.Beta.elicit(alpha=2, beta=5, name="beta"),
        dr.StretchedBeta.elicit(0, 5, 10, name="stretched_beta"),
        dr.PERT.elicit(0, 5, 10, name="pert"),
        dr.Bernoulli.elicit(0.3, name="bernoulli"),
        dr.Binomial.elicit(10, 0.3, name="binomial"),
        dr.Geometric.elicit(0.3, name="geometric"),
        dr.NegativeBinomial.elicit(5, 0.3, name="negative_binomial"),
        dr.Poisson.elicit(3.0, name="poisson"),
    ],
)
def test_concrete_distribution_json_round_trip(dist: dr.Distribution) -> None:
    assert_model_json_round_trip(dist)


def test_correlation_matrix_json_round_trip() -> None:
    assert_model_json_round_trip(dr.CorrelationMatrix.from_n_corr(3, 0.25))


@pytest.mark.parametrize(
    "obj",
    [
        dr.UvMixture.elicit(
            components=(dr.Normal.elicit(0, 1), dr.Normal.elicit(10, 12)),
            weights=(0.25, 0.75),
            name="mixture",
        ),
        dr.GaussianCopula.from_distributions_and_correlation(
            [dr.Normal.elicit(0, 1), dr.LogNormal.elicit(1, 3)],
            0.5,
        ),
        dr.StudentTCopula.from_distributions_and_correlation(
            [dr.Normal.elicit(0, 1), dr.LogNormal.elicit(1, 3)],
            0.5,
            nu=5,
        ),
        dr.DTree(
            dr.DecisionNode(
                "Choose",
                {
                    "A": dr.ChanceNode("Chance", {"Win": (0.2, 10), "Lose": (0.8, 0)}),
                    "B": 1,
                },
            ),
            name="tree",
        ),
    ],
)
def test_nested_pydantic_objects_json_round_trip(obj: BaseModel) -> None:
    assert_model_json_round_trip(obj)


class DistributionHolder(BaseModel):
    value: dr.Distribution


class UvDistributionHolder(BaseModel):
    value: dr.UvDistribution


class UnitBoundedContinuousHolder(BaseModel):
    value: dr.UvUnitBoundedContinuous


class CopulaHolder(BaseModel):
    value: dr.Copula


@pytest.mark.parametrize(
    ("holder_cls", "value"),
    [
        (DistributionHolder, dr.Constant.elicit(1)),
        (DistributionHolder, dr.Normal.elicit(0, 1)),
        (UvDistributionHolder, dr.Constant.elicit(1)),
        (UvDistributionHolder, dr.Normal.elicit(0, 1)),
        (
            UvDistributionHolder,
            dr.UvMixture.elicit(
                (dr.Normal.elicit(0, 1), dr.Normal.elicit(10, 12)), (1, 2)
            ),
        ),
        (UnitBoundedContinuousHolder, dr.Beta.elicit(alpha=2, beta=5)),
        (UnitBoundedContinuousHolder, dr.LogitNormal.elicit(0.1, 0.3)),
        (
            CopulaHolder,
            dr.GaussianCopula.from_distributions_and_correlation(
                [dr.Normal.elicit(0, 1), dr.Normal.elicit(10, 12)],
                0.25,
            ),
        ),
        (
            CopulaHolder,
            dr.StudentTCopula.from_distributions_and_correlation(
                [dr.Normal.elicit(0, 1), dr.Normal.elicit(10, 12)],
                0.25,
                nu=5,
            ),
        ),
    ],
)
def test_abstract_base_typed_fields_json_round_trip(
    holder_cls: type[BaseModel], value: Any
) -> None:
    holder = holder_cls(value=value)

    restored = holder_cls.model_validate_json(holder.model_dump_json())

    assert type(restored.value) is type(value)
    assert restored.model_dump(mode="json") == holder.model_dump(mode="json")


def test_domain_typed_distribution_field_rejects_incompatible_distribution() -> None:
    with pytest.raises(ValidationError):
        UnitBoundedContinuousHolder(value=dr.Normal.elicit(0, 1))

    with pytest.raises(ValidationError):
        UnitBoundedContinuousHolder.model_validate(
            {"value": dr.Normal.elicit(0, 1).model_dump(mode="json")}
        )


def test_decision_tree_with_distribution_and_model_outcomes_json_round_trip() -> None:
    dist = dr.Normal.elicit(90, 110, name="x")
    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "Distribution": dist,
                "Model": dist - 10,
                "Fixed": 80,
            },
        ),
        name="tree_with_uncertainty",
    )

    assert_model_json_round_trip(tree)


def test_mc_model_json_round_trip_with_pydantic_type_adapter() -> None:
    model = (dr.Normal.elicit(0, 1, name="x") + 10) * dr.LogNormal.elicit(
        1, 3, name="y"
    )
    model.sample(size=10, seed=123)

    adapter = TypeAdapter(type(model))
    dumped = adapter.dump_python(model, mode="json")
    restored = adapter.validate_json(adapter.dump_json(model))

    assert type(restored) is type(model)
    assert dumped["op"] == "multiply"
    assert "_cached_samples" not in dumped
    assert "_cached_size" not in dumped
    assert "_cached_seed" not in dumped
    assert adapter.dump_python(restored, mode="json") == dumped


def test_where_model_json_round_trip_with_pydantic_type_adapter() -> None:
    model = dr.where(dr.Normal.elicit(-1, 1, name="x") > 0, 1, 0)

    adapter = TypeAdapter(type(model))
    dumped = adapter.dump_python(model, mode="json")
    restored = adapter.validate_json(adapter.dump_json(model))

    assert dumped["op"] == "where"
    assert dumped["operands"][0]["op"] == "greater"
    assert adapter.dump_python(restored, mode="json") == dumped
