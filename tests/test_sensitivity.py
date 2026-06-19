import numpy as np
import pandas as pd
import pytest

import drisk as dr


def test_mc_model_sensitivity_varies_distribution_leaves_one_at_a_time() -> None:
    x = dr.Normal(params={"mu": 10.0, "sigma": 2.0}, name="x")
    y = dr.Normal(params={"mu": 100.0, "sigma": 10.0}, name="y")
    model = x + 2 * y + 5

    analysis = dr.OneAtATimeSensitivity(model, percentiles=(90, 50, 10), precision=None)
    sensitivity = analysis.evaluate()

    assert analysis.target is model

    assert sensitivity["variable"].tolist() == ["x", "x", "x", "y", "y", "y"]
    assert sensitivity["percentile"].tolist() == ["p90", "p50", "p10"] * 2
    assert sensitivity["baseline"].unique().tolist() == [pytest.approx(215.0)]

    x_p90 = float(x.ppf(0.10))
    y_p10 = float(y.ppf(0.90))
    x_row = sensitivity[
        (sensitivity["variable"] == "x") & (sensitivity["percentile"] == "p90")
    ].iloc[0]
    y_row = sensitivity[
        (sensitivity["variable"] == "y") & (sensitivity["percentile"] == "p10")
    ].iloc[0]

    assert x_row["input_value"] == pytest.approx(x_p90)
    assert x_row["output_value"] == pytest.approx(x_p90 + 2 * 100 + 5)
    assert y_row["input_value"] == pytest.approx(y_p10)
    assert y_row["output_value"] == pytest.approx(10 + 2 * y_p10 + 5)
    assert y_row["delta"] == pytest.approx(y_row["output_value"] - 215.0)


def test_one_at_a_time_sensitivity_is_json_serializable() -> None:
    x = dr.Normal(params={"mu": 10.0, "sigma": 2.0}, name="x")
    y = dr.Normal(params={"mu": 100.0, "sigma": 10.0}, name="y")
    model = x + y
    analysis = dr.OneAtATimeSensitivity(
        target=model,
        percentiles=(90, 50, 10),
        precision=None,
    )

    restored = dr.OneAtATimeSensitivity.model_validate_json(analysis.model_dump_json())

    assert restored.model_dump(mode="json") == analysis.model_dump(mode="json")
    pd.testing.assert_frame_equal(restored.evaluate(), analysis.evaluate())


def test_mc_model_sensitivity_uses_default_percentiles_and_rounding() -> None:
    x = dr.Normal(params={"mu": 0.0, "sigma": 1.0}, name="x")
    model = x * 10

    sensitivity = dr.one_at_a_time(model)

    assert sensitivity["percentile"].tolist() == [
        "p99",
        "p90",
        "p75",
        "p50",
        "p25",
        "p10",
        "p1",
    ]
    assert sensitivity.loc[0, "input_value"] == round(float(x.ppf(0.01)), 2)
    assert sensitivity.loc[0, "output_value"] == round(float(x.ppf(0.01)) * 10, 2)


def test_mc_model_sensitivity_reused_distribution_is_one_variable() -> None:
    x = dr.Normal(params={"mu": 10.0, "sigma": 2.0}, name="x")
    model = x + x

    sensitivity = model.sensitivity(percentiles=(90, 10), precision=None)

    assert sensitivity["variable"].tolist() == ["x", "x"]
    assert sensitivity.loc[0, "output_value"] == pytest.approx(2 * float(x.ppf(0.10)))
    assert sensitivity.loc[1, "output_value"] == pytest.approx(2 * float(x.ppf(0.90)))


def test_mc_model_sensitivity_omits_constants() -> None:
    model = dr.MCModel.from_operation(dr.MCOperation.ADD, 1, 2)

    sensitivity = model.sensitivity()

    assert isinstance(sensitivity, pd.DataFrame)
    assert sensitivity.empty
    assert sensitivity.columns.tolist() == [
        "target",
        "variable",
        "percentile",
        "input_value",
        "output_value",
        "baseline",
        "delta",
    ]


def test_mc_model_sensitivity_requires_scalar_output() -> None:
    x = dr.Normal(params={"mu": 0.0, "sigma": 1.0}, name="x")
    model = x + np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="requires scalar model outputs"):
        model.sensitivity()


def test_one_at_a_time_sensitivity_plot_returns_ranked_tornado_axes() -> None:
    import matplotlib.pyplot as plt

    x = dr.Normal(params={"mu": 10.0, "sigma": 1.0}, name="x")
    y = dr.Normal(params={"mu": 20.0, "sigma": 1.0}, name="y")
    model = x + 10 * y
    fig, ax = plt.subplots()

    returned = model.plot_sensitivity(ax=ax, precision=None)

    assert returned is ax
    assert [label.get_text() for label in ax.get_yticklabels()] == ["y", "x"]
    assert ax.get_xlabel() == "output_value"
    assert ax.get_title() == "Sensitivity tornado"
    assert (
        len(ax.patches) == 8
    )  # negative and positive halves for two bar bands and variables
    patch_colors = {patch.get_facecolor() for patch in ax.patches}
    assert len(patch_colors) == 2
    assert all(patch.get_edgecolor() == (1.0, 1.0, 1.0, 1.0) for patch in ax.patches)
    assert all(patch.get_linewidth() == 1 for patch in ax.patches)
    assert len(ax.collections) >= 2  # negative and positive p99-p1 line collections
    assert all(collection.get_zorder() == -100 for collection in ax.collections)
    assert len(ax.lines) == 1  # centered p50 dashed line
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[0].get_xdata()[0] == pytest.approx(210.0)

    plt.close(fig)


def test_mc_model_sensitivity_analysis_returns_serializable_analysis() -> None:
    x = dr.Normal(params={"mu": 10.0, "sigma": 1.0}, name="x")
    analysis = (x * 2).sensitivity_analysis(percentiles=(90, 50, 10))

    assert isinstance(analysis, dr.OneAtATimeSensitivity)
    assert analysis.percentiles == (90, 50, 10)
