"""One-at-a-time sensitivity analysis."""

from __future__ import annotations

from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from drisk.distributions import Distribution
from drisk.distributions.registry import concrete_distribution_types
from drisk.models import MCModel
from drisk.summary import DEFAULT_PERCENTILES

from ._evaluate import evaluate_mc_model
from ._inputs import (
    distribution_key,
    distribution_label,
    distribution_percentile,
    percentile_column,
    percentile_scenario_values,
)

SENSITIVITY_COLUMNS = [
    "target",
    "variable",
    "percentile",
    "input_value",
    "output_value",
    "baseline",
    "delta",
]


class OneAtATimeSensitivity(BaseModel):
    """Serializable one-at-a-time sensitivity analysis configuration."""

    target: SerializeAsAny[MCModel]
    percentiles: tuple[float | int, ...] = DEFAULT_PERCENTILES
    center_percentile: float | int = 50
    precision: int | None = 2

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, target: MCModel | None = None, **data: Any) -> None:
        """Allow notebook-friendly positional target construction."""
        if target is not None and "target" not in data:
            data["target"] = target
        super().__init__(**data)

    @model_validator(mode="after")
    def restore_target_operands(self) -> Self:
        """Restore MCModel operand objects after JSON validation."""
        self.target = _restore_mc_model(self.target)
        return self

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate sensitivity results as a tidy dataframe.

        All distribution leaves are fixed to ``center_percentile`` for the
        baseline. Each distribution is then varied through ``percentiles`` while
        all other distribution leaves remain fixed at the centre value.
        Constants are not included as sensitivity variables.
        """
        distributions = self.target.distributions()
        center_values = percentile_scenario_values(
            distributions, self.center_percentile
        )
        baseline = evaluate_mc_model(self.target, center_values)

        rows: list[dict[str, float | str]] = []
        target_label = self.target.name or "value"

        for index, distribution in enumerate(distributions):
            variable = distribution_label(distribution, index)
            key = distribution_key(distribution)
            for percentile in self.percentiles:
                fixed_values = dict(center_values)
                input_value = distribution_percentile(distribution, percentile)
                fixed_values[key] = input_value
                output_value = evaluate_mc_model(self.target, fixed_values)
                rows.append(
                    {
                        "target": target_label,
                        "variable": variable,
                        "percentile": percentile_column(percentile),
                        "input_value": input_value,
                        "output_value": output_value,
                        "baseline": baseline,
                        "delta": output_value - baseline,
                    }
                )

        frame = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
        if self.precision is not None and not frame.empty:
            numeric_columns = ["input_value", "output_value", "baseline", "delta"]
            frame[numeric_columns] = frame[numeric_columns].round(self.precision)
        return frame

    def plot(
        self,
        ax: Any = None,
        *,
        show: bool = False,
        positive_color: str = "seagreen",
        negative_color: str = "firebrick",
        baseline_kwargs: dict[str, Any] | None = None,
        bar_kwargs: dict[str, Any] | None = None,
        line_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Plot one-at-a-time results as a tornado chart.

        Variables are ranked by their largest absolute effect on the centered
        result. Symmetric percentile pairs are drawn from widest to narrowest:
        an outer ``p99``-``p1`` pair is shown as a line, while inner pairs such
        as ``p90``-``p10`` and ``p75``-``p25`` are shown as progressively thicker
        horizontal bars. Impacts to the right of the centered result are green;
        impacts to the left are red. The centered result is shown as a black
        dashed vertical line.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        frame = self.evaluate()
        if frame.empty:
            ax.set_title(f"{self.target.name or 'Sensitivity'} tornado")
            ax.set_xlabel("output_value")
            if show:
                import matplotlib.pyplot as plt

                plt.show()
            return ax

        baseline = float(frame["baseline"].iloc[0])
        ranked_variables = (
            frame.groupby("variable")["delta"]
            .apply(lambda values: float(values.abs().max()))
            .sort_values(ascending=False)
            .index.tolist()
        )
        y_positions = {variable: i for i, variable in enumerate(ranked_variables)}
        result_by_variable = {
            variable: variable_frame.set_index("percentile")["output_value"].to_dict()
            for variable, variable_frame in frame.groupby("variable", sort=False)
        }

        pairs = _percentile_pairs(self.percentiles)
        bar_pairs = [pair for pair in pairs if pair != (99.0, 1.0)]
        bar_heights = _bar_heights(bar_pairs)

        seen_labels: set[str] = set()
        for high, low in pairs:
            pair_label = f"p{high:g}-p{low:g}"
            label = pair_label if pair_label not in seen_labels else None
            seen_labels.add(pair_label)
            for variable in ranked_variables:
                variable_results = result_by_variable[variable]
                high_label = percentile_column(high)
                low_label = percentile_column(low)
                if (
                    high_label not in variable_results
                    or low_label not in variable_results
                ):
                    continue
                x1 = float(variable_results[high_label])
                x2 = float(variable_results[low_label])
                left = min(x1, x2)
                width = abs(x2 - x1)
                y = y_positions[variable]
                right = left + width
                if (high, low) == (99.0, 1.0):
                    _draw_split_line(
                        ax,
                        y=y,
                        left=left,
                        right=right,
                        baseline=baseline,
                        positive_color=positive_color,
                        negative_color=negative_color,
                        label=label,
                        line_kwargs=line_kwargs,
                    )
                else:
                    _draw_split_bar(
                        ax,
                        y=y,
                        left=left,
                        right=right,
                        baseline=baseline,
                        height=bar_heights[(high, low)],
                        positive_color=positive_color,
                        negative_color=negative_color,
                        label=label,
                        bar_kwargs=bar_kwargs,
                    )
                label = None

        default_baseline_kwargs = {
            "color": "black",
            "linestyle": "--",
            "linewidth": 1.5,
            "label": percentile_column(self.center_percentile),
        }
        ax.axvline(baseline, **{**default_baseline_kwargs, **(baseline_kwargs or {})})
        ax.set_yticks(range(len(ranked_variables)), labels=ranked_variables)
        ax.invert_yaxis()
        ax.set_xlabel(self.target.name or "output_value")
        ax.set_title(f"{self.target.name or 'Sensitivity'} tornado")

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax


def _draw_split_bar(
    ax: Any,
    *,
    y: float,
    left: float,
    right: float,
    baseline: float,
    height: float,
    positive_color: str,
    negative_color: str,
    label: str | None,
    bar_kwargs: dict[str, Any] | None,
) -> None:
    """Draw an interval bar split into negative/positive baseline impacts."""
    kwargs = {"edgecolor": "white", "linewidth": 1, **(bar_kwargs or {})}
    label_used = False
    if left < baseline:
        negative_right = min(right, baseline)
        ax.barh(
            y,
            negative_right - left,
            left=left,
            height=height,
            color=negative_color,
            label=label,
            **kwargs,
        )
        label_used = label is not None
    if right > baseline:
        positive_left = max(left, baseline)
        ax.barh(
            y,
            right - positive_left,
            left=positive_left,
            height=height,
            color=positive_color,
            label=None if label_used else label,
            **kwargs,
        )


def _draw_split_line(
    ax: Any,
    *,
    y: float,
    left: float,
    right: float,
    baseline: float,
    positive_color: str,
    negative_color: str,
    label: str | None,
    line_kwargs: dict[str, Any] | None,
) -> None:
    """Draw an interval line split into negative/positive baseline impacts."""
    kwargs = {"zorder": -100, **(line_kwargs or {})}
    label_used = False
    if left < baseline:
        negative_right = min(right, baseline)
        ax.hlines(
            y,
            left,
            negative_right,
            color=negative_color,
            linewidth=2,
            label=label,
            **kwargs,
        )
        label_used = label is not None
    if right > baseline:
        positive_left = max(left, baseline)
        ax.hlines(
            y,
            positive_left,
            right,
            color=positive_color,
            linewidth=2,
            label=None if label_used else label,
            **kwargs,
        )


def _percentile_pairs(
    percentiles: tuple[float | int, ...],
) -> list[tuple[float, float]]:
    """Return symmetric percentile pairs, sorted widest to narrowest."""
    percentile_values = {float(percentile) for percentile in percentiles}
    pairs = [
        (percentile, 100 - percentile)
        for percentile in percentile_values
        if percentile > 50 and 100 - percentile in percentile_values
    ]
    return sorted(pairs, key=lambda pair: pair[0], reverse=True)


def _bar_heights(pairs: list[tuple[float, float]]) -> dict[tuple[float, float], float]:
    """Return progressively thicker bar heights for inner percentile bands."""
    if not pairs:
        return {}
    if len(pairs) == 1:
        return {pairs[0]: 0.35}
    heights = [0.22 + i * (0.28 / (len(pairs) - 1)) for i in range(len(pairs))]
    return dict(zip(pairs, heights, strict=True))


def _restore_mc_model(model: MCModel) -> MCModel:
    """Recursively restore serialized MCModel operands to model objects."""
    model.operands = tuple(_restore_operand(operand) for operand in model.operands)
    return model


def _restore_operand(operand: Any) -> Any:
    """Restore a serialized MCModel operand when Pydantic left it as a dict."""
    if isinstance(operand, MCModel):
        return _restore_mc_model(operand)

    if isinstance(operand, Distribution):
        return operand

    if isinstance(operand, dict):
        if "op" in operand and "operands" in operand:
            return _restore_mc_model(MCModel.model_validate(operand))
        if "dist_type" in operand:
            for distribution_type in concrete_distribution_types():
                if (
                    operand["dist_type"]
                    == distribution_type.model_fields["dist_type"].default
                ):
                    return distribution_type.model_validate(operand)

    return operand


def one_at_a_time(
    target: MCModel,
    *,
    percentiles: list[float | int] | tuple[float | int, ...] = DEFAULT_PERCENTILES,
    center_percentile: float | int = 50,
    precision: int | None = 2,
) -> pd.DataFrame:
    """Run one-at-a-time sensitivity analysis for a target."""
    return OneAtATimeSensitivity(
        target,
        percentiles=tuple(percentiles),
        center_percentile=center_percentile,
        precision=precision,
    ).evaluate()
