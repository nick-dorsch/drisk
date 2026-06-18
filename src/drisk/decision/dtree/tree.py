"""Decision tree model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from drisk.random import SeedLike
from drisk.summary import percentile_label

from ._plotting import build_tree_layout, draw_tree_layout, set_tree_limits
from .nodes.base import DTreeNode
from .nodes.factory import as_node


class DTree(BaseModel):
    """A sampleable, serializable decision tree."""

    root: SerializeAsAny[DTreeNode]
    name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, root: Any = None, name: str | None = None, **data: Any) -> None:
        if root is not None and "root" not in data:
            data["root"] = root
        if name is not None and "name" not in data:
            data["name"] = name
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def coerce_root(cls, data: Any) -> Any:
        if isinstance(data, dict) and "root" in data:
            copied = dict(data)
            copied["root"] = as_node(copied["root"])
            return copied
        return data

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        """Return the rolled-back expected value of the tree."""
        return self.root.expected_value(size=size, seed=seed)

    def rollback(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
        precision: int | None = 2,
        node_types: str | tuple[str, ...] | None = "decision",
    ) -> pd.DataFrame:
        """Return a rollback table indexed by node.

        By default, only decision-node branches are shown. Pass ``node_types=None`` to
        include decision, chance, and outcome nodes, or pass a node type / tuple of
        node types to filter the table.
        """
        frame = pd.DataFrame(self.root.rollback_rows(size=size, seed=seed))
        if node_types is not None:
            included = (node_types,) if isinstance(node_types, str) else node_types
            frame = frame[frame["node_type"].isin(included)]
        if precision is not None and "expected_value" in frame:
            frame["expected_value"] = frame["expected_value"].round(precision)
        return frame.set_index("node")

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Sample outcomes under the rollback-selected policy, without perfect information."""
        return self.root.sample(size=size, seed=seed)

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Alias for :meth:`sample`."""
        return self.sample(size=size, seed=seed)

    def summary(
        self,
        *,
        size: int | tuple[int, ...] = 10_000,
        seed: SeedLike = None,
        percentiles: list[float | int] | tuple[float | int, ...] = (90, 50, 10),
        precision: int | None = 2,
    ) -> pd.DataFrame:
        """Summarize simulated outcomes under the rollback-selected policy."""
        samples = np.ravel(self.sample(size=size, seed=seed))
        values: dict[str, float] = {"mean": float(np.mean(samples))}
        percentile_values = np.percentile(samples, percentiles)
        values.update(
            {
                percentile_label(percentile): float(value)
                for percentile, value in zip(
                    percentiles, percentile_values, strict=True
                )
            }
        )
        index_label = self.name or "value"
        summary = pd.DataFrame(values, index=pd.Index([index_label], name="metric"))
        if precision is not None:
            summary = summary.round(precision)
        return summary

    def plot(
        self,
        ax: Any = None,
        *,
        size: int | tuple[int, ...] = 10_000,
        seed: SeedLike = None,
        bins: int | str = 80,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot sampled tree outcomes as an empirical CDF with a histogram."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        samples = np.sort(np.ravel(self.sample(size=size, seed=seed)))
        ecdf = np.arange(1, samples.size + 1) / samples.size
        (ecdf_line,) = ax.plot(samples, ecdf, **kwargs)

        hist_ax = ax.twinx()
        hist_ax.hist(samples, bins=bins, color=ecdf_line.get_color(), alpha=0.2)
        hist_ax.set_yticks([])
        hist_ax.spines["right"].set_visible(False)

        ax.set_xlabel(self.name or "value")
        ax.set_ylabel("cumulative probability")
        ax.set_ylim(bottom=0, top=1)
        ax.set_title(self.name or "Decision tree outcome")

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax

    def plot_tree(
        self,
        ax: Any = None,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
        show_expected_values: bool = True,
        show_probabilities: bool = True,
        show_selected: bool = True,
        precision: int = 2,
        show: bool = False,
    ) -> Any:
        """Plot the decision tree structure from top to bottom."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(figsize=(10, 6))

        layout = build_tree_layout(self.root)
        draw_tree_layout(
            ax,
            layout,
            size=size,
            seed=seed,
            show_expected_values=show_expected_values,
            show_probabilities=show_probabilities,
            show_selected=show_selected,
            precision=precision,
        )

        ax.set_title(self.name or "Decision tree")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        set_tree_limits(ax, layout)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax
