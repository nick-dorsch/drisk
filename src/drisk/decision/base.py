"""Decision tree support for business decision analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from drisk.distributions import Distribution
from drisk.models import MCModel
from drisk.random import SeedLike, get_rng
from drisk.summary import percentile_label

ValueLike = Any


class DTreeNode(BaseModel, ABC):
    """Abstract base class for decision tree nodes."""

    node_type: str
    name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        """Return the expected value of this node."""
        pass

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        """Generate outcome samples from this node."""
        pass

    @abstractmethod
    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        """Return tidy rollback rows for this node and descendants."""
        pass


class OutcomeNode(DTreeNode):
    """Terminal node containing a scalar, distribution, or Monte Carlo model value."""

    node_type: Literal["outcome"] = "outcome"
    value: ValueLike

    def __init__(
        self, value: ValueLike = 0, name: str | None = None, **data: Any
    ) -> None:
        if "value" not in data:
            data["value"] = value
        if name is not None and "name" not in data:
            data["name"] = name
        super().__init__(**data)

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        samples = np.ravel(
            _sample_value(self.value, size=10_000 if size is None else size, seed=seed)
        )
        return float(np.mean(samples))

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        return _sample_value(self.value, size=size, seed=seed)

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "node": self.name or "outcome",
                "node_type": self.node_type,
                "branch": None,
                "probability": None,
                "expected_value": self.expected_value(size=size, seed=seed),
                "selected": None,
            }
        ]


class DecisionBranch(BaseModel):
    """A branch leaving a decision node."""

    name: str
    node: SerializeAsAny[DTreeNode]
    value: ValueLike = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def coerce_node(cls, data: Any) -> Any:
        if isinstance(data, dict) and "node" in data:
            copied = dict(data)
            copied["node"] = as_node(copied["node"])
            return copied
        return data


class ChanceBranch(BaseModel):
    """A probability-weighted branch leaving a chance node."""

    name: str
    probability: float = Field(ge=0)
    node: SerializeAsAny[DTreeNode]
    value: ValueLike = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def coerce_node(cls, data: Any) -> Any:
        if isinstance(data, dict) and "node" in data:
            copied = dict(data)
            copied["node"] = as_node(copied["node"])
            return copied
        return data


class DecisionNode(DTreeNode):
    """Decision node that selects the branch with the highest expected value."""

    node_type: Literal["decision"] = "decision"
    branches: tuple[DecisionBranch, ...]

    def __init__(
        self,
        name: str | None = None,
        branches: Any = None,
        **data: Any,
    ) -> None:
        if name is not None and "name" not in data:
            data["name"] = name
        if branches is not None and "branches" not in data:
            data["branches"] = branches
        super().__init__(**data)

    @field_validator("branches", mode="before")
    @classmethod
    def coerce_branches(cls, branches: Any) -> tuple[DecisionBranch, ...]:
        return tuple(_coerce_decision_branches(branches))

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: tuple[DecisionBranch, ...]
    ) -> tuple[DecisionBranch, ...]:
        if not branches:
            raise ValueError("DecisionNode requires at least one branch")
        return branches

    def selected_branch(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> DecisionBranch:
        values = [
            _branch_expected_value(branch, size=size, seed=seed)
            for branch in self.branches
        ]
        return self.branches[int(np.argmax(values))]

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        return _branch_expected_value(
            self.selected_branch(size=size, seed=seed), size=size, seed=seed
        )

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        branch = self.selected_branch(size=size, seed=seed)
        return _sample_branch(branch, size=size, seed=seed)

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        values = [
            _branch_expected_value(branch, size=size, seed=seed)
            for branch in self.branches
        ]
        selected_index = int(np.argmax(values))
        rows = [
            {
                "node": self.name or "decision",
                "node_type": self.node_type,
                "branch": branch.name,
                "probability": None,
                "expected_value": value,
                "selected": i == selected_index,
            }
            for i, (branch, value) in enumerate(zip(self.branches, values, strict=True))
        ]
        for branch in self.branches:
            rows.extend(branch.node.rollback_rows(size=size, seed=seed))
        return rows


class ChanceNode(DTreeNode):
    """Chance node that follows branches according to their probabilities."""

    node_type: Literal["chance"] = "chance"
    branches: tuple[ChanceBranch, ...]

    def __init__(
        self,
        name: str | None = None,
        branches: Any = None,
        **data: Any,
    ) -> None:
        if name is not None and "name" not in data:
            data["name"] = name
        if branches is not None and "branches" not in data:
            data["branches"] = branches
        super().__init__(**data)

    @field_validator("branches", mode="before")
    @classmethod
    def coerce_branches(cls, branches: Any) -> tuple[ChanceBranch, ...]:
        return tuple(_coerce_chance_branches(branches))

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: tuple[ChanceBranch, ...]
    ) -> tuple[ChanceBranch, ...]:
        if not branches:
            raise ValueError("ChanceNode requires at least one branch")
        probability_sum = sum(branch.probability for branch in branches)
        if not np.isclose(probability_sum, 1.0):
            raise ValueError(
                f"ChanceNode branch probabilities must sum to 1, got {probability_sum}"
            )
        return branches

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        return float(
            sum(
                branch.probability
                * _branch_expected_value(branch, size=size, seed=seed)
                for branch in self.branches
            )
        )

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        rng = get_rng(seed)
        probabilities = [branch.probability for branch in self.branches]
        choices = rng.choice(len(self.branches), size=size, p=probabilities)
        branch_samples = [
            _sample_branch(branch, size=size, seed=rng) for branch in self.branches
        ]
        result = np.empty(np.shape(choices), dtype=float)
        for i, samples in enumerate(branch_samples):
            result[choices == i] = np.asarray(samples, dtype=float)[choices == i]
        return result

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        rows = [
            {
                "node": self.name or "chance",
                "node_type": self.node_type,
                "branch": branch.name,
                "probability": branch.probability,
                "expected_value": _branch_expected_value(branch, size=size, seed=seed),
                "selected": None,
            }
            for branch in self.branches
        ]
        for branch in self.branches:
            rows.extend(branch.node.rollback_rows(size=size, seed=seed))
        return rows


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
    ) -> pd.DataFrame:
        """Return a tidy rollback table with selected decision branches."""
        frame = pd.DataFrame(self.root.rollback_rows(size=size, seed=seed))
        if precision is not None and "expected_value" in frame:
            frame["expected_value"] = frame["expected_value"].round(precision)
        return frame

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

        layout = _build_tree_layout(self.root)
        _draw_tree_layout(
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
        _set_tree_limits(ax, layout)

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax


@dataclass
class _TreeLayoutNode:
    node: DTreeNode
    x: float
    y: float
    depth: int
    children: list[tuple[DecisionBranch | ChanceBranch, _TreeLayoutNode]] = field(
        default_factory=list
    )


def as_node(value: Any) -> DTreeNode:
    """Return ``value`` as a decision tree node."""
    if isinstance(value, DTreeNode):
        return value
    if isinstance(value, dict) and "node_type" in value:
        node_type = value["node_type"]
        if node_type == "decision":
            return DecisionNode.model_validate(value)
        if node_type == "chance":
            return ChanceNode.model_validate(value)
        if node_type == "outcome":
            return OutcomeNode.model_validate(value)
        raise ValueError(f"Unknown decision tree node_type: {node_type}")
    return OutcomeNode(value=value)


def _build_tree_layout(root: DTreeNode) -> _TreeLayoutNode:
    leaf_counter = [0]

    def build(node: DTreeNode, depth: int) -> _TreeLayoutNode:
        children = [
            (branch, build(branch.node, depth + 1)) for branch in _node_branches(node)
        ]
        if children:
            x = float(np.mean([child.x for _, child in children]))
        else:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
        return _TreeLayoutNode(
            node=node, x=x * 2.0, y=-depth * 1.7, depth=depth, children=children
        )

    return build(root, 0)


def _node_branches(node: DTreeNode) -> tuple[DecisionBranch | ChanceBranch, ...]:
    if isinstance(node, DecisionNode | ChanceNode):
        return node.branches
    return ()


def _draw_tree_layout(
    ax: Any,
    layout: _TreeLayoutNode,
    *,
    size: int | tuple[int, ...] | None,
    seed: SeedLike,
    show_expected_values: bool,
    show_probabilities: bool,
    show_selected: bool,
    precision: int,
) -> None:
    from matplotlib.patches import Circle, Rectangle, RegularPolygon

    node_colors = {
        "decision": "#2E7D32",
        "chance": "#7B1E3A",
        "outcome": "#2F6DAE",
    }
    selected_branch = None
    if show_selected and isinstance(layout.node, DecisionNode):
        selected_branch = layout.node.selected_branch(size=size, seed=seed)

    for branch, child in layout.children:
        is_selected = branch is selected_branch
        edge_color = "#222222" if is_selected else "#8A8A8A"
        edge_width = 2.6 if is_selected else 1.4
        ax.plot(
            [layout.x, child.x],
            [layout.y - 0.22, child.y + 0.28],
            color=edge_color,
            linewidth=edge_width,
            zorder=1,
        )
        _draw_branch_label(
            ax,
            branch,
            x=(layout.x + child.x) / 2,
            y=(layout.y + child.y) / 2 + 0.12,
            show_probabilities=show_probabilities,
            selected=is_selected,
        )
        _draw_tree_layout(
            ax,
            child,
            size=size,
            seed=seed,
            show_expected_values=show_expected_values,
            show_probabilities=show_probabilities,
            show_selected=show_selected,
            precision=precision,
        )

    color = node_colors.get(layout.node.node_type, "#4C78A8")
    if isinstance(layout.node, DecisionNode):
        patch = Rectangle(
            (layout.x - 0.26, layout.y - 0.26),
            0.52,
            0.52,
            facecolor=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
    elif isinstance(layout.node, ChanceNode):
        patch = Circle(
            (layout.x, layout.y),
            radius=0.30,
            facecolor=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
    else:
        patch = RegularPolygon(
            (layout.x, layout.y),
            numVertices=3,
            radius=0.36,
            orientation=0,
            facecolor=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
    ax.add_patch(patch)
    _draw_node_label(
        ax,
        layout.node,
        x=layout.x,
        y=layout.y,
        size=size,
        seed=seed,
        show_expected_values=show_expected_values,
        precision=precision,
    )


def _draw_node_label(
    ax: Any,
    node: DTreeNode,
    *,
    x: float,
    y: float,
    size: int | tuple[int, ...] | None,
    seed: SeedLike,
    show_expected_values: bool,
    precision: int,
) -> None:
    label = node.name or node.node_type.title()
    if show_expected_values:
        value = node.expected_value(size=size, seed=seed)
        label = f"{label}\nEV {_format_tree_number(value, precision)}"
    ax.text(
        x,
        y - 0.48,
        label,
        ha="center",
        va="top",
        fontsize=9,
        color="#222222",
        zorder=4,
    )


def _draw_branch_label(
    ax: Any,
    branch: DecisionBranch | ChanceBranch,
    *,
    x: float,
    y: float,
    show_probabilities: bool,
    selected: bool,
) -> None:
    label = branch.name
    if show_probabilities and isinstance(branch, ChanceBranch):
        label = f"{label} ({branch.probability:.0%})"
    weight = "bold" if selected else "normal"
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight=weight,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
        zorder=5,
    )


def _set_tree_limits(ax: Any, layout: _TreeLayoutNode) -> None:
    nodes = list(_walk_layout(layout))
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 0.8)


def _walk_layout(layout: _TreeLayoutNode) -> list[_TreeLayoutNode]:
    nodes = [layout]
    for _, child in layout.children:
        nodes.extend(_walk_layout(child))
    return nodes


def _format_tree_number(value: float, precision: int) -> str:
    rounded = round(value, precision)
    if precision == 0:
        return f"{rounded:,.0f}"
    return f"{rounded:,.{precision}f}"


def _coerce_decision_branches(branches: Any) -> list[DecisionBranch]:
    if isinstance(branches, dict):
        return [
            DecisionBranch(name=str(name), node=as_node(value))
            for name, value in branches.items()
        ]
    return [
        branch
        if isinstance(branch, DecisionBranch)
        else DecisionBranch.model_validate(branch)
        for branch in branches
    ]


def _coerce_chance_branches(branches: Any) -> list[ChanceBranch]:
    if isinstance(branches, dict):
        coerced = []
        for name, spec in branches.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                probability, value = spec
                coerced.append(
                    ChanceBranch(
                        name=str(name), probability=probability, node=as_node(value)
                    )
                )
            else:
                coerced.append(ChanceBranch.model_validate({"name": name, **spec}))
        return coerced
    return [
        branch
        if isinstance(branch, ChanceBranch)
        else ChanceBranch.model_validate(branch)
        for branch in branches
    ]


def _branch_expected_value(
    branch: DecisionBranch | ChanceBranch,
    *,
    size: int | tuple[int, ...] | None = None,
    seed: SeedLike = None,
) -> float:
    rng = get_rng(seed)
    return _value_expected_value(
        branch.value, size=size, seed=rng
    ) + branch.node.expected_value(
        size=size,
        seed=rng,
    )


def _value_expected_value(
    value: ValueLike,
    *,
    size: int | tuple[int, ...] | None = None,
    seed: SeedLike = None,
) -> float:
    samples = np.ravel(
        _sample_value(value, size=10_000 if size is None else size, seed=seed)
    )
    return float(np.mean(samples))


def _sample_branch(
    branch: DecisionBranch | ChanceBranch,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    rng = get_rng(seed)
    return _sample_value(branch.value, size=size, seed=rng) + branch.node.sample(
        size=size, seed=rng
    )


def _sample_value(
    value: ValueLike,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    if isinstance(value, Distribution | MCModel):
        return np.asarray(value.sample(size=size, seed=seed), dtype=float)
    if isinstance(value, DTree):
        return np.asarray(value.sample(size=size, seed=seed), dtype=float)
    return np.broadcast_to(np.asarray(value, dtype=float), size)
