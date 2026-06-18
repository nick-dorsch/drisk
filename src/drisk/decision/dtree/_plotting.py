"""Decision tree structure plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from drisk.random import SeedLike

from .branches import ChanceBranch, DecisionBranch
from .nodes.base import DTreeNode
from .nodes.chance import ChanceNode
from .nodes.decision import DecisionNode


@dataclass
class TreeLayoutNode:
    node: DTreeNode
    x: float
    y: float
    depth: int
    children: list[tuple[DecisionBranch | ChanceBranch, TreeLayoutNode]] = field(
        default_factory=list
    )


def build_tree_layout(root: DTreeNode) -> TreeLayoutNode:
    x_spacing = 2.0
    y_spacing = 1.7

    def subtree_width(node: DTreeNode) -> int:
        widths = [subtree_width(branch.node) for branch in node_branches(node)]
        return max(1, sum(widths))

    def build(node: DTreeNode, depth: int, center: float) -> TreeLayoutNode:
        branches = node_branches(node)
        child_widths = [subtree_width(branch.node) for branch in branches]
        children: list[tuple[DecisionBranch | ChanceBranch, TreeLayoutNode]] = []

        for branch, child_center in zip(
            branches, child_centers(center, child_widths), strict=True
        ):
            children.append((branch, build(branch.node, depth + 1, child_center)))

        return TreeLayoutNode(
            node=node,
            x=center * x_spacing,
            y=-depth * y_spacing,
            depth=depth,
            children=children,
        )

    return build(root, 0, 0.0)


def child_centers(parent_center: float, child_widths: list[int]) -> list[float]:
    if not child_widths:
        return []

    positions = [0.0]
    for left_width, right_width in zip(child_widths, child_widths[1:], strict=False):
        positions.append(positions[-1] + (left_width + right_width) / 2)

    center_offset = (positions[0] + positions[-1]) / 2
    return [parent_center + position - center_offset for position in positions]


def draw_tree_layout(
    ax: Any,
    layout: TreeLayoutNode,
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
    node_scale = 0.75
    selected_branch = None
    if show_selected and isinstance(layout.node, DecisionNode):
        selected_branch = layout.node.selected_branch(size=size, seed=seed)

    for branch, child in layout.children:
        is_selected = branch is selected_branch
        edge_color = "#222222" if is_selected else "#8A8A8A"
        edge_width = 2.6 if is_selected else 1.4
        ax.plot(
            [layout.x, child.x],
            [layout.y - 0.22 * node_scale, child.y + 0.28 * node_scale],
            color=edge_color,
            linewidth=edge_width,
            zorder=1,
        )
        draw_branch_label(
            ax,
            branch,
            x=(layout.x + child.x) / 2,
            y=(layout.y + child.y) / 2 + 0.12,
            show_probabilities=show_probabilities,
            selected=is_selected,
        )
        draw_tree_layout(
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
            (layout.x - 0.26 * node_scale, layout.y - 0.26 * node_scale),
            0.52 * node_scale,
            0.52 * node_scale,
            facecolor=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
    elif isinstance(layout.node, ChanceNode):
        patch = Circle(
            (layout.x, layout.y),
            radius=0.30 * node_scale,
            facecolor=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
    else:
        patch = RegularPolygon(
            (layout.x, layout.y),
            numVertices=3,
            radius=0.36 * node_scale,
            orientation=0,
            facecolor=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
    ax.add_patch(patch)
    draw_node_label(
        ax,
        layout.node,
        x=layout.x,
        y=layout.y,
        size=size,
        seed=seed,
        show_expected_values=show_expected_values,
        precision=precision,
    )


def set_tree_limits(ax: Any, layout: TreeLayoutNode) -> None:
    nodes = list(walk_layout(layout))
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    x_extent = max(abs(min(xs)), abs(max(xs))) + 1.0
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 0.8)


def node_branches(node: DTreeNode) -> tuple[DecisionBranch | ChanceBranch, ...]:
    if isinstance(node, DecisionNode | ChanceNode):
        return node.branches
    return ()


def draw_node_label(
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
    ax.text(
        x + 0.28,
        y + 0.24,
        label,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
        zorder=4,
    )
    if show_expected_values:
        value = node.expected_value(size=size, seed=seed)
        ax.text(
            x,
            y - 0.48,
            f"EV {format_tree_number(value, precision)}",
            ha="center",
            va="top",
            fontsize=9,
            color="#222222",
            zorder=4,
        )


def draw_branch_label(
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


def walk_layout(layout: TreeLayoutNode) -> list[TreeLayoutNode]:
    nodes = [layout]
    for _, child in layout.children:
        nodes.extend(walk_layout(child))
    return nodes


def format_tree_number(value: float, precision: int) -> str:
    rounded = round(value, precision)
    if precision == 0:
        return f"{rounded:,.0f}"
    return f"{rounded:,.{precision}f}"
