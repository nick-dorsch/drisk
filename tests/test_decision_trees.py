import numpy as np
import pytest

import drisk as dr


def test_decision_tree_dict_api_rolls_back_to_best_branch() -> None:
    tree = dr.DTree(
        dr.DecisionNode(
            "Drill?",
            {
                "Drill": dr.ChanceNode(
                    "Reservoir",
                    {
                        "Success": (0.5, 100),
                        "Dry": (0.5, -20),
                    },
                ),
                "Do nothing": 0,
            },
        )
    )

    rollback = tree.rollback(precision=None)

    assert tree.expected_value() == pytest.approx(40)
    drill = rollback.loc[(rollback.index == "Drill?") & (rollback["branch"] == "Drill")]
    do_nothing = rollback.loc[
        (rollback.index == "Drill?") & (rollback["branch"] == "Do nothing")
    ]
    assert rollback.index.name == "node"
    assert set(rollback["node_type"]) == {"decision"}
    assert drill.iloc[0]["selected"] is True
    assert do_nothing.iloc[0]["selected"] is False


def test_decision_tree_sampling_uses_selected_policy_without_perfect_information() -> (
    None
):
    tree = dr.DTree(
        dr.DecisionNode(
            "Invest?",
            {
                "Risky": dr.ChanceNode(
                    "Market",
                    {
                        "Good": (0.5, 100),
                        "Bad": (0.5, -200),
                    },
                ),
                "Safe": 10,
            },
        )
    )

    samples = tree.sample(size=200, seed=123)

    # If decisions used perfect information sample-by-sample, outcomes would include 100.
    assert np.all(samples == 10)
    assert tree.expected_value() == pytest.approx(10)


def test_decision_tree_rollback_can_include_all_node_types() -> None:
    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "A": dr.ChanceNode("Chance", {"Win": (0.5, 10), "Lose": (0.5, 0)}),
                "B": 1,
            },
        )
    )

    rollback = tree.rollback(node_types=None)

    assert rollback.index.name == "node"
    assert set(rollback["node_type"]) == {"decision", "chance", "outcome"}


def test_chance_node_sampling_follows_probabilities() -> None:
    tree = dr.DTree(
        dr.ChanceNode(
            "Coin",
            {
                "Heads": (0.25, 1),
                "Tails": (0.75, 0),
            },
        )
    )

    samples = tree.sample(size=10_000, seed=123)

    assert np.mean(samples) == pytest.approx(0.25, abs=0.02)


def test_decision_tree_accepts_distribution_and_model_outcomes() -> None:
    x = dr.Normal.elicit(90, 110, name="x")
    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "Uncertain": x - 10,
                "Fixed": 80,
            },
        ),
        name="Example",
    )

    rollback = tree.rollback(size=2_000, seed=123, precision=None)
    summary = tree.summary(size=100, seed=123, precision=None)

    assert rollback.loc[rollback["branch"] == "Uncertain", "selected"].iloc[0] is True
    assert rollback.index.tolist() == ["Choose", "Choose"]
    assert summary.index.tolist() == ["Example"]
    assert tree.sample(size=5, seed=123).shape == (5,)


def test_branch_values_are_added_to_downstream_values() -> None:
    tree = dr.DTree(
        dr.DecisionNode(
            "Bid?",
            branches=[
                dr.DecisionBranch(
                    name="Bid",
                    value=-10,
                    node=dr.ChanceNode(
                        "Win?",
                        {
                            "Win": (0.5, 50),
                            "Lose": (0.5, 0),
                        },
                    ),
                ),
                dr.DecisionBranch(name="Pass", node=0),
            ],
        )
    )

    assert tree.expected_value() == pytest.approx(15)


def test_nodes_are_pydantic_serializable() -> None:
    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "A": dr.ChanceNode("Chance", {"Win": (0.2, 10), "Lose": (0.8, 0)}),
                "B": 1,
            },
        ),
        name="Serializable",
    )

    dumped = tree.model_dump()

    assert dumped["name"] == "Serializable"
    assert dumped["root"]["node_type"] == "decision"
    assert dumped["root"]["branches"][0]["node"]["node_type"] == "chance"

    restored = dr.DTree.model_validate(dumped)

    assert isinstance(restored.root, dr.DecisionNode)
    assert isinstance(restored.root.branches[0].node, dr.ChanceNode)
    assert restored.expected_value() == pytest.approx(tree.expected_value())


def test_plot_tree_draws_node_shapes_and_branch_labels() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, RegularPolygon

    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "A": dr.ChanceNode("Chance", {"Win": (0.2, 10), "Lose": (0.8, 0)}),
                "B": 1,
            },
        )
    )
    fig, ax = plt.subplots()

    returned = tree.plot_tree(ax=ax, size=100, seed=123, precision=0)

    assert returned is ax
    assert any(isinstance(patch, Rectangle) for patch in ax.patches)
    assert any(isinstance(patch, Circle) for patch in ax.patches)
    assert any(isinstance(patch, RegularPolygon) for patch in ax.patches)
    assert any("Win (20%)" in text.get_text() for text in ax.texts)
    decision_patch = next(patch for patch in ax.patches if isinstance(patch, Rectangle))
    assert decision_patch.get_x() == pytest.approx(-0.195)
    assert decision_patch.get_y() == pytest.approx(-0.195)
    assert decision_patch.get_width() == pytest.approx(0.39)
    assert ax.get_xlim()[0] == pytest.approx(-ax.get_xlim()[1])

    plt.close(fig)


def test_tree_layout_centres_children_around_their_parent() -> None:
    from drisk.decision.dtree._plotting import build_tree_layout

    tree = dr.DTree(
        dr.DecisionNode(
            "Choose",
            {
                "Complex": dr.ChanceNode(
                    "Chance",
                    {
                        "A": (0.25, 1),
                        "B": (0.25, 2),
                        "C": (0.25, 3),
                        "D": (0.25, 4),
                    },
                ),
                "Simple": 0,
            },
        )
    )

    layout = build_tree_layout(tree.root)
    complex_layout = layout.children[0][1]
    simple_layout = layout.children[1][1]
    outcome_xs = [child.x for _, child in complex_layout.children]

    assert layout.x == pytest.approx(0)
    assert complex_layout.x == pytest.approx(-simple_layout.x)
    assert min(outcome_xs) + max(outcome_xs) == pytest.approx(2 * complex_layout.x)


def test_chance_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        dr.ChanceNode("Bad", {"A": (0.2, 1), "B": (0.2, 0)})
