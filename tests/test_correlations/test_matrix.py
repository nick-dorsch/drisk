import numpy as np
import pytest
from pydantic import ValidationError

from drisk.correlations import CorrelationMatrix


def test_correlation_matrix_validates_and_roundtrips_numpy() -> None:
    matrix = [[1.0, 0.5], [0.5, 1.0]]

    corr = CorrelationMatrix(matrix=matrix)

    assert corr.matrix == matrix
    np.testing.assert_array_equal(corr.to_numpy(), np.array(matrix))
    np.testing.assert_array_equal(
        CorrelationMatrix.from_numpy(np.array(matrix)).to_numpy(), np.array(matrix)
    )


def test_correlation_matrix_from_n_corr() -> None:
    corr = CorrelationMatrix.from_n_corr(3, 0.25)

    assert corr.matrix == [
        [1.0, 0.25, 0.25],
        [0.25, 1.0, 0.25],
        [0.25, 0.25, 1.0],
    ]


def test_correlation_matrix_rejects_invalid_shape_and_values() -> None:
    with pytest.raises(ValidationError):
        CorrelationMatrix(matrix=[])

    with pytest.raises(ValidationError):
        CorrelationMatrix(matrix=[[1.0, 0.5], [0.5, 1.0], [0.0, 0.0]])

    with pytest.raises(ValidationError):
        CorrelationMatrix(matrix=[[1.0, 0.5], [0.5, 0.9]])

    with pytest.raises(ValidationError):
        CorrelationMatrix(matrix=[[1.0, 0.5], [0.3, 1.0]])

    with pytest.raises(ValidationError):
        CorrelationMatrix(matrix=[[1.0, 1.2], [1.2, 1.0]])


def test_correlation_matrix_rejects_non_psd_matrix() -> None:
    with pytest.raises(ValidationError):
        CorrelationMatrix(
            matrix=[
                [1.0, 0.9, 0.9],
                [0.9, 1.0, -0.9],
                [0.9, -0.9, 1.0],
            ]
        )


def test_correlation_matrix_plot_returns_annotated_heatmap() -> None:
    import matplotlib.pyplot as plt

    corr = CorrelationMatrix(matrix=[[1.0, -0.25], [-0.25, 1.0]])
    _, ax = plt.subplots()

    returned_ax = corr.plot(ax=ax, labels=["a", "b"])

    assert returned_ax is ax
    assert len(ax.images) == 1
    assert len(ax.texts) == 4

    plt.close(ax.figure)
