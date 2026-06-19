"""Correlation matrix validation and helpers."""

from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class CorrelationMatrix(BaseModel):
    """Represent and validate a numeric correlation matrix."""

    matrix: list[list[float]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, matrix: list[list[float]]) -> list[list[float]]:
        """Validate shape, correlation bounds, symmetry, and PSD-ness."""
        if not matrix:
            raise ValueError("Matrix cannot be empty")

        arr = np.asarray(matrix, dtype=float)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {arr.shape}")

        diagonal = np.diag(arr)
        if not np.allclose(diagonal, 1.0):
            bad_indices = np.where(~np.isclose(diagonal, 1.0))[0]
            i = int(bad_indices[0])
            raise ValueError(
                f"Diagonal element at ({i}, {i}) must be 1.0, got {diagonal[i]}"
            )

        if not np.allclose(arr, arr.T):
            diff = np.abs(arr - arr.T)
            i, j = np.unravel_index(np.argmax(diff), diff.shape)
            if i > j:
                i, j = j, i
            raise ValueError(
                f"Matrix is not symmetric: ({i}, {j})={arr[i, j]} != ({j}, {i})={arr[j, i]}"
            )

        if not np.all((arr >= -1.0) & (arr <= 1.0)):
            bad_mask = (arr < -1.0) | (arr > 1.0)
            i, j = np.unravel_index(np.argmax(bad_mask), arr.shape)
            raise ValueError(
                f"Correlation value at ({i}, {j}) must be between -1 and 1, got {arr[i, j]}"
            )

        eigenvalues = np.linalg.eigvalsh(arr)
        min_eigenvalue = float(np.min(eigenvalues))
        if min_eigenvalue < -1e-10:
            raise ValueError(
                "Correlation matrix must be positive semidefinite; "
                f"minimum eigenvalue is {min_eigenvalue}"
            )

        return matrix

    @classmethod
    def from_n_corr(cls, n: int, corr: float) -> Self:
        """Create an ``n`` by ``n`` matrix with a shared off-diagonal correlation."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if not (-1.0 <= corr <= 1.0):
            raise ValueError(f"Correlation value must be between -1 and 1, got {corr}")

        matrix = [[1.0 if i == j else float(corr) for j in range(n)] for i in range(n)]
        return cls(matrix=matrix)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Self:
        """Create a correlation matrix from a NumPy array."""
        if arr.ndim != 2:
            raise ValueError(f"Array must be 2-dimensional, got {arr.ndim}")
        return cls(matrix=arr.tolist())

    def to_numpy(self) -> np.ndarray:
        """Return the correlation matrix as a NumPy array."""
        return np.asarray(self.matrix, dtype=float)

    def plot(
        self,
        ax: Any = None,
        *,
        labels: list[str] | None = None,
        cmap: str = "Spectral",
        show: bool = False,
        colorbar: bool = True,
        **imshow_kwargs: Any,
    ) -> Any:
        """
        Plot the correlation matrix as an annotated heatmap.

        Returns the Matplotlib ``Axes`` object. Importing Matplotlib is deferred
        so non-plotting use stays lightweight. Extra keyword arguments are
        passed to ``imshow``.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        arr = self.to_numpy()
        n = arr.shape[0]

        if labels is not None and len(labels) != n:
            raise ValueError(f"labels must have length {n}, got {len(labels)}")

        image = ax.imshow(
            arr,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            **imshow_kwargs,
        )

        tick_labels = labels if labels is not None else [str(i) for i in range(n)]
        ax.set_xticks(np.arange(n), labels=tick_labels, rotation=90)
        ax.tick_params(
            axis="x", bottom=True, labelbottom=True, top=False, labeltop=False
        )
        ax.set_yticks(np.arange(n), labels=tick_labels)
        ax.grid(False)
        ax.set_title("Correlation matrix")

        for i in range(n):
            for j in range(n):
                text_color = "white" if abs(arr[i, j]) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{arr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

        if colorbar:
            ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.figure.tight_layout()

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return ax
