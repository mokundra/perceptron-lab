"""Visualization functions for perceptron experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from perceptron_lab.model import Perceptron


def plot_decision_boundary(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    model: Perceptron,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot two classes and the perceptron's decision boundary."""
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
        raise ValueError("X must contain 2D points")

    if y.shape != (X.shape[0],):
        raise ValueError("y must contain one label per point")

    if model.n_features != 2:
        raise ValueError("Visualization requires exactly two features")

    if np.allclose(model.w, 0.0):
        raise ValueError("Cannot draw a boundary with zero weights")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Class 0",
        alpha=0.8,
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Class 1",
        alpha=0.8,
    )

    w1, w2 = float(model.w[0]), float(model.w[1])

    if np.isclose(w2, 0.0):
        ax.axvline(
            -model.bias / w1,
            linestyle="--",
            label="Decision boundary",
        )
    else:
        x_line = np.linspace(
            float(X[:, 0].min()) - 1,
            float(X[:, 0].max()) + 1,
            200,
        )

        y_line = -(w1 * x_line + model.bias) / w2

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            label="Decision boundary",
        )

    ax.set_title("Perceptron decision boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def plot_training_errors(
    model: Perceptron,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot the number of training mistakes per epoch."""
    if not model.errors_history:
        raise ValueError("Train the model before plotting mistakes")

    fig, ax = plt.subplots(figsize=(7, 4))

    epochs = range(1, len(model.errors_history) + 1)

    ax.plot(
        epochs,
        model.errors_history,
        marker="o",
    )

    ax.set_title("Training mistakes by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of mistakes")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
