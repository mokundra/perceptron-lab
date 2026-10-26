from pathlib import Path

import numpy as np
import pytest

from perceptron_lab.datasets import generate_synthetic_data
from perceptron_lab.model import Perceptron
from perceptron_lab.visualization import (
    plot_decision_boundary,
    plot_training_errors,
)


def test_decision_boundary_save(tmp_path: Path) -> None:
    """Check that the decision boundary plot is saved."""
    X, y = generate_synthetic_data(n_samples=60, seed=42)

    model = Perceptron(n_features=2, epochs=10)
    model.train(X, y)

    file_path = tmp_path / "plots" / "boundary.png"

    plot_decision_boundary(
        X,
        y,
        model,
        save_path=file_path,
        show=False,
    )

    assert file_path.is_file()
    assert file_path.stat().st_size > 0


def test_vertical_decision_boundary(tmp_path: Path) -> None:
    """Check plotting when the second weight is zero."""
    model = Perceptron(n_features=2)

    model.w = np.array([1.0, 0.0])
    model.bias = -0.5

    X = np.array(
        [
            [-2.0, -1.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 1], dtype=np.int64)

    file_path = tmp_path / "vertical.png"

    plot_decision_boundary(
        X,
        y,
        model,
        save_path=file_path,
        show=False,
    )

    assert file_path.is_file()
    assert file_path.stat().st_size > 0


def test_training_errors_save(tmp_path: Path) -> None:
    """Check that the training error graph is saved."""
    X, y = generate_synthetic_data(n_samples=60, seed=42)

    model = Perceptron(n_features=2, epochs=10)
    model.train(X, y)

    file_path = tmp_path / "errors.png"

    plot_training_errors(
        model,
        save_path=file_path,
        show=False,
    )

    assert file_path.is_file()
    assert file_path.stat().st_size > 0


def test_zero_weights_boundary() -> None:
    """Check rejection of an undefined decision boundary."""
    X, y = generate_synthetic_data(n_samples=20)

    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="zero weights"):
        plot_decision_boundary(X, y, model, show=False)


def test_invalid_feature_dimensions() -> None:
    """Check rejection of data with three features."""
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1], dtype=np.int64)

    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="2D"):
        plot_decision_boundary(X, y, model, show=False)


def test_plot_errors_before_training() -> None:
    """Check that training history is required."""
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError, match="Train the model"):
        plot_training_errors(model, show=False)
