"""Tests for the binary perceptron model."""

import numpy as np
import pytest

from perceptron_lab.model import Perceptron


def test_initialization() -> None:
    """Check initial model parameters."""
    model = Perceptron(n_features=3)

    assert len(model.w) == 3
    assert np.all(model.w == 0)
    assert model.bias == 0.0
    assert model.errors_history == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-5.0, 0),
        (0.0, 1),
        (5.0, 1),
    ],
)
def test_activation(value: float, expected: int) -> None:
    """Check the threshold activation function  ."""
    model = Perceptron(n_features=2)

    assert model.activate(value) == expected


def test_prediction() -> None:
    """Check predictions with manually assigned weights."""
    model = Perceptron(n_features=2)

    model.w = np.array([1.0, 2.0])
    model.bias = -3.0

    assert model.predict(np.array([2.0, 1.0])) == 1
    assert model.predict(np.array([0.0, 0.0])) == 0


def test_training() -> None:
    """Check learning on linearly separable data."""
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    model = Perceptron(n_features=2, epochs=10)
    model.train(X, y)

    assert model.score(X, y) == pytest.approx(1.0)
    assert len(model.errors_history) == 10
    assert model.errors_history[-1] == 0


def test_score() -> None:
    """Check classification accuracy."""
    model = Perceptron(n_features=2)

    model.w = np.array([1.0, 1.0])

    X = np.array(
        [
            [2.0, 1.0],
            [-2.0, -1.0],
        ]
    )
    y = np.array([1, 0], dtype=np.int64)

    assert model.score(X, y) == pytest.approx(1.0)


def test_invalid_features() -> None:
    """Check rejection of incorrect feature dimensions."""
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError):
        model.predict(np.array([1.0, 2.0, 3.0]))
