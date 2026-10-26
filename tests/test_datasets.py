"""Tests for synthetic and real classification datasets."""

import numpy as np
import pytest

from perceptron_lab.datasets import (
    generate_overlapping_data,
    generate_synthetic_data,
    load_iris_data,
)
from perceptron_lab.model import Perceptron


def test_synthetic_data_shape() -> None:
    """Check the dimensions of generated features and labels."""
    X, y = generate_synthetic_data(n_samples=200, seed=42)

    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64


def test_class_balance() -> None:
    """Check class counts for an odd number of samples."""
    _, y = generate_synthetic_data(n_samples=101, seed=42)

    assert np.count_nonzero(y == 0) == 50
    assert np.count_nonzero(y == 1) == 51


def test_reproducibility() -> None:
    """Check that the same seed produces identical datasets."""
    X1, y1 = generate_synthetic_data(n_samples=100, seed=42)
    X2, y2 = generate_synthetic_data(n_samples=100, seed=42)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize("n_samples", [-5, 0, 1])
def test_invalid_sample_count(n_samples: int) -> None:
    """Check rejection of invalid sample counts."""
    with pytest.raises(ValueError):
        generate_synthetic_data(n_samples=n_samples)


def test_iris_data() -> None:
    """Check the dimensions and labels of the Iris dataset."""
    X, y = load_iris_data()

    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert np.count_nonzero(y == 0) == 50
    assert np.count_nonzero(y == 1) == 50
    assert np.all(np.isfinite(X))


def test_training_on_synthetic_data() -> None:
    """Check that the perceptron learns the generated dataset."""
    X, y = generate_synthetic_data(n_samples=200, seed=42)

    model = Perceptron(n_features=2, epochs=50)
    model.train(X, y)

    assert model.score(X, y) == pytest.approx(1.0)


def test_overlapping_data_shape() -> None:
    """Check dimensions and class balance of noisy data."""
    X, y = generate_overlapping_data(n_samples=101, seed=42)

    assert X.shape == (101, 2)
    assert y.shape == (101,)
    assert np.count_nonzero(y == 0) == 50
    assert np.count_nonzero(y == 1) == 51
    assert np.all(np.isfinite(X))


def test_overlapping_data_reproducibility() -> None:
    """Check reproducibility with the same random seed."""
    X1, y1 = generate_overlapping_data(seed=42)
    X2, y2 = generate_overlapping_data(seed=42)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_noise_increases_spread() -> None:
    """Check that increasing noise increases class dispersion."""
    X_low, y_low = generate_overlapping_data(
        n_samples=200,
        noise_std=0.2,
        seed=42,
    )

    X_high, y_high = generate_overlapping_data(
        n_samples=200,
        noise_std=2.0,
        seed=42,
    )

    spread_low = np.std(X_low[y_low == 0, 0])
    spread_high = np.std(X_high[y_high == 0, 0])

    assert spread_high > spread_low


@pytest.mark.parametrize(
    "noise_std",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_noise(noise_std: float) -> None:
    """Check rejection of invalid noise values."""
    with pytest.raises(ValueError):
        generate_overlapping_data(noise_std=noise_std)
