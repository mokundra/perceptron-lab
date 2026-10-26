"""Functions for generating and loading classification datasets."""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris  # type: ignore[import-untyped]


def generate_synthetic_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two linearly separable classes of 2D points."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    rng = np.random.default_rng(seed)

    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    class_0 = rng.uniform(-3.0, -1.0, size=(n_class_0, 2))
    class_1 = rng.uniform(1.0, 3.0, size=(n_class_1, 2))

    X = np.vstack((class_0, class_1))

    y = np.concatenate(
        (
            np.zeros(n_class_0, dtype=np.int64),
            np.ones(n_class_1, dtype=np.int64),
        )
    )

    indices = rng.permutation(n_samples)

    return X[indices], y[indices]


def load_iris_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load two Iris species using petal length and width."""
    iris = load_iris()

    features = np.asarray(iris.data, dtype=np.float64)
    labels = np.asarray(iris.target, dtype=np.int64)

    mask = labels < 2

    X = features[mask, 2:4]
    y = labels[mask]

    return X, y


def generate_overlapping_data(
    n_samples: int = 200,
    noise_std: float = 2.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two Gaussian classes with potentially overlapping regions."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and positive")

    rng = np.random.default_rng(seed)

    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    class_0 = rng.normal(
        loc=-1.5,
        scale=noise_std,
        size=(n_class_0, 2),
    )

    class_1 = rng.normal(
        loc=1.5,
        scale=noise_std,
        size=(n_class_1, 2),
    )

    X = np.vstack((class_0, class_1))

    y = np.concatenate(
        (
            np.zeros(n_class_0, dtype=np.int64),
            np.ones(n_class_1, dtype=np.int64),
        )
    )

    indices = rng.permutation(n_samples)

    return X[indices], y[indices]


def load_kaggle_penguins(
    csv_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load two penguin species from a Kaggle CSV file."""
    features: list[list[float]] = []
    labels: list[int] = []

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "species",
            "culmen_length_mm",
            "culmen_depth_mm",
        }

        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV is missing required columns")

        for row in reader:
            species = row["species"]

            if species not in ("Adelie", "Chinstrap"):
                continue

            try:
                length = float(row["culmen_length_mm"])
                depth = float(row["culmen_depth_mm"])
            except (ValueError, TypeError):
                continue

            if not np.isfinite(length) or not np.isfinite(depth):
                continue

            features.append([length, depth])
            labels.append(0 if species == "Adelie" else 1)

    if set(labels) != {0, 1}:
        raise ValueError("Both penguin species must be present")

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    return X, y
