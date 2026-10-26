"""Run perceptron experiments on synthetic and Iris datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from perceptron_lab.datasets import (
    generate_overlapping_data,
    generate_synthetic_data,
    load_iris_data,
    load_kaggle_penguins,
)
from perceptron_lab.model import Perceptron
from perceptron_lab.visualization import (
    plot_decision_boundary,
    plot_training_errors,
)


@dataclass(frozen=True)
class ExperimentResult:
    """Store accuracy scores and training statistics."""

    dataset: str
    n_train: int
    n_test: int
    train_accuracy: float
    test_accuracy: float
    errors_per_epoch: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    precision: float
    recall: float
    f1: float


def _run_experiment(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    dataset_name: str,
    learning_rate: float,
    epochs: int,
    seed: int,
    plots_dir: Path | None,
    show_plots: bool,
    standardize: bool = False,
) -> ExperimentResult:
    """Train, evaluate, and visualize a binary classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    if standardize:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        if np.any(std == 0):
            raise ValueError("Cannot standardize constant features")

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    model = Perceptron(
        n_features=X.shape[1],
        learning_rate=learning_rate,
        epochs=epochs,
    )

    model.train(X_train, y_train)

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for x, target in zip(X_test, y_test, strict=True):
        predicted = model.predict(x)

        if target == 0 and predicted == 0:
            tn += 1
        elif target == 0 and predicted == 1:
            fp += 1
        elif target == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    result = ExperimentResult(
        dataset=dataset_name,
        n_train=len(y_train),
        n_test=len(y_test),
        train_accuracy=model.score(X_train, y_train),
        test_accuracy=model.score(X_test, y_test),
        errors_per_epoch=tuple(model.errors_history),
        confusion_matrix=((tn, fp), (fn, tp)),
        precision=precision,
        recall=recall,
        f1=f1,
    )

    if plots_dir is not None or show_plots:
        boundary_path = (
            plots_dir / f"{dataset_name}_boundary.png"
            if plots_dir is not None
            else None
        )

        errors_path = (
            plots_dir / f"{dataset_name}_errors.png" if plots_dir is not None else None
        )

        plot_decision_boundary(
            X_train,
            y_train,
            model,
            save_path=boundary_path,
            show=show_plots,
        )

        plot_training_errors(
            model,
            save_path=errors_path,
            show=show_plots,
        )

    return result


def run_synthetic_experiment(
    n_samples: int = 200,
    learning_rate: float = 0.1,
    epochs: int = 50,
    seed: int = 42,
    plots_dir: Path | None = Path("plots"),
    show_plots: bool = False,
) -> ExperimentResult:
    """Run an experiment on generated linearly separable data."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10")

    X, y = generate_synthetic_data(n_samples=n_samples, seed=seed)

    return _run_experiment(
        X,
        y,
        dataset_name="synthetic",
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        plots_dir=plots_dir,
        show_plots=show_plots,
    )


def run_iris_experiment(
    learning_rate: float = 0.1,
    epochs: int = 50,
    seed: int = 42,
    plots_dir: Path | None = Path("plots"),
    show_plots: bool = False,
) -> ExperimentResult:
    """Run an experiment on two classes from the Iris dataset."""
    X, y = load_iris_data()

    return _run_experiment(
        X,
        y,
        dataset_name="iris",
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        plots_dir=plots_dir,
        show_plots=show_plots,
    )


def run_noisy_experiment(
    n_samples: int = 200,
    noise_std: float = 2.0,
    learning_rate: float = 0.1,
    epochs: int = 50,
    seed: int = 42,
    plots_dir: Path | None = Path("plots"),
    show_plots: bool = False,
) -> ExperimentResult:
    """Run an experiment on potentially overlapping classes."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10")

    X, y = generate_overlapping_data(
        n_samples=n_samples,
        noise_std=noise_std,
        seed=seed,
    )

    return _run_experiment(
        X,
        y,
        dataset_name="noisy",
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        plots_dir=plots_dir,
        show_plots=show_plots,
    )


def run_penguins_experiment(
    csv_path: Path = Path("data/penguins_size.csv"),
    learning_rate: float = 0.1,
    epochs: int = 50,
    seed: int = 42,
    plots_dir: Path | None = Path("plots"),
    show_plots: bool = False,
) -> ExperimentResult:
    """Train and evaluate the perceptron on Kaggle penguin data."""
    X, y = load_kaggle_penguins(csv_path)

    return _run_experiment(
        X,
        y,
        dataset_name="penguins",
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        plots_dir=plots_dir,
        show_plots=show_plots,
        standardize=True,
    )
