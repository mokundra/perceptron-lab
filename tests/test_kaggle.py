
"""Tests for Kaggle penguin data and experiments."""

import sys
from pathlib import Path

import numpy as np
import pytest

from perceptron_lab.cli import main
from perceptron_lab.datasets import load_kaggle_penguins
from perceptron_lab.experiments import run_penguins_experiment


@pytest.fixture
def penguins_csv(tmp_path: Path) -> Path:
    """Create a temporary penguin dataset."""
    csv_path = tmp_path / "penguins_size.csv"
    rows = ["species,culmen_length_mm,culmen_depth_mm"]

    for i in range(12):
        rows.append(f"Adelie,{38 + i * 0.1:.2f},{18 + i * 0.05:.2f}")
        rows.append(f"Chinstrap,{48 + i * 0.1:.2f},{19 + i * 0.05:.2f}")

    rows.extend(["Gentoo,45,15", "Adelie,NA,18"])
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def test_load_penguins(penguins_csv: Path) -> None:
    """Check data shape, filtering, and binary classes."""
    X, y = load_kaggle_penguins(penguins_csv)

    assert X.shape == (24, 2)
    assert y.shape == (24,)
    assert np.all(np.isfinite(X))
    assert np.count_nonzero(y == 0) == 12
    assert np.count_nonzero(y == 1) == 12


def test_missing_csv_columns(tmp_path: Path) -> None:
    """Reject a CSV with missing required columns."""
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("species,length\nAdelie,39\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_kaggle_penguins(csv_path)


def test_penguins_experiment(penguins_csv: Path) -> None:
    """Check training, evaluation, and classification metrics."""
    result = run_penguins_experiment(
        csv_path=penguins_csv,
        epochs=20,
        plots_dir=None,
    )

    assert result.n_train == 19
    assert result.n_test == 5
    assert len(result.errors_per_epoch) == 20
    assert 0 <= result.test_accuracy <= 1
    assert 0 <= result.f1 <= 1
    assert sum(map(sum, result.confusion_matrix)) == result.n_test


def test_penguins_cli(
    penguins_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check Kaggle dataset execution through CLI."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perceptron",
            "--dataset",
            "penguins",
            "--data-path",
            str(penguins_csv),
            "--epochs",
            "10",
            "--no-save-plots",
        ],
    )

    main()
    output = capsys.readouterr().out

    assert "Dataset: penguins" in output
    assert "Test accuracy:" in output
    assert "F1-score:" in output
    assert "Confusion matrix:" in output
