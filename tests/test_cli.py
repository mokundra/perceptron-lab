"""Tests for the perceptron command-line interface."""

import sys

import pytest

from perceptron_lab.cli import main, parse_args


def test_default_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check default command-line arguments."""
    monkeypatch.setattr(sys, "argv", ["perceptron"])

    args = parse_args()

    assert args.dataset == "synthetic"
    assert args.samples == 200
    assert args.lr == 0.1
    assert args.epochs == 50
    assert args.seed == 42


def test_synthetic_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check a complete synthetic experiment through CLI."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perceptron",
            "--dataset",
            "synthetic",
            "--samples",
            "40",
            "--epochs",
            "10",
            "--no-save-plots",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Dataset: synthetic" in output
    assert "Training samples: 32" in output
    assert "Test samples: 8" in output
    assert "Train accuracy:" in output
    assert "Test accuracy:" in output


def test_iris_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check a complete Iris experiment through CLI."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["perceptron", "--dataset", "iris", "--no-save-plots"],
    )

    main()

    output = capsys.readouterr().out

    assert "Dataset: iris" in output
    assert "Training samples: 80" in output
    assert "Test samples: 20" in output


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("--lr", "-0.5"),
        ("--epochs", "0"),
        ("--samples", "5"),
        ("--seed", "-1"),
    ],
)
def test_invalid_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: str,
) -> None:
    """Check rejection of invalid command-line values."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["perceptron", argument, value],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_noisy_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check a noisy experiment through the command-line interface."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perceptron",
            "--dataset",
            "noisy",
            "--samples",
            "40",
            "--noise",
            "2.0",
            "--epochs",
            "10",
            "--no-save-plots",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Dataset: noisy" in output
    assert "Training samples: 32" in output
    assert "Test samples: 8" in output
    assert "Train accuracy:" in output
    assert "Test accuracy:" in output


@pytest.mark.parametrize(
    "noise",
    ["0", "-1", "nan", "inf"],
)
def test_invalid_noise_cli(
    monkeypatch: pytest.MonkeyPatch,
    noise: str,
) -> None:
    """Check rejection of invalid noise levels."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["perceptron", "--dataset", "noisy", "--noise", noise],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
