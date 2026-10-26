"""Command-line interface for perceptron experiments."""

import argparse
import math
from pathlib import Path

from perceptron_lab.experiments import (
    run_iris_experiment,
    run_noisy_experiment,
    run_penguins_experiment,
    run_synthetic_experiment,
)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a binary perceptron."
    )

    parser.add_argument(
        "--dataset",
        choices=("synthetic", "iris", "noisy", "penguins"),
        default="synthetic",
        help="Dataset to use.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of synthetic samples.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate.",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=2.0,
        help="Noise standard deviation for the noisy dataset.",
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/penguins_size.csv"),
        help="Path to the Kaggle penguins CSV file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots on screen.",
    )

    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Disable saving plots.",
    )

    args = parser.parse_args()

    if args.samples < 10:
        parser.error("--samples must be at least 10")

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")

    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")

    if not math.isfinite(args.noise) or args.noise <= 0:
        parser.error("--noise must be finite and positive")

    return args


def main() -> None:
    """Run the selected experiment and print its results."""
    args = parse_args()

    plots_dir = None if args.no_save_plots else Path("plots")

    try:
        if args.dataset == "synthetic":
            result = run_synthetic_experiment(
                n_samples=args.samples,
                learning_rate=args.lr,
                epochs=args.epochs,
                seed=args.seed,
                plots_dir=plots_dir,
                show_plots=args.show_plots,
            )
        elif args.dataset == "noisy":
            result = run_noisy_experiment(
                n_samples=args.samples,
                noise_std=args.noise,
                learning_rate=args.lr,
                epochs=args.epochs,
                seed=args.seed,
                plots_dir=plots_dir,
                show_plots=args.show_plots,
            )
        elif args.dataset == "iris":
            result = run_iris_experiment(
                learning_rate=args.lr,
                epochs=args.epochs,
                seed=args.seed,
                plots_dir=plots_dir,
                show_plots=args.show_plots,
            )
        else:
            result = run_penguins_experiment(
                csv_path=args.data_path,
                learning_rate=args.lr,
                epochs=args.epochs,
                seed=args.seed,
                plots_dir=plots_dir,
                show_plots=args.show_plots,
            )

    except (ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from None

    print(f"Dataset: {result.dataset}")
    print(f"Training samples: {result.n_train}")
    print(f"Test samples: {result.n_test}")
    print(f"Train accuracy: {result.train_accuracy:.2%}")
    print(f"Test accuracy: {result.test_accuracy:.2%}")
    print(f"First epoch mistakes: {result.errors_per_epoch[0]}")
    print(f"Precision: {result.precision:.2%}")
    print(f"Recall: {result.recall:.2%}")
    print(f"F1-score: {result.f1:.2%}")
    print(f"Confusion matrix: {result.confusion_matrix}")

    if plots_dir is not None:
        print(f"Plots saved to: {plots_dir}")
