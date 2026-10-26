"""Binary perceptron for classification into classes 0 and 1."""

import numpy as np
from numpy.typing import NDArray


class Perceptron:
    """A single-layer classifier trained with the perceptron learning rule."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Initialize model parameters and zero weights."""
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w: NDArray[np.float64] = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.errors_history: list[int] = []

    def activate(self, z: float) -> int:
        """Return 1 for nonnegative input and 0 otherwise."""
        return 1 if z >= 0 else 0

    def _validate_features(self, x: NDArray[np.float64]) -> None:
        """Check the shape and values of a single input object."""
        if x.ndim != 1 or x.shape[0] != self.n_features:
            raise ValueError(f"x must have exactly {self.n_features} features")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite numbers")

    def _validate_dataset(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> None:
        """Check feature matrix and binary target labels."""
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n_samples, {self.n_features})")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must contain one label per sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numbers")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only classes 0 and 1")

    def predict(self, x: NDArray[np.float64]) -> int:
        """Predict the binary class of one feature vector."""
        self._validate_features(x)

        z = float(np.dot(self.w, x) + self.bias)

        return self.activate(z)

    def train(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """Update weights for the specified number of epochs."""
        self._validate_dataset(X, y)
        self.errors_history.clear()

        for _ in range(self.epochs):
            mistakes = 0

            for x, target in zip(X, y, strict=True):
                error = int(target) - self.predict(x)

                if error != 0:
                    self.w += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
                    mistakes += 1

            self.errors_history.append(mistakes)

    def score(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        """Return the proportion of correctly classified samples."""
        self._validate_dataset(X, y)

        correct = sum(
            self.predict(x) == int(target) for x, target in zip(X, y, strict=True)
        )

        return correct / len(y)
