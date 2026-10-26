# Perceptron Lab

An educational implementation of a **binary perceptron from scratch** in Python, with data generation, training visualizations, a command-line interface, automated tests, and experiments on real datasets.

The perceptron algorithm is implemented using NumPy; `scikit-learn` is used for loading Iris and splitting datasets, **not** for the classifier itself.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

Install project and development dependencies:

```bash
uv sync --locked --dev
```

## Run experiments

```bash
# Linearly separable synthetic data
uv run perceptron --dataset synthetic --samples 200 --lr 0.1 --epochs 50

# Two classes from the Iris dataset
uv run perceptron --dataset iris --epochs 50

# Synthetic data with overlapping classes
uv run perceptron --dataset noisy --samples 200 --noise 2.0 --epochs 50

# Kaggle Palmer Penguins: Adelie vs Chinstrap
uv run perceptron --dataset penguins --epochs 100
```

Common options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--dataset` | `synthetic`, `iris`, `noisy`, or `penguins` | `synthetic` |
| `--samples` | Number of generated samples | `200` |
| `--lr` | Positive learning rate | `0.1` |
| `--epochs` | Training epochs | `50` |
| `--seed` | Random seed | `42` |
| `--noise` | Standard deviation for noisy data | `2.0` |
| `--data-path` | Path to the penguins CSV | `data/penguins_size.csv` |
| `--show-plots` | Display plots interactively | Off |
| `--no-save-plots` | Do not save plots | Off |

Plots are saved in `plots/` by default. Use `--show-plots` to open them interactively, or `--no-save-plots` to disable saving.

Alternatively, run the package as a module:

```bash
uv run python -m perceptron_lab --dataset synthetic
```

## Kaggle dataset setup

Download the Palmer Penguins `penguins_size.csv` dataset from Kaggle and place it at:

```text
data/penguins_size.csv
```

You may also select a different location:

```bash
uv run perceptron --dataset penguins --data-path path/to/penguins_size.csv --epochs 100
```

The project uses two species, **Adelie (0)** and **Chinstrap (1)**, and two numerical measurements: `culmen_length_mm` and `culmen_depth_mm`. It skips rows with missing or nonfinite required measurements. For this CSV, 219 usable records were found.

Kaggle features are standardized using only training-set statistics, then the same transformation is applied to the test set to avoid data leakage. Data are split 80%/20% with stratification and a reproducible random seed.

The unit tests generate temporary CSVs; running the test suite **does not require downloading the Kaggle file**.

## Algorithm

For an input feature vector $x$, the model predicts class 1 when

$$
w \cdot x + b \geq 0
$$

and class 0 otherwise. After a classification error, parameters are updated using

$$
w \leftarrow w + \eta(y - \hat{y})x
$$

$$
b \leftarrow b + \eta(y - \hat{y})
$$

where $\eta$ is the learning rate. The implementation tracks the number of mistakes in every epoch and supports any positive number of numerical input features, while the decision-boundary plot is limited to two dimensions.

## Example results

The following results were observed with the **Palmer Penguins** dataset, two selected species and two features, `seed=42`, `epochs=100`, and `learning_rate=0.1`:

| Metric | Result |
| --- | ---: |
| Train samples | 175 |
| Test samples | 44 |
| Training accuracy | 96.57% |
| Test accuracy | 100.00% |
| Precision (Chinstrap) | 100.00% |
| Recall (Chinstrap) | 100.00% |
| F1-score (Chinstrap) | 100.00% |
| Confusion matrix `((TN, FP), (FN, TP))` | `((30, 0), (0, 14))` |

These results describe **one** split of the data. Perfect test accuracy on 44 observations does **not** guarantee perfect generalization. Experiments on overlapping synthetic classes show that the classic perceptron does not necessarily converge to zero training mistakes when the data are not linearly separable.

## Automated checks

```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src tests
```

The repository also includes a GitHub Actions workflow in `.github/workflows/ci.yml` to perform these checks on pushes and pull requests.

## Project structure

```text
perceptron-lab/
├── .github/workflows/ci.yml
├── data/                    # Kaggle CSV (download separately)
├── plots/                   # Generated plots
├── src/perceptron_lab/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── model.py
│   ├── datasets.py
│   ├── experiments.py
│   └── visualization.py
├── tests/
│   ├── test_cli.py
│   ├── test_datasets.py
│   ├── test_kaggle.py
│   ├── test_model.py
│   └── test_visualization.py
├── pyproject.toml
└── uv.lock
```


