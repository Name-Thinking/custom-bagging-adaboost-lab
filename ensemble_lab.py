# %% [markdown]
# # Ensemble Learning Lab: Single Decision Tree vs. Custom Bagging (Parallel) vs. Custom AdaBoost SAMME (Sequential)
#
# - Dataset: Iris via Hugging Face `datasets` (`scikit-learn/iris`), with fallback to `sklearn.datasets.load_iris()`
# - Split: 80% Train / 20% Test (`stratify=y`, fixed `random_state`)
# - Models:
#   1. Single Model: `DecisionTreeClassifier`
#   2. Custom Bagging: Bootstrap sampling (with replacement) + `joblib.Parallel` & `delayed` + Majority Vote
#   3. Custom Boosting: Sequential AdaBoost SAMME (`alpha = ln((1 - err) / err) + ln(K - 1)`) + Weighted Voting

# %%
from __future__ import annotations

import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Cap BLAS thread pools per process before importing NumPy/scikit-learn to avoid
# Windows paging-file OOM errors when joblib spawns multiple loky workers.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
SPECIES_TO_LABEL = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


@dataclass(frozen=True)
class DatasetBundle:
    """Container for loaded Iris features, labels, and provenance metadata."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    source: str


def load_iris_dataset(hf_dataset_id: str = "scikit-learn/iris") -> DatasetBundle:
    """Load the Iris dataset from Hugging Face datasets, falling back to sklearn.

    Drops the ``Id`` column when loading ``scikit-learn/iris`` so row order
    cannot leak the target label.
    """
    try:
        from datasets import load_dataset

        hf_ds = load_dataset(hf_dataset_id, split="train")
        df = hf_ds.to_pandas()

        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        if not all(col in df.columns for col in [*FEATURE_COLUMNS, "Species"]):
            raise ValueError(
                f"Unexpected schema in HF dataset '{hf_dataset_id}': {list(df.columns)}"
            )

        y_mapped = df["Species"].map(SPECIES_TO_LABEL)
        if y_mapped.isna().any():
            raise ValueError("Found unmapped species labels in Hugging Face Iris dataset.")

        X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = y_mapped.to_numpy(dtype=np.int64)
        return DatasetBundle(
            X=X,
            y=y,
            feature_names=FEATURE_COLUMNS.copy(),
            target_names=TARGET_NAMES.copy(),
            source=f"Hugging Face datasets ({hf_dataset_id})",
        )
    except Exception as exc:
        print(
            f"[Fallback] Failed to load '{hf_dataset_id}' from Hugging Face ({exc}). "
            "Using sklearn.datasets.load_iris()."
        )
        iris = load_iris()
        return DatasetBundle(
            X=np.asarray(iris.data, dtype=np.float64),
            y=np.asarray(iris.target, dtype=np.int64),
            feature_names=list(iris.feature_names),
            target_names=list(iris.target_names),
            source="sklearn.datasets.load_iris() [fallback]",
        )


def _validate_fit_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate 2D non-empty feature matrix X and matching 1D (or column) target y."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)

    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError(f"X must be a non-empty 2D array, got shape {X_arr.shape}.")
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    elif y_arr.ndim != 1:
        raise ValueError(
            f"y must be a 1D array or (n_samples, 1) column vector, got shape {y_arr.shape}."
        )
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, got {X_arr.shape[0]} and {y_arr.shape[0]}."
        )
    return X_arr, y_arr


def _spawn_seeds(random_state: int, n_estimators: int) -> list[int]:
    """Generate deterministic per-estimator uint32 seeds from a base random_state."""
    seed_seq = np.random.SeedSequence(random_state)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_seq.spawn(n_estimators)
    ]


def _fit_bootstrap_tree(
    X: np.ndarray,
    y_idx: np.ndarray,
    seed: int,
    max_depth: int | None = None,
) -> DecisionTreeClassifier:
    """Top-level worker function for Windows/Jupyter-safe joblib parallel bagging."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    bootstrap_indices = rng.integers(0, n_samples, size=n_samples)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(X[bootstrap_indices], y_idx[bootstrap_indices])
    return tree


class CustomBaggingClassifier:
    """From-scratch Bagging Classifier using bootstrap sampling, joblib Parallel, and majority voting."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
        batch_size: int | str | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.batch_size = batch_size
        self.classes_: np.ndarray = np.array([])
        self.estimators_: list[DecisionTreeClassifier] = []

    def _resolve_batch_size(self) -> int | str:
        if self.batch_size is not None:
            return self.batch_size
        effective_workers = max(1, joblib.effective_n_jobs(self.n_jobs))
        # One coarse batch per worker minimizes Windows IPC round-trips on small datasets.
        return max(1, math.ceil(self.n_estimators / effective_workers))

    def fit(self, X: np.ndarray, y: np.ndarray) -> CustomBaggingClassifier:
        X_arr, y_arr = _validate_fit_inputs(X, y)

        self.classes_ = np.unique(y_arr)
        y_idx = np.searchsorted(self.classes_, y_arr)

        seeds = _spawn_seeds(self.random_state, self.n_estimators)

        batch_size = self._resolve_batch_size()
        self.estimators_ = Parallel(n_jobs=self.n_jobs, batch_size=batch_size)(
            delayed(_fit_bootstrap_tree)(X_arr, y_idx, seed, self.max_depth)
            for seed in seeds
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_arr = np.asarray(X, dtype=np.float64)
        all_preds = np.asarray(
            [tree.predict(X_arr) for tree in self.estimators_],
            dtype=np.int64,
        )

        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)
        vote_counts = np.zeros((n_samples, n_classes), dtype=np.int64)
        for class_idx in range(n_classes):
            vote_counts[:, class_idx] = np.sum(all_preds == class_idx, axis=0)

        return self.classes_[np.argmax(vote_counts, axis=1)]


class CustomAdaBoostSAMME:
    """From-scratch Multi-class AdaBoost (SAMME) using sequential decision stumps."""

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
        eps: float = 1e-12,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.eps = eps
        self.classes_: np.ndarray = np.array([])
        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimator_weights_: list[float] = []
        self.estimator_errors_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> CustomAdaBoostSAMME:
        X_arr, y_arr = _validate_fit_inputs(X, y)

        n_samples = X_arr.shape[0]
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("AdaBoost SAMME requires at least 2 distinct classes.")

        y_idx = np.searchsorted(self.classes_, y_arr)
        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        seeds = _spawn_seeds(self.random_state, self.n_estimators)

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        random_guess_err_limit = 1.0 - (1.0 / n_classes)

        for idx, seed in enumerate(seeds):
            stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
            stump.fit(X_arr, y_idx, sample_weight=sample_weight)
            y_pred = stump.predict(X_arr)

            incorrect = (y_pred != y_idx).astype(np.float64)
            err = float(np.dot(sample_weight, incorrect) / np.sum(sample_weight))

            if err <= 0.0:
                alpha = (
                    1.0
                    if idx == 0
                    else float(np.log((1.0 - self.eps) / self.eps) + np.log(n_classes - 1))
                )
                self.estimators_.append(stump)
                self.estimator_weights_.append(alpha)
                self.estimator_errors_.append(0.0)
                break

            # Stop if weighted error is no better than multi-class random guessing (alpha <= 0).
            if err >= random_guess_err_limit - self.eps:
                if not self.estimators_:
                    raise RuntimeError(
                        f"First base learner achieved err={err:.6f} >= 1 - 1/K ({random_guess_err_limit:.6f})."
                    )
                break

            err_safe = float(np.clip(err, self.eps, 1.0 - self.eps))
            # SAMME formula: alpha = ln((1 - err) / err) + ln(K - 1)
            alpha = float(np.log((1.0 - err_safe) / err_safe) + np.log(n_classes - 1))

            sample_weight = sample_weight * np.exp(alpha * incorrect)
            weight_sum = float(np.sum(sample_weight))
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                break
            sample_weight /= weight_sum

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        class_scores = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        row_idx = np.arange(n_samples)

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            pred_idx = stump.predict(X_arr)
            class_scores[row_idx, pred_idx] += alpha

        return self.classes_[np.argmax(class_scores, axis=1)]


def evaluate_model(
    name: str,
    model_or_factory: Any | Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """Train and test one model across n_repeats runs and record median timings."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")

    train_times: list[float] = []
    infer_times: list[float] = []
    y_pred: np.ndarray | None = None
    fitted_model: Any = None

    for _ in range(n_repeats):
        model = model_or_factory() if callable(model_or_factory) else model_or_factory

        t_train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t_train_start)

        t_infer_start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_times.append(time.perf_counter() - t_infer_start)

        fitted_model = model

    acc = float(accuracy_score(y_test, y_pred))
    fitted_estimators = len(fitted_model.estimators_) if hasattr(fitted_model, "estimators_") else 1

    return {
        "Model": name,
        "Base Learners": fitted_estimators,
        "Accuracy": acc,
        "Accuracy (%)": acc * 100.0,
        "Training Time (s)": float(statistics.median(train_times)),
        "Inference Time (s)": float(statistics.median(infer_times)),
    }


def _warmup_worker(X_tiny: np.ndarray, y_tiny: np.ndarray, seed: int) -> int:
    """Top-level worker that initializes sklearn tree C-extensions in each loky process."""
    _fit_bootstrap_tree(X_tiny, y_tiny, seed=seed, max_depth=1)
    return os.getpid()


def _warmup_joblib_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_jobs: int,
) -> tuple[float, int]:
    """Initialize all joblib loky worker processes once before timing parallel training."""
    t0 = time.perf_counter()
    n_workers = max(1, joblib.effective_n_jobs(n_jobs))
    X_tiny = X_train[:12]
    y_tiny = y_train[:12]
    pids = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_warmup_worker)(X_tiny, y_tiny, i) for i in range(n_workers * 4)
    )
    return time.perf_counter() - t0, len(set(pids))


def _build_model_factories(
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    include_sequential_bagging: bool = False,
) -> list[tuple[str, Callable[[], Any]]]:
    """Return ordered model factories for deterministic repeated timing benchmarks."""
    factories: list[tuple[str, Callable[[], Any]]] = [
        (
            "Single DecisionTree",
            lambda: DecisionTreeClassifier(random_state=random_state),
        ),
        (
            "Custom Bagging (Parallel)",
            lambda: CustomBaggingClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
        ),
        (
            "Custom AdaBoost SAMME (Sequential)",
            lambda: CustomAdaBoostSAMME(
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        ),
    ]
    if include_sequential_bagging:
        factories.append(
            (
                "Custom Bagging (Sequential, n_jobs=1)",
                lambda: CustomBaggingClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    n_jobs=1,
                    random_state=random_state,
                ),
            )
        )
    return factories


def _prepare_benchmark_context(
    random_state: int,
    n_jobs: int,
    warm_pool: bool,
) -> tuple[DatasetBundle, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, int]]:
    """Load Iris, create stratified 80/20 split, and optionally warm the joblib worker pool."""
    bundle = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=random_state,
        stratify=bundle.y,
    )
    warmup_info = _warmup_joblib_pool(X_train, y_train, n_jobs) if warm_pool else (0.0, 0)
    return bundle, X_train, X_test, y_train, y_test, warmup_info


def run_benchmark(
    n_estimators: int = 500,
    random_state: int = 42,
    n_jobs: int = -1,
    warm_pool: bool = True,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, DatasetBundle, tuple[float, int]]:
    """Load Iris, split 80/20, and benchmark Single Tree, Custom Bagging, and Custom SAMME."""
    if not 100 <= n_estimators <= 500:
        raise ValueError("Lab specification requires n_estimators in range [100, 500].")

    bundle, X_train, X_test, y_train, y_test, warmup_info = _prepare_benchmark_context(
        random_state=random_state,
        n_jobs=n_jobs,
        warm_pool=warm_pool,
    )
    models = _build_model_factories(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)

    results = [
        evaluate_model(name, factory, X_train, y_train, X_test, y_test, n_repeats=n_repeats)
        for name, factory in models
    ]
    results_df = pd.DataFrame(results)
    return results_df, bundle, warmup_info


def run_scaling_benchmark(
    estimator_counts: tuple[int, ...] = (100, 200, 300, 400, 500),
    random_state: int = 42,
    n_jobs: int = -1,
    warm_pool: bool = True,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, DatasetBundle, tuple[float, int]]:
    """Measure median Accuracy, Training Time, and Inference Time across n_estimators in [100, 500]."""
    for n in estimator_counts:
        if not 100 <= n <= 500:
            raise ValueError(f"All estimator counts must lie in [100, 500], got {n}.")

    bundle, X_train, X_test, y_train, y_test, warmup_info = _prepare_benchmark_context(
        random_state=random_state,
        n_jobs=n_jobs,
        warm_pool=warm_pool,
    )
    rows: list[dict[str, Any]] = []

    for n in estimator_counts:
        factories = _build_model_factories(
            n_estimators=n,
            random_state=random_state,
            n_jobs=n_jobs,
            include_sequential_bagging=True,
        )
        for name, factory in factories:
            metrics = evaluate_model(
                name,
                factory,
                X_train,
                y_train,
                X_test,
                y_test,
                n_repeats=n_repeats,
            )
            metrics["n_estimators"] = n
            rows.append(metrics)

    return pd.DataFrame(rows), bundle, warmup_info


def format_results_table(df: pd.DataFrame) -> str:
    """Format benchmark metrics into a readable table."""
    display_df = pd.DataFrame(
        {
            "Model": df["Model"],
            "Estimators": df["Base Learners"],
            "Accuracy": df["Accuracy (%)"].map(lambda v: f"{v:.2f}%"),
            "Training Time (s)": df["Training Time (s)"].map(lambda v: f"{v:.6f}"),
            "Inference Time (s)": df["Inference Time (s)"].map(lambda v: f"{v:.6f}"),
        }
    )
    try:
        from tabulate import tabulate

        return tabulate(display_df, headers="keys", tablefmt="github", showindex=False)
    except ImportError:
        return display_df.to_string(index=False)


def format_benchmark_report(
    results_df: pd.DataFrame,
    dataset_info: DatasetBundle,
    warmup_info: tuple[float, int],
    n_estimators: int,
    random_state: int,
    n_repeats: int,
) -> str:
    """Format the full benchmark header and results table shared by CLI output and out.txt."""
    warmup_s, warmed_workers = warmup_info
    return (
        "=== Ensemble Learning Lab (Iris 80/20 Split) ===\n"
        f"Dataset Source   : {dataset_info.source}\n"
        f"Shape            : {dataset_info.X.shape[0]} rows x {dataset_info.X.shape[1]} features\n"
        f"CPU Cores        : {os.cpu_count()}\n"
        f"n_estimators     : {n_estimators} | random_state: {random_state} | timing: median of {n_repeats} runs\n"
        f"Joblib Pool Warm : {warmup_s:.4f} s across {warmed_workers} worker processes "
        "(one-time Windows spawn excluded from fit timing)\n\n"
        f"{format_results_table(results_df)}"
    )


if __name__ == "__main__":
    N_ESTIMATORS = 500
    RANDOM_STATE = 42
    N_REPEATS = 5

    summary_df, dataset_info, warmup_info = run_benchmark(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        warm_pool=True,
        n_repeats=N_REPEATS,
    )

    print(
        format_benchmark_report(
            results_df=summary_df,
            dataset_info=dataset_info,
            warmup_info=warmup_info,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_repeats=N_REPEATS,
        )
    )
