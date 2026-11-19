"""Generate unslopped, reproducible visualizations for the Ensemble Learning Lab."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ensemble_lab import (
    CustomAdaBoostSAMME,
    CustomBaggingClassifier,
    DatasetBundle,
    format_benchmark_report,
    load_iris_dataset,
    run_scaling_benchmark,
)

OUTPUT_DIR = Path(__file__).resolve().parent

# Cohesive light-paper palette (Graphite ramp + 3 semantic series accents)
TEXT_PRIMARY = "#171717"
TEXT_SECONDARY = "#525252"
GRID_COLOR = "#e5e5e5"
SPINE_COLOR = "#d4d4d4"

COLOR_TREE = "#2563eb"
COLOR_BAGGING = "#dc2626"
COLOR_BOOSTING = "#16a34a"

ESTIMATOR_COUNTS = (100, 200, 300, 400, 500)


def setup_plot_style() -> None:
    """Configure clean matplotlib defaults with zero missing-font warnings."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "xtick.color": TEXT_SECONDARY,
            "ytick.labelsize": 9,
            "ytick.color": TEXT_SECONDARY,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#e5e5e5",
            "legend.fontsize": 8.5,
        }
    )


def _style_axis(ax: matplotlib.axes.Axes) -> None:
    """Apply quiet axis styling with hidden top/right spines."""
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def plot_scaling_dashboard(
    scaling_df: pd.DataFrame,
    save_path: Path,
    n_repeats: int = 5,
    n_test_samples: int = 30,
    show_figure: bool = False,
) -> None:
    """Create a 2x2 dashboard matching Figure 1: Accuracy, Train Time, Inference Time, and Speedup."""
    import matplotlib.pyplot as plt

    required_models = (
        "Single DecisionTree",
        "Custom Bagging (Parallel)",
        "Custom Bagging (Sequential, n_jobs=1)",
        "Custom AdaBoost SAMME (Sequential)",
    )
    model_frames: dict[str, pd.DataFrame] = {}
    ns: list[int] | None = None
    for model_name in required_models:
        sub = scaling_df[scaling_df["Model"] == model_name].sort_values("n_estimators")
        if sub.empty:
            raise ValueError(f"Missing scaling benchmark rows for model '{model_name}'.")
        model_ns = [int(v) for v in sub["n_estimators"].to_numpy()]
        if ns is None:
            ns = model_ns
        elif model_ns != ns:
            raise ValueError(
                f"Mismatched n_estimators for model '{model_name}': expected {ns}, got {model_ns}."
            )
        model_frames[model_name] = sub

    assert ns is not None

    def _series(model_name: str, col: str) -> np.ndarray:
        return model_frames[model_name][col].to_numpy(dtype=np.float64)

    acc_tree = _series("Single DecisionTree", "Accuracy")
    acc_bag = _series("Custom Bagging (Parallel)", "Accuracy")
    acc_ada = _series("Custom AdaBoost SAMME (Sequential)", "Accuracy")

    train_tree = _series("Single DecisionTree", "Training Time (s)")
    train_bag_par = _series("Custom Bagging (Parallel)", "Training Time (s)")
    train_bag_seq = _series("Custom Bagging (Sequential, n_jobs=1)", "Training Time (s)")
    train_ada = _series("Custom AdaBoost SAMME (Sequential)", "Training Time (s)")

    inf_tree_ms = _series("Single DecisionTree", "Inference Time (s)") * 1000.0
    inf_bag_ms = _series("Custom Bagging (Parallel)", "Inference Time (s)") * 1000.0
    inf_ada_ms = _series("Custom AdaBoost SAMME (Sequential)", "Inference Time (s)") * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    ax_acc, ax_train = axes[0]
    ax_inf, ax_spd = axes[1]

    # Panel 1: Test Accuracy
    ax_acc.plot(ns, acc_tree, color=COLOR_TREE, linestyle="--", marker="o", markersize=5.5, linewidth=1.8, label="Single Decision Tree")
    ax_acc.plot(ns, acc_bag, color=COLOR_BAGGING, linestyle="-", marker="s", markersize=5.5, linewidth=1.8, label="Custom Bagging (Parallel)")
    ax_acc.plot(ns, acc_ada, color=COLOR_BOOSTING, linestyle="-", marker="^", markersize=6.0, linewidth=1.8, label="Custom AdaBoost SAMME (Sequential)")
    ax_acc.set_title("Test Accuracy", pad=10)
    ax_acc.set_xlabel("Number of base learners (n_estimators)")
    ax_acc.set_ylabel("Test accuracy")
    ax_acc.set_xticks(ns)
    ax_acc.set_ylim(0.800, 1.005)
    ax_acc.set_yticks(np.arange(0.800, 1.001, 0.025))
    ax_acc.legend(loc="lower right", framealpha=0.95)
    if np.allclose(acc_tree, acc_ada):
        anchor_idx = min(1, len(ns) - 1)
        anchor_x = ns[anchor_idx]
        anchor_y = float(acc_ada[anchor_idx])
        text_x = ns[0] + (ns[-1] - ns[0]) * 0.05 if len(ns) > 1 else anchor_x
        ax_acc.annotate(
            "Single Decision Tree = Custom AdaBoost SAMME (overlapping)",
            xy=(anchor_x, anchor_y),
            xytext=(text_x, anchor_y - 0.036),
            fontsize=8.5,
            color=TEXT_SECONDARY,
            arrowprops={"arrowstyle": "->", "color": "#737373", "lw": 0.9},
        )

    # Panel 2: Training Time
    ax_train.plot(ns, train_tree, color=COLOR_TREE, linestyle="--", marker="o", markersize=5.5, linewidth=1.8, label="Single Decision Tree")
    ax_train.plot(ns, train_bag_par, color=COLOR_BAGGING, linestyle="-", marker="s", markersize=5.5, linewidth=1.8, label="Custom Bagging (Parallel)")
    ax_train.plot(ns, train_ada, color=COLOR_BOOSTING, linestyle="-", marker="^", markersize=6.0, linewidth=1.8, label="Custom AdaBoost SAMME (Sequential)")
    ax_train.set_title(f"Training Time (median of {n_repeats} runs)", pad=10)
    ax_train.set_xlabel("Number of base learners (n_estimators)")
    ax_train.set_ylabel("Seconds")
    ax_train.set_xticks(ns)
    ax_train.set_ylim(-0.02, max(train_ada.max(), train_bag_par.max()) * 1.18)
    ax_train.legend(loc="upper left", framealpha=0.95)

    # Panel 3: Inference Time (ms)
    ax_inf.plot(ns, inf_tree_ms, color=COLOR_TREE, linestyle="--", marker="o", markersize=5.5, linewidth=1.8, label="Single Decision Tree")
    ax_inf.plot(ns, inf_bag_ms, color=COLOR_BAGGING, linestyle="-", marker="s", markersize=5.5, linewidth=1.8, label="Custom Bagging (Parallel)")
    ax_inf.plot(ns, inf_ada_ms, color=COLOR_BOOSTING, linestyle="-", marker="^", markersize=6.0, linewidth=1.8, label="Custom AdaBoost SAMME (Sequential)")
    ax_inf.set_title(f"Inference Time, {n_test_samples} test samples (median of {n_repeats})", pad=10)
    ax_inf.set_xlabel("Number of base learners (n_estimators)")
    ax_inf.set_ylabel("Milliseconds")
    ax_inf.set_xticks(ns)
    ax_inf.set_ylim(-2.0, max(inf_ada_ms.max(), inf_bag_ms.max()) * 1.2)
    ax_inf.legend(loc="upper left", framealpha=0.95)

    # Panel 4: Bagging Sequential vs Parallel Speedup
    ax_spd.plot(ns, train_bag_seq, color=COLOR_TREE, linestyle="-", marker="o", markersize=5.5, linewidth=1.8, label="Sequential (n_jobs = 1)")
    ax_spd.plot(ns, train_bag_par, color=COLOR_BAGGING, linestyle="-", marker="s", markersize=5.5, linewidth=1.8, label="Parallel (n_jobs = -1)")
    ax_spd.set_title(f"Bagging training: Sequential vs Parallel (median of {n_repeats})", pad=10)
    ax_spd.set_xlabel("Number of base learners (n_estimators)")
    ax_spd.set_ylabel("Seconds")
    ax_spd.set_xticks(ns)
    ax_spd.set_ylim(0.0, max(train_bag_seq.max(), train_bag_par.max()) * 1.18)
    ax_spd.legend(loc="upper left", framealpha=0.95)

    last_n = ns[-1]
    speedup_last = float(train_bag_seq[-1] / train_bag_par[-1]) if train_bag_par[-1] > 0 else 0.0
    text_x_spd = ns[0] + (last_n - ns[0]) * 0.55 if len(ns) > 1 else last_n
    ax_spd.annotate(
        f"Speedup at n={last_n}: {speedup_last:.2f}x",
        xy=(last_n, float(train_bag_par[-1])),
        xytext=(text_x_spd, float(train_bag_par[-1]) * 0.48),
        fontsize=9,
        fontweight="bold",
        color=COLOR_BAGGING,
        arrowprops={"arrowstyle": "->", "color": COLOR_BAGGING, "lw": 1.0},
    )

    for ax in axes.ravel():
        _style_axis(ax)

    fig.suptitle("Ensemble Learning on Iris: Single Tree vs Bagging vs AdaBoost SAMME", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.4, w_pad=2.4)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close(fig)


PRIMARY_MODEL_ORDER = (
    "Single DecisionTree",
    "Custom Bagging (Parallel)",
    "Custom AdaBoost SAMME (Sequential)",
)
SEQUENTIAL_BAGGING_MODEL = "Custom Bagging (Sequential, n_jobs=1)"


def _select_primary_models(scaling_df: pd.DataFrame, n_estimators: int) -> pd.DataFrame:
    """Filter scaling results to the three primary models for a given n_estimators."""
    return scaling_df[
        (scaling_df["n_estimators"] == n_estimators)
        & (scaling_df["Model"] != SEQUENTIAL_BAGGING_MODEL)
    ].copy()


def plot_metrics_comparison(
    scaling_df: pd.DataFrame,
    save_path: Path,
    n_estimators: int = 500,
    n_repeats: int = 5,
) -> None:
    """Plot unslopped n=500 summary bars derived directly from the median-of-5 benchmark."""
    import matplotlib.pyplot as plt

    sub = _select_primary_models(scaling_df, n_estimators=n_estimators)
    sub["Model"] = pd.Categorical(sub["Model"], categories=list(PRIMARY_MODEL_ORDER), ordered=True)
    sub = sub.sort_values("Model")

    labels = ["Single Tree", "Custom Bagging", "AdaBoost SAMME"]
    colors = [COLOR_TREE, COLOR_BAGGING, COLOR_BOOSTING]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    # 1. Accuracy (full 0-100 scale to avoid truncated bar distortion)
    acc_values = sub["Accuracy (%)"].to_numpy()
    bars0 = axes[0].bar(labels, acc_values, color=colors, width=0.52, alpha=0.88)
    axes[0].set_title("Test Accuracy (%)", pad=10)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 108)
    for bar in bars0:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.5, color=TEXT_PRIMARY)

    # 2. Training Time (median of 5)
    train_times = sub["Training Time (s)"].to_numpy()
    bars1 = axes[1].bar(labels, train_times, color=colors, width=0.52, alpha=0.88)
    axes[1].set_title(f"Training Time (median of {n_repeats})", pad=10)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_ylim(0, max(train_times) * 1.22)
    for bar in bars1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + max(train_times) * 0.025, f"{h:.4f} s", ha="center", va="bottom", fontweight="bold", fontsize=9.5, color=TEXT_PRIMARY)

    # 3. Inference Time (median of 5)
    infer_ms = sub["Inference Time (s)"].to_numpy() * 1000.0
    bars2 = axes[2].bar(labels, infer_ms, color=colors, width=0.52, alpha=0.88)
    axes[2].set_title(f"Inference Time (median of {n_repeats})", pad=10)
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_ylim(0, max(infer_ms) * 1.22)
    for bar in bars2:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, h + max(infer_ms) * 0.025, f"{h:.2f} ms", ha="center", va="bottom", fontweight="bold", fontsize=9.5, color=TEXT_PRIMARY)

    for ax in axes:
        _style_axis(ax)

    fig.suptitle(f"Model Comparison at n_estimators={n_estimators} (Iris 80/20 Split, median of {n_repeats} runs)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.2)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_decision_boundaries(save_path: Path, n_estimators: int = 500, random_state: int = 42) -> None:
    """Plot 2D Petal slice of the actual 4-feature models so reported test accuracies stay 100% consistent."""
    import matplotlib.pyplot as plt

    bundle = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X, bundle.y, test_size=0.2, random_state=random_state, stratify=bundle.y
    )

    # Hold Sepal Length (col 0) and Sepal Width (col 1) at training medians while sweeping Petal features (cols 2, 3)
    sepal_medians = np.median(X_train[:, :2], axis=0)
    x_min, x_max = bundle.X[:, 2].min() - 0.4, bundle.X[:, 2].max() + 0.4
    y_min, y_max = bundle.X[:, 3].min() - 0.4, bundle.X[:, 3].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025), np.arange(y_min, y_max, 0.025))
    grid_4d = np.column_stack(
        [
            np.full(xx.size, sepal_medians[0]),
            np.full(xx.size, sepal_medians[1]),
            xx.ravel(),
            yy.ravel(),
        ]
    )

    models = [
        ("Single DecisionTree", DecisionTreeClassifier(random_state=random_state)),
        ("Custom Bagging (500 trees)", CustomBaggingClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
        ("Custom AdaBoost SAMME (500 stumps)", CustomAdaBoostSAMME(n_estimators=n_estimators, random_state=random_state)),
    ]

    cmap_light = ListedColormap(["#fee2e2", "#dbeafe", "#dcfce7"])
    cmap_bold = ["#dc2626", "#2563eb", "#16a34a"]
    class_levels = np.arange(len(cmap_bold) + 1) - 0.5
    class_norm = BoundaryNorm(class_levels, ncolors=len(cmap_bold))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), sharex=True, sharey=True)

    for ax, (title, model) in zip(axes, models):
        model.fit(X_train, y_train)
        full_test_acc = accuracy_score(y_test, model.predict(X_test)) * 100.0
        Z = model.predict(grid_4d).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=class_levels, cmap=cmap_light, norm=class_norm, alpha=0.85)
        ax.contour(xx, yy, Z, levels=class_levels, colors="#9ca3af", linewidths=0.7)

        for class_idx, color in enumerate(cmap_bold):
            train_mask = y_train == class_idx
            ax.scatter(
                X_train[train_mask, 2],
                X_train[train_mask, 3],
                c=color,
                edgecolor="#ffffff",
                linewidth=0.6,
                s=30,
                alpha=0.45,
                label=f"Train {bundle.target_names[class_idx]}" if ax == axes[0] else None,
            )
            test_mask = y_test == class_idx
            ax.scatter(
                X_test[test_mask, 2],
                X_test[test_mask, 3],
                c=color,
                edgecolor="#171717",
                linewidth=1.1,
                s=58,
                marker="o",
                label=f"Test {bundle.target_names[class_idx]}" if ax == axes[0] else None,
            )

        ax.set_title(f"{title}\n4D Test Accuracy: {full_test_acc:.2f}%", pad=8)
        ax.set_xlabel("Petal Length (cm)")
        _style_axis(ax)

    axes[0].set_ylabel("Petal Width (cm)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=6, frameon=True)
    fig.suptitle(
        "Decision Boundaries (2D Petal Slice; Sepal Length & Width fixed at train median)",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84], w_pad=1.6)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for chart generation."""
    parser = argparse.ArgumentParser(description="Generate Ensemble Learning Lab charts.")
    parser.add_argument("--show", action="store_true", help="Display Figure 1 interactively.")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats per estimator count.")
    return parser


def _write_summary_report(
    scaling_df: pd.DataFrame,
    bundle: DatasetBundle,
    warmup_info: tuple[float, int],
    output_path: Path,
    n_estimators: int = 500,
    random_state: int = 42,
    n_repeats: int = 5,
) -> Path:
    """Write the n=500 primary-model benchmark summary table to out.txt."""
    sub_primary = _select_primary_models(scaling_df, n_estimators=n_estimators)
    out_content = (
        format_benchmark_report(
            results_df=sub_primary,
            dataset_info=bundle,
            warmup_info=warmup_info,
            n_estimators=n_estimators,
            random_state=random_state,
            n_repeats=n_repeats,
        )
        + "\n"
    )
    output_path.write_text(out_content, encoding="utf-8")
    return output_path


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.show:
        matplotlib.use("Agg")

    setup_plot_style()

    print(f"1. Running scaling benchmark ({ESTIMATOR_COUNTS}, median of {args.repeats} runs)...")
    scaling_df, bundle, warmup_info = run_scaling_benchmark(
        estimator_counts=ESTIMATOR_COUNTS,
        random_state=42,
        n_jobs=-1,
        warm_pool=True,
        n_repeats=args.repeats,
    )
    warmup_s, workers = warmup_info
    print(f"   -> Warmed {workers} worker processes in {warmup_s:.3f} s")

    scaling_file = OUTPUT_DIR / "estimators_scaling.png"
    plot_scaling_dashboard(
        scaling_df,
        scaling_file,
        n_repeats=args.repeats,
        n_test_samples=int(round(len(bundle.y) * 0.2)),
        show_figure=args.show,
    )
    print(f"   -> Saved: {scaling_file}")

    print("2. Generating n=500 summary bar comparison...")
    metrics_file = OUTPUT_DIR / "ensemble_comparison.png"
    plot_metrics_comparison(scaling_df, metrics_file, n_estimators=500, n_repeats=args.repeats)
    print(f"   -> Saved: {metrics_file}")

    # Synchronize out.txt with the exact n=500 median-of-5 benchmark metrics
    _write_summary_report(
        scaling_df=scaling_df,
        bundle=bundle,
        warmup_info=warmup_info,
        output_path=OUTPUT_DIR / "out.txt",
        n_estimators=500,
        random_state=42,
        n_repeats=args.repeats,
    )
    print("   -> Synchronized out.txt with exact 5-run median metrics")

    print("3. Generating consistent 4D-slice Decision Boundaries plot...")
    boundary_file = OUTPUT_DIR / "decision_boundaries.png"
    plot_decision_boundaries(boundary_file, n_estimators=500, random_state=42)
    print(f"   -> Saved: {boundary_file}")

    print("\nALL PLOTS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
