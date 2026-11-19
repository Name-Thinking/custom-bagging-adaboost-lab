import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib.axes import Axes

import ensemble_lab
import plot_results


class TestDataLoading(unittest.TestCase):
    def test_load_iris_drops_id_and_returns_4_features(self) -> None:
        from sklearn.datasets import load_iris

        iris = load_iris()
        species_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        fake_df = pd.DataFrame(iris.data, columns=ensemble_lab.FEATURE_COLUMNS)
        fake_df.insert(0, "Id", np.arange(1, len(fake_df) + 1))
        fake_df["Species"] = [species_names[int(idx)] for idx in iris.target]

        class _FakeHFDataset:
            def to_pandas(self) -> pd.DataFrame:
                return fake_df.copy()

        with patch("datasets.load_dataset", return_value=_FakeHFDataset()) as mock_load:
            bundle = ensemble_lab.load_iris_dataset()

        mock_load.assert_called_once_with("scikit-learn/iris", split="train")
        self.assertEqual(bundle.X.shape, (150, 4))
        self.assertEqual(bundle.y.shape, (150,))
        self.assertEqual(bundle.feature_names, ensemble_lab.FEATURE_COLUMNS)
        self.assertEqual(len(bundle.target_names), 3)
        self.assertNotIn("Id", bundle.feature_names)
        self.assertNotIn("id", bundle.feature_names)
        self.assertIn("Hugging Face", bundle.source)

    def test_load_iris_fallback_to_sklearn_on_hf_error(self) -> None:
        with patch("datasets.load_dataset", side_effect=RuntimeError("offline")):
            bundle = ensemble_lab.load_iris_dataset()
        self.assertEqual(bundle.X.shape, (150, 4))
        self.assertEqual(bundle.y.shape, (150,))
        self.assertEqual(len(bundle.feature_names), 4)
        self.assertEqual(len(bundle.target_names), 3)
        self.assertIn("fallback", bundle.source)


class TestCustomBaggingClassifier(unittest.TestCase):
    def test_bootstrap_sample_size_and_replacement(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(60, 4))
        y = np.repeat([0, 1, 2], 20)

        tree = ensemble_lab._fit_bootstrap_tree(X, y, seed=42, max_depth=None)
        self.assertEqual(tree.tree_.n_node_samples[0], len(X))

        draw_rng = np.random.default_rng(42)
        indices = draw_rng.integers(0, len(X), size=len(X))
        self.assertEqual(len(indices), len(X))
        self.assertLess(len(np.unique(indices)), len(X))

    def test_parallel_fit_determinism_with_string_labels(self) -> None:
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [5.0, 5.0],
                [5.1, 5.1],
                [10.0, 10.0],
                [10.1, 10.1],
            ]
        )
        y = np.array(["a", "a", "b", "b", "c", "c"])

        m1 = ensemble_lab.CustomBaggingClassifier(n_estimators=20, n_jobs=2, random_state=7).fit(X, y)
        m2 = ensemble_lab.CustomBaggingClassifier(n_estimators=20, n_jobs=2, random_state=7).fit(X, y)

        self.assertEqual(len(m1.estimators_), 20)
        preds1 = m1.predict(X)
        preds2 = m2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_array_equal(preds1, y)

    def test_majority_vote_with_custom_estimators(self) -> None:
        class _StubTree:
            def __init__(self, output: np.ndarray) -> None:
                self._output = output

            def predict(self, X: np.ndarray) -> np.ndarray:
                return self._output

        model = ensemble_lab.CustomBaggingClassifier(n_estimators=3)
        model.classes_ = np.array([10, 20, 30])
        model.estimators_ = [
            _StubTree(np.array([2, 0])),
            _StubTree(np.array([2, 1])),
            _StubTree(np.array([0, 1])),
        ]
        preds = model.predict(np.zeros((2, 2)))
        np.testing.assert_array_equal(preds, np.array([30, 20]))

    def test_fit_rejects_mismatched_sample_counts_and_empty_input(self) -> None:
        X_short = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y_long = np.array([0, 0, 1, 1, 2, 2])
        with self.assertRaises(ValueError):
            ensemble_lab.CustomBaggingClassifier(n_estimators=5, n_jobs=1, random_state=42).fit(X_short, y_long)

        X_long = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        y_short = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            ensemble_lab.CustomBaggingClassifier(n_estimators=5, n_jobs=1, random_state=42).fit(X_long, y_short)

        with self.assertRaises(ValueError):
            ensemble_lab.CustomBaggingClassifier(n_estimators=5, n_jobs=1, random_state=42).fit(
                np.empty((0, 4)), np.empty((0,))
            )


class TestCustomAdaBoostSAMME(unittest.TestCase):
    def test_samme_alpha_exact_formula(self) -> None:
        X = np.array([[0.0], [0.5], [2.0], [2.5], [4.0], [4.5]])
        y = np.array([0, 0, 1, 1, 2, 2])

        model = ensemble_lab.CustomAdaBoostSAMME(n_estimators=1, random_state=42).fit(X, y)
        self.assertEqual(len(model.estimators_), 1)
        self.assertEqual(model.estimators_[0].max_depth, 1)

        err = model.estimator_errors_[0]
        self.assertAlmostEqual(err, 1.0 / 3.0, places=6)

        expected_alpha = math.log((1.0 - err) / err) + math.log(3 - 1)
        self.assertAlmostEqual(model.estimator_weights_[0], expected_alpha, places=6)

    def test_weighted_voting_prefers_higher_alpha(self) -> None:
        class _StubStump:
            def __init__(self, pred: int) -> None:
                self._pred = pred

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.full(X.shape[0], self._pred, dtype=np.int64)

        model = ensemble_lab.CustomAdaBoostSAMME(n_estimators=3)
        model.classes_ = np.array(["setosa", "versicolor", "virginica"])
        model.estimators_ = [_StubStump(0), _StubStump(0), _StubStump(2)]
        model.estimator_weights_ = [0.3, 0.3, 1.5]

        preds = model.predict(np.zeros((2, 4)))
        np.testing.assert_array_equal(preds, np.array(["virginica", "virginica"]))

    def test_handles_zero_error_stump_without_nan(self) -> None:
        X = np.array([[0.0], [0.1], [10.0], [10.1]])
        y = np.array([0, 0, 1, 1])

        model = ensemble_lab.CustomAdaBoostSAMME(n_estimators=10, random_state=42).fit(X, y)
        self.assertEqual(len(model.estimators_), 1)
        self.assertTrue(np.isfinite(model.estimator_weights_[0]))
        np.testing.assert_array_equal(model.predict(X), y)

    def test_requires_at_least_two_classes(self) -> None:
        X = np.zeros((4, 2))
        y = np.zeros(4, dtype=int)
        with self.assertRaises(ValueError):
            ensemble_lab.CustomAdaBoostSAMME(n_estimators=5).fit(X, y)

    def test_fit_handles_2d_column_target_and_rejects_invalid_shapes(self) -> None:
        X = np.array([[0.0], [0.5], [2.0], [2.5], [4.0], [4.5]])
        y_col = np.array([[0], [0], [1], [1], [2], [2]])

        model = ensemble_lab.CustomAdaBoostSAMME(n_estimators=5, random_state=42).fit(X, y_col)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (6,))
        np.testing.assert_array_equal(preds, y_col.ravel())

        with self.assertRaises(ValueError):
            ensemble_lab.CustomAdaBoostSAMME(n_estimators=5).fit(np.empty((0, 4)), np.array([0, 1]))

        with self.assertRaises(ValueError):
            ensemble_lab.CustomAdaBoostSAMME(n_estimators=5).fit(X, np.zeros((6, 2), dtype=int))


class TestEndToEndLab(unittest.TestCase):
    def test_run_benchmark_100_estimators(self) -> None:
        df, bundle, _ = ensemble_lab.run_benchmark(
            n_estimators=100,
            random_state=42,
            n_jobs=2,
            warm_pool=False,
            n_repeats=1,
        )
        self.assertEqual(len(df), 3)
        self.assertEqual(bundle.X.shape, (150, 4))
        self.assertTrue((df["Accuracy"] >= 0.8).all())
        self.assertTrue((df["Training Time (s)"] > 0.0).all())
        self.assertTrue((df["Inference Time (s)"] > 0.0).all())

    def test_run_scaling_benchmark_range_validation(self) -> None:
        df, _, _ = ensemble_lab.run_scaling_benchmark(
            estimator_counts=(100, 200),
            random_state=42,
            n_jobs=2,
            warm_pool=False,
            n_repeats=1,
        )
        self.assertEqual(len(df), 8)
        self.assertEqual(set(df["n_estimators"]), {100, 200})

        with self.assertRaises(ValueError):
            ensemble_lab.run_scaling_benchmark(estimator_counts=(50, 100), warm_pool=False, n_repeats=1)


class TestPlotResultsRegressions(unittest.TestCase):
    def test_output_dir_is_module_relative_and_no_external_artifact_dir(self) -> None:
        expected_dir = Path(plot_results.__file__).resolve().parent
        self.assertEqual(plot_results.OUTPUT_DIR, expected_dir)
        self.assertFalse(hasattr(plot_results, "ARTIFACT_DIR"))
        source_text = Path(plot_results.__file__).read_text(encoding="utf-8")
        self.assertNotIn(r"d:\Bock", source_text)

    def test_plot_scaling_dashboard_accepts_non_default_estimator_counts(self) -> None:
        def _make_df(counts: tuple[int, ...]) -> pd.DataFrame:
            rows = []
            for n in counts:
                rows.extend(
                    [
                        {
                            "Model": "Single DecisionTree",
                            "n_estimators": n,
                            "Accuracy": 0.93,
                            "Training Time (s)": 0.002,
                            "Inference Time (s)": 0.0004,
                        },
                        {
                            "Model": "Custom Bagging (Parallel)",
                            "n_estimators": n,
                            "Accuracy": 0.96,
                            "Training Time (s)": 0.05 * (n / 100),
                            "Inference Time (s)": 0.01 * (n / 100),
                        },
                        {
                            "Model": "Custom Bagging (Sequential, n_jobs=1)",
                            "n_estimators": n,
                            "Accuracy": 0.96,
                            "Training Time (s)": 0.15 * (n / 100),
                            "Inference Time (s)": 0.01 * (n / 100),
                        },
                        {
                            "Model": "Custom AdaBoost SAMME (Sequential)",
                            "n_estimators": n,
                            "Accuracy": 0.93,
                            "Training Time (s)": 0.18 * (n / 100),
                            "Inference Time (s)": 0.01 * (n / 100),
                        },
                    ]
                )
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            annotations_2pt: list[str] = []
            orig_annotate = Axes.annotate

            def _capture_annotate(ax, text, *args, **kwargs):
                annotations_2pt.append(text)
                return orig_annotate(ax, text, *args, **kwargs)

            out_2pt = Path(tmpdir) / "scaling_2pt.png"
            with patch.object(Axes, "annotate", autospec=True, side_effect=_capture_annotate):
                plot_results.plot_scaling_dashboard(_make_df((100, 200)), out_2pt)

            self.assertTrue(out_2pt.exists())
            self.assertTrue(any("n=200" in text for text in annotations_2pt))
            self.assertFalse(any("n=500" in text for text in annotations_2pt))

            out_1pt = Path(tmpdir) / "scaling_1pt.png"
            plot_results.plot_scaling_dashboard(_make_df((100,)), out_1pt)
            self.assertTrue(out_1pt.exists())

    def test_plot_decision_boundaries_preserves_class_colors_when_subset_predicted(self) -> None:
        from sklearn.datasets import load_iris

        iris = load_iris()
        local_bundle = ensemble_lab.DatasetBundle(
            X=np.asarray(iris.data, dtype=np.float64),
            y=np.asarray(iris.target, dtype=np.int64),
            feature_names=list(iris.feature_names),
            target_names=list(iris.target_names),
            source="sklearn.datasets.load_iris() [test]",
        )

        captured_contours = []
        orig_contourf = Axes.contourf

        def _capture_contourf(ax, *args, **kwargs):
            cf = orig_contourf(ax, *args, **kwargs)
            captured_contours.append(cf)
            return cf

        orig_samme_predict = ensemble_lab.CustomAdaBoostSAMME.predict

        def _subset_predict(self_model, X):
            # Simulate a 2D slice where the model predicts only classes {0, 1}
            preds = orig_samme_predict(self_model, X)
            return np.where(preds == 2, 1, preds)

        orig_bagging_cls = plot_results.CustomBaggingClassifier

        def _single_job_bagging(*args, **kwargs):
            kwargs["n_jobs"] = 1
            return orig_bagging_cls(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "boundaries.png"
            with (
                patch.object(plot_results, "load_iris_dataset", return_value=local_bundle),
                patch.object(plot_results, "CustomBaggingClassifier", side_effect=_single_job_bagging),
                patch.object(ensemble_lab.CustomAdaBoostSAMME, "predict", autospec=True, side_effect=_subset_predict),
                patch.object(Axes, "contourf", autospec=True, side_effect=_capture_contourf),
            ):
                plot_results.plot_decision_boundaries(out_path, n_estimators=5, random_state=42)

            self.assertEqual(len(captured_contours), 3)
            # Third panel is CustomAdaBoostSAMME which only predicted classes {0, 1}
            subset_cf = captured_contours[2]
            self.assertEqual(matplotlib.colors.to_hex(subset_cf.cmap(subset_cf.norm(0))), "#fee2e2")
            self.assertEqual(matplotlib.colors.to_hex(subset_cf.cmap(subset_cf.norm(1))), "#dbeafe")
            self.assertEqual(matplotlib.colors.to_hex(subset_cf.cmap(subset_cf.norm(2))), "#dcfce7")


class TestRefactoringCharacterization(unittest.TestCase):
    @staticmethod
    def _sample_scaling_df() -> pd.DataFrame:
        rows = []
        for n in (100, 200, 300, 400, 500):
            rows.extend(
                [
                    {
                        "Model": "Single DecisionTree",
                        "Base Learners": 1,
                        "Accuracy": 0.9333333333333333,
                        "Accuracy (%)": 93.33333333333333,
                        "Training Time (s)": 0.002158,
                        "Inference Time (s)": 0.000377,
                        "n_estimators": n,
                    },
                    {
                        "Model": "Custom Bagging (Parallel)",
                        "Base Learners": n,
                        "Accuracy": 0.9666666666666667,
                        "Accuracy (%)": 96.66666666666667,
                        "Training Time (s)": 0.282031 * (n / 500),
                        "Inference Time (s)": 0.086583 * (n / 500),
                        "n_estimators": n,
                    },
                    {
                        "Model": "Custom AdaBoost SAMME (Sequential)",
                        "Base Learners": n,
                        "Accuracy": 0.9333333333333333,
                        "Accuracy (%)": 93.33333333333333,
                        "Training Time (s)": 0.954677 * (n / 500),
                        "Inference Time (s)": 0.087733 * (n / 500),
                        "n_estimators": n,
                    },
                    {
                        "Model": "Custom Bagging (Sequential, n_jobs=1)",
                        "Base Learners": n,
                        "Accuracy": 0.9666666666666667,
                        "Accuracy (%)": 96.66666666666667,
                        "Training Time (s)": 0.850000 * (n / 500),
                        "Inference Time (s)": 0.086583 * (n / 500),
                        "n_estimators": n,
                    },
                ]
            )
        return pd.DataFrame(rows)

    def test_format_results_table_output_contract(self) -> None:
        df = self._sample_scaling_df()
        sub_500 = df[
            (df["n_estimators"] == 500)
            & (df["Model"] != "Custom Bagging (Sequential, n_jobs=1)")
        ].copy()
        table = ensemble_lab.format_results_table(sub_500)

        for token in (
            "Model",
            "Estimators",
            "Accuracy",
            "Training Time (s)",
            "Inference Time (s)",
            "Single DecisionTree",
            "Custom Bagging (Parallel)",
            "Custom AdaBoost SAMME (Sequential)",
            "93.33%",
            "96.67%",
            "0.282031",
            "0.954677",
        ):
            self.assertIn(token, table)

    def test_plot_metrics_comparison_writes_png(self) -> None:
        df = self._sample_scaling_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "ensemble_comparison.png"
            plot_results.plot_metrics_comparison(df, out_png, n_estimators=500, n_repeats=5)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)

    def test_plot_results_main_writes_expected_out_txt_and_pngs(self) -> None:
        df = self._sample_scaling_df()
        fake_bundle = ensemble_lab.DatasetBundle(
            X=np.zeros((150, 4), dtype=np.float64),
            y=np.repeat([0, 1, 2], 50),
            feature_names=["f1", "f2", "f3", "f4"],
            target_names=["setosa", "versicolor", "virginica"],
            source="Mock Dataset Source",
        )
        fake_warmup = (1.2345, 4)

        def _fake_boundaries(save_path: Path, n_estimators: int = 500, random_state: int = 42) -> None:
            save_path.write_bytes(b"PNG_STUB")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with (
                patch.object(plot_results, "OUTPUT_DIR", out_dir),
                patch.object(
                    plot_results,
                    "run_scaling_benchmark",
                    return_value=(df, fake_bundle, fake_warmup),
                ) as mock_scale,
                patch.object(plot_results, "plot_decision_boundaries", side_effect=_fake_boundaries) as mock_db,
                patch("sys.argv", ["plot_results.py", "--repeats", "3"]),
            ):
                plot_results.main()

            mock_scale.assert_called_once_with(
                estimator_counts=plot_results.ESTIMATOR_COUNTS,
                random_state=42,
                n_jobs=-1,
                warm_pool=True,
                n_repeats=3,
            )
            mock_db.assert_called_once_with(out_dir / "decision_boundaries.png", n_estimators=500, random_state=42)

            self.assertTrue((out_dir / "estimators_scaling.png").exists())
            self.assertTrue((out_dir / "ensemble_comparison.png").exists())
            self.assertTrue((out_dir / "decision_boundaries.png").exists())

            out_txt = (out_dir / "out.txt").read_text(encoding="utf-8")
            self.assertTrue(out_txt.startswith("=== Ensemble Learning Lab (Iris 80/20 Split) ===\n"))
            self.assertIn("Dataset Source   : Mock Dataset Source\n", out_txt)
            self.assertIn("Shape            : 150 rows x 4 features\n", out_txt)
            self.assertIn("n_estimators     : 500 | random_state: 42 | timing: median of 3 runs\n", out_txt)
            self.assertIn(
                "Joblib Pool Warm : 1.2345 s across 4 worker processes "
                "(one-time Windows spawn excluded from fit timing)\n\n",
                out_txt,
            )
            self.assertNotIn("Custom Bagging (Sequential, n_jobs=1)", out_txt)
            self.assertTrue(out_txt.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
