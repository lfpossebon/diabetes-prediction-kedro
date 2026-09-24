"""Unit tests for the modelling and refit nodes, on a small synthetic master table."""

import numpy as np
import pandas as pd
import pytest

from diabetes.pipelines.modelling.nodes import (
    analyse_thresholds,
    evaluate_model,
    optimize_hyperparameters,
    select_champion,
    train_model,
)
from diabetes.pipelines.refit.nodes import refit_model

SPLITS = ["train", "test", "validate"]
N_ROWS = 120


@pytest.fixture
def master_table() -> pd.DataFrame:
    """Outcome driven by Glucose, so any sensible model beats chance."""
    rng = np.random.default_rng(0)
    glucose = rng.normal(size=N_ROWS)
    return pd.DataFrame(
        {
            "Outcome": (glucose + rng.normal(scale=0.5, size=N_ROWS) > 0).astype(int),
            "Glucose": glucose,
            "BMI": rng.normal(size=N_ROWS),
            "NEW_BMI_Obese": rng.integers(0, 2, size=N_ROWS),
            "split": np.resize(SPLITS, N_ROWS),
        }
    )


BASELINE = {
    "class_path": "sklearn.linear_model.LogisticRegression",
    "target_column": "Outcome",
    "train_splits": ["train"],
    "eval_splits": SPLITS,
    "init_args": {"max_iter": 1000},
}

OPTIMIZATION = {
    **BASELINE,
    "class_path": "sklearn.ensemble.RandomForestClassifier",
    "train_splits": ["train", "test"],
    "init_args": {"random_state": 0},
    "cv": 3,
    "scoring": "roc_auc",
    "n_jobs": 1,
    "param_grid": {"n_estimators": [10], "max_depth": [None, 3]},
}

DECISION = {"threshold": 0.5}
EVALUATION = {"n_bootstrap": 100, "confidence": 0.95, "random_state": 0}
SELECTION = {"split": "validate", "metric": "roc_auc", "min_improvement": 0.0}
THRESHOLDS = {"cv": 3, "random_state": 0, "thresholds": [0.2, 0.4, 0.6, 0.8]}


def test_train_and_evaluate_every_split(master_table):
    artifact = train_model(master_table, BASELINE)

    metrics = evaluate_model(artifact, master_table, DECISION, EVALUATION)

    assert set(metrics) == set(SPLITS)
    # Every column but the target and the split label is a feature.
    assert artifact["feature_columns"] == ["Glucose", "BMI", "NEW_BMI_Obese"]
    assert metrics["validate"]["roc_auc"] > 0.5  # noqa: PLR2004 - better than chance
    assert metrics["train"]["n_samples"] == N_ROWS // 3


def test_splits_the_model_was_fitted_on_are_flagged_in_sample(master_table):
    """Regression: the tuned model's test score was reported as a holdout."""
    artifact = optimize_hyperparameters(master_table, OPTIMIZATION)

    metrics = evaluate_model(artifact, master_table, DECISION, EVALUATION)

    assert metrics["train"]["in_sample"] is True
    assert metrics["test"]["in_sample"] is True
    assert metrics["validate"]["in_sample"] is False


def test_roc_auc_comes_with_a_confidence_interval(master_table):
    artifact = train_model(master_table, BASELINE)

    validate = evaluate_model(artifact, master_table, DECISION, EVALUATION)["validate"]

    low, high = validate["roc_auc_ci"]
    assert low <= validate["roc_auc"] <= high
    assert high - low > 0


def test_grid_search_returns_the_same_artifact_shape(master_table):
    artifact = optimize_hyperparameters(master_table, OPTIMIZATION)

    assert artifact["best_params"]["max_depth"] in (None, 3)
    metrics = evaluate_model(artifact, master_table, DECISION, EVALUATION)
    assert set(metrics) == set(SPLITS)


def test_lower_threshold_trades_precision_for_recall(master_table):
    """The cut-off comes from params:decision, and every metric follows it."""
    artifact = train_model(master_table, BASELINE)

    strict = evaluate_model(artifact, master_table, {"threshold": 0.8}, EVALUATION)
    lenient = evaluate_model(artifact, master_table, {"threshold": 0.2}, EVALUATION)
    strict, lenient = strict["validate"], lenient["validate"]

    assert lenient["threshold"] == pytest.approx(0.2)
    assert lenient["recall"] > strict["recall"]
    assert lenient["confusion_matrix"]["fn"] < strict["confusion_matrix"]["fn"]
    cm = lenient["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == lenient["n_samples"]


class TestSelectChampion:
    @staticmethod
    def _metrics(score: float, in_sample: bool = False) -> dict:
        return {
            "validate": {"roc_auc": score, "roc_auc_ci": [0, 1], "in_sample": in_sample}
        }

    @pytest.fixture
    def models(self, master_table):
        return train_model(master_table, BASELINE), optimize_hyperparameters(
            master_table, OPTIMIZATION
        )

    def test_challenger_that_wins_is_promoted(self, models):
        baseline, optimized = models

        champion, report = select_champion(
            baseline, self._metrics(0.80), optimized, self._metrics(0.85), SELECTION
        )

        assert champion is optimized
        assert report["champion"] == "optimized"
        assert report["estimator"] == "RandomForestClassifier"

    def test_tie_keeps_the_simpler_baseline(self, models):
        baseline, optimized = models

        champion, report = select_champion(
            baseline, self._metrics(0.85), optimized, self._metrics(0.85), SELECTION
        )

        assert champion is baseline
        assert report["champion"] == "baseline"

    def test_challenger_must_clear_the_margin(self, models):
        baseline, optimized = models

        champion, _ = select_champion(
            baseline,
            self._metrics(0.80),
            optimized,
            self._metrics(0.81),
            {**SELECTION, "min_improvement": 0.02},
        )

        assert champion is baseline

    def test_in_sample_split_is_refused(self, models):
        baseline, optimized = models

        with pytest.raises(ValueError, match="holdout"):
            select_champion(
                baseline,
                self._metrics(0.80),
                optimized,
                self._metrics(0.99, in_sample=True),
                SELECTION,
            )


def test_threshold_curve_is_out_of_fold_on_the_training_rows(master_table):
    artifact = train_model(master_table, BASELINE)

    curve = analyse_thresholds(artifact, master_table, {"threshold": 0.4}, THRESHOLDS)

    assert curve["n_samples"] == (master_table["split"] == "train").sum()
    assert curve["configured_threshold"]["threshold"] == pytest.approx(0.4)
    # Raising the cut-off can only flag fewer patients and miss more of them.
    recall = [p["recall"] for p in curve["curve"]]
    flagged = [p["flagged_rate"] for p in curve["curve"]]
    assert recall == sorted(recall, reverse=True)
    assert flagged == sorted(flagged, reverse=True)


def test_refit_keeps_hyperparameters_and_uses_every_split(master_table):
    source = train_model(master_table, {**BASELINE, "init_args": {"C": 0.5}})

    production = refit_model(master_table, source, {"train_splits": SPLITS})

    assert production["estimator"] is not source["estimator"]
    assert production["estimator"].get_params()["C"] == pytest.approx(0.5)
    assert production["estimator"].n_features_in_ == len(production["feature_columns"])
