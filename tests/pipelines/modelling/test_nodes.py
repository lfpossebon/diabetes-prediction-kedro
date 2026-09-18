"""Unit tests for the modelling and refit nodes, on a small synthetic master table."""

import numpy as np
import pandas as pd
import pytest

from diabetes.pipelines.modelling.nodes import (
    evaluate_model,
    optimize_hyperparameters,
    train_model,
)
from diabetes.pipelines.refit.nodes import refit_model

COLUMNS = {"numerical": ["Glucose", "BMI"], "categorical": ["NEW_BMI"]}
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
            "NEW_BMI": rng.integers(0, 4, size=N_ROWS),
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


def test_train_and_evaluate_every_split(master_table):
    artifact = train_model(master_table, COLUMNS, BASELINE)

    metrics = evaluate_model(artifact, master_table)

    assert set(metrics) == set(SPLITS)
    assert artifact["feature_columns"] == ["Glucose", "BMI", "NEW_BMI"]
    assert metrics["validate"]["roc_auc"] > 0.5  # noqa: PLR2004 - better than chance
    assert metrics["train"]["n_samples"] == N_ROWS // 3


def test_grid_search_returns_the_same_artifact_shape(master_table):
    params = {
        **BASELINE,
        "class_path": "sklearn.ensemble.RandomForestClassifier",
        "train_splits": ["train", "test"],
        "init_args": {"random_state": 0},
        "cv": 3,
        "scoring": "roc_auc",
        "param_grid": {"n_estimators": [10], "max_depth": [None, 3]},
    }

    artifact = optimize_hyperparameters(master_table, COLUMNS, params)

    assert artifact["best_params"]["max_depth"] in (None, 3)
    assert set(evaluate_model(artifact, master_table)) == set(SPLITS)


def test_refit_keeps_hyperparameters_and_uses_every_split(master_table):
    source = train_model(master_table, COLUMNS, {**BASELINE, "init_args": {"C": 0.5}})

    production = refit_model(master_table, COLUMNS, source, {"train_splits": SPLITS})

    assert production["estimator"] is not source["estimator"]
    assert production["estimator"].get_params()["C"] == pytest.approx(0.5)
    assert production["estimator"].n_features_in_ == len(production["feature_columns"])
