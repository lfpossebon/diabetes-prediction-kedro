"""Unit tests for the modelling and refit nodes, on a small synthetic master table."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from diabetes.pipelines.modelling.nodes import (
    analyse_thresholds,
    evaluate_model,
    optimize_hyperparameters,
    select_champion,
    train_model,
)
from diabetes.pipelines.refit.nodes import refit_model, summarise_odds_ratios

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
GUIDELINE = {"column": "Glucose", "threshold": 140}
EVALUATION = {"n_bootstrap": 100, "confidence": 0.95, "random_state": 0}
SELECTION = {"split": "validate", "min_improvement": 0.0, **EVALUATION}
THRESHOLDS = {
    "cv": 3,
    "random_state": 0,
    "thresholds": [0.2, 0.4, 0.6, 0.8],
    "reference_prevalences": [0.10, 0.05],
}


def test_train_and_evaluate_every_split(master_table):
    artifact = train_model(master_table, BASELINE)

    metrics = evaluate_model(artifact, master_table, DECISION, EVALUATION)

    assert set(metrics) == set(SPLITS)
    # Every column but the target and the split label is a feature.
    assert artifact["feature_columns"] == ["Glucose", "BMI", "NEW_BMI_Obese"]
    assert metrics["validate"]["roc_auc"] > 0.5  # noqa: PLR2004 - better than chance
    assert metrics["train"]["n_samples"] == N_ROWS // 3


def test_configured_features_are_the_only_ones_used(master_table):
    """The parsimonious baseline fits on its own list, not on the whole table."""
    artifact = train_model(master_table, {**BASELINE, "features": ["Glucose"]})

    assert artifact["features"] == ["Glucose"]
    assert artifact["feature_columns"] == ["Glucose"]
    assert artifact["estimator"].n_features_in_ == 1


def test_unknown_configured_feature_fails_loudly(master_table):
    with pytest.raises(KeyError, match="Insulin"):
        train_model(master_table, {**BASELINE, "features": ["Glucose", "Insulin"]})


def test_clinical_and_calibration_metrics(master_table):
    artifact = train_model(master_table, BASELINE)

    validate = evaluate_model(artifact, master_table, DECISION, EVALUATION)["validate"]

    cm = validate["confusion_matrix"]
    assert validate["specificity"] == pytest.approx(cm["tn"] / (cm["tn"] + cm["fp"]))
    assert validate["npv"] == pytest.approx(cm["tn"] / (cm["tn"] + cm["fn"]))
    assert 0.0 <= validate["brier"] <= 0.25  # noqa: PLR2004 - better than a coin
    assert validate["observed_expected"] > 0
    assert validate["calibration_slope"] > 0


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


class _RiskFromColumn:
    """Stand-in classifier whose risk is a noisy logistic of one column.

    ``noise`` sets how much random error is added before the logistic, so the
    tests can build a sharp model, a weak one, or two nearly equal ones.
    """

    def __init__(self, column: str, noise: float, seed: int) -> None:
        self.column, self.noise, self.seed = column, noise, seed

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        score = X[self.column].to_numpy() + rng.normal(scale=self.noise, size=len(X))
        p = 1 / (1 + np.exp(-score))
        return np.column_stack([1 - p, p])


def _artifact(noise: float, seed: int = 0, train_splits=("train",)) -> dict:
    return {
        "estimator": _RiskFromColumn("Glucose", noise, seed),
        "target_column": "Outcome",
        "feature_columns": ["Glucose"],
        "train_splits": list(train_splits),
    }


class TestSelectChampion:
    def test_clearly_better_challenger_is_promoted(self, master_table):
        baseline, optimized = _artifact(noise=20.0), _artifact(noise=0.0)

        champion, report = select_champion(baseline, optimized, master_table, SELECTION)

        assert champion is optimized
        assert report["champion"] == "optimized"
        assert report["difference"]["ci"][0] > 0

    def test_identical_models_keep_the_simpler_baseline(self, master_table):
        baseline, optimized = _artifact(noise=0.0), _artifact(noise=0.0)

        champion, report = select_champion(baseline, optimized, master_table, SELECTION)

        assert champion is baseline
        assert report["difference"]["estimate"] == pytest.approx(0.0)

    def test_a_point_gain_inside_the_noise_keeps_the_baseline(self, master_table):
        """Regression: a higher AUC on ~100 patients used to promote the challenger,
        however small the gap."""
        baseline, optimized = _artifact(noise=0.6, seed=1), _artifact(noise=0.5)

        champion, report = select_champion(baseline, optimized, master_table, SELECTION)

        low, _ = report["difference"]["ci"]
        assert report["difference"]["estimate"] > 0
        assert low <= 0
        assert champion is baseline

    def test_challenger_must_clear_the_margin(self, master_table):
        baseline, optimized = _artifact(noise=20.0), _artifact(noise=0.0)

        champion, _ = select_champion(
            baseline, optimized, master_table, {**SELECTION, "min_improvement": 0.9}
        )

        assert champion is baseline

    def test_in_sample_split_is_refused(self, master_table):
        baseline = _artifact(noise=0.0)
        optimized = _artifact(noise=0.0, train_splits=("train", "validate"))

        with pytest.raises(ValueError, match="holdout"):
            select_champion(baseline, optimized, master_table, SELECTION)


def _measured(master_table: pd.DataFrame) -> pd.DataFrame:
    """The same rows before scaling: Glucose back in mg/dL, one value unmeasured."""
    measured = master_table.assign(Glucose=120 + 30 * master_table["Glucose"])
    measured.loc[0, "Glucose"] = np.nan
    return measured


def _analyse(master_table: pd.DataFrame, threshold: float = 0.4) -> dict:
    artifact = train_model(master_table, BASELINE)
    return analyse_thresholds(
        artifact,
        master_table,
        _measured(master_table),
        {"threshold": threshold},
        GUIDELINE,
        THRESHOLDS,
    )


def test_threshold_curve_is_out_of_fold_on_the_training_rows(master_table):
    curve = _analyse(master_table)

    assert curve["n_samples"] == (master_table["split"] == "train").sum()
    assert curve["configured_threshold"]["threshold"] == pytest.approx(0.4)
    # Raising the cut-off can only flag fewer patients and miss more of them.
    sensitivity = [p["sensitivity"] for p in curve["curve"]]
    flagged = [p["flagged_rate"] for p in curve["curve"]]
    assert sensitivity == sorted(sensitivity, reverse=True)
    assert flagged == sorted(flagged, reverse=True)


def test_threshold_curve_carries_the_decision_analysis(master_table):
    curve = _analyse(master_table)

    prevalence = curve["positives"] / curve["n_samples"]
    for point in curve["curve"]:
        t = point["threshold"]
        # Net benefit written through sensitivity and specificity.
        expected = point["sensitivity"] * prevalence - (1 - point["specificity"]) * (
            1 - prevalence
        ) * t / (1 - t)
        assert point["net_benefit"] == pytest.approx(expected)
        # PPV falls and NPV rises as the disease gets rarer.
        high, low = point["at_prevalence"]
        assert (high["prevalence"], low["prevalence"]) == (0.10, 0.05)
        assert low["ppv"] <= high["ppv"] <= point["ppv"]
        assert low["npv"] >= high["npv"] >= point["npv"]
    # Referring everyone pays off less as the cut-off rises.
    refer_all = [p["net_benefit_all"] for p in curve["curve"]]
    assert refer_all == sorted(refer_all, reverse=True)


def test_guideline_rule_is_scored_on_the_measured_glucose(master_table):
    """Glucose >= 140 mg/dL on the training rows, an unmeasured one excluded."""
    curve = _analyse(master_table)

    measured = _measured(master_table)
    train = measured[measured["split"] == "train"]
    guideline = curve["referral_strategies"]["guideline"]
    assert curve["referral_strategies"]["guideline_rule"] == "Glucose >= 140"
    igt = train["Glucose"] >= GUIDELINE["threshold"]
    assert guideline["flagged_rate"] == pytest.approx(igt.mean())
    assert guideline["threshold"] == pytest.approx(0.4)


def test_model_or_guideline_flags_everyone_either_one_flags(master_table):
    """The served rule only adds referrals, so it misses no more patients than
    the model or the guideline alone."""
    curve = _analyse(master_table)

    model = curve["configured_threshold"]
    guideline = curve["referral_strategies"]["guideline"]
    served = curve["referral_strategies"]["model_or_guideline"]
    assert served["flagged_rate"] >= max(
        model["flagged_rate"], guideline["flagged_rate"]
    )
    assert served["missed"] <= min(model["missed"], guideline["missed"])


def test_measured_rows_must_match_the_master_table(master_table):
    artifact = train_model(master_table, BASELINE)
    shuffled = _measured(master_table).assign(
        split=np.roll(master_table["split"].to_numpy(), 1)
    )

    with pytest.raises(ValueError, match="same rows"):
        analyse_thresholds(
            artifact, master_table, shuffled, {"threshold": 0.4}, GUIDELINE, THRESHOLDS
        )


def test_refit_keeps_hyperparameters_and_uses_every_split(master_table):
    source = train_model(master_table, {**BASELINE, "init_args": {"C": 0.5}})

    production = refit_model(master_table, source, {"train_splits": SPLITS})

    assert production["estimator"] is not source["estimator"]
    assert production["estimator"].get_params()["C"] == pytest.approx(0.5)
    assert production["estimator"].n_features_in_ == len(production["feature_columns"])


def test_refit_keeps_the_champions_configured_features(master_table):
    """The production table has every column; the parsimonious champion must
    still see only its own."""
    source = train_model(master_table, {**BASELINE, "features": ["Glucose", "BMI"]})

    production = refit_model(master_table, source, {"train_splits": SPLITS})

    assert production["feature_columns"] == ["Glucose", "BMI"]
    assert production["estimator"].n_features_in_ == 2  # noqa: PLR2004


class TestOddsRatios:
    @pytest.fixture
    def production(self, master_table):
        source = train_model(
            master_table, {**BASELINE, "features": ["Glucose", "NEW_BMI_Obese"]}
        )
        return refit_model(master_table, source, {"train_splits": SPLITS})

    def test_scaled_columns_are_reported_per_iqr_and_per_unit(self, production):
        iqr = 20.0
        scaler = RobustScaler().fit(np.array([[0.0], [10.0], [20.0], [30.0], [40.0]]))

        report = summarise_odds_ratios(production, {"Glucose": scaler})

        glucose = report["odds_ratios"][0]
        coef = production["estimator"].coef_[0, 0]
        assert glucose["feature"] == "Glucose"
        assert glucose["iqr"] == pytest.approx(iqr)
        assert glucose["odds_ratio_per_iqr"] == pytest.approx(np.exp(coef))
        assert glucose["odds_ratio_per_unit"] == pytest.approx(np.exp(coef / iqr))

    def test_unscaled_columns_are_per_unit_only(self, production):
        report = summarise_odds_ratios(production, {})

        flag = report["odds_ratios"][1]
        assert flag["iqr"] is None
        assert flag["odds_ratio_per_iqr"] is None
        assert flag["odds_ratio_per_unit"] == pytest.approx(
            np.exp(production["estimator"].coef_[0, 1])
        )

    def test_model_without_coefficients_is_reported_not_failed(self, master_table):
        source = optimize_hyperparameters(master_table, OPTIMIZATION)
        production = refit_model(master_table, source, {"train_splits": SPLITS})

        report = summarise_odds_ratios(production, {})

        assert report["model"] == "RandomForestClassifier"
        assert report["odds_ratios"] is None
