"""Unit tests for the inference nodes."""

import numpy as np
import pandas as pd
import pytest

from diabetes.pipelines.inference.nodes import (
    apply_diagnostic_criteria,
    apply_guideline_referral,
    predict,
    to_dataframe,
)


class _StubEstimator:
    """Minimal stand-in for a fitted classifier."""

    def predict(self, X):
        return np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[0.8, 0.2], [0.3, 0.7]])


class TestToDataFrame:
    def test_dataframe_passes_through(self):
        df = pd.DataFrame({"a": [1, 2]})

        assert to_dataframe(df) is df

    def test_list_of_records_becomes_a_dataframe(self):
        """This is the path an HTTP payload would take in an online API."""
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

        out = to_dataframe(records)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == len(records)


DECISION = {"threshold": 0.5}


class TestPredict:
    def test_returns_one_record_per_row_with_positive_class_probability(self):
        artifact = {"estimator": _StubEstimator(), "feature_columns": ["a"]}
        data = pd.DataFrame({"a": [1.0, 2.0], "ignored": [9, 9]})

        out = predict(artifact, data, DECISION)

        assert out == [
            {
                "index": 0,
                "prediction": 0,
                "probability": 0.2,
                "decision_basis": "model",
            },
            {
                "index": 1,
                "prediction": 1,
                "probability": 0.7,
                "decision_basis": "model",
            },
        ]

    def test_threshold_comes_from_parameters_not_the_estimator(self):
        """Lowering the cut-off flags the 0.2 row that predict() at 0.5 would miss."""
        artifact = {"estimator": _StubEstimator(), "feature_columns": ["a"]}
        data = pd.DataFrame({"a": [1.0, 2.0]})

        out = predict(artifact, data, {"threshold": 0.15})

        assert [r["prediction"] for r in out] == [1, 1]

    def test_values_are_plain_python_types_for_json_serialization(self):
        artifact = {"estimator": _StubEstimator(), "feature_columns": ["a"]}
        data = pd.DataFrame({"a": [1.0, 2.0]})

        out = predict(artifact, data, DECISION)

        assert all(type(r["prediction"]) is int for r in out)
        assert all(type(r["probability"]) is float for r in out)


GUIDELINE = {"column": "Glucose", "threshold": 140}

CRITERIA = [
    {"column": "Glucose", "threshold": 200},
    {"column": "FastingGlucose", "threshold": 126},
    {"column": "HbA1c", "threshold": 6.5},
]


def _model_output(*predictions: int) -> list[dict]:
    return [
        {"index": i, "prediction": p, "probability": 0.1, "decision_basis": "model"}
        for i, p in enumerate(predictions)
    ]


class TestApplyGuidelineReferral:
    @pytest.mark.parametrize(
        ("glucose", "referred"),
        [(139.9, False), (140.0, True), (199.0, True), (0.0, False), (np.nan, False)],
    )
    def test_impaired_glucose_tolerance_is_always_flagged(self, glucose, referred):
        """0 is the CSV's "not measured" and NaN an absent value: neither is IGT."""
        raw = pd.DataFrame({"Glucose": [glucose]})

        (record,) = apply_guideline_referral(_model_output(0), raw, GUIDELINE)

        assert record["prediction"] == int(referred)
        assert (record["decision_basis"] == "impaired_glucose_tolerance") is referred

    def test_keeps_the_probability_and_never_unflags(self):
        """The guideline adds referrals; a patient the model flagged stays
        flagged, and on the model's own grounds."""
        raw = pd.DataFrame({"Glucose": [150.0, 150.0, 100.0]})

        igt, flagged, normal = apply_guideline_referral(
            _model_output(0, 1, 1), raw, GUIDELINE
        )

        assert igt == {
            "index": 0,
            "prediction": 1,
            "probability": 0.1,
            "decision_basis": "impaired_glucose_tolerance",
        }
        assert flagged["decision_basis"] == normal["decision_basis"] == "model"
        assert flagged["prediction"] == normal["prediction"] == 1

    def test_rows_and_predictions_must_line_up(self):
        raw = pd.DataFrame({"Glucose": [120.0, 130.0]})

        with pytest.raises(ValueError, match="zip"):
            apply_guideline_referral(_model_output(0), raw, GUIDELINE)


class TestApplyDiagnosticCriteria:
    @pytest.mark.parametrize(
        ("measured", "met"),
        [
            ({"Glucose": 199.0}, []),
            ({"Glucose": 200.0}, ["Glucose >= 200"]),
            ({"Glucose": np.nan}, []),
            ({"Glucose": 120.0, "FastingGlucose": 125.9}, []),
            ({"Glucose": 120.0, "FastingGlucose": 126.0}, ["FastingGlucose >= 126"]),
            ({"Glucose": 120.0, "HbA1c": 6.4}, []),
            ({"Glucose": 120.0, "HbA1c": 6.5}, ["HbA1c >= 6.5"]),
            (
                {"Glucose": 210.0, "FastingGlucose": 140.0, "HbA1c": 7.0},
                ["Glucose >= 200", "FastingGlucose >= 126", "HbA1c >= 6.5"],
            ),
        ],
    )
    def test_any_measured_criterion_is_a_diagnosis(self, measured, met):
        """An unmeasured value (NaN, or a column the input lacks) is not one."""
        raw = pd.DataFrame({k: [v] for k, v in measured.items()})

        (record,) = apply_diagnostic_criteria(_model_output(0), raw, CRITERIA)

        assert record["criteria_met"] == met
        assert (record["decision_basis"] == "diagnostic_criterion") is bool(met)

    def test_diagnosed_patient_gets_no_model_probability(self):
        raw = pd.DataFrame({"Glucose": [120.0, 120.0], "HbA1c": [6.8, None]})
        referred = [
            {
                "index": 0,
                "prediction": 0,
                "probability": 0.1,
                "decision_basis": "model",
            },
            {
                "index": 1,
                "prediction": 1,
                "probability": 0.1,
                "decision_basis": "impaired_glucose_tolerance",
            },
        ]

        diagnosed, scored = apply_diagnostic_criteria(referred, raw, CRITERIA)

        assert diagnosed == {
            "index": 0,
            "prediction": 1,
            "probability": None,
            "decision_basis": "diagnostic_criterion",
            "criteria_met": ["HbA1c >= 6.5"],
        }
        assert scored == {**referred[1], "criteria_met": []}

    def test_rows_and_predictions_must_line_up(self):
        raw = pd.DataFrame({"Glucose": [120.0, 130.0]})

        with pytest.raises(ValueError, match="zip"):
            apply_diagnostic_criteria(_model_output(0), raw, CRITERIA)
