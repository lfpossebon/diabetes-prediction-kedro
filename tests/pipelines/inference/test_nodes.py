"""Unit tests for the inference nodes."""

import numpy as np
import pandas as pd

from diabetes.pipelines.inference.nodes import predict, to_dataframe


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
            {"index": 0, "prediction": 0, "probability": 0.2},
            {"index": 1, "prediction": 1, "probability": 0.7},
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
