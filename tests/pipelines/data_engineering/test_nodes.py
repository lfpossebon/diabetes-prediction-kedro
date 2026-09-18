"""Unit tests for the data_engineering nodes, using synthetic DataFrames."""

import numpy as np
import pandas as pd
import pytest

from diabetes.pipelines.data_engineering.nodes import (
    add_split_column,
    clean_data,
    create_features,
    fit_encoders,
    fit_imputers,
    fit_outlier_caps,
    fit_scalers,
    transform_encoders,
    transform_imputers,
    transform_outlier_caps,
    transform_scalers,
)

RAW = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

COLUMNS = {
    "target": "Outcome",
    "raw": RAW,
    "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "numerical": ["Glucose", "BMI"],
    "categorical": ["NEW_BMI"],
}

FEATURES = {
    "senior_age": 50,
    "bmi": {
        "bins": [0, 18.5, 24.9, 29.9, 1000],
        "labels": ["Underweight", "Healthy", "Overweight", "Obese"],
    },
    "glucose": {
        "bins": [0, 140, 200, 1000],
        "labels": ["Normal", "Prediabetes", "Diabetes"],
    },
    "glucose_level": {
        "bins": [0, 70, 100, 126, 1000],
        "labels": ["low", "normal", "hidden", "high"],
    },
    "insulin_normal_range": [16, 166],
}

OUTLIERS = {"q1": 0.05, "q3": 0.95, "iqr_factor": 1.5}

EXTREME_GLUCOSE = 500.0


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pregnancies": [0, 3, 1],
            "Glucose": [148, 0, 89],
            "BloodPressure": [72, 66, 0],
            "SkinThickness": [35, 0, 23],
            "Insulin": [0, 0, 94],
            "BMI": [33.6, 26.6, 0.0],
            "DiabetesPedigreeFunction": [0.627, 0.351, 0.167],
            "Age": [50, 31, 21],
            "Outcome": [1, 0, 0],
            "PatientName": ["a", "b", "c"],
        }
    )


class TestCleanData:
    def test_keeps_only_target_and_raw_columns(self, raw_df):
        out = clean_data(raw_df, COLUMNS)

        assert list(out.columns) == ["Outcome", *RAW]

    def test_zeros_become_missing_only_where_configured(self, raw_df):
        out = clean_data(raw_df, COLUMNS)

        assert np.isnan(out["Glucose"].iloc[1])
        assert np.isnan(out["BMI"].iloc[2])
        # Zero pregnancies is a real value, not a missing one.
        assert out["Pregnancies"].iloc[0] == 0

    def test_missing_target_is_tolerated(self, raw_df):
        """Inference data may have no Outcome column; the same node must still work."""
        out = clean_data(raw_df.drop(columns=["Outcome"]), COLUMNS)

        assert "Outcome" not in out.columns
        assert len(out) == len(raw_df)

    def test_absent_raw_columns_are_added_as_missing(self, raw_df):
        """An API request may omit a measurement; it becomes NaN to be imputed."""
        out = clean_data(raw_df.drop(columns=["SkinThickness"]), COLUMNS)

        assert out["SkinThickness"].isna().all()

    def test_input_is_not_mutated(self, raw_df):
        before = raw_df.copy()
        clean_data(raw_df, COLUMNS)

        pd.testing.assert_frame_equal(raw_df, before)


class TestAddSplitColumn:
    def test_proportions_must_sum_to_one(self):
        df = pd.DataFrame({"x": range(10)})

        with pytest.raises(ValueError, match="must sum to 1.0"):
            add_split_column(
                df, {"train": 0.7, "test": 0.2, "validate": 0.2, "random_state": 42}
            )

    def test_same_seed_is_reproducible(self):
        df = pd.DataFrame({"x": range(500)})
        split = {"train": 0.7, "test": 0.15, "validate": 0.15, "random_state": 42}

        first = add_split_column(df, split)
        second = add_split_column(df, split)

        pd.testing.assert_series_equal(first["split"], second["split"])


@pytest.fixture
def split_df() -> pd.DataFrame:
    """A frame whose test row is extreme — the leakage probe."""
    return pd.DataFrame(
        {
            "Glucose": [100.0, np.nan, 120.0, 110.0, EXTREME_GLUCOSE],
            "BMI": [20.0, 30.0, np.nan, 25.0, 90.0],
            "NEW_BMI": ["Healthy", "Obese", "Healthy", "Overweight", "Underweight"],
            "Outcome": [0, 1, 0, 1, 1],
            "split": ["train", "train", "train", "train", "test"],
        }
    )


class TestFitTransformSeparation:
    def test_imputer_statistics_come_from_the_train_split_only(self, split_df):
        fill = fit_imputers(
            split_df,
            COLUMNS,
            {"columns": ["numerical"], "strategy": "median", "split_to_fit": ["train"]},
        )

        # Glucose on train is [100, 120, 110] -> median 110; the 500 in test must not shift it.
        assert fill["Glucose"] == pytest.approx(110.0)

    def test_imputation_fills_every_gap(self, split_df):
        fill = fit_imputers(
            split_df,
            COLUMNS,
            {"columns": ["numerical"], "strategy": "median", "split_to_fit": ["train"]},
        )

        out = transform_imputers(split_df, fill)

        assert out[["Glucose", "BMI"]].notna().all().all()
        assert split_df["Glucose"].isna().any(), "input must not be mutated"

    def test_outlier_caps_ignore_the_test_split(self, split_df):
        caps = fit_outlier_caps(
            split_df.fillna(110.0),
            COLUMNS,
            {"columns": ["numerical"], "split_to_fit": ["train"]},
            OUTLIERS,
        )
        out = transform_outlier_caps(split_df.fillna(110.0), caps)

        lower, upper = caps["Glucose"]
        assert upper < EXTREME_GLUCOSE
        assert out["Glucose"].max() == pytest.approx(upper)
        assert out["Glucose"].min() >= lower

    def test_unseen_category_is_encoded_as_minus_one(self, split_df):
        """'Underweight' exists only in the test split, so the encoder must not know it."""
        encoders = fit_encoders(
            split_df, COLUMNS, {"columns": ["categorical"], "split_to_fit": ["train"]}
        )

        out = transform_encoders(split_df, encoders)

        assert "Underweight" not in set(encoders["NEW_BMI"].categories_[0])
        assert out["NEW_BMI"].iloc[4] == -1
        assert out["NEW_BMI"].dtype.kind in "iu"

    def test_scaler_statistics_come_from_the_train_split_only(self, split_df):
        filled = split_df.fillna(110.0)
        scalers = fit_scalers(
            filled, COLUMNS, {"columns": ["numerical"], "split_to_fit": ["train"]}
        )

        out = transform_scalers(filled, scalers)

        # RobustScaler centres on the train median, so that row scales to 0.
        assert scalers["Glucose"].center_[0] == pytest.approx(110.0)
        assert out["Glucose"].iloc[3] == pytest.approx(0.0)


class TestCreateFeatures:
    @pytest.fixture
    def clean_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Pregnancies": [2.0, 5.0, 0.0],
                "Glucose": [65.0, 150.0, 110.0],
                "Insulin": [10.0, 100.0, 200.0],
                "BMI": [22.0, 35.0, 27.0],
                "Age": [25.0, 60.0, 50.0],
            }
        )

    def test_categories(self, clean_df):
        out = create_features(clean_df, FEATURES)

        assert out["NEW_AGE_CAT"].tolist() == ["mature", "senior", "senior"]
        assert out["NEW_BMI"].tolist() == ["Healthy", "Obese", "Overweight"]
        assert out["NEW_GLUCOSE"].tolist() == ["Normal", "Prediabetes", "Normal"]
        assert out["NEW_AGE_GLUCOSE_NOM"].tolist() == [
            "lowmature",
            "highsenior",
            "hiddensenior",
        ]

    def test_age_bmi_uses_exclusive_bmi_bins(self, clean_df):
        """Regression: the notebook's 'BMI > 18.5' rule labelled everyone obese."""
        out = create_features(clean_df, FEATURES)

        assert out["NEW_AGE_BMI_NOM"].tolist() == [
            "healthymature",
            "obesesenior",
            "overweightsenior",
        ]

    def test_insulin_score_is_never_empty(self, clean_df):
        """Regression: the notebook returned None for normal insulin values."""
        out = create_features(clean_df, FEATURES)

        assert out["NEW_INSULIN_SCORE"].tolist() == ["Abnormal", "Normal", "Abnormal"]

    def test_interactions(self, clean_df):
        out = create_features(clean_df, FEATURES)

        assert out["GLUCOSE_X_INSULIN"].tolist() == [650.0, 15000.0, 22000.0]
        assert out["GLUCOSE_X_PREGNANCIES"].tolist() == [130.0, 750.0, 0.0]

    def test_input_is_not_mutated(self, clean_df):
        before = clean_df.copy()
        create_features(clean_df, FEATURES)

        pd.testing.assert_frame_equal(clean_df, before)
