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
    "Age",
]

COLUMNS = {
    "target": "Outcome",
    "raw": RAW,
    "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "missing_indicator": ["Insulin"],
    "numerical": ["Glucose", "BMI"],
    "categorical": ["NEW_BMI"],
}

FEATURES = {
    "senior_age": 50,
    "bmi": {
        "bins": [0, 18.5, 25, 30, 1000],
        "labels": ["Underweight", "Healthy", "Overweight", "Obese"],
    },
    "glucose": {
        "bins": [0, 140, 200, 1000],
        "labels": ["Normal", "Prediabetes", "Diabetes"],
    },
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
    def test_keeps_only_target_raw_columns_and_missing_flags(self, raw_df):
        """A CSV column left out of ``raw`` is dropped: the pedigree function,
        which no clinic can compute, never reaches the model."""
        out = clean_data(raw_df, COLUMNS)

        assert list(out.columns) == ["Outcome", *RAW, "INSULIN_MISSING"]
        assert "DiabetesPedigreeFunction" not in out.columns

    def test_missing_flag_is_recorded_before_imputation(self, raw_df):
        """Insulin = 0 means "not measured"; the flag must keep that information."""
        out = clean_data(raw_df, COLUMNS)

        assert out["INSULIN_MISSING"].tolist() == [1, 1, 0]

    def test_absent_column_is_flagged_missing(self, raw_df):
        out = clean_data(raw_df.drop(columns=["Insulin"]), COLUMNS)

        assert out["INSULIN_MISSING"].eq(1).all()

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

    def test_stratified_splits_keep_the_class_balance(self):
        n_rows, n_positive = 200, 60
        df = pd.DataFrame({"Outcome": [1] * n_positive + [0] * (n_rows - n_positive)})
        split = {
            "train": 0.7,
            "test": 0.15,
            "validate": 0.15,
            "random_state": 42,
            "stratify_by": "Outcome",
        }

        out = add_split_column(df, split)

        prevalence = out.groupby("split")["Outcome"].mean()
        assert prevalence.to_dict() == pytest.approx(
            {"train": 0.3, "test": 0.3, "validate": 0.3}, abs=0.01
        )
        assert out["split"].value_counts().to_dict() == {
            "train": 140,
            "test": 30,
            "validate": 30,
        }


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

    def test_categories_become_one_indicator_column_each(self, split_df):
        """Nominal categories must not be given an order: one 0/1 column each."""
        encoders = fit_encoders(
            split_df, COLUMNS, {"columns": ["categorical"], "split_to_fit": ["train"]}
        )

        out = transform_encoders(split_df, encoders)

        dummies = ["NEW_BMI_Healthy", "NEW_BMI_Obese", "NEW_BMI_Overweight"]
        assert "NEW_BMI" not in out.columns
        assert [c for c in out.columns if c.startswith("NEW_BMI_")] == dummies
        assert out.loc[0, dummies].tolist() == [1, 0, 0]
        assert out[dummies].dtypes.map(lambda t: t.kind in "iu").all()

    def test_unseen_category_is_all_zeros(self, split_df):
        """'Underweight' exists only in the test split, so the encoder must not know it."""
        encoders = fit_encoders(
            split_df, COLUMNS, {"columns": ["categorical"], "split_to_fit": ["train"]}
        )

        out = transform_encoders(split_df, encoders)

        assert "Underweight" not in set(encoders["NEW_BMI"].categories_[0])
        assert out.filter(like="NEW_BMI_").iloc[4].eq(0).all()

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
            "normalmature",
            "prediabetessenior",
            "normalsenior",
        ]

    def test_age_bmi_uses_exclusive_bmi_bins(self, clean_df):
        """Regression: the notebook's 'BMI > 18.5' rule labelled everyone obese."""
        out = create_features(clean_df, FEATURES)

        assert out["NEW_AGE_BMI_NOM"].tolist() == [
            "healthymature",
            "obesesenior",
            "overweightsenior",
        ]

    def test_features_without_a_physiological_basis_are_not_built(self, clean_df):
        """The insulin score and both interactions were dropped on clinical grounds."""
        out = create_features(clean_df, FEATURES)

        dropped = {"NEW_INSULIN_SCORE", "GLUCOSE_X_INSULIN", "GLUCOSE_X_PREGNANCIES"}
        assert not dropped & set(out.columns)

    @pytest.mark.parametrize(
        ("glucose", "expected"),
        [
            (139.9, "Normal"),
            (140.0, "Prediabetes"),
            (199.9, "Prediabetes"),
            (200.0, "Diabetes"),
        ],
    )
    def test_glucose_uses_the_2_hour_ogtt_cut_offs(self, glucose, expected):
        """WHO/ADA: >= 140 is impaired glucose tolerance, >= 200 is diabetes."""
        df = pd.DataFrame({"Glucose": [glucose], "BMI": [27.0], "Age": [30.0]})

        out = create_features(df, FEATURES)

        assert out["NEW_GLUCOSE"].iloc[0] == expected
        assert out["NEW_AGE_GLUCOSE_NOM"].iloc[0] == f"{expected.lower()}mature"

    def test_126_is_a_normal_2_hour_glucose(self):
        """Regression: fasting cut-offs labelled a normal 2-hour 126 mg/dL "high"."""
        df = pd.DataFrame({"Glucose": [126.0], "BMI": [27.0], "Age": [30.0]})

        out = create_features(df, FEATURES)

        assert out["NEW_AGE_GLUCOSE_NOM"].iloc[0] == "normalmature"

    @pytest.mark.parametrize(
        ("bmi", "expected"),
        [
            (18.4, "Underweight"),
            (18.5, "Healthy"),
            (24.95, "Healthy"),
            (25.0, "Overweight"),
            (29.95, "Overweight"),
            (30.0, "Obese"),
        ],
    )
    def test_bmi_uses_the_who_classes(self, bmi, expected):
        """Regression: the 24.9 / 29.9 edges put 24.95 in overweight and 29.95
        in obese."""
        df = pd.DataFrame({"Glucose": [100.0], "BMI": [bmi], "Age": [30.0]})

        out = create_features(df, FEATURES)

        assert out["NEW_BMI"].iloc[0] == expected

    def test_input_is_not_mutated(self, clean_df):
        before = clean_df.copy()
        create_features(clean_df, FEATURES)

        pd.testing.assert_frame_equal(clean_df, before)
