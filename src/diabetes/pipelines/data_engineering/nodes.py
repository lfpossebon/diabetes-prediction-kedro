"""Nodes for the 'data_engineering' pipeline.

The pipeline follows the fit/transform split: ``fit_*`` nodes learn only from
the splits listed in ``split_to_fit`` (normally just ``train``) and return
artefacts, while ``transform_*`` nodes apply those artefacts to every row.
Keeping the two apart is what prevents statistics leakage, and it is what lets
the refit and inference pipelines reuse these same functions.

The notebook fitted the imputer, the outlier thresholds, the encoders and the
scaler on the full dataset *before* splitting; here each of them is a
``fit_*`` / ``transform_*`` pair. ``create_features`` is the only stateless
step: it works row by row, so it cannot leak.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

logger = logging.getLogger(__name__)


def _expand_column_groups(
    columns: dict[str, Any],
    group_keys: list[str],
) -> list[str]:
    """Flatten the requested column groups into a single list of column names.

    Args:
        columns: Column groups dictionary mapping group keys to either a single
            column name or a list of column names.
        group_keys: Group keys to expand, e.g. ``["raw"]``.

    Returns:
        Flat list of column names.
    """
    return [
        item
        for key in group_keys
        for item in (columns[key] if isinstance(columns[key], list) else [columns[key]])
    ]


def _fitting_rows(df_in: pd.DataFrame, split_to_fit: list[str]) -> pd.DataFrame:
    """Rows a ``fit_*`` node is allowed to learn from."""
    return df_in.loc[df_in["split"].isin(split_to_fit)]


def clean_data(
    raw_data: pd.DataFrame,
    columns: dict[str, Any],
) -> pd.DataFrame:
    """Select the raw measurements, coerce them to numbers and flag missing zeros.

    The dataset has no real NaNs: an unmeasured Glucose, BloodPressure,
    SkinThickness, Insulin or BMI is stored as ``0``. Those zeros become NaN
    here so the imputer can fill them later.

    Columns listed in ``missing_indicator`` also get a 0/1 ``<COL>_MISSING``
    flag, recorded *before* imputation erases the information. Insulin is
    missing for about half of the patients, so the median imputation alone
    would make them indistinguishable from patients measured at the median.

    Raw columns absent from the input are added as NaN (and imputed later),
    so an API request with a missing field is still scored. The target is kept
    only if present, which lets the same function serve training data and
    inference data.

    Args:
        raw_data: Raw input DataFrame.
        columns: Column groups. Recognised keys:
            - ``target`` (str): Target column name.
            - ``raw`` (list[str]): Raw measurement column names.
            - ``zero_as_missing`` (list[str]): Columns where 0 means missing.
            - ``missing_indicator`` (list[str], optional): Columns that get a
              ``<COL>_MISSING`` flag.

    Returns:
        DataFrame with the target (if present), the raw columns and the
        missing-value flags.
    """
    target = columns["target"]
    keep = ([target] if target in raw_data.columns else []) + columns["raw"]

    df_out = raw_data.reindex(columns=keep).copy()

    for c in columns["raw"]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    zero_cols = columns["zero_as_missing"]
    df_out[zero_cols] = df_out[zero_cols].mask(df_out[zero_cols] == 0)

    for c in columns.get("missing_indicator", []):
        df_out[f"{c.upper()}_MISSING"] = df_out[c].isna().astype(int)

    logger.info(
        "Cleaned data: %d rows, %d columns, %d missing values",
        len(df_out),
        len(df_out.columns),
        int(df_out[columns["raw"]].isna().sum().sum()),
    )

    return df_out


def add_split_column(
    df_in: pd.DataFrame,
    split: dict[str, Any],
) -> pd.DataFrame:
    """Randomly assign each row to a train, test, or validate split.

    With ``stratify_by`` set, the proportions are applied inside each class,
    so every split keeps the dataset's diabetes prevalence. On ~100-row
    holdouts, an unstratified draw can shift it by several points and move
    the metrics with it.

    Args:
        df_in: Cleaned input DataFrame.
        split: Split configuration with keys:
            - ``train`` (float): Proportion of rows for training.
            - ``test`` (float): Proportion of rows for testing.
            - ``validate`` (float): Proportion of rows for validation.
            - ``random_state`` (int): Seed for reproducible assignment.
            - ``stratify_by`` (str, optional): Column whose class balance
              every split must preserve.

    Returns:
        Input DataFrame with an added ``split`` column whose values are
        ``"train"``, ``"test"``, or ``"validate"``.

    Raises:
        ValueError: If ``train + test + validate`` does not sum to 1.0.
    """
    total = split["train"] + split["test"] + split["validate"]

    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    rng = np.random.default_rng(split["random_state"])
    stratify_by = split.get("stratify_by")
    strata = (
        df_in.groupby(stratify_by).indices.values()
        if stratify_by
        else [np.arange(len(df_in))]
    )

    labels = np.empty(len(df_in), dtype=object)
    for positions in strata:
        shuffled = rng.permutation(positions)
        n_train = round(len(shuffled) * split["train"])
        n_test = round(len(shuffled) * split["test"])
        labels[shuffled[:n_train]] = "train"
        labels[shuffled[n_train : n_train + n_test]] = "test"
        labels[shuffled[n_train + n_test :]] = "validate"

    df_out = df_in.assign(split=labels)

    logger.info(
        "Split counts — train: %d, test: %d, validate: %d",
        (labels == "train").sum(),
        (labels == "test").sum(),
        (labels == "validate").sum(),
    )

    return df_out


def fit_imputers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, float]:
    """Learn one fill value per column from the specified splits.

    Args:
        df_in: DataFrame with a ``split`` column and the columns to impute.
        columns: Column groups dictionary mapping group keys to column names.
        params: Imputer configuration with keys:
            - ``columns`` (list[str]): Group keys from ``columns`` to impute.
            - ``strategy`` (str): ``"median"`` or ``"mean"``.
            - ``split_to_fit`` (list[str]): Split labels used for fitting.

    Returns:
        Mapping of column names to their fill value.
    """
    fit_df = _fitting_rows(df_in, params["split_to_fit"])
    strategy = params.get("strategy", "median")

    fill_values = {
        col: float(getattr(fit_df[col], strategy)())
        for col in _expand_column_groups(columns, params["columns"])
    }

    logger.info("Fitted %s imputers for %d columns", strategy, len(fill_values))
    return fill_values


def transform_imputers(
    df_in: pd.DataFrame,
    fill_values: dict[str, float],
) -> pd.DataFrame:
    """Fill missing values with the fitted per-column values.

    Args:
        df_in: DataFrame containing the columns to impute.
        fill_values: Mapping of column names to fill values.

    Returns:
        DataFrame with no missing values in the imputed columns.
    """
    present = {col: v for col, v in fill_values.items() if col in df_in.columns}
    return df_in.fillna(value=present)


def fit_outlier_caps(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
    outliers: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """Learn lower and upper outlier thresholds per column from the specified splits.

    Same rule as the notebook's ``outlier_thresholds``: with ``q1``/``q3``
    quantiles and ``iqr = q3 - q1``, the limits are ``q1 - k*iqr`` and
    ``q3 + k*iqr``.

    Args:
        df_in: DataFrame with a ``split`` column and the columns to cap.
        columns: Column groups dictionary mapping group keys to column names.
        params: Cap configuration with keys:
            - ``columns`` (list[str]): Group keys from ``columns`` to cap.
            - ``split_to_fit`` (list[str]): Split labels used for fitting.
        outliers: Threshold configuration with keys ``q1``, ``q3`` and
            ``iqr_factor``.

    Returns:
        Mapping of column names to ``(lower, upper)`` limits.
    """
    fit_df = _fitting_rows(df_in, params["split_to_fit"])
    k = outliers["iqr_factor"]

    caps: dict[str, tuple[float, float]] = {}
    for col in _expand_column_groups(columns, params["columns"]):
        q1 = fit_df[col].quantile(outliers["q1"])
        q3 = fit_df[col].quantile(outliers["q3"])
        iqr = q3 - q1
        caps[col] = (float(q1 - k * iqr), float(q3 + k * iqr))

    logger.info("Fitted outlier caps for %d columns", len(caps))
    return caps


def transform_outlier_caps(
    df_in: pd.DataFrame,
    caps: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Clip each capped column to its fitted limits.

    Args:
        df_in: DataFrame containing the columns to cap.
        caps: Mapping of column names to ``(lower, upper)`` limits.

    Returns:
        DataFrame with every capped column inside its limits.
    """
    df_out = df_in.copy()

    for col, (lower, upper) in caps.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    return df_out


def _cut(values: pd.Series, spec: dict[str, Any]) -> pd.Series:
    """Bin a numeric series into string labels, with left-closed bins.

    Clinical cut-offs are stated as lower bounds ("BMI >= 30", "2-hour glucose
    >= 140"), so each bin includes its left edge: ``[25, 30)`` is overweight
    and 30 is already obese.
    """
    return pd.cut(
        values,
        bins=spec["bins"],
        labels=spec["labels"],
        right=False,
        include_lowest=True,
    ).astype(str)


def create_features(
    df_in: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Add the notebook's engineered features, computed row by row.

    Every glucose category uses the 2-hour OGTT cut-offs (140 / 200 mg/dL),
    because that is what ``Glucose`` measures. The notebook built
    ``NEW_AGE_GLUCOSE_NOM`` from fasting cut-offs (100 / 126), which called a
    normal 2-hour value of 126 "high".

    ``NEW_AGE_BMI_NOM`` fixes a notebook bug: its obese rule used
    ``BMI > 18.5`` and overwrote every other category. The BMI part now comes
    from the same exclusive bins as ``NEW_BMI``.

    The notebook's ``NEW_INSULIN_SCORE`` and its ``GLUCOSE_X_INSULIN`` /
    ``GLUCOSE_X_PREGNANCIES`` interactions are not built: none of them has a
    physiological basis for 2-hour values (see ``columns`` in parameters.yml).

    Args:
        df_in: Imputed and capped DataFrame with the raw measurement columns.
        params: Feature configuration with keys ``senior_age``, ``bmi`` and
            ``glucose`` (the last two with ``bins`` and ``labels``).

    Returns:
        DataFrame with the new ``NEW_*`` categorical columns.
    """
    df_out = df_in.copy()

    age_cat = np.where(df_out["Age"] >= params["senior_age"], "senior", "mature")
    bmi_cat = _cut(df_out["BMI"], params["bmi"])
    glucose_cat = _cut(df_out["Glucose"], params["glucose"])

    df_out["NEW_AGE_CAT"] = age_cat
    df_out["NEW_BMI"] = bmi_cat
    df_out["NEW_GLUCOSE"] = glucose_cat
    df_out["NEW_AGE_BMI_NOM"] = bmi_cat.str.lower() + age_cat
    df_out["NEW_AGE_GLUCOSE_NOM"] = glucose_cat.str.lower() + age_cat

    logger.info("Created features: %d columns", len(df_out.columns))
    return df_out


def fit_encoders(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, OneHotEncoder]:
    """Fit a OneHotEncoder for each categorical column on the specified splits.

    The categories are nominal (``obesesenior``, ``prediabetesmature`` ...), so
    an integer code would hand a linear model an order that does not exist.
    One 0/1 column per category, as the notebook's ``get_dummies`` did.

    Only rows belonging to the splits in ``params["split_to_fit"]`` are used.
    A category never seen during fitting becomes all zeros instead of
    raising, so an unusual API request is still scored.

    Args:
        df_in: DataFrame with a ``split`` column and all feature columns.
        columns: Column groups dictionary mapping group keys to column names.
        params: Encoder configuration with keys:
            - ``columns`` (list[str]): Group keys from ``columns`` to encode.
            - ``split_to_fit`` (list[str]): Split labels used for fitting.

    Returns:
        Mapping of column names to their fitted ``OneHotEncoder`` instances.
    """
    fit_df = _fitting_rows(df_in, params["split_to_fit"])
    encoders: dict[str, OneHotEncoder] = {}

    for col in _expand_column_groups(columns, params["columns"]):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
        encoder.fit(fit_df[[col]].astype(str))
        encoders[col] = encoder

    logger.info("Fitted one-hot encoders for %d columns", len(encoders))
    return encoders


def transform_encoders(
    df_in: pd.DataFrame,
    encoders: dict[str, OneHotEncoder],
) -> pd.DataFrame:
    """Replace each categorical column with one 0/1 column per fitted category.

    Args:
        df_in: DataFrame containing the columns to encode.
        encoders: Mapping of column names to fitted ``OneHotEncoder`` instances.

    Returns:
        DataFrame where each encoded column ``C`` is replaced by ``C_<category>``
        indicator columns.
    """
    encoded = [
        pd.DataFrame(
            encoder.transform(df_in[[col]].astype(str)),
            columns=encoder.get_feature_names_out(),
            index=df_in.index,
        )
        for col, encoder in encoders.items()
        if col in df_in.columns
    ]
    present = [col for col in encoders if col in df_in.columns]

    return pd.concat([df_in.drop(columns=present), *encoded], axis=1)


def fit_scalers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, RobustScaler]:
    """Fit a RobustScaler for each numerical column on the specified splits.

    RobustScaler (median and IQR) is the notebook's choice; it suits these
    skewed clinical measurements better than a mean/std scaler.

    Args:
        df_in: DataFrame with a ``split`` column and all feature columns.
        columns: Column groups dictionary mapping group keys to column names.
        params: Scaler configuration with keys:
            - ``columns`` (list[str]): Group keys from ``columns`` to scale.
            - ``split_to_fit`` (list[str]): Split labels used for fitting.

    Returns:
        Mapping of column names to their fitted ``RobustScaler`` instances.
    """
    fit_df = _fitting_rows(df_in, params["split_to_fit"])
    scalers: dict[str, RobustScaler] = {}

    for col in _expand_column_groups(columns, params["columns"]):
        scaler = RobustScaler()
        scaler.fit(fit_df[[col]])
        scalers[col] = scaler

    logger.info("Fitted scalers for %d columns", len(scalers))
    return scalers


def transform_scalers(
    df_in: pd.DataFrame,
    scalers: dict[str, RobustScaler],
) -> pd.DataFrame:
    """Apply fitted scalers to numerical columns.

    Args:
        df_in: DataFrame containing the columns to scale.
        scalers: Mapping of column names to fitted scaler instances.

    Returns:
        DataFrame with each scaled column replaced by its scaled values.
    """
    df_out = df_in.copy()

    for col, scaler in scalers.items():
        if col in df_out.columns:
            df_out[col] = scaler.transform(df_out[[col]]).ravel()

    return df_out
