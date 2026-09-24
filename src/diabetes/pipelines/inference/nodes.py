"""Nodes for the 'inference' pipeline.

Only four functions are new here: ``to_dataframe``, ``predict``,
``apply_guideline_referral`` and ``apply_diagnostic_criteria``. Cleaning,
imputation, outlier capping, feature creation, encoding and scaling reuse the
data_engineering nodes, applied with the ``production_*`` artefacts. No
fitting happens anywhere in this pipeline.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(
    raw_data: pd.DataFrame | list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert raw input data to a DataFrame if it is not one already.

    Accepting both shapes is what lets the same pipeline serve batch scoring
    (a CSVDataset yields a DataFrame) and online scoring (an HTTP payload
    injected as a list of records).

    Args:
        raw_data: Input data as either a DataFrame (e.g. from a CSVDataset)
            or a list of dictionaries (e.g. from a JSONDataset).

    Returns:
        DataFrame built from the input data.
    """
    if isinstance(raw_data, pd.DataFrame):
        return raw_data
    return pd.DataFrame(raw_data)


def predict(
    model_artifact: dict[str, Any],
    inference_data: pd.DataFrame,
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score a fully-processed DataFrame and return predictions with probabilities.

    The 0/1 prediction applies the business threshold from ``params:decision``
    rather than ``estimator.predict``, whose cut-off is a fixed 0.5. Changing
    the trade-off between missed cases and false alarms is then a YAML change,
    for batch and API alike.

    Args:
        model_artifact: Model artifact dict containing ``estimator`` and
            ``feature_columns`` keys.
        inference_data: Encoded and scaled DataFrame ready for scoring.
        decision: Decision configuration with key ``threshold`` (float):
            minimum positive-class probability to predict 1.

    Returns:
        List of dictionaries, each with ``index`` (int, row number),
        ``prediction`` (int), ``probability`` (float, positive-class score)
        and ``decision_basis`` (``"model"``; the clinical rules downstream may
        change it).
    """
    estimator = model_artifact["estimator"]
    feature_cols = model_artifact["feature_columns"]

    X = inference_data[feature_cols]
    probabilities = estimator.predict_proba(X)[:, 1]
    predictions = (probabilities >= decision["threshold"]).astype(int)

    result = [
        {
            "index": idx,
            "prediction": int(p),
            "probability": float(prob),
            "decision_basis": "model",
        }
        for idx, (p, prob) in enumerate(zip(predictions, probabilities, strict=True))
    ]

    logger.info("Generated predictions for %d samples", len(result))
    return result


def _measured(raw_data: pd.DataFrame, column: str) -> pd.Series:
    """A measurement as received, NaN where it is absent or not a number.

    The clinical rules read the input before cleaning, imputation or outlier
    capping could move a value. A column the input does not carry (the study
    CSV has no fasting glucose) is all NaN, and a NaN never meets a rule: an
    unmeasured value is neither a diagnosis nor a referral. A ``0``, the
    CSV's "not measured", is below every threshold, so it never meets one
    either.
    """
    if column not in raw_data.columns:
        return pd.Series(np.nan, index=raw_data.index)
    return pd.to_numeric(raw_data[column], errors="coerce")


def apply_guideline_referral(
    predictions: list[dict[str, Any]],
    raw_data: pd.DataFrame,
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flag every patient the guidelines already refer, whatever her score.

    A 2-hour OGTT glucose of 140-199 mg/dL is impaired glucose tolerance, a
    form of prediabetes that the ADA refers to lifestyle prevention. The model
    exists to add referrals among women with a normal 2-hour glucose, where
    half of the future cases are, not to overrule the guideline: a patient at
    or above ``rule["threshold"]`` whom the model did not flag gets
    ``prediction = 1`` and ``decision_basis = "impaired_glucose_tolerance"``.
    Her probability is kept, since the model is within its range there.

    A patient at 200 or more also meets the rule here; the diagnostic
    criteria, applied next, then take her out of the model's hands.

    Args:
        predictions: Records returned by ``predict``, in row order.
        raw_data: The input rows as received, before ``clean_data``.
        rule: Rule configuration with keys ``column`` (str) and
            ``threshold`` (float).

    Returns:
        The records with the rule applied.
    """
    meets = _measured(raw_data, rule["column"]).ge(rule["threshold"]).to_numpy()

    result = [
        {**record, "prediction": 1, "decision_basis": "impaired_glucose_tolerance"}
        if referred and record["prediction"] == 0
        else record
        for record, referred in zip(predictions, meets, strict=True)
    ]

    logger.info(
        "%d of %d patients flagged by the guideline alone (%s >= %s)",
        sum(r["decision_basis"] == "impaired_glucose_tolerance" for r in result),
        len(result),
        rule["column"],
        rule["threshold"],
    )
    return result


def apply_diagnostic_criteria(
    predictions: list[dict[str, Any]],
    raw_data: pd.DataFrame,
    criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report a patient who already meets a diagnostic criterion as such.

    A 2-hour OGTT glucose >= 200 mg/dL, a fasting glucose >= 126 mg/dL or an
    HbA1c >= 6.5% is diabetes by the ADA criteria: the patient already has the
    outcome the model forecasts. Her record gets ``prediction = 1``,
    ``probability = None``, ``decision_basis = "diagnostic_criterion"`` and,
    in ``criteria_met``, the criteria she met, so the clinician knows which
    result to confirm. Every other record keeps its decision and gets an empty
    ``criteria_met``.

    Args:
        predictions: Records after ``apply_guideline_referral``, in row order.
        raw_data: The input rows as received, before ``clean_data``.
        criteria: One dict per criterion, with keys ``column`` (str) and
            ``threshold`` (float).

    Returns:
        The records with ``criteria_met`` added and the criteria applied.
    """
    met_by_row = [[] for _ in range(len(raw_data))]
    for criterion in criteria:
        meets = _measured(raw_data, criterion["column"]).ge(criterion["threshold"])
        label = f"{criterion['column']} >= {criterion['threshold']:g}"
        for row in np.flatnonzero(meets.to_numpy()):
            met_by_row[row].append(label)

    result = [
        {
            **record,
            "prediction": 1,
            "probability": None,
            "decision_basis": "diagnostic_criterion",
            "criteria_met": met,
        }
        if met
        else {**record, "criteria_met": []}
        for record, met in zip(predictions, met_by_row, strict=True)
    ]

    logger.info(
        "%d of %d patients meet a diagnostic criterion",
        sum(bool(met) for met in met_by_row),
        len(result),
    )
    return result
