"""Nodes for the 'inference' pipeline.

Only two functions are new here: ``to_dataframe`` and ``predict``. Cleaning,
imputation, outlier capping, feature creation, encoding and scaling reuse the
data_engineering nodes, applied with the
``production_*`` artefacts. No fitting happens anywhere in this pipeline.
"""

import logging
from typing import Any

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
) -> list[dict[str, Any]]:
    """Score a fully-processed DataFrame and return predictions with probabilities.

    Args:
        model_artifact: Model artifact dict containing ``estimator`` and
            ``feature_columns`` keys.
        inference_data: Encoded and scaled DataFrame ready for scoring.

    Returns:
        List of dictionaries, each with ``index`` (int, row number),
        ``prediction`` (int), and ``probability`` (float, positive-class
        score).
    """
    estimator = model_artifact["estimator"]
    feature_cols = model_artifact["feature_columns"]

    X = inference_data[feature_cols]
    predictions = estimator.predict(X)
    probabilities = estimator.predict_proba(X)[:, 1]

    result = [
        {"index": idx, "prediction": int(p), "probability": float(prob)}
        for idx, (p, prob) in enumerate(zip(predictions, probabilities, strict=True))
    ]

    logger.info("Generated predictions for %d samples", len(result))
    return result
