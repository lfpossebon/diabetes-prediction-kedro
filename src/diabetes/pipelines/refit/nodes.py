"""Nodes for the 'refit' pipeline.

During modelling the artefacts are fitted on the train split only, so that
evaluation stays honest. Once the approach is validated, production artefacts
are refitted on *every* split to squeeze out the extra data.

The imputers, outlier caps, encoders and scalers need no new code — the refit
pipeline reuses the data_engineering functions with a wider ``split_to_fit``.
Only the model needs dedicated nodes: one to refit it, one to publish its odds
ratios.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler

from diabetes.pipelines.modelling.nodes import SPLIT_COLUMN, feature_columns

logger = logging.getLogger(__name__)


def refit_model(
    master_table: pd.DataFrame,
    champion_model: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Retrain the champion model on all data splits for production use.

    A fresh, unfitted clone of the champion's estimator — same class, same
    hyperparameters — is fitted on the splits specified in
    ``params["train_splits"]`` (typically all splits).

    A champion configured with a ``features`` list keeps exactly those
    columns. Otherwise the feature columns are read from the *production*
    master table: its one-hot columns come from the production encoders, which
    may know a category the train-only encoders never saw.

    Args:
        master_table: Table processed with the *production* artefacts, with a
            ``split`` column and all feature and target columns.
        champion_model: Model artifact chosen by ``select_champion``, whose
            estimator class and hyperparameters are reused.
        params: Refit configuration with keys:
            - ``train_splits`` (list[str]): Split labels to train on.

    Returns:
        Production model artifact dict with keys ``estimator``,
        ``target_column``, ``feature_columns`` and ``train_splits``.
    """
    target = champion_model["target_column"]
    feature_cols = feature_columns(master_table, target, champion_model.get("features"))

    train_df = master_table[master_table[SPLIT_COLUMN].isin(params["train_splits"])]
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    estimator = clone(champion_model["estimator"])
    estimator.fit(X_train, y_train)

    logger.info(
        "Refitted %s on %d samples (%d features) using all splits",
        type(estimator).__name__,
        len(X_train),
        X_train.shape[1],
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "features": champion_model.get("features"),
        "feature_columns": feature_cols,
        "train_splits": list(params["train_splits"]),
    }


def summarise_odds_ratios(
    model_artifact: dict[str, Any],
    scalers: dict[str, RobustScaler],
) -> dict[str, Any]:
    """Odds ratios of the production model, in clinical units.

    The model sees RobustScaler output, ``(x - median) / IQR``, so each
    logistic-regression coefficient is a log odds ratio per IQR of the raw
    measurement. Dividing it by the IQR gives the odds ratio per natural unit:
    per mg/dL of glucose, per kg/m² of BMI, per year of age. A column that was
    not scaled (a one-hot flag) is per unit already. The estimates are L2
    penalised (sklearn's default), so they are slightly shrunk towards 1.

    The champion is chosen at run time and may be a model without
    coefficients, such as a random forest; the report then says so instead of
    failing.

    Args:
        model_artifact: Production artifact returned by ``refit_model``.
        scalers: Production scalers, keyed by column name.

    Returns:
        Dict with the model name, the ``intercept`` and one ``odds_ratios``
        entry per feature, or ``odds_ratios = None`` for a model without
        coefficients.
    """
    estimator = model_artifact["estimator"]
    report: dict[str, Any] = {"model": type(estimator).__name__}

    if not hasattr(estimator, "coef_"):
        logger.info("%s has no coefficients: no odds ratios", report["model"])
        return {**report, "intercept": None, "odds_ratios": None}

    rows = []
    for col, coef in zip(
        model_artifact["feature_columns"], estimator.coef_[0], strict=True
    ):
        iqr = float(scalers[col].scale_[0]) if col in scalers else None
        rows.append(
            {
                "feature": col,
                "coefficient": float(coef),
                "iqr": iqr,
                "odds_ratio_per_iqr": float(np.exp(coef)) if iqr else None,
                "odds_ratio_per_unit": float(np.exp(coef / iqr if iqr else coef)),
            }
        )

    logger.info(
        "Odds ratios per unit: %s",
        ", ".join(f"{r['feature']} {r['odds_ratio_per_unit']:.3f}" for r in rows),
    )
    return {**report, "intercept": float(estimator.intercept_[0]), "odds_ratios": rows}
