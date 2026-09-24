"""Nodes for the 'refit' pipeline.

During modelling the artefacts are fitted on the train split only, so that
evaluation stays honest. Once the approach is validated, production artefacts
are refitted on *every* split to squeeze out the extra data.

The imputers, outlier caps, encoders and scalers need no new code — the refit
pipeline reuses the data_engineering functions with a wider ``split_to_fit``.
Only the model needs a dedicated node.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.base import clone

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

    The feature columns are read from the *production* master table: its
    one-hot columns come from the production encoders, which may know a
    category the train-only encoders never saw.

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
    feature_cols = feature_columns(master_table, target)

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
        "feature_columns": feature_cols,
        "train_splits": list(params["train_splits"]),
    }
