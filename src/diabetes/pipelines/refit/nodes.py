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

logger = logging.getLogger(__name__)


def refit_model(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    optimized_model: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Retrain the optimized model on all data splits for production use.

    Extracts the estimator class and best hyperparameters from the
    ``optimized_model`` artifact, then fits a fresh instance on the
    splits specified in ``params["train_splits"]`` (typically all splits).

    Args:
        master_table: Table processed with the *production* artefacts, with a
            ``split`` column and all feature and target columns.
        columns: Column groups dictionary with ``numerical`` and
            ``categorical`` keys defining the feature columns.
        optimized_model: Model artifact produced by
            ``optimize_hyperparameters``, containing a fitted ``estimator``
            whose class and hyperparameters are reused.
        params: Refit configuration with keys:
            - ``train_splits`` (list[str]): Split labels to train on.

    Returns:
        Production model artifact dict with keys ``estimator``,
        ``target_column``, and ``feature_columns``.
    """
    source_estimator = optimized_model["estimator"]
    target = optimized_model["target_column"]
    feature_cols = columns["numerical"] + columns["categorical"]

    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    estimator = type(source_estimator)(**source_estimator.get_params())
    estimator.fit(X_train, y_train)

    logger.info(
        "Refitted %s on %d samples (%d features) using all splits",
        type(source_estimator).__name__,
        len(X_train),
        X_train.shape[1],
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "feature_columns": feature_cols,
    }
